"""Full condition prefix + sparse target for the pretrained LLaDA Llama blocks.

Conditioning modules are registered separately from the pretrained Transformer.
Dense mode is the numerical reference:
all input queries are evaluated, but inactive target KEYS are excluded. Sparse
mode removes inactive queries before embedding/projections/FFN and keeps the
same original absolute RoPE positions. Both return only active-target logits.
"""

from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class PrefixLayout:
    prefix_length: int
    active_positions: torch.Tensor  # absolute positions in THIS CFG branch
    # Conditioning metadata: bbox before halo/pseudo-mask and original sequence type IDs.
    region_size: object = None
    token_types: object = None
    position_ids: object = None
    mapping_version: str = "lumina_roles_v1"

    def validate(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "Prefix editing supports one sample per forward; loop over variable-size samples"
            )
        if not 0 < self.prefix_length <= input_ids.shape[1]:
            raise ValueError("invalid prefix length")
        a = self.active_positions
        if a.dtype != torch.long or a.ndim != 1 or a.device != input_ids.device:
            raise ValueError("active_positions must be device-local 1D int64")
        if a.numel() and (
            a.min() < self.prefix_length
            or a.max() >= input_ids.shape[1]
            or not bool((a[1:] > a[:-1]).all())
        ):
            raise ValueError(
                "active target positions must be sorted, unique and outside prefix"
            )


@dataclass
class ConditionKVCache:
    """Generation-local cache, keyed independently per CFG branch.

    K is stored AFTER RoPE; V before attention. Only prefix slices are cloned,
    so no target backing storage is retained. Prefix identity and model version
    are checked on each reuse; callers must clear in a finally block.
    """

    branches: dict = field(default_factory=dict)

    def clear(self):
        self.branches.clear()

    def prepare(self, branch, model, prefix, semantics=None):
        signature = (
            id(model),
            str(prefix.device),
            str(next(model.parameters()).dtype),
            tuple((id(p), p._version) for p in model.parameters()),
            semantics,
        )
        old = self.branches.get(branch)
        phase = getattr(getattr(model, "live_type_rope", None), "phase", None)
        if (
            old is not None
            and phase is not None
            and old.get("rotary_phase_snapshot") is not None
        ):
            captured = old["rotary_phase_snapshot"]
            if (
                captured.dtype != phase.dtype
                or captured.device != phase.device
                or captured.shape != phase.shape
                or not torch.equal(captured, phase)
            ):
                raise ValueError(
                    "rotary phase parameter changed: clear condition cache"
                )
        if old is not None and (
            old["signature"] != signature or not torch.equal(old["prefix"], prefix)
        ):
            raise ValueError(
                "condition or model changed: clear cache or use a distinct CFG branch key"
            )
        if old is None:
            old = dict(
                signature=signature,
                prefix=prefix.detach().clone(),
                layers={},
                ready=False,
            )
            if phase is not None:
                old["rotary_phase_snapshot"] = phase.detach().clone()
            self.branches[branch] = old
        return old


def _rope(block, q, k, positions, rope_length, rotary=None):
    # Preserve Lumina's full-precision RoPE formula, indexing original positions
    # for BOTH Q and current K; cached K must never be rotated a second time.
    if rotary is not None and rotary["mode"] == "three_axis":
        module = rotary["module"]
        sin, cos = module.three_axis_trig(
            rotary["position_ids"].index_select(0, positions)
        )
        return module.three_axis_rotate(q, sin, cos), module.three_axis_rotate(
            k, sin, cos
        )
    original_q, original_k = q.dtype, k.dtype
    if block.config.rope_full_precision:
        q, k = q.float(), k.float()
    sin, cos = block.rotary_emb.get_rotary_embedding(rope_length, q.device)
    sin = sin.index_select(-2, positions).to(q.dtype)
    cos = cos.index_select(-2, positions).to(q.dtype)
    if rotary is not None and rotary["mode"] == "pretrained_type_phase":
        types = rotary["token_types"].index_select(0, positions)
        sin, cos = rotary["module"].compatible_trig(sin, cos, types)
    q = block.rotary_emb.apply_rotary_pos_emb(sin, cos, q)
    k = block.rotary_emb.apply_rotary_pos_emb(sin, cos, k)
    return q.to(original_q), k.to(original_k)


def prefix_block(
    block,
    x,
    positions,
    prefix_length,
    active_positions,
    dense,
    entry=None,
    layer=0,
    rope_length=None,
    rotary=None,
):
    B, T, D = x.shape
    normalized = block.attn_norm(x)
    q, k, v = (
        block.q_proj(normalized),
        block.k_proj(normalized),
        block.v_proj(normalized),
    )
    if block.q_norm is not None and block.k_norm is not None:
        q, k = block.q_norm(q).to(k.dtype), block.k_norm(k).to(k.dtype)
    hd = D // block.config.n_heads
    q = q.view(B, T, block.config.n_heads, hd).transpose(1, 2)
    k = k.view(B, T, block.config.effective_n_kv_heads, hd).transpose(1, 2)
    v = v.view(B, T, block.config.effective_n_kv_heads, hd).transpose(1, 2)
    q, k = (
        _rope(block, q, k, positions, rope_length)
        if rotary is None
        else _rope(block, q, k, positions, rope_length, rotary)
    )
    reused = entry is not None and entry["ready"]
    if reused:
        pk, pv = entry["layers"][layer]
        if pk.device != k.device or pk.dtype != k.dtype or pv.dtype != v.dtype:
            raise ValueError("condition cache device/dtype mismatch")
        k = torch.cat((pk, k), dim=-2)
        v = torch.cat((pv, v), dim=-2)
    elif entry is not None:
        entry["layers"][layer] = (
            k[:, :, :prefix_length].detach().clone(),
            v[:, :, :prefix_length].detach().clone(),
        )
    dropout = block.config.attention_dropout if block.training else 0.0
    if dense:
        # Evaluate ALL queries/projections/FFNs, gathering effective keys only.
        # Split SDPA avoids a dense mask forcing a different CUDA kernel from
        # sparse mode; this is still dense Transformer token computation.
        key_positions = torch.cat(
            (torch.arange(prefix_length, device=x.device), active_positions)
        )
        effective_k = k.index_select(2, key_positions)
        effective_v = v.index_select(2, key_positions)
        pa = block._scaled_dot_product_attention(
            q[:, :, :prefix_length],
            k[:, :, :prefix_length],
            v[:, :, :prefix_length],
            dropout_p=dropout,
        )
        ta = block._scaled_dot_product_attention(
            q[:, :, prefix_length:], effective_k, effective_v, dropout_p=dropout
        )
        att = torch.cat((pa, ta), dim=-2)
    elif reused:
        att = block._scaled_dot_product_attention(q, k, v, dropout_p=dropout)
    else:
        # Split calls exactly express prefix isolation without quadratic masks.
        pa = block._scaled_dot_product_attention(
            q[:, :, :prefix_length],
            k[:, :, :prefix_length],
            v[:, :, :prefix_length],
            dropout_p=dropout,
        )
        ta = (
            block._scaled_dot_product_attention(
                q[:, :, prefix_length:], k, v, dropout_p=dropout
            )
            if T > prefix_length
            else q[:, :, prefix_length:]
        )
        att = torch.cat((pa, ta), dim=-2)
    x = x + block.dropout(
        block.attn_out(att.transpose(1, 2).contiguous().view(B, T, D))
    )
    residual = x
    normalized = block.ff_norm(x)
    x = block.act(block.ff_proj(normalized)) * block.up_proj(normalized)
    return residual + block.dropout(block.ff_out(x))


def forward_prefix(
    wrapper,
    input_ids,
    layout,
    mode="sparse",
    condition_cache=None,
    branch="cond",
    vocab_slice=None,
    gradient_checkpointing=False,
    region_conditioning=None,
    token_type_encoding=None,
    token_type_embedding=None,
    token_type_rope=None,
    rope_mode=None,
):
    """Return active logits in layout order; scatter is performed by sampler.

    Dense and sparse paths have identical effective attention relationships.
    Cache is inference-only and independent of Lumina's Max Logit cache.
    """
    layout.validate(input_ids)
    if mode not in ("dense", "sparse"):
        raise ValueError("mode must be dense or sparse")
    core = wrapper.model
    if core.config.block_group_size != 1 or str(core.config.block_type) != "llama":
        raise NotImplementedError(
            "Prefix editing requires the official ungrouped Llama blocks"
        )
    if condition_cache is not None and (
        wrapper.training or torch.is_grad_enabled() or mode != "sparse"
    ):
        raise ValueError("condition cache requires eval + no_grad + sparse mode")
    n = input_ids.shape[1]
    p = layout.prefix_length
    a = layout.active_positions
    spec = getattr(wrapper.config, "live_mgt_conditioning", {})
    region_enabled = (
        spec.get("region", False)
        if region_conditioning is None
        else region_conditioning
    )
    if token_type_embedding is not None:
        if (
            token_type_encoding is not None
            and token_type_encoding != token_type_embedding
        ):
            raise ValueError("conflicting embedding aliases")
        token_type_encoding = token_type_embedding
    type_enabled = (
        spec.get("token_type", False)
        if token_type_encoding is None
        else token_type_encoding
    )
    rope_spec = getattr(wrapper.config, "live_mgt_rope", {})
    rope_enabled = (
        rope_spec.get("enabled", False) if token_type_rope is None else token_type_rope
    )
    rope_mode = (
        rope_spec.get("mode", "pretrained_type_phase")
        if rope_mode is None
        else rope_mode
    )
    if rope_mode not in ("original", "pretrained_type_phase", "three_axis"):
        raise ValueError("unknown rope mode")
    rotary = None
    if rope_enabled and rope_mode != "original":
        if not hasattr(wrapper, "live_type_rope"):
            raise ValueError("enable_type_rope() required")
        if layout.token_types is None:
            raise ValueError("rotary token types required")
        if (
            layout.token_types.shape != (n,)
            or (layout.token_types < 0).any()
            or (layout.token_types > 2).any()
        ):
            raise ValueError("invalid rotary type IDs")
        if layout.mapping_version != "lumina_roles_v1":
            raise ValueError("unknown role mapping version")
        if rope_mode == "three_axis" and (
            layout.position_ids is None or layout.position_ids.shape != (n, 3)
        ):
            raise ValueError("three-axis coordinate metadata required")
        if rope_mode == "three_axis" and not torch.equal(
            layout.position_ids[:, 0], layout.token_types
        ):
            raise ValueError("coordinate type axis and token_types disagree")
        rotary = dict(
            mode=rope_mode,
            module=wrapper.live_type_rope,
            token_types=layout.token_types,
            position_ids=layout.position_ids,
        )
    semantics = (
        bool(rope_enabled),
        rope_mode if rope_enabled else "original",
        layout.mapping_version,
        tuple(rope_spec.get("axes_dims", ())),
        rope_spec.get("theta", 10000.0),
        (tuple(wrapper.live_type_rope.axes_dims), wrapper.live_type_rope.theta)
        if hasattr(wrapper, "live_type_rope")
        else None,
    )
    if (region_enabled or type_enabled) and not hasattr(wrapper, "live_conditioning"):
        raise ValueError("enable_conditioning() required before conditioned forward")
    # Region size is target-only; type changes may affect prefix and must
    # invalidate the prefix cache even when token IDs remain unchanged.
    cache_branch = (branch, bool(type_enabled))
    if (type_enabled or rotary is not None) and layout.token_types is None:
        raise ValueError("Conditioning requires token_types metadata")
    if condition_cache is not None and (type_enabled or rotary is not None):
        prefix_types = layout.token_types[:p].to(input_ids.device)
        old = condition_cache.branches.get(cache_branch if type_enabled else branch)
        if old is not None and (
            old.get("prefix_types") is None
            or not torch.equal(old["prefix_types"], prefix_types)
        ):
            raise ValueError("prefix token types changed: clear condition cache")
        if (
            old is not None
            and rope_mode == "three_axis"
            and rotary is not None
            and (
                old.get("prefix_coordinates") is None
                or not torch.equal(old["prefix_coordinates"], layout.position_ids[:p])
            )
        ):
            raise ValueError("prefix rotary coordinates changed: clear condition cache")
    entry = (
        condition_cache.prepare(
            cache_branch if type_enabled else branch,
            wrapper,
            input_ids[:, :p],
            semantics,
        )
        if condition_cache is not None
        else None
    )
    if entry is not None and (type_enabled or rotary is not None):
        entry["prefix_types"] = prefix_types.detach().clone()
    if entry is not None and rotary is not None and rope_mode == "three_axis":
        entry["prefix_coordinates"] = layout.position_ids[:p].detach().clone()
    reused = entry is not None and entry["ready"]
    if mode == "dense":
        positions = torch.arange(n, device=input_ids.device)
    elif reused:
        positions = a
    else:
        positions = torch.cat((torch.arange(p, device=input_ids.device), a))
    # Empty edits produce no predictions; do not run any Transformer blocks.
    if not a.numel():
        head = (
            core.transformer.wte.weight
            if core.config.weight_tying
            else core.transformer.ff_out.weight
        )
        size = head.shape[0] if vocab_slice is None else vocab_slice[1] - vocab_slice[0]
        return CausalLMOutputWithPast(logits=head.new_empty((1, 0, size)))
    x = core.transformer.wte(input_ids.index_select(1, positions))
    if core.config.input_emb_norm:
        x = x * (core.config.d_model**0.5)
    if region_enabled or type_enabled:
        x = wrapper.live_conditioning(
            x, positions, p, layout, region_enabled, type_enabled
        )
    x = core.transformer.emb_drop(x)
    for i, block in enumerate(core.transformer.blocks):
        if gradient_checkpointing and wrapper.training:
            # Capture per-iteration block/index; no cache mutation in training.
            def run(t, block=block, i=i):
                return block(
                    t,
                    live_context=(positions, p, a, mode == "dense", None, i, n, rotary),
                )[0]

            x = checkpoint(run, x, use_reentrant=False)
        else:
            x = block(
                x, live_context=(positions, p, a, mode == "dense", entry, i, n, rotary)
            )[0]
    if entry is not None:
        entry["ready"] = True
    # Project only active rows even in dense reference mode.
    x = x.index_select(1, a) if mode == "dense" else (x if reused else x[:, p:])
    x = core.transformer.ln_f(x)
    weight = (
        core.transformer.wte.weight
        if core.config.weight_tying
        else core.transformer.ff_out.weight
    )
    bias = None if core.config.weight_tying else core.transformer.ff_out.bias
    if vocab_slice is not None:
        lo, hi = vocab_slice
        weight = weight[lo:hi]
        if bias is not None:
            bias = bias[lo:hi]
    logits = F.linear(x, weight, bias)
    if core.config.scale_logits:
        logits = logits * (1 / core.config.d_model**0.5)
    return CausalLMOutputWithPast(logits=logits)
