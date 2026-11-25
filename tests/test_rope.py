import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pytest
import torch

from livemgt_training.predict import sequence_inputs
from model import LLaDAForMultiModalGeneration
from model.livemgt import ConditionKVCache, PrefixLayout, _rope
from model.livemgt_conditioning import bbox_size, enable_conditioning
from model.livemgt_rope import TokenTypeRotary, enable_type_rope
from utils.edit_mask import pixel_mask_to_vq, sequence_position_metadata, vq_positions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def inputs(grid=(3, 4), active=None):
    h, w = grid
    ref = [126349] + sum(([126360] * w + [126084] for _ in range(h)), []) + [126350]
    prefix = [1, 2] + ref + [3]
    p = len(prefix)
    tail = (
        [126354, 126349]
        + sum(([126336] * w + [126084] for _ in range(h)), [])
        + [126350, 126355]
    )
    # Tiny vocabulary accepts arbitrary valid IDs for these direct model tests;
    # metadata is computed before replacing IDs with [0,255] equivalents.
    ids = torch.tensor([prefix + tail], device=DEVICE)
    types, coords, version = sequence_position_metadata(ids, p, grid)
    if active is None:
        active = torch.zeros(h, w, dtype=torch.bool, device=DEVICE)
        active[0, 0] = True
        active[-1, -1] = True
    layout = PrefixLayout(
        p,
        vq_positions(h, w, p + 2, DEVICE)[active.flatten()],
        bbox_size(active),
        types,
        coords,
        version,
    )
    return ids % 256, layout


@pytest.mark.parametrize("full_precision", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_init_bit_exact_and_gradient(model, full_precision, dtype):
    model = model.to(dtype)
    ids, l = inputs()
    for b in model.model.transformer.blocks:
        b.config.rope_full_precision = full_precision
    with torch.no_grad():
        expected = model(ids, infer=True, live_layout=l).logits
    module = enable_type_rope(model)
    with torch.no_grad():
        actual = model(ids, infer=True, live_layout=l).logits
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    model.train()
    model(ids, infer=True, live_layout=l).logits.float().square().mean().backward()
    assert (
        module.phase.grad is not None
        and torch.isfinite(module.phase.grad).all()
        and module.phase.grad.norm() > 0
    )


def test_types_rotate_qk_not_v(model):
    block = model.model.transformer.blocks[0]
    module = enable_type_rope(model)
    with torch.no_grad():
        module.phase[1].fill_(0.25)
        module.phase[2].fill_(-0.3)
    q = torch.randn(1, 4, 1, 16, device=DEVICE)
    k = torch.randn(1, 2, 1, 16, device=DEVICE)
    v = k.clone()
    pos = torch.tensor([5], device=DEVICE)
    kwargs = lambda t: dict(
        mode="pretrained_type_phase",
        module=module,
        token_types=torch.full((8,), t, device=DEVICE),
        position_ids=None,
    )
    ref = _rope(block, q, k, pos, 8, kwargs(1))
    target = _rope(block, q, k, pos, 8, kwargs(2))
    assert not torch.equal(ref[0], target[0]) and not torch.equal(ref[1], target[1])
    assert torch.equal(k, v)  # no input mutation; V has no rotation call
    torch.testing.assert_close(
        ref[0].norm(dim=-1), q.norm(dim=-1), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("head_dim", [8, 16, 128])
def test_three_axis_flux_exact_reference(head_dim):
    from diffusers.models.embeddings import FluxPosEmbed, apply_rotary_emb

    m = TokenTypeRotary(head_dim).to(DEVICE)
    ids = torch.tensor([[0, 0, 0], [1, 3, 4], [2, 3, 4], [2, -1, -1]], device=DEVICE)
    sin, cos = m.three_axis_trig(ids)
    ec, es = FluxPosEmbed(theta=10000, axes_dim=list(m.axes_dims)).to(DEVICE)(ids)
    torch.testing.assert_close(sin[0, 0], es, atol=0, rtol=0)
    torch.testing.assert_close(cos[0, 0], ec, atol=0, rtol=0)
    x = torch.randn(1, 2, 4, head_dim, device=DEVICE)
    torch.testing.assert_close(
        m.three_axis_rotate(x, sin, cos), apply_rotary_emb(x, (ec, es)), atol=0, rtol=0
    )
    assert sum(m.axes_dims) == head_dim and all(d % 2 == 0 for d in m.axes_dims)
    if head_dim == 128:
        assert m.axes_dims == (16, 56, 56)


@pytest.mark.parametrize("mode", ["pretrained_type_phase", "three_axis"])
@pytest.mark.parametrize("grid", [(3, 4), (4, 6)])
@pytest.mark.parametrize("halo", [0, 1])
def test_dense_sparse_cache_original_positions(model, mode, grid, halo):
    active = torch.zeros(grid, dtype=torch.bool, device=DEVICE)
    active[1, 1] = True
    active = pixel_mask_to_vq(active, grid, halo)
    ids, l = inputs(grid, active)
    module = enable_type_rope(model, mode=mode)
    with torch.no_grad():
        module.phase.normal_(std=0.05)
    cache = ConditionKVCache()
    shapes = []
    h = model.model.transformer.blocks[0].q_proj.register_forward_pre_hook(
        lambda m, a: shapes.append(a[0].shape[1])
    )
    with torch.no_grad():
        dense = model(ids, infer=True, live_layout=l, mode="dense").logits
        sparse = model(ids, infer=True, live_layout=l).logits
        model(ids, infer=True, live_layout=l, condition_cache=cache)
        cached = model(ids, infer=True, live_layout=l, condition_cache=cache).logits
    torch.testing.assert_close(dense, sparse, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(sparse, cached, atol=3e-6, rtol=3e-5)
    assert shapes == [
        ids.shape[1],
        l.prefix_length + int(active.sum()),
        l.prefix_length + int(active.sum()),
        int(active.sum()),
    ]
    assert (
        l.active_positions.tolist()
        == vq_positions(*grid, l.prefix_length + 2, DEVICE)[active.flatten()].tolist()
    )
    assert l.position_ids[l.active_positions, 1:].tolist() == active.nonzero().tolist()
    h.remove()


def test_cache_invalidations_and_prefix_independence(model):
    ids, l = inputs()
    module = enable_type_rope(model)
    enable_conditioning(model)
    with torch.no_grad():
        module.phase.normal_(std=0.1)
        model.live_conditioning.region_mlp[-1].weight.normal_(std=0.01)

    def fresh():
        c = ConditionKVCache()
        model(ids, infer=True, live_layout=l, condition_cache=c)
        return c

    with torch.no_grad():
        c = fresh()
        old_k = c.branches[("cond", True)]["layers"][0][0].clone()
        l.region_size = (2, 4)
        model(ids, infer=True, live_layout=l, condition_cache=c)
        torch.testing.assert_close(
            old_k, c.branches[("cond", True)]["layers"][0][0], atol=0, rtol=0
        )
        with pytest.raises(ValueError):
            model(
                ids,
                infer=True,
                live_layout=l,
                condition_cache=c,
                rope_mode="three_axis",
            )
        c = fresh()
        module.phase.add_(0.1)
        with pytest.raises(ValueError):
            model(ids, infer=True, live_layout=l, condition_cache=c)
        c = fresh()
        l.token_types = l.token_types.clone()
        l.token_types[0] = 1
        with pytest.raises(ValueError):
            model(ids, infer=True, live_layout=l, condition_cache=c)
        l.token_types[0] = 0
        c = fresh()
        l.mapping_version = "unsupported"
        with pytest.raises(ValueError):
            model(ids, infer=True, live_layout=l, condition_cache=c)


@pytest.mark.parametrize("branch", ["cond", "uncond_text", "uncond_image"])
def test_shared_mapping_training_cfg_and_specials(branch):
    class Tok:
        def __init__(self, text):
            self.input_ids = [1, 2, 3] if "<uncondition>" not in text else [4, 5]

    source = torch.full((1, 3, 4), 126360, device=DEVICE)
    mask = torch.ones_like(source, dtype=torch.bool)
    batch = dict(source=source, original=mask, instruction="edit")
    ids, l = sequence_inputs(Tok, batch, source, mask, branch)
    t, c, v = sequence_position_metadata(ids, l.prefix_length, (3, 4))
    assert torch.equal(t, l.token_types) and torch.equal(c, l.position_ids)
    if branch == "uncond_image":
        assert (t[: l.prefix_length] == 0).all()
    else:
        assert (t == 1).sum() == 3 * 5 + 2
        ref = (t == 1).nonzero().flatten()
        assert c[ref[0]].tolist() == [1, -1, -1]
        assert c[ref[-1]].tolist() == [1, 3, 4]
    assert c[l.prefix_length].tolist() == [2, -2, -2]
    assert c[l.prefix_length + 1].tolist() == [2, -1, -1]
    assert c[-2].tolist() == [2, 3, 4] and c[-1].tolist() == [2, 4, 4]
    assert (c[l.active_positions, 1:] == mask[0].nonzero()).all()


@pytest.mark.parametrize("mode", ["pretrained_type_phase", "three_axis"])
@pytest.mark.parametrize("shape", ["empty", "full", "horizontal", "vertical"])
def test_mask_shapes_empty_full(model, mode, shape):
    active = torch.zeros(3, 4, dtype=torch.bool, device=DEVICE)
    if shape == "full":
        active[:] = True
    if shape == "horizontal":
        active[1, :3] = True
    if shape == "vertical":
        active[:, 1] = True
    ids, l = inputs(active=active)
    enable_type_rope(model, mode=mode)
    with torch.no_grad():
        out = model(ids, infer=True, live_layout=l).logits
    assert out.shape == (1, int(active.sum()), 256) and torch.isfinite(out).all()


def test_hf_roundtrip_and_refuse_discard(model, tmp_path):
    ids, l = inputs()
    phase = enable_type_rope(model)
    with torch.no_grad():
        phase.phase.normal_(std=0.1)
    model.save_pretrained(tmp_path)
    restored = LLaDAForMultiModalGeneration.from_pretrained(tmp_path).to(DEVICE).eval()
    with torch.no_grad():
        a = model(ids, infer=True, live_layout=l).logits
        b = restored(ids, infer=True, live_layout=l).logits
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg.pop("live_mgt_rope")
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="refusing to discard"):
        LLaDAForMultiModalGeneration.from_pretrained(tmp_path)


def test_old_hf_config_new_phase_missing_initializes_zero(model, tmp_path):
    model.save_pretrained(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["live_mgt_rope"] = dict(
        enabled=True,
        mode="pretrained_type_phase",
        axes_dims=[2, 6, 8],
        theta=10000.0,
        mapping_version="lumina_roles_v1",
    )
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    restored, info = LLaDAForMultiModalGeneration.from_pretrained(
        tmp_path, output_loading_info=True
    )
    assert "live_type_rope.phase" in info["missing_keys"]
    assert restored.live_type_rope.phase.count_nonzero() == 0


def test_prefix_kv_same_area_different_bbox_nonzero_phase(model):
    module = enable_type_rope(model)
    condition = enable_conditioning(model)
    with torch.no_grad():
        module.phase.normal_(std=0.1)
        condition.region_mlp[-1].weight.normal_(std=0.03)
    masks = []
    a = torch.zeros(4, 6, dtype=torch.bool, device=DEVICE)
    a[1, 1:5] = True
    masks.append(a)
    a = torch.zeros_like(a)
    a[:, 2] = True
    masks.append(a)
    caches = []
    with torch.no_grad():
        for mask in masks:
            ids, l = inputs((4, 6), mask)
            cache = ConditionKVCache()
            model(ids, infer=True, live_layout=l, condition_cache=cache)
            caches.append(cache)
    for layer in range(3):
        for a, b in zip(
            caches[0].branches[("cond", True)]["layers"][layer],
            caches[1].branches[("cond", True)]["layers"][layer],
        ):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_cache_enable_disable_and_coordinates(model):
    ids, l = inputs()
    enable_type_rope(model)
    cache = ConditionKVCache()
    with torch.no_grad():
        model(
            ids, infer=True, live_layout=l, condition_cache=cache, token_type_rope=False
        )
        with pytest.raises(ValueError):
            model(
                ids,
                infer=True,
                live_layout=l,
                condition_cache=cache,
                token_type_rope=True,
            )
        cache.clear()
        model(
            ids,
            infer=True,
            live_layout=l,
            condition_cache=cache,
            rope_mode="three_axis",
        )
        l.position_ids = l.position_ids.clone()
        l.position_ids[0, 1] += 1
        with pytest.raises(ValueError):
            model(
                ids,
                infer=True,
                live_layout=l,
                condition_cache=cache,
                rope_mode="three_axis",
            )


def test_sampler_and_training_metadata_identical():
    from generators.image_to_image_generator import generate_i2i

    class Tok:
        def __init__(self, text):
            self.input_ids = [1, 2, 3] if "<uncondition>" not in text else [4, 5]

    source = torch.full((1, 3, 4), 126360, device=DEVICE)
    region = torch.ones_like(source, dtype=torch.bool)
    batch = dict(source=source, original=region, instruction="edit")
    built = {
        b: sequence_inputs(Tok, batch, torch.full_like(source, 126336), region, b)
        for b in ["cond", "uncond_text", "uncond_image"]
    }

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1, device=DEVICE))
            self.seen = {}

        def forward(self, ids, **kw):
            from transformers.modeling_outputs import CausalLMOutputWithPast

            l = kw["live_layout"]
            self.seen[kw["branch"]] = l
            return CausalLMOutputWithPast(
                logits=torch.zeros(1, l.active_positions.numel(), 8192, device=DEVICE)
            )

    spy = Spy()
    ids, l = built["cond"]
    ut, lt = built["uncond_text"]
    ui, li = built["uncond_image"]
    generate_i2i(
        spy,
        ids,
        seq_len=12,
        newline_every=4,
        timesteps=1,
        temperature=0,
        cfg_scale=2.5,
        cfg_img=4,
        uncon_text=ut[:, : lt.prefix_length],
        uncon_image=ui[:, : li.prefix_length],
        code_start=l.prefix_length + 2,
        text_vocab_size=126356,
        edit_mask=region[0],
        source_vq=source,
        prefix_mode="sparse",
    )
    for branch in built:
        expected = built[branch][1]
        actual = spy.seen[branch]
        assert torch.equal(expected.token_types, actual.token_types) and torch.equal(
            expected.position_ids, actual.position_ids
        )


def test_hf_bf16_keeps_phase_fp32(model, tmp_path):
    enable_type_rope(model)
    with torch.no_grad():
        model.live_type_rope.phase.fill_(0.012345678)
    model.save_pretrained(tmp_path)
    restored = LLaDAForMultiModalGeneration.from_pretrained(
        tmp_path, torch_dtype=torch.bfloat16
    )
    assert restored.live_type_rope.phase.dtype == torch.float32
    torch.testing.assert_close(
        restored.live_type_rope.phase, model.live_type_rope.phase.cpu(), atol=0, rtol=0
    )


def test_shared_three_branch_cache_nonzero_phase(model):
    class Tok:
        def __init__(self, text):
            self.input_ids = [1, 2, 3] if "<uncondition>" not in text else [4, 5]

    source = torch.full((1, 3, 4), 126360, device=DEVICE)
    mask = torch.ones_like(source, dtype=torch.bool)
    batch = dict(source=source, original=mask, instruction="edit")
    m = enable_type_rope(model)
    with torch.no_grad():
        m.phase.normal_(std=0.05)
    cache = ConditionKVCache()
    with torch.no_grad():
        for branch in ["cond", "uncond_text", "uncond_image"]:
            ids, l = sequence_inputs(Tok, batch, source, mask, branch)
            ids = ids % 256
            model(ids, infer=True, live_layout=l, condition_cache=cache, branch=branch)
            cached = model(
                ids, infer=True, live_layout=l, condition_cache=cache, branch=branch
            ).logits
            direct = model(ids, infer=True, live_layout=l, branch=branch).logits
            torch.testing.assert_close(cached, direct, atol=3e-6, rtol=3e-5)
    assert set(cache.branches) == {"cond", "uncond_text", "uncond_image"}
    assert cache.branches["uncond_image"]["prefix_types"].eq(0).all()


def test_cache_rejects_phase_data_mutation_bypassing_tensor_version(model):
    ids, l = inputs()
    module = enable_type_rope(model)
    cache = ConditionKVCache()
    with torch.no_grad():
        model(ids, infer=True, live_layout=l, condition_cache=cache)
    version = module.phase._version
    module.phase.data.fill_(0.1)
    assert module.phase._version == version
    with torch.no_grad(), pytest.raises(ValueError, match="phase parameter changed"):
        model(ids, infer=True, live_layout=l, condition_cache=cache)


def test_compatible_small_head_does_not_require_three_axis_split():
    module = TokenTypeRotary(4).to(DEVICE)
    sin = torch.zeros(1, 1, 3, 4, device=DEVICE)
    cos = torch.ones_like(sin)
    types = torch.arange(3, device=DEVICE)
    new_sin, new_cos = module.compatible_trig(sin, cos, types)
    torch.testing.assert_close(sin, new_sin, atol=0, rtol=0)
    torch.testing.assert_close(cos, new_cos, atol=0, rtol=0)
    new_sin.sum().backward()
    assert (module.phase.grad != 0).all()
    with pytest.raises(ValueError, match="too small"):
        module.three_axis_trig(torch.zeros(3, 3, device=DEVICE))


def test_multistep_sampler_nonzero_rotary_dense_sparse_cached():
    from generators.image_to_image_generator import generate_i2i
    from model.configuration_llada import LLaDAConfig

    torch.manual_seed(17)
    c = LLaDAConfig(
        d_model=16,
        n_heads=2,
        n_kv_heads=2,
        n_layers=2,
        mlp_hidden_size=32,
        vocab_size=134548,
        embedding_size=134548,
        max_sequence_length=128,
        block_type="llama",
        rope=True,
        weight_tying=True,
        activation_type="silu",
        layer_norm_type="rms",
        include_bias=False,
        flash_attention=False,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
    )
    m = LLaDAForMultiModalGeneration(c).to(DEVICE).eval()
    m.model.reset_parameters()
    module = enable_type_rope(m)
    with torch.no_grad():
        module.phase.normal_(std=0.05)

    class Tok:
        def __init__(self, text):
            self.input_ids = [1, 2, 3] if "<uncondition>" not in text else [4, 5]

    source = torch.arange(12, device=DEVICE).reshape(1, 3, 4) + 126356
    region = torch.zeros_like(source, dtype=torch.bool)
    region[:, 0, [0, 3]] = True
    region[:, -1, [0, 3]] = True
    batch = dict(source=source, original=region, instruction="edit")
    masked = torch.where(region, 126336, source)
    built = {
        b: sequence_inputs(Tok, batch, masked, region, b)
        for b in ["cond", "uncond_text", "uncond_image"]
    }
    ids, l = built["cond"]
    ut, lt = built["uncond_text"]
    ui, li = built["uncond_image"]
    outputs = []
    for mode, cached in [("dense", False), ("sparse", False), ("sparse", True)]:
        out = generate_i2i(
            m,
            ids,
            seq_len=12,
            newline_every=4,
            timesteps=3,
            temperature=0,
            cfg_scale=2.5,
            cfg_img=4,
            uncon_text=ut[:, : lt.prefix_length],
            uncon_image=ui[:, : li.prefix_length],
            code_start=l.prefix_length + 2,
            text_vocab_size=126356,
            edit_mask=region[0],
            source_vq=source,
            prefix_mode=mode,
            condition_cache=cached,
        )
        assert torch.equal(
            out[:, ~region.flatten()], source.flatten(1)[:, ~region.flatten()]
        )
        outputs.append(out)
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[1], outputs[2])


def test_image_null_branch_does_not_invent_reference_from_text_specials():
    class Tok:
        def __init__(self, text):
            self.input_ids = [1, 126349, 126350, 2]

    source = torch.full((1, 3, 4), 126360, device=DEVICE)
    region = torch.ones_like(source, dtype=torch.bool)
    batch = dict(
        source=source, original=region, instruction="literal image markers in text"
    )
    ids, l = sequence_inputs(Tok, batch, source, region, "uncond_image")
    assert l.token_types[: l.prefix_length].eq(0).all()
    assert l.position_ids[: l.prefix_length].eq(0).all()
