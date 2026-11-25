import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch

from generators.image_to_image_generator import generate_i2i
from model import LLaDAForMultiModalGeneration
from model.configuration_llada import LLaDAConfig
from model.livemgt import ConditionKVCache, PrefixLayout, _rope
from model.modeling_llada import LLaDAModelLM
from utils.edit_mask import pixel_mask_to_vq, vq_positions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model():
    torch.manual_seed(45)
    c = LLaDAConfig(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=3,
        mlp_hidden_size=128,
        vocab_size=256,
        embedding_size=256,
        max_sequence_length=128,
        block_type="llama",
        rope=True,
        rope_full_precision=True,
        weight_tying=False,
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
    return m


def input_data():
    ids = torch.arange(24, device=DEVICE)[None]
    a = torch.tensor([8, 10, 17, 22], device=DEVICE)
    return ids, PrefixLayout(7, a)


@pytest.mark.parametrize("count", [0, 1, 4, 17])
def test_dense_sparse(model, count):
    ids, _ = input_data()
    a = torch.arange(7, 24, device=DEVICE)[:count]
    l = PrefixLayout(7, a)
    with torch.no_grad():
        d = model(ids, infer=True, live_layout=l, mode="dense").logits
        s = model(ids, infer=True, live_layout=l, mode="sparse").logits
    torch.testing.assert_close(d, s, atol=3e-6, rtol=3e-5)


def test_sparse_compute_and_original_positions(model):
    ids, l = input_data()
    lengths = []
    handles = [
        b.q_proj.register_forward_pre_hook(lambda m, a: lengths.append(a[0].shape[1]))
        for b in model.model.transformer.blocks
    ]
    # Inspect position use and compare to full-sequence RoPE at selected positions.
    b = model.model.transformer.blocks[0]
    q = torch.randn(1, 4, 24, 16, device=DEVICE)
    k = torch.randn(1, 2, 24, 16, device=DEVICE)
    all_q, all_k = b.rotary_emb(q, k)
    pos = torch.cat((torch.arange(7, device=DEVICE), l.active_positions))
    sq, sk = _rope(b, q.index_select(2, pos), k.index_select(2, pos), pos, 24)
    torch.testing.assert_close(sq, all_q.index_select(2, pos))
    torch.testing.assert_close(sk, all_k.index_select(2, pos))
    with torch.no_grad():
        model(ids, infer=True, live_layout=l).logits
    for h in handles:
        h.remove()
    assert lengths == [11] * 3  # 7 prefix + 4 target, not 24


def test_prefix_isolation_and_default(model):
    ids, l = input_data()
    changed = ids.clone()
    changed[:, 7:] += 40
    mask = torch.ones(1, 1, 24, 24, dtype=torch.bool, device=DEVICE)
    mask[:, :, :7, 7:] = False
    with torch.no_grad():
        o = LLaDAModelLM.forward(model, ids, attention_bias=mask).logits
        c = LLaDAModelLM.forward(model, changed, attention_bias=mask).logits
        un = LLaDAModelLM.forward(model, ids).logits
        all_allowed = LLaDAModelLM.forward(
            model, ids, attention_bias=torch.ones_like(mask)
        ).logits
        additive = LLaDAModelLM.forward(
            model,
            ids,
            attention_bias=torch.zeros_like(mask, dtype=torch.float).masked_fill(
                ~mask, float("-inf")
            ),
        ).logits
        two_dim = LLaDAModelLM.forward(model, ids, attention_bias=mask[0, 0]).logits
        wrapper = model(ids, infer=True, attention_bias=mask).logits
    torch.testing.assert_close(o[:, :7], c[:, :7], atol=1e-6, rtol=1e-5)
    assert (o[:, 7:] - c[:, 7:]).abs().max() > 1e-3
    torch.testing.assert_close(un, all_allowed, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(o, additive, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(o, two_dim, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(o, wrapper, atol=1e-6, rtol=1e-5)


def test_cache_cfg_and_mutation(model):
    ids, l = input_data()
    cache = ConditionKVCache()
    lengths = []
    h = model.model.transformer.blocks[0].q_proj.register_forward_pre_hook(
        lambda m, a: lengths.append(a[0].shape[1])
    )
    with torch.no_grad():
        for branch in ["cond", "uncond_text", "uncond_image"]:
            branch_ids = ids.clone()
            branch_ids[:, :7] += len(branch)
            model(
                branch_ids,
                infer=True,
                live_layout=l,
                condition_cache=cache,
                branch=branch,
            ).logits
            branch_ids[:, l.active_positions] += 1
            reused = model(
                branch_ids,
                infer=True,
                live_layout=l,
                condition_cache=cache,
                branch=branch,
            ).logits
            fresh = model(branch_ids, infer=True, live_layout=l).logits
            torch.testing.assert_close(reused, fresh, atol=3e-6, rtol=3e-5)
            assert all(
                k.shape[-2] == 7 for k, v in cache.branches[branch]["layers"].values()
            )
        assert len(cache.branches) == 3
        with pytest.raises(ValueError, match="condition or model changed"):
            model(ids, infer=True, live_layout=l, condition_cache=cache, branch="cond")
    h.remove()
    assert lengths == [11, 4, 11] * 3
    cache.clear()
    assert not cache.branches


def test_pixel_mask_and_newline():
    m = torch.zeros(64, 96, dtype=torch.bool)
    m[0, 0] = True
    m[-1, -1] = True
    a = pixel_mask_to_vq(m, (4, 6))
    assert a.sum() == 2
    assert a[0, 0] and a[-1, -1]
    assert pixel_mask_to_vq(m, (4, 6), halo=1).sum() == 8
    pos = vq_positions(4, 6, 13)
    assert pos.tolist() == [13 + r * 7 + c for r in range(4) for c in range(6)]
    for h in [-1, True, 1.2]:
        with pytest.raises(ValueError):
            pixel_mask_to_vq(m, (4, 6), h)


class ToyImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, ids, **kw):
        self.calls += 1
        from transformers.modeling_outputs import CausalLMOutputWithPast

        logits = torch.zeros(*ids.shape, 134548, device=ids.device)
        logits[..., 126357] = 10
        return CausalLMOutputWithPast(logits=logits)


@pytest.mark.parametrize("fraction", [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("cfg", [(0, 0), (2.5, 0), (0, 4), (2.5, 4)])
def test_mask_sampler_edges(fraction, cfg):
    # Rectangular grid and delimiters; outside IDs are checked after each step.
    m = ToyImageModel()
    source = torch.arange(12)[None] + 126356
    tail = (
        [126354, 126349]
        + sum(([126336] * 4 + [126084] for _ in range(3)), [])
        + [126350, 126355]
    )
    prompt = torch.tensor([[10, 11, 12] + tail])
    active = torch.zeros(3, 4, dtype=torch.bool)
    active.flatten()[: round(12 * fraction)] = True
    out = generate_i2i(
        m,
        prompt,
        seq_len=12,
        newline_every=4,
        timesteps=3,
        temperature=0.0,
        cfg_scale=cfg[0],
        cfg_img=cfg[1],
        uncon_text=torch.tensor([[9, 8]]),
        uncon_image=torch.tensor([[7]]),
        code_start=5,
        text_vocab_size=126356,
        edit_mask=active,
        source_vq=source,
    )
    assert out.shape == (1, 12)
    assert torch.equal(out[:, ~active.flatten()], source[:, ~active.flatten()])
    assert (out[:, active.flatten()] == 126357).all()
    assert torch.equal(prompt, torch.tensor([[10, 11, 12] + tail]))
    if fraction == 0:
        assert m.calls == 0


def test_training_gradient_save_resume(model, tmp_path):
    model.train()
    ids, l = input_data()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    d = model(ids, infer=True, live_layout=l, mode="dense").logits
    s = model(ids, infer=True, live_layout=l, gradient_checkpointing=True).logits
    torch.testing.assert_close(d, s, atol=3e-6, rtol=3e-5)
    loss = torch.nn.functional.cross_entropy(
        s.flatten(0, 1), torch.ones(4, dtype=torch.long, device=DEVICE)
    )
    loss.backward()
    assert model.model.transformer.blocks[0].q_proj.weight.grad.norm() > 0
    opt.step()
    opt.zero_grad()
    torch.save(
        dict(model=model.state_dict(), optimizer=opt.state_dict()),
        tmp_path / "state.pt",
    )
    state = torch.load(tmp_path / "state.pt", weights_only=True)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    assert opt.state_dict()["state"]


def test_default_text_train_and_variable_lengths(model):
    # Original task wrapper loss and mixed-length padding still work.
    ids = [[1, 2, 3], [4, 5]]
    labels = [[-100, 2, 3], [-100, 5]]
    loss = model(ids, labels=labels)
    assert torch.isfinite(loss)
    loss.backward()
    with torch.no_grad():
        out = model(torch.tensor([[1, 2, 3]], device=DEVICE), infer=True).logits
    assert out.shape == (1, 3, 256) and torch.isfinite(out).all()


@pytest.mark.parametrize("prefix_len", [1, 5, 9])
def test_cache_changed_active_sets(model, prefix_len):
    ids = torch.arange(30, device=DEVICE)[None]
    cache = ConditionKVCache()
    with torch.no_grad():
        for count in [1, 5, 10, 0]:
            l = PrefixLayout(prefix_len, torch.arange(12, 12 + count, device=DEVICE))
            c = model(
                ids,
                infer=True,
                live_layout=l,
                condition_cache=cache,
                branch=str(prefix_len),
            ).logits
            s = model(ids, infer=True, live_layout=l).logits
            torch.testing.assert_close(c, s, atol=3e-6, rtol=3e-5)
    cache.clear()


@pytest.fixture(scope="module")
def image_tiny_model():
    torch.manual_seed(7)
    c = LLaDAConfig(
        d_model=16,
        n_heads=2,
        n_layers=2,
        n_kv_heads=2,
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
    return m


@pytest.mark.parametrize("count", [0, 1, 2, 4, 8, 16])
@pytest.mark.parametrize("cfg", [(0, 0), (2.5, 4)])
def test_real_prefix_sampler(image_tiny_model, count, cfg, monkeypatch):
    from model import livemgt

    cleared = []
    real_clear = livemgt.ConditionKVCache.clear

    def record_clear(self):
        real_clear(self)
        cleared.append(not self.branches)

    monkeypatch.setattr(livemgt.ConditionKVCache, "clear", record_clear)
    source = torch.arange(16, device=DEVICE)[None] + 126356
    tail = (
        [126354, 126349]
        + sum(([126336] * 4 + [126084] for _ in range(4)), [])
        + [126350, 126355]
    )
    prefix = (
        [7, 8, 126349]
        + sum(([126360] * 4 + [126084] for _ in range(4)), [])
        + [126350, 9]
    )
    active = torch.zeros(4, 4, dtype=torch.bool)
    active.flatten()[:count] = True
    outputs = []
    for mode, cached in [("dense", False), ("sparse", False), ("sparse", True)]:
        outputs.append(
            generate_i2i(
                image_tiny_model,
                torch.tensor([prefix + tail], device=DEVICE),
                seq_len=16,
                newline_every=4,
                timesteps=3,
                temperature=0.0,
                cfg_scale=cfg[0],
                cfg_img=cfg[1],
                uncon_text=torch.tensor([[10] + prefix[2:]], device=DEVICE),
                uncon_image=torch.tensor([[11, 12]], device=DEVICE),
                code_start=len(prefix) + 2,
                text_vocab_size=126356,
                edit_mask=active,
                source_vq=source,
                prefix_mode=mode,
                condition_cache=cached,
            )
        )
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[1], outputs[2])
    assert torch.equal(
        outputs[0][:, ~active.flatten()], source[:, ~active.flatten().to(DEVICE)]
    )
    assert cleared == [True]


def test_generation_cache_cleanup_on_exception(monkeypatch):
    from model import livemgt

    cleared = []

    def clear(self):
        self.branches.clear()
        cleared.append(True)

    monkeypatch.setattr(livemgt.ConditionKVCache, "clear", clear)
    with pytest.raises(ValueError):
        generate_i2i(
            ToyImageModel(),
            torch.tensor([[126336]]),
            seq_len=1,
            newline_every=1,
            timesteps=1,
            uncon_text=torch.tensor([[1]]),
            uncon_image=torch.tensor([[2]]),
            code_start=0,
            source_vq=torch.tensor([[0]]),
            edit_mask=torch.ones(1, 1),
            prefix_mode="sparse",
            condition_cache=True,
        )
    assert cleared == [True]
