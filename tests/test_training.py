"""Core checks for the production precision, sampler and checkpoint policies."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference.edit import edit_tokens
from livemgt_training.core import generation_region, gt_ce
from livemgt_training.full import checkpoint, data
from livemgt_training.full.config import load, resolve_batch, validate
from livemgt_training.full.validation import independent_rng
from livemgt_training.precision import autocast, compute_dtype, configure

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load(ROOT / "configs/formal/sft.json")


def test_formal_defaults_and_batch():
    c = validate(config(), 4)
    assert c["resolution"] == c["validation_resolution"] == 512
    assert (
        c["mixed_precision"] == "no"
        and c["inference_dtype"] == c["full_export_dtype"] == "float32"
    )
    assert c["sft_updates"] == 20000 and c["dimo_updates"] == 6000 and c["halo"] == 1
    assert compute_dtype(c) == torch.float32 and c["resolved_global_batch_size"] == 32
    c["global_batch_size"] = 64
    assert resolve_batch(c, 4)["gradient_accumulation"] == 16
    c["global_batch_size"] = 65
    with pytest.raises(ValueError, match="divisible"):
        resolve_batch(c, 4)
    c = config()
    c["mixed_precision"] = "bf16"
    c["inference_dtype"] = "bfloat16"
    assert compute_dtype(validate(c, 2)) == torch.bfloat16


def test_fp32_has_no_autocast_or_tf32():
    configure(False)
    assert (
        not torch.backends.cuda.matmul.allow_tf32
        and not torch.backends.cudnn.allow_tf32
    )
    with autocast(config()):
        assert not torch.is_autocast_enabled()


def test_gt_ce_only_M():
    m = torch.zeros(1, 4, 5, dtype=torch.bool)
    m[:, 1, 2] = True
    assert generation_region(m, True, 1).sum() == 9
    logits = torch.randn(1, 4, 5, 8192, requires_grad=True)
    gt_ce(logits, torch.full((1, 4, 5), 126356), m).backward()
    assert logits.grad[m].count_nonzero() and not logits.grad[~m].count_nonzero()


def test_manifest_replay_and_rank_slices(tmp_path, monkeypatch):
    rows = [
        dict(dataset_name=k, sample_uid=f"{k}:{i}")
        for k in ("interedit", "magicbrush", "maskedit")
        for i in range(32)
    ]
    manifest = tmp_path / "data.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = config()
    c.update(
        train_manifest=str(manifest),
        validation_manifest=str(manifest),
        data_mode="mixture",
        data_mixture={"interedit": 0.9, "magicbrush": 0.05, "maskedit": 0.05},
    )
    monkeypatch.setattr(
        data,
        "encode",
        lambda row, *a: dict(
            uid=row["sample_uid"], original=torch.ones(1, 2, 2, dtype=torch.bool)
        ),
    )
    streams = [data.Stream(c, None, r, 2) for r in range(2)]
    seen = [set(), set()]
    for _ in range(128):
        for r, s in enumerate(streams):
            seen[r].add(s.next()["uid"])
    assert not seen[0] & seen[1]
    for r, s in enumerate(streams):
        resumed = data.Stream(c, None, r, 2, s.cursor)
        assert [resumed.next()["uid"] for _ in range(20)] == [
            s.next()["uid"] for _ in range(20)
        ]
    vals = [data.Stream(c, None, r, 2, validation=True) for r in range(2)]
    assert vals[0].next()["uid"] == vals[1].next()["uid"]


def test_retention_only_after_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda: None)
    for i in (1, 2, 3, 4):
        folder = tmp_path / f"checkpoint-{i}"
        folder.mkdir()
        (folder / "COMPLETE").write_text("done")
    (tmp_path / "checkpoint-5").mkdir()
    checkpoint.prune(tmp_path, keep=2, milestones=[1])
    assert {p.name for p in tmp_path.iterdir()} == {
        "checkpoint-1",
        "checkpoint-3",
        "checkpoint-4",
        "checkpoint-5",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG stream")
def test_validation_rng_isolation():
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    expected = (random.random(), float(np.random.rand()), torch.rand(2, device="cuda"))
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    with independent_rng(500, torch.device("cuda", 0)):
        random.random()
        np.random.rand()
        torch.rand(100, device="cuda")
    actual = (random.random(), float(np.random.rand()), torch.rand(2, device="cuda"))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)


def test_dimo_conditional_only_and_steps(monkeypatch):
    import inference.edit as edit

    source = torch.full((1, 2, 2), 126360, dtype=torch.long)
    batch = dict(
        source=source,
        original=torch.ones_like(source, dtype=torch.bool),
        instruction="change",
    )

    def ids(*a):
        return torch.ones(1, 20, dtype=torch.long), SimpleNamespace(prefix_length=8)

    monkeypatch.setattr(edit, "sequence_inputs", ids)
    received = {}

    def sample(model, ids, **kwargs):
        received.update(kwargs)
        return source.flatten(1)

    monkeypatch.setattr(edit, "generate_i2i", sample)
    edit_tokens(None, None, batch, phase="dimo", text_cfg=2.5, image_cfg=4)
    assert (
        received["timesteps"] == 1 and received["cfg_scale"] == received["cfg_img"] == 0
    )
    edit_tokens(None, None, batch, phase="sft")
    assert (
        received["timesteps"] == 18
        and received["cfg_scale"] == 2.5
        and received["cfg_img"] == 4
    )
    with pytest.raises(ValueError):
        edit_tokens(None, None, batch, phase="dimo", steps=18)


def test_fp32_flash_dispatch_uses_sdpa():
    from model.modeling_llada import LLaDALlamaBlock

    def unsupported(*args, **kwargs):
        raise AssertionError("FP32 passed to Flash Attention")

    block = SimpleNamespace(flash_attn_func=unsupported)
    q = torch.randn(1, 2, 4, 8)
    actual = LLaDALlamaBlock._scaled_dot_product_attention(block, q, q, q)
    expected = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_master_export_meta_materialization():
    from accelerate import init_empty_weights

    from model import LLaDAForMultiModalGeneration
    from model.configuration_llada import LLaDAConfig
    from model.livemgt_conditioning import enable_conditioning
    from model.livemgt_rope import enable_type_rope

    config = LLaDAConfig(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        mlp_hidden_size=64,
        vocab_size=256,
        embedding_size=256,
        block_type="llama",
        rope=True,
        weight_tying=False,
        activation_type="silu",
        layer_norm_type="rms",
        include_bias=False,
    )
    original = LLaDAForMultiModalGeneration(config)
    original.model.reset_parameters()
    enable_conditioning(original)
    enable_type_rope(original)
    state = original.state_dict()
    with init_empty_weights():
        empty = LLaDAForMultiModalGeneration(original.config)
    assert all(p.is_meta for p in empty.parameters())
    empty.load_state_dict(state, strict=True, assign=True)
    assert not any(p.is_meta for p in empty.parameters())
    for n, v in empty.state_dict().items():
        torch.testing.assert_close(v, state[n], atol=0, rtol=0)
