import copy
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pytest
import torch

from livemgt_training.adapters import RoleBank, install_trainables
from livemgt_training.core import (
    fkl_surrogate,
    generation_region,
    gt_ce,
    lumina_cfg,
    pseudo_mask,
    region_weight_map,
    signed_mask_distance,
    teacher_edit_evidence,
    visual_logits,
)
from model import LLaDAForMultiModalGeneration
from model.livemgt import ConditionKVCache, PrefixLayout
from model.livemgt_conditioning import bbox_size, enable_conditioning

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def layout():
    ids = torch.arange(24, device=DEVICE)[None]
    types = torch.zeros(24, dtype=torch.long, device=DEVICE)
    types[3:7] = 1
    types[7:] = 2
    return ids, PrefixLayout(
        7, torch.tensor([8, 10, 17, 22], device=DEVICE), (2, 3), types
    )


@pytest.mark.parametrize(
    "region,types", [(False, False), (True, False), (False, True), (True, True)]
)
def test_p3_zero_initialization(model, region, types):
    ids, l = layout()
    with torch.no_grad():
        before = model(ids, infer=True, live_layout=l).logits
    enable_conditioning(model)
    with torch.no_grad():
        after = model(
            ids,
            infer=True,
            live_layout=l,
            region_conditioning=region,
            token_type_encoding=types,
        ).logits
    torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.parametrize("region,types", [(True, False), (False, True), (True, True)])
def test_p3_backward_cache_dense_positions(model, region, types):
    ids, l = layout()
    module = enable_conditioning(model)
    # Trainable nonzero settings test more than just zero-init identity.
    with torch.no_grad():
        module.region_mlp[-1].weight.normal_(std=0.002)
        module.token_type.weight.normal_(std=0.002)
    model.train()
    logits = model(
        ids,
        infer=True,
        live_layout=l,
        region_conditioning=region,
        token_type_encoding=types,
        gradient_checkpointing=True,
    ).logits
    logits.square().mean().backward()
    if region:
        assert module.region_mlp[-1].weight.grad.norm() > 0
    if types:
        assert module.token_type.weight.grad.norm() > 0
    model.eval()
    cache = ConditionKVCache()
    options = dict(
        infer=True, live_layout=l, region_conditioning=region, token_type_encoding=types
    )
    with torch.no_grad():
        s = model(ids, **options).logits
        d = model(ids, **options, mode="dense").logits
        model(ids, **options, condition_cache=cache)
        c = model(ids, **options, condition_cache=cache).logits
    torch.testing.assert_close(s, d, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(s, c, atol=3e-6, rtol=3e-5)
    # P0-P2 RoPE-index test remains part of the joint regression suite.
    cache.clear()


def test_prefix_independent_of_bbox(model):
    ids, l = layout()
    m = enable_conditioning(model)
    with torch.no_grad():
        m.region_mlp[-1].weight.normal_(std=0.1)
    ca = ConditionKVCache()
    cb = ConditionKVCache()
    with torch.no_grad():
        model(ids, infer=True, live_layout=l, condition_cache=ca)
        other = copy.copy(l)
        other.region_size = (6, 8)
        model(ids, infer=True, live_layout=other, condition_cache=cb)
    for layer in range(3):
        for a, b in zip(
            ca.branches[("cond", True)]["layers"][layer],
            cb.branches[("cond", True)]["layers"][layer],
        ):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_p3_checkpoint_and_default_path(model, tmp_path):
    ids, l = layout()
    with torch.no_grad():
        original = model(ids, infer=True).logits
    m = enable_conditioning(model)
    with torch.no_grad():
        m.token_type.weight.normal_(std=0.01)
        m.region_mlp[-1].weight.normal_(std=0.01)
    model.save_pretrained(tmp_path)
    restored = LLaDAForMultiModalGeneration.from_pretrained(tmp_path).to(DEVICE).eval()
    with torch.no_grad():
        expected = model(ids, infer=True, live_layout=l).logits
        actual = restored(ids, infer=True, live_layout=l).logits
        default = model(ids, infer=True).logits
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    torch.testing.assert_close(default, original, atol=0, rtol=0)


def test_bbox_and_cache_types(model):
    m = torch.zeros(6, 7, dtype=torch.bool)
    m[1:4, 2:6] = True
    assert bbox_size(m) == (3, 4) and bbox_size(torch.zeros_like(m)) == (0, 0)
    ids, l = layout()
    enable_conditioning(model)
    cache = ConditionKVCache()
    with torch.no_grad():
        model(ids, infer=True, live_layout=l, condition_cache=cache)
        l.token_types = l.token_types.clone()
        l.token_types[0] = 1
        with pytest.raises(ValueError, match="types changed"):
            model(ids, infer=True, live_layout=l, condition_cache=cache)


def upstream():
    root = os.environ.get("EDITMGT_ROOT")
    if not root:
        pytest.skip("optional upstream parity: set EDITMGT_ROOT")
    sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(
        "editmgt_dimo_core", str(Path(root) / "src/dimo/core.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cbc_upstream_exact_parity():
    original = torch.zeros(1, 9, 11, dtype=torch.bool, device=DEVICE)
    original[:, 2:7, 3:8] = True
    region = generation_region(original, True, 1)
    c = dict(
        enabled=True,
        core_weight=1.0,
        boundary_weight=2.0,
        evidence_weight=0.5,
        boundary_sigma=1.0,
        core_margin=1.0,
        weighted_pseudo_mask=True,
        evidence_chunk_size=3,
    )
    up = upstream()
    distance = signed_mask_distance(original, region)
    torch.testing.assert_close(
        distance, up.signed_mask_distance(original, region), atol=0, rtol=0
    )
    assert (distance[original] > 0).all() and (distance[region & ~original] < 0).all()
    conditional = torch.randn(1, 9, 11, 13, device=DEVICE)
    unconditional = torch.randn_like(conditional)
    evidence = teacher_edit_evidence(conditional, unconditional, 3)
    torch.testing.assert_close(
        evidence,
        up.teacher_edit_evidence(conditional, unconditional, 3),
        atol=0,
        rtol=0,
    )
    weights = region_weight_map(original, region, c, evidence, distance)
    torch.testing.assert_close(
        weights,
        up.region_weight_map(original, region, c, evidence, distance),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(weights[region].mean(), torch.ones((), device=DEVICE))
    torch.manual_seed(99)
    q = pseudo_mask(region, torch.tensor([0.4], device=DEVICE), weights)
    torch.manual_seed(99)
    reference = up.pseudo_mask(region, torch.tensor([0.4], device=DEVICE), weights)
    assert torch.equal(q, reference) and not (q & ~region).any()
    assert q.sum() == round(int(region.sum()) * 0.4)


def test_fkl_direction_and_visual_offset():
    # Unified text classes carry extreme scores; slicing must ignore them.
    z = torch.randn(1, 2, 2, 134548, device=DEVICE, requires_grad=True)
    with torch.no_grad():
        z[..., :126356] = 1000
    t = torch.randn(1, 2, 2, 8192, device=DEVICE, requires_grad=True)
    a = torch.randn_like(t, requires_grad=True)
    q = torch.tensor([[[True, False], [True, False]]], device=DEVICE)
    w = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]], device=DEVICE)
    loss = fkl_surrogate(visual_logits(z), t, a, q, w)
    loss.backward()
    expected = (
        (a.detach().float().softmax(-1) - t.detach().float().softmax(-1))
        * q[..., None]
        * w[..., None]
        / q.sum()
    )
    torch.testing.assert_close(z.grad[..., 126356:], expected, atol=1e-7, rtol=1e-5)
    assert (
        z.grad[..., :126356].count_nonzero() == 0 and t.grad is None and a.grad is None
    )
    # Explicit two-class descent direction: ascent teacher class, descent aux class.
    small = torch.zeros(1, 1, 1, 2, requires_grad=True, device=DEVICE)
    teacher = torch.tensor([[[[10.0, -10.0]]]], device=DEVICE)
    aux = -teacher
    fkl_surrogate(
        small, teacher, aux, torch.ones(1, 1, 1, dtype=torch.bool, device=DEVICE)
    ).backward()
    assert small.grad[..., 0] < 0 and small.grad[..., 1] > 0


def test_aux_gt_offsets_and_masks():
    logits = torch.randn(1, 2, 2, 8192, device=DEVICE, requires_grad=True)
    target = torch.full((1, 2, 2), 126359, device=DEVICE)
    q = torch.tensor([[[1, 0], [0, 0]]], device=DEVICE).bool()
    loss = gt_ce(logits, target, q)
    loss.backward()
    assert logits.grad[q].count_nonzero() > 0 and logits.grad[~q].count_nonzero() == 0
    with pytest.raises(ValueError):
        gt_ce(logits, target - 126356, q)


def test_cfg6_equivalence_and_three_branch_formula():
    c = torch.randn(2, 3)
    u = torch.randn_like(c)
    i = torch.randn_like(c)
    torch.testing.assert_close(lumina_cfg(c, u, i, 5.0, 0.0), u + 6 * (c - u))
    torch.testing.assert_close(
        lumina_cfg(c, u, i, 2.5, 4), c + 2.5 * (c - u) + 4 * (c - i)
    )


def test_role_bank_independence_and_ema(model):
    c = dict(p3=dict(region=True, token_type=True), lora_rank=2, lora_alpha=4)
    params = install_trainables(model, c)
    bank = RoleBank(model, params)
    teacher = copy.deepcopy(bank.states["teacher"])
    aux = copy.deepcopy(bank.states["auxiliary"])
    bank.activate("student", True)
    with torch.no_grad():
        for p in params.values():
            p.add_(1.0)
    bank.save_role()
    rate = bank.ema_update(0.9999, 0, True)
    assert rate == 0.1
    for name in params:
        torch.testing.assert_close(
            bank.states["teacher"][name], teacher[name], atol=0, rtol=0
        )
        torch.testing.assert_close(
            bank.states["auxiliary"][name], aux[name], atol=0, rtol=0
        )
        torch.testing.assert_close(
            bank.states["ema"][name],
            teacher[name] * 0.1 + bank.states["student"][name] * 0.9,
        )
    bank.graph_live = True
    with pytest.raises(RuntimeError):
        bank.activate("auxiliary")


def test_mask_ratio_and_initialization_upstream():
    from livemgt_training.core import one_step_initial, sample_mask_ratio

    torch.manual_seed(15)
    actual = sample_mask_ratio(100, DEVICE)
    torch.manual_seed(15)
    expected = torch.cos(torch.rand(100, device=DEVICE) * torch.pi * 0.5)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    original = torch.zeros(1, 5, 5, dtype=torch.bool, device=DEVICE)
    original[:, 2, 2] = True
    region = generation_region(original, True, 1)
    source = torch.full((1, 5, 5), 126359, device=DEVICE)
    initial = one_step_initial(source, region, 1.0)
    assert (initial[region] == 126336).all() and torch.equal(
        initial[~region], source[~region]
    )
    half = one_step_initial(source, region, 0.5)
    assert (half[region] == 126336).sum() == round(int(region.sum()) * 0.5)
    assert torch.equal(half[~region], source[~region])
    assert (
        (half[region] == 126336) | ((half[region] >= 126356) & (half[region] < 134548))
    ).all()


def test_existing_p3_sft_module_and_freeze_policy(model):
    enable_conditioning(model)
    c = dict(p3=dict(region=True, token_type=True), lora_rank=2, lora_alpha=4)
    params = install_trainables(model, c)
    assert any(n.startswith("live_conditioning") for n in params)
    bank = RoleBank(
        model, params, freeze_policy={"student": ["live_conditioning.height"]}
    )
    bank.activate("student", True)
    assert not model.live_conditioning.height.weight.requires_grad
    assert model.live_conditioning.width.weight.requires_grad
    bank.activate("teacher")
    assert all(not p.requires_grad for p in params.values())
