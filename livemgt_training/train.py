"""Optional single-process LoRA training, independent of the full-parameter backend."""

import argparse, copy, hashlib, json, random, time, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from diffusers import VQModel
from model import LLaDAForMultiModalGeneration
from .config import load_config
from .data import (
    read_manifest,
    audit_splits,
    audit_official_eval,
    ReplayStream,
    encode_triplet,
)
from .adapters import install_trainables, RoleBank
from .engine import DiMOEngine, grad_stats, state_delta
from .core import generation_region, pseudo_mask, gt_ce
from .predict import predict_grid
from .checkpoint import save_checkpoint, load_checkpoint


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optimizers_and_schedulers(params, c):
    optimizers = {
        role: torch.optim.AdamW(
            list(params.values()), lr=c[f"{role}_lr"], weight_decay=c["weight_decay"]
        )
        for role in ["student", "auxiliary"]
    }
    warmup = c.get("warmup_steps", 0)
    schedulers = {
        role: torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: min(1.0, step / warmup) if warmup else 1.0
        )
        for role, opt in optimizers.items()
    }
    return optimizers, schedulers


def sft_update(bank, tokenizer, batch, c, optimizer, scheduler):
    bank.activate("student", train=True)
    region = generation_region(batch["original"], True, c["distillation_halo"])
    ratio = torch.cos(torch.rand(1, device=region.device) * torch.pi / 2)
    q = pseudo_mask(region, ratio)
    clean = torch.where(region, batch["target"], batch["source"])
    corrupted = torch.where(q, 126336, clean)
    draw = random.random()
    drop = c["cfg_dropout"]
    branch = (
        "uncond_image"
        if draw < drop["image"]
        else ("uncond_text" if draw < drop["image"] + drop["text"] else "cond")
    )
    logits = predict_grid(
        bank.model,
        tokenizer,
        batch,
        corrupted,
        region,
        branch,
        gradient_checkpointing=c["gradient_checkpointing"],
    )
    loss = gt_ce(logits, batch["target"], q)
    bank.graph_live = True
    try:
        loss.backward()
    finally:
        bank.graph_live = False
    stats = grad_stats(bank.params.values())
    rope_grad = bank.params.get("live_type_rope.phase")
    rope_grad_norm = (
        float(rope_grad.grad.norm())
        if rope_grad is not None and rope_grad.grad is not None
        else None
    )
    if not stats["finite"] or not stats["norm"]:
        raise RuntimeError("SFT gradients invalid")
    torch.nn.utils.clip_grad_norm_(list(bank.params.values()), c["max_grad_norm"])
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    bank.save_role()
    return dict(
        loss=float(loss),
        gradient=stats,
        type_rope_gradient_norm=rope_grad_norm,
        type_rope_parameter_norm=float(rope_grad.norm())
        if rope_grad is not None
        else None,
        branch=branch,
        uid=batch["uid"],
        corrupted=int(q.sum()),
        region=int(region.sum()),
    )


@torch.no_grad()
def validate_sft(bank, tokenizer, batch, c):
    bank.activate("student")
    region = generation_region(batch["original"], True, c["distillation_halo"])
    tokens = torch.where(region, 126336, batch["source"])
    logits = predict_grid(bank.model, tokenizer, batch, tokens, region)
    return dict(
        uid=batch["uid"],
        loss=float(gt_ce(logits, batch["target"], region)),
        mask_ratio=1.0,
        optimizer_data=False,
    )


def main(args):
    """Explicit single-process LoRA mode; full training uses full/train.py."""
    from livemgt_training.precision import configure
    from inference.edit import edit_tokens, load_vae, save_images

    c = load_config(args.config)
    configure(False)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    steps = c["planning"][
        "sft_initial_budget_updates"
        if args.phase == "sft"
        else "dimo_initial_budget_updates"
    ]
    if args.max_updates is not None:
        steps = min(steps, args.max_updates)
    if args.phase == "dimo" and (
        c["teacher_status"] != "FORMAL_SFT_TRAINED"
        or not (c["teacher_adapter"] or c.get("teacher_merged_base", False))
    ):
        raise ValueError("LoRA DiMO requires an explicit formal SFT Teacher")
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])
    random.seed(c["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        c["base_checkpoint"], local_files_only=True
    )
    dtype = torch.float32 if c.get("mixed_precision", "no") == "no" else torch.bfloat16
    model = (
        LLaDAForMultiModalGeneration.from_pretrained(
            c["base_checkpoint"],
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    params = install_trainables(model, c)
    teacher = None
    if c["teacher_adapter"]:
        payload = torch.load(
            c["teacher_adapter"], map_location="cpu", weights_only=False
        )
        if payload["origin"].get("run_mode") != "formal_sft":
            raise ValueError("Teacher adapter lacks formal SFT provenance")
        teacher = payload["roles"]["student"]
    bank = RoleBank(model, params, teacher, c.get("freeze_adapter_prefixes"))
    opts, sch = optimizers_and_schedulers(params, c)
    if args.phase == "sft":
        opts["student"].param_groups[0].update(lr=c["sft_lr"], initial_lr=c["sft_lr"])
        sch["student"].base_lrs = [c["sft_lr"]]
    vae = load_vae(
        c.get("vq_checkpoint") or c["base_checkpoint"], "cuda", torch.float32
    )
    progress = dict(student_updates=0, auxiliary_updates=0, consumed_batches=0)
    origin = dict(
        base=c["base_checkpoint"],
        train_manifest_sha256=digest_file(c["train_manifest"]),
        validation_manifest_sha256=digest_file(c["validation_manifest"]),
        phase=args.phase,
        run_mode="formal_" + args.phase,
        type_rope=c.get("type_rope"),
        resolution=c["resolution"],
        mixed_precision=c.get("mixed_precision", "no"),
        p3=c["p3"],
        lora_rank=c["lora_rank"],
    )
    if args.resume:
        progress = load_checkpoint(args.resume, bank, opts, sch, origin)
    stream = ReplayStream(
        read_manifest(c["train_manifest"]),
        vae,
        c["resolution"],
        progress["consumed_batches"],
        c.get("asset_roots", {}),
    )
    engine = DiMOEngine(bank, tokenizer, c, opts, sch)
    validation = read_manifest(c["validation_manifest"])
    for step in range(progress["student_updates"] + 1, steps + 1):
        batch = stream.next()
        rec = (
            sft_update(bank, tokenizer, batch, c, opts["student"], sch["student"])
            if args.phase == "sft"
            else engine.update(batch, stream.next, step - 1)
        )
        progress.update(
            student_updates=step,
            auxiliary_updates=step if args.phase == "dimo" else 0,
            consumed_batches=stream.offset,
        )
        print(
            json.dumps({"step": step, "loss": rec.get("loss", rec.get("fkl"))}),
            flush=True,
        )
        if step % c["validation_every"] == 0:
            from .full.validation import independent_rng

            with independent_rng(c["seed"], torch.device("cuda", 0)):
                bank.activate("student")
                vb = encode_triplet(
                    validation[0], vae, c["resolution"], c.get("asset_roots", {})
                )
                result = edit_tokens(
                    model,
                    tokenizer,
                    vb,
                    phase=args.phase,
                    halo=c["distillation_halo"],
                    cache=False,
                    generator=torch.Generator(device="cuda").manual_seed(c["seed"]),
                )
                save_images(
                    vb,
                    result,
                    vae,
                    out / "validation" / f"{step}.png",
                    c["resolution"],
                    c["resolution"],
                    target=vb["target"],
                )
        if step % c["save_every"] == 0 or step == steps:
            save_checkpoint(
                out / f"checkpoint-{step}.pt", bank, opts, sch, progress, c, origin
            )
    (out / "config.json").write_text(json.dumps(c, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Optional single-process LoRA SFT / DiMO")
    p.add_argument("--config", required=True)
    p.add_argument("--phase", choices=["sft", "dimo"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--max-updates", type=int)
    main(p.parse_args())
