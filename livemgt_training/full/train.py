"""Full-parameter SFT / Region-DiMO with explicit distributed and precision policies."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from inference.edit import load_vae
from livemgt_training.precision import compute_dtype, configure

from . import checkpoint
from .config import load, validate
from .data import Stream, digest
from .engine import Engine
from .runtime import (
    CPUEMA,
    build_model,
    collective_call,
    initialize,
    memory,
    optimizer,
    scheduler,
    wrap,
)
from .validation import validate as validate_images


def content_origin(config, phase, resume=None):
    """One startup hash of frozen assets and manifest versions, never per step."""

    def identify():
        meta = (
            json.loads((Path(resume) / "metadata.json").read_text()) if resume else None
        )
        base = (
            meta["origin"]["base"]
            if meta
            else checkpoint.asset_identity(config["base_checkpoint"])
        )
        teacher = None
        if phase == "dimo":
            teacher = (
                checkpoint.asset_identity(config["teacher_checkpoint"])
                if config.get("teacher_checkpoint")
                else meta["origin"].get("teacher")
                if meta
                else None
            )
            if not teacher:
                raise ValueError("DiMO requires an identified Teacher source")
        vq = checkpoint.asset_identity(
            config.get("vq_checkpoint") or config["base_checkpoint"], "vqvae"
        )
        versions = dict(config.get("data_versions", {}))
        for name, path in config.get("data_version_files", {}).items():
            versions[name] = digest(path)
        for name, path in config["asset_roots"].items():
            provenance = Path(path) / "DATA_PROVENANCE.json"
            if provenance.is_file():
                versions["provenance:" + name] = digest(provenance)
        return dict(
            base=base,
            teacher=teacher,
            vq_vae=vq,
            data_versions=versions,
            train=digest(config["train_manifest"]),
            validation=digest(config["validation_manifest"]),
            phase=phase,
        )

    origin = checkpoint.rank_zero(identify)
    objects = [origin]
    dist.broadcast_object_list(objects, src=0)
    origin = objects[0]
    keys = (
        "halo",
        "supervise_target_halo",
        "rope_mode",
        "token_type_embedding",
        "region_conditioning",
        "cbc",
        "gt_ce",
        "teacher_text_cfg",
        "teacher_image_cfg",
        "resolution",
        "validation_resolution",
        "learning_rate",
        "warmup",
        "weight_decay",
        "seed",
        "data_mode",
        "data_mixture",
        "cfg_dropout",
        "student_initial_mask_ratio",
        "auxiliary_label_smoothing",
        "evidence_mode",
        "max_grad_norm",
        "mixed_precision",
        "parameter_dtype",
        "tf32",
        "adam_betas",
        "adam_epsilon",
        "optimizer",
        "micro_batch_size",
        "gradient_accumulation",
        "resolved_global_batch_size",
        "backend",
        "sft_updates",
        "dimo_updates",
        "ema_decay",
        "ema_warmup",
    )
    origin["semantics"] = {k: config[k] for k in keys}
    origin["semantics"]["sampler_version"] = 2
    return origin


def log(output, event, **values):
    # Every rank enters this wrapper, including for rank-zero-only disk writes.
    def write():
        line = json.dumps(dict(event=event, **values))
        print(line, flush=True)
        with (output / "train.jsonl").open("a") as f:
            f.write(line + "\n")

    checkpoint.rank_zero(write)


def main(args):
    rank, world, device = initialize()
    try:
        config = validate(load(args.config, args.set), world)
        configure(config.get("tf32", False))
        origin = content_origin(config, args.phase, args.resume)
        if args.resume:
            meta = collective_call(
                lambda: json.loads((Path(args.resume) / "metadata.json").read_text())
            )
            rules = checkpoint.resume_rules(
                meta,
                origin,
                world,
                allow_world_change=args.allow_world_size_change,
                allow_batch_change=args.allow_batch_size_change,
                allow_backend_change=args.allow_backend_change,
            )
            if rank == 0:
                print(json.dumps({"resume_rules": rules}), flush=True)
        steps = config["sft_updates" if args.phase == "sft" else "dimo_updates"]
        if args.max_updates is not None:
            if args.max_updates < 1:
                raise ValueError("--max-updates must be positive")
            steps = min(steps, args.max_updates)
        output = Path(args.output)

        def prepare_output():
            if args.resume:
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.mkdir(parents=True, exist_ok=False)
            name = (
                "config.json"
                if not (output / "config.json").exists()
                else f"config-resume-{time.time_ns()}.json"
            )
            (output / name).write_text(json.dumps(config, indent=2) + "\n")

        checkpoint.rank_zero(prepare_output)
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        if args.resume:
            config["_resume_model_config"] = str(
                Path(args.resume) / "model_config.json"
            )
        if args.phase == "dimo" and not args.allow_untrained_teacher:
            provenance = origin["teacher"].get("provenance", {})
            if config["teacher_status"] != "FORMAL_SFT_TRAINED" or not provenance.get(
                "formal_sft"
            ):
                raise ValueError(
                    "DiMO requires formal SFT Teacher provenance; --allow-untrained-teacher is development only and must also be explicit on resume"
                )
            teacher_vq = provenance.get("origin", {}).get("vq_vae")
            if teacher_vq is None or teacher_vq != origin["vq_vae"]:
                raise ValueError(
                    "Teacher SFT provenance VQ-VAE identity is missing or differs from the current VQ-VAE"
                )
        tokenizer_path = (
            (Path(args.resume) / "tokenizer")
            if args.resume
            else (
                config["teacher_checkpoint"]
                if args.phase == "dimo"
                else config["base_checkpoint"]
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        models = {}
        opts = {}
        schedulers = {}
        roles = (
            ("teacher", "student", "auxiliary")
            if args.phase == "dimo"
            else ("student",)
        )
        for role in roles:
            role_config = dict(config)
            if args.phase == "dimo" and not args.resume:
                role_config["base_checkpoint"] = config["teacher_checkpoint"]
            raw = build_model(role_config, device, teacher=role == "teacher")
            models[role] = wrap(raw, config, device, teacher=role == "teacher")
            if role != "teacher":
                opts[role] = optimizer(models[role], config)
                schedulers[role] = scheduler(opts[role], config)
            log(
                output,
                "model",
                role=role,
                backend=config["backend"],
                teacher_offload=config["teacher_offload"]
                if role == "teacher"
                else False,
                parameter_dtype=config["parameter_dtype"],
                compute_dtype=str(compute_dtype(config)),
            )
        ema = CPUEMA(models["student"], config)
        vae = load_vae(
            config.get("vq_checkpoint") or config["base_checkpoint"],
            device,
            compute_dtype(config),
        )
        progress = dict(student_updates=0, auxiliary_updates=0, global_sample_cursor=0)
        if args.resume:
            progress, recovery = checkpoint.load(
                args.resume,
                models,
                opts,
                schedulers,
                ema,
                origin,
                args.allow_world_size_change,
                mmap_weights=config.get("checkpoint_mmap", False),
                allow_batch_change=args.allow_batch_size_change,
                allow_backend_change=args.allow_backend_change,
                stream_required=True,
            )
        else:
            torch.manual_seed(config["seed"] + rank)
            np.random.seed(config["seed"] + rank)
            random.seed(config["seed"] + rank)
        stream = collective_call(
            lambda: Stream(
                config,
                vae,
                rank,
                world,
                progress["global_sample_cursor"],
                state=recovery["sampler_state"] if args.resume else None,
                reset_counts=recovery["world_size_changed"] if args.resume else False,
            )
        )
        validation = collective_call(
            lambda: Stream(config, vae, rank, world, validation=True)
        )
        engine = Engine(models, opts, schedulers, ema, tokenizer, config)
        last_saved = (
            progress["student_updates"]
            if (
                output / f'checkpoint-{progress["student_updates"]}' / "COMPLETE"
            ).exists()
            else -1
        )
        for step in range(progress["student_updates"] + 1, steps + 1):
            start = time.perf_counter()
            values = (
                engine.sft(stream, step)
                if args.phase == "sft"
                else engine.dimo(stream, step)
            )
            progress = dict(
                student_updates=step,
                auxiliary_updates=step if args.phase == "dimo" else 0,
                global_sample_cursor=stream.cursor,
            )
            if step == 1 or step % config["log_interval"] == 0:
                # Average scalar metrics only; no model fingerprints or sample UID gathers.
                keys = list(values)
                scalars = torch.tensor([values[k] for k in keys], device=device)
                dist.all_reduce(scalars)
                scalars /= world
                metrics = dict(zip(keys, scalars.tolist()))
                log(
                    output,
                    "update",
                    step=step,
                    seconds=time.perf_counter() - start,
                    **metrics,
                )
            if step % config["memory_interval"] == 0:
                log(output, "memory", step=step, **memory())
            if step % config["validation_interval"] == 0:
                validate_images(
                    models["student"],
                    tokenizer,
                    validation,
                    vae,
                    config,
                    output,
                    args.phase,
                    step,
                    ema if config["validation_weights"] == "ema" else None,
                )
                log(
                    output,
                    "validation",
                    step=step,
                    samples=config["validation_samples"],
                )
            if step % config["checkpoint_interval"] == 0 or step == steps:
                checkpoint.save(
                    output / f"checkpoint-{step}",
                    models,
                    opts,
                    schedulers,
                    ema,
                    progress,
                    config,
                    origin,
                    tokenizer,
                    stream=stream,
                    final=step == steps,
                )
                last_saved = step
                checkpoint.prune(
                    output, config["checkpoint_keep"], config["checkpoint_milestones"]
                )
                log(output, "checkpoint", step=step)
        # A fresh output with zero remaining updates still receives a final state.
        if last_saved != progress["student_updates"]:
            checkpoint.save(
                output / f'checkpoint-{progress["student_updates"]}',
                models,
                opts,
                schedulers,
                ema,
                progress,
                config,
                origin,
                tokenizer,
                stream=stream,
                final=True,
            )
        if args.export:
            checkpoint.export(
                output / f'{args.export}-export-{progress["student_updates"]}',
                models["student"],
                dtype=getattr(torch, config["full_export_dtype"]),
                ema=ema if args.export == "ema" else None,
                inference_dtype=config["inference_dtype"],
            )

            def write_provenance():
                target = output / f'{args.export}-export-{progress["student_updates"]}'
                tokenizer.save_pretrained(target)
                checkpoint.atomic_text(
                    target / "TRAINING_PROVENANCE.json",
                    json.dumps(
                        checkpoint.training_provenance(
                            config, progress, origin, args.export
                        ),
                        indent=2,
                    )
                    + "\n",
                )

            checkpoint.rank_zero(write_provenance)
        log(output, "finished", **progress)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--phase", choices=["sft", "dimo"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--resume")
    p.add_argument(
        "--allow-world-size-change",
        action="store_true",
        help="Repartition rank-owned source slices, reset source counts and reseed rank RNG; not exact data replay",
    )
    p.add_argument(
        "--allow-batch-size-change",
        action="store_true",
        help="Explicitly permit changed microbatch/accumulation/global batch; keep optimizer moments and stream cursor",
    )
    p.add_argument(
        "--allow-backend-change",
        action="store_true",
        help="Explicitly repartition portable named optimizer states onto the requested backend",
    )
    p.add_argument(
        "--max-updates",
        type=int,
        help="Explicitly cap updates for development; never implicit",
    )
    p.add_argument(
        "--allow-untrained-teacher",
        action="store_true",
        help="Development only; not a valid quality Teacher",
    )
    p.add_argument("--export", choices=["student", "ema"])
    main(p.parse_args())


if __name__ == "__main__":
    cli()
