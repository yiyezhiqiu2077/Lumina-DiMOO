"""Portable full-training contract. Local paths belong only in explicit overrides."""

import json
import math
from pathlib import Path

VALID_COMBINATIONS = {
    ("ddp", "adamw"),
    ("ddp_zero", "adamw"),
    ("fsdp", "adamw"),
    ("ddp", "adamw8bit"),
}


def resolve_batch(c, world):
    micro = c["micro_batch_size"]
    accum = c["gradient_accumulation"]
    if type(micro) != int or micro < 1 or type(accum) != int or accum < 1:
        raise ValueError("batch sizes must be positive integers")
    if c.get("global_batch_size") is not None:
        g = c["global_batch_size"]
        if type(g) != int or g < 1 or g % (world * micro):
            raise ValueError(
                "Global Batch Size must be divisible by World Size * Micro Batch Size"
            )
        accum = g // (world * micro)
    c["gradient_accumulation"] = accum
    c["resolved_global_batch_size"] = world * micro * accum
    return c


def validate(c, world):
    if c.get("mode") != "full":
        raise ValueError("full backend requires mode=full; no LoRA fallback")
    c = resolve_batch(c, world)
    pair = (c["backend"], c["optimizer"])
    if pair not in VALID_COMBINATIONS:
        raise ValueError(
            f"Unverified backend/optimizer combination {pair}; adamw8bit is enabled only for explicitly tested DDP"
        )
    if c["optimizer"] == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ValueError(
                "adamw8bit requires explicitly installed bitsandbytes==0.45.5"
            ) from e
        if bnb.__version__ != "0.45.5":
            raise ValueError(
                "unverified bitsandbytes version; validated version is 0.45.5"
            )
    if c["mixed_precision"] not in ("no", "bf16") or c["parameter_dtype"] != "float32":
        raise ValueError("use FP32 parameters with mixed_precision=no or explicit bf16")
    if c.get("inference_dtype", "float32") not in ("float32", "bfloat16"):
        raise ValueError("unsupported inference dtype")
    expected = "float32" if c["mixed_precision"] == "no" else "bfloat16"
    if c.get("inference_dtype", "float32") != expected:
        raise ValueError(
            "inference_dtype must match training compute for in-place distributed validation; use explicit standalone inference to select a different dtype"
        )
    if c.get("tf32", False):
        raise ValueError("TF32 is disabled for this training contract")
    if c.get("checkpoint_keep", 2) < 1:
        raise ValueError("checkpoint_keep must be positive")
    if c.get("validation_weights", "student") not in ("student", "ema"):
        raise ValueError("validation_weights must be student or ema")
    if any(
        c[k] < 1
        for k in (
            "checkpoint_interval",
            "validation_interval",
            "log_interval",
            "memory_interval",
        )
    ):
        raise ValueError("intervals must be positive")
    if c["training_cpu_offload"] and (
        c["backend"] != "fsdp" or c["gradient_accumulation"] != 1
    ):
        raise ValueError(
            "training CPU offload requires fsdp and accumulation=1; FSDP1 offload accumulation is unsupported"
        )
    if c["teacher_offload"] and c["backend"] != "fsdp":
        raise ValueError("teacher parameter CPU offload is validated only with FSDP")
    if c["checkpoint_format"] != "full":
        raise ValueError("only portable full checkpoint format is enabled")
    if c.get("gan", {}).get("enabled"):
        raise ValueError("GAN disabled: no validated discriminator")
    for k in ["resolution", "validation_resolution"]:
        if type(c[k]) != int or c[k] < 16 or c[k] % 16:
            raise ValueError(f"{k} must be a positive multiple of VQ stride 16")
    if c["ema_device"] not in ["cpu", "cuda"]:
        raise ValueError("ema_device must be cpu or cuda")
    if c["full_export_dtype"] not in ["float32", "bfloat16"]:
        raise ValueError("unsupported export dtype")
    if c["data_mode"] not in ["manifest", "mixture"]:
        raise ValueError("data_mode must be manifest or mixture")
    if c["backend"] == "fsdp" and not c["fsdp_sync_each_microbatch"]:
        raise ValueError("only synchronization on every FSDP microbatch is validated")
    if c["max_grad_norm"] <= 0 or c["warmup"] < 0:
        raise ValueError("invalid clipping/warmup")
    if c["scheduler"] != "constant_with_warmup":
        raise ValueError("unsupported scheduler")
    if c["halo"] < 0 or type(c["halo"]) != int:
        raise ValueError("halo must be a nonnegative token-grid radius")
    if (
        not c["region_conditioning"]
        or c["token_type_embedding"]
        or not c["token_type_rope"]
        or c["rope_mode"] != "pretrained_type_phase"
    ):
        raise ValueError(
            "formal C contract: RSC + compatible type phase, no type embedding"
        )
    if c["learning_rate"] <= 0 or not math.isfinite(c["learning_rate"]):
        raise ValueError("invalid learning rate")
    drop = c["cfg_dropout"]
    if any(not 0 <= drop[x] <= 1 for x in ["text", "image"]) or sum(drop.values()) > 1:
        raise ValueError("invalid exclusive CFG dropout")
    return resolve_batch(c, world)


def load(path, overrides=()):
    c = json.loads(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        target = c
        parts = key.split(".")
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    return c
