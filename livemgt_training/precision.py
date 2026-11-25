"""One precision policy shared by training, validation and inference."""

from contextlib import nullcontext

import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def configure(tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")


def compute_dtype(config):
    mode = config.get("mixed_precision", "no")
    if mode not in ("no", "bf16"):
        raise ValueError("mixed_precision must be no or bf16")
    return torch.float32 if mode == "no" else torch.bfloat16


def autocast(config, device_type="cuda"):
    if compute_dtype(config) == torch.float32:
        return nullcontext()
    return torch.autocast(device_type, dtype=torch.bfloat16)


def inference_dtype(config, override=None):
    name = override or config.get("inference_dtype", "float32")
    if name not in DTYPES:
        raise ValueError("inference_dtype must be float32 or bfloat16")
    return DTYPES[name]
