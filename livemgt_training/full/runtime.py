"""Explicit DDP / sharded optimizers / FSDP1 FULL_SHARD integration."""

import contextlib
import os
import resource
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel import DistributedDataParallel as DDP

from livemgt_training.precision import compute_dtype
from model import LLaDAForMultiModalGeneration
from model.configuration_llada import LLaDAConfig
from model.livemgt_conditioning import enable_conditioning
from model.livemgt_rope import enable_type_rope
from model.modeling_llada import LLaDALlamaBlock


def initialize():
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "25012")
        dist.init_process_group(
            "nccl",
            rank=int(os.environ.get("RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            timeout=timedelta(minutes=20),
        )
    torch.set_num_threads(int(os.environ.get("TRAIN_CPU_THREADS", "4")))
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)


def build_model(c, device, teacher=False):
    if c.get("_resume_model_config"):
        architecture = LLaDAConfig.from_pretrained(
            c["_resume_model_config"], local_files_only=True
        )
        architecture.init_device = "cpu"
        model = LLaDAForMultiModalGeneration(architecture)
    else:
        if not c["base_checkpoint"]:
            raise ValueError("set base_checkpoint to a local official/SFT checkpoint")
        model = LLaDAForMultiModalGeneration.from_pretrained(
            c["base_checkpoint"],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    enable_conditioning(model, region=True, token_type=False)
    enable_type_rope(model, enabled=True, mode="pretrained_type_phase")
    model = model.float()
    model.requires_grad_(not teacher)
    # A disabled embedding has no place in the optimized set. All Transformer
    # parameters, RSC and phase remain trainable for Student/Auxiliary.
    model.live_conditioning.token_type.requires_grad_(False)
    return model


def unwrap(model):
    return model.module if isinstance(model, (DDP, FSDP)) else model


def wrap(model, c, device, teacher=False):
    if c["backend"] == "fsdp":
        # Root owns embedding/head and RSC/phase. Their direct sliced weight
        # access occurs inside root forward, so root all-gather covers it.
        # Block calls are real FSDP.forward calls, never .module.forward.
        return FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy({LLaDALlamaBlock}),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=(
                MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                )
                if compute_dtype(c) == torch.bfloat16
                else None
            ),
            cpu_offload=CPUOffload(
                offload_params=bool(
                    c["teacher_offload"] if teacher else c["training_cpu_offload"]
                )
            ),
            use_orig_params=True,
            device_id=device,
            limit_all_gathers=True,
            sync_module_states=True,
        )
    model = model.to(device)
    return (
        model
        if teacher
        else DDP(
            model,
            device_ids=[device.index],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    )


def optimizer(model, c):
    params = [p for p in model.parameters() if p.requires_grad]
    kwargs = dict(
        lr=c["learning_rate"],
        weight_decay=c["weight_decay"],
        betas=tuple(c.get("adam_betas", [0.9, 0.999])),
        eps=c.get("adam_epsilon", 1e-8),
    )
    if c["optimizer"] == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, **kwargs)
    if c["backend"] == "ddp_zero":
        from torch.distributed.optim import ZeroRedundancyOptimizer

        return ZeroRedundancyOptimizer(
            params, optimizer_class=torch.optim.AdamW, foreach=False, **kwargs
        )
    return torch.optim.AdamW(params, foreach=False, **kwargs)


def scheduler(opt, c):
    # Linear warmup begins at a nonzero first-update learning rate.
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda completed: min(1.0, (completed + 1) / max(1, c["warmup"]))
    )


def no_sync(model, c, last):
    # FSDP deliberately syncs each microbatch: no_sync would keep FULL gradients
    # unsharded across 8 microbatches. Accumulation is still exact averaging.
    if c["backend"] != "fsdp" and not last:
        return model.no_sync()
    return contextlib.nullcontext()


def named(model):
    return {
        n.replace("_fsdp_wrapped_module.", "").removeprefix("module."): p
        for n, p in model.named_parameters()
    }


class CPUEMA:
    def __init__(self, student, c):
        self.c = c
        target = "cpu" if c["ema_device"] == "cpu" else torch.cuda.current_device()
        if c["ema_device"] not in ["cpu", "cuda"]:
            raise ValueError("ema_device must be cpu or cuda")
        self.state = {
            n: p.detach().to(target, copy=True).float()
            for n, p in named(student).items()
        }

    @torch.no_grad()
    def update(self, student, step):
        """step is the one-based optimizer update that just completed."""
        from livemgt_training.core import ema_decay_at_step

        rate = ema_decay_at_step(
            self.c["ema_decay"], step - 1 if self.c["ema_warmup"] else None
        )
        for n, p in named(student).items():
            e = self.state[n]
            v = p.detach().to(e.device, dtype=torch.float32)
            e.mul_(rate).add_(v, alpha=1 - rate)
        return rate


def memory():
    import psutil

    p = psutil.Process()
    cpu = p.memory_info()
    return dict(
        rank=dist.get_rank(),
        allocated_gib=torch.cuda.memory_allocated() / 2**30,
        reserved_gib=torch.cuda.memory_reserved() / 2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        cpu_rss_gib=cpu.rss / 2**30,
        cpu_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        cpu_available_gib=psutil.virtual_memory().available / 2**30,
    )


def collective_call(function):
    """Run local work and propagate its failure before the next model collective."""
    value = None
    error = None
    try:
        value = function()
    except Exception as exc:
        error = exc
    if dist.is_initialized():
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        failed = torch.tensor(int(error is not None), device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, repr(error) if error else None)
            raise RuntimeError(f"distributed operation failed: {errors}") from error
    elif error is not None:
        raise error
    return value


def check_finite(value, label):
    """All ranks stop before backward/sampling if any rank has NaN/Inf."""

    def check():
        if not torch.isfinite(value.detach()).all():
            raise FloatingPointError(f"nonfinite {label}")

    collective_call(check)
