"""Small-model integration check for changed training/checkpoint code.

Example: torchrun --standalone --nproc_per_node=2 -m tests.distributed_worker
  --backend fsdp --output /tmp/lumina-test
This is a test fixture, never an 8B training result or production entrypoint.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from inference.edit import edit_tokens
from livemgt_training.full import checkpoint
from livemgt_training.full.config import load, validate
from livemgt_training.full.engine import Engine
from livemgt_training.full.runtime import (
    CPUEMA,
    initialize,
    optimizer,
    scheduler,
    wrap,
)
from livemgt_training.precision import configure
from model import LLaDAForMultiModalGeneration
from model.configuration_llada import LLaDAConfig
from model.livemgt_conditioning import enable_conditioning
from model.livemgt_rope import enable_type_rope


def main(a):
    rank, world, device = initialize()
    configure(False)
    c = load(Path(__file__).resolve().parents[1] / "configs/formal/sft.json")
    c.update(backend=a.backend, gradient_accumulation=2, gradient_checkpointing=True)
    validate(c, world)
    output = Path(a.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    models = {}
    opts = {}
    sch = {}
    for role in ("teacher", "student", "auxiliary"):
        torch.manual_seed(42)
        config = LLaDAConfig(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            mlp_hidden_size=64,
            vocab_size=134548,
            embedding_size=134548,
            max_sequence_length=4096,
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
        raw = LLaDAForMultiModalGeneration(config)
        raw.model.reset_parameters()
        enable_conditioning(raw, region=True, token_type=False)
        enable_type_rope(raw)
        raw.requires_grad_(role != "teacher")
        raw.live_conditioning.token_type.requires_grad_(False)
        models[role] = wrap(raw, c, device, teacher=role == "teacher")
        if role != "teacher":
            opts[role] = optimizer(models[role], c)
            sch[role] = scheduler(opts[role], c)
    ema = CPUEMA(models["student"], c)
    tokenizer = lambda text: SimpleNamespace(input_ids=[1, 2, 3])

    class Stream:
        cursor = 0

        def next(self):
            self.cursor += world
            source = torch.full(
                (1, 4, 4), 126360 + rank + self.cursor, dtype=torch.long, device=device
            )
            mask = torch.zeros_like(source, dtype=torch.bool)
            mask[:, 1:3, 1:3] = True
            return dict(
                source=source,
                target=source + 1,
                original=mask,
                instruction="change color",
                uid=str(self.cursor + rank),
            )

    stream = Stream()
    engine = Engine(models, opts, sch, ema, tokenizer, c)
    calls = [0]
    handle = models["student"].register_forward_pre_hook(
        lambda *a: calls.__setitem__(0, calls[0] + 1)
    )
    sft = engine.sft(stream, 1)
    dimo = engine.dimo(stream, 2)
    progress = dict(
        student_updates=2, auxiliary_updates=1, global_sample_cursor=stream.cursor
    )
    origin = {
        "phase": "dimo",
        "teacher": {"fixture": "small-full-model-seed42"},
        "semantics": {"mixed_precision": "no"},
    }
    checkpoint.save(
        output / "checkpoint-2", models, opts, sch, ema, progress, c, origin
    )
    restored, _ = checkpoint.load(
        output / "checkpoint-2", models, opts, sch, ema, origin, mmap_weights=a.mmap
    )
    assert restored == progress
    # Exercise immutable Teacher reuse, final retention and Student/EMA exports.
    checkpoint.save(
        output / "checkpoint-3", models, opts, sch, ema, progress, c, origin
    )
    assert not (output / "checkpoint-3" / "teacher.pt").exists()
    checkpoint.export(output / "ema-export", models["student"], ema=ema)
    model = models["student"]
    model.eval()
    batch = stream.next()
    with torch.no_grad():
        sample = edit_tokens(
            model,
            tokenizer,
            batch,
            phase="sft",
            steps=2,
            cache=False,
            generator=torch.Generator(device=device).manual_seed(12),
        )
    from unittest.mock import patch

    from livemgt_training.full import validation

    class FixedValidation:
        cursor = 0

        def next(self):
            self.cursor += world
            source = torch.full((1, 4, 4), 126360, dtype=torch.long, device=device)
            original = torch.zeros_like(source, dtype=torch.bool)
            original[:, 1:3, 1:3] = True
            return dict(
                source=source,
                target=source + 1,
                original=original,
                instruction="change color",
                uid="fixed-validation",
            )

    validation_stream = FixedValidation()
    validation_config = dict(c, sft_sampling_steps=2)
    before_rng = torch.cuda.get_rng_state().clone()
    with patch.object(validation, "save_images", lambda *args, **kwargs: None):
        validation.validate(
            model,
            tokenizer,
            validation_stream,
            None,
            validation_config,
            output,
            "sft",
            2,
            ema,
        )
    assert validation_stream.cursor == 0 and torch.equal(
        before_rng, torch.cuda.get_rng_state()
    )
    handle.remove()
    metrics = {
        "fixture": "small model, not Lumina 8B",
        "backend": a.backend,
        "dtype": str(next(model.parameters()).dtype),
        "sft": sft,
        "dimo": dimo,
        "wrapper_forward_calls": calls[0],
        "restored_progress": restored,
        "sample_shape": list(sample.shape),
        "mmap": a.mmap,
        "fixed_validation_rng_restored": True,
    }
    (output / f"rank{rank}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ddp", "ddp_zero", "fsdp"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mmap", action="store_true")
    main(p.parse_args())
