"""Fixed image-edit validation with independent RNG and no extra model replica."""

import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from inference.edit import edit_tokens, save_images
from livemgt_training.precision import compute_dtype, inference_dtype

from .checkpoint import ema_weights, rank_zero


@contextmanager
def independent_rng(seed, device):
    python, numpy = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[device]):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        try:
            yield torch.Generator(device=device).manual_seed(seed)
        finally:
            random.setstate(python)
            np.random.set_state(numpy)


@torch.no_grad()
def validate(model, tokenizer, stream, vae, config, output, phase, step, ema=None):
    if inference_dtype(config) != compute_dtype(config):
        raise ValueError(
            "validation inference_dtype differs from wrapped model compute; select matching precision or standalone inference"
        )
    device = torch.device("cuda", torch.cuda.current_device())
    was_training = model.training
    model.eval()
    original_cursor = stream.cursor
    stream.cursor = 0
    try:
        with independent_rng(config["seed"] + 100000, device), ema_weights(model, ema):
            for index in range(config.get("validation_samples", 1)):
                batch = stream.next()
                generator = torch.Generator(device=device).manual_seed(
                    config["seed"] + 100000 + index
                )
                # Every rank participates in FSDP forwards, using the same fixed sample.
                result = edit_tokens(
                    model,
                    tokenizer,
                    batch,
                    phase=phase,
                    steps=config.get("sft_sampling_steps", 18) if phase == "sft" else 1,
                    text_cfg=config.get("inference_text_cfg", 2.5),
                    image_cfg=config.get("inference_image_cfg", 4.0),
                    halo=config["halo"],
                    cache=False,
                    generator=generator,
                    precision=config,
                )

                def save_example():
                    folder = Path(output) / "validation" / f"step-{step}"
                    folder.mkdir(parents=True, exist_ok=True)
                    save_images(
                        batch,
                        result,
                        vae,
                        folder / f"{index}.png",
                        config["validation_resolution"],
                        config["validation_resolution"],
                        target=batch["target"],
                    )
                    (folder / f"{index}.json").write_text(
                        json.dumps(
                            {
                                "uid": batch["uid"],
                                "instruction": batch["instruction"],
                                "phase": phase,
                                "weights": "ema" if ema else "student",
                                "steps": config.get("sft_sampling_steps", 18)
                                if phase == "sft"
                                else 1,
                                "resolution": config["validation_resolution"],
                                "seed": config["seed"] + 100000 + index,
                                "cache": False,
                            },
                            indent=2,
                        )
                        + "\n"
                    )

                # All ranks propagate a rank-zero image/JSON write failure together.
                rank_zero(save_example)
    finally:
        stream.cursor = original_cursor
        model.train(was_training)
