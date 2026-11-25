"""Mask-guided editing shared by command-line inference and periodic validation."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import VQModel
from PIL import Image
from transformers import AutoTokenizer

from generators.image_to_image_generator import generate_i2i
from livemgt_training.precision import DTYPES, autocast, configure, inference_dtype
from livemgt_training.predict import sequence_inputs
from model import LLaDAForMultiModalGeneration
from utils.edit_mask import pixel_mask_to_vq
from utils.image_utils import decode_vq_to_image, encode_img_with_breaks


@torch.no_grad()
def prepare_source(source, mask, instruction, vae, height=512, width=512):
    stride = 2 ** (len(vae.config.block_out_channels) - 1)
    if height % stride or width % stride:
        raise ValueError("image dimensions must be multiples of VQ stride")
    image = (
        Image.open(source)
        .convert("RGB")
        .resize((width, height), Image.Resampling.LANCZOS)
    )
    binary = (
        Image.open(mask).convert("L").resize(image.size, Image.Resampling.NEAREST)
        if mask
        else Image.new("L", image.size, 255)
    )
    codes = encode_img_with_breaks(image, vae)
    codes = torch.tensor([v for v in codes if 126356 <= v < 134548], device=vae.device)
    grid = (height // stride, width // stride)
    batch = dict(
        source=codes.reshape(1, *grid),
        original=pixel_mask_to_vq(binary, grid).to(vae.device)[None],
        instruction=instruction,
    )
    return batch, image, binary


@torch.no_grad()
def edit_tokens(
    model,
    tokenizer,
    batch,
    *,
    phase="sft",
    steps=None,
    text_cfg=2.5,
    image_cfg=4.0,
    halo=1,
    cache=True,
    generator=None,
    precision=None,
):
    """SFT: original multi-step MaskGit. DiMO: one conditional-only sampling step."""
    if phase not in ("sft", "dimo"):
        raise ValueError("phase must be sft or dimo")
    if phase == "dimo":
        if steps not in (None, 1):
            raise ValueError("DiMO inference is one step")
        steps, text_cfg, image_cfg = 1, 0.0, 0.0
    else:
        steps = 18 if steps is None else steps
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    source = batch["source"]
    if not batch["original"].any():
        return source.reshape(1, -1).clone()
    from livemgt_training.core import generation_region

    region = generation_region(batch["original"], True, halo)
    initial = torch.where(region, 126336, source)
    ids, layout = sequence_inputs(tokenizer, batch, initial, region)
    ut, utl = sequence_inputs(tokenizer, batch, initial, region, "uncond_text")
    ui, uil = sequence_inputs(tokenizer, batch, initial, region, "uncond_image")
    h, w = source.shape[-2:]
    with autocast(precision or {"mixed_precision": "no"}):
        return generate_i2i(
            model,
            ids,
            seq_len=h * w,
            newline_every=w,
            timesteps=steps,
            cfg_scale=text_cfg,
            cfg_img=image_cfg,
            uncon_text=ut[:, : utl.prefix_length],
            uncon_image=ui[:, : uil.prefix_length],
            code_start=layout.prefix_length + 2,
            source_vq=source.reshape(1, -1),
            edit_mask=batch["original"][0],
            halo=halo,
            prefix_mode="sparse",
            condition_cache=cache,
            generator=generator,
            region_conditioning=True,
            token_type_embedding=False,
            token_type_rope=True,
            rope_mode="pretrained_type_phase",
        )


def save_images(
    batch, result, vae, output, height, width, source=None, mask=None, target=None
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = decode_vq_to_image(result, None, "", height, width, vae)
    image.save(output)
    source = source if source is not None else batch.get("source_image")
    mask = mask if mask is not None else batch.get("mask_image")
    if source is None:
        source = decode_vq_to_image(
            batch["source"].reshape(1, -1), None, "", height, width, vae
        )
    if mask is None:
        import numpy as np

        mask = Image.fromarray(
            (batch["original"][0].cpu().numpy().astype(np.uint8) * 255)
        ).resize((width, height), Image.Resampling.NEAREST)
    source.save(output.with_name(output.stem + "_source.png"))
    mask.save(output.with_name(output.stem + "_mask.png"))
    tiles = [source, mask.convert("RGB"), image]
    if target is not None:
        gt = batch.get("target_image")
        if gt is None:
            gt = decode_vq_to_image(target.reshape(1, -1), None, "", height, width, vae)
        gt.save(output.with_name(output.stem + "_gt.png"))
        tiles.append(gt)
    canvas = Image.new("RGB", (width * len(tiles), height))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * width, 0))
    canvas.save(output.with_name(output.stem + "_comparison.png"))
    return image


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", "--model", required=True)
    p.add_argument(
        "--vq-vae",
        "--vae_ckpt",
        dest="vq_vae",
        required=True,
        help="Directory containing vqvae/ (or a VQModel directory itself)",
    )
    p.add_argument("--source", "--image_path", dest="source", required=True)
    p.add_argument("--mask", "--edit_mask", dest="mask")
    p.add_argument("--instruction", "--prompt", dest="instruction", required=True)
    p.add_argument("--phase", choices=["sft", "dimo"], default="sft")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--steps", "--timesteps", dest="steps", type=int)
    p.add_argument(
        "--text-cfg", "--cfg_scale", dest="text_cfg", type=float, default=2.5
    )
    p.add_argument(
        "--image-cfg", "--cfg_img", dest="image_cfg", type=float, default=4.0
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--halo", type=int, default=1)
    p.add_argument("--cache", choices=["on", "off"], default="on")
    p.add_argument(
        "--precision",
        choices=list(DTYPES),
        help="Overrides inference_dtype from --config or the model export",
    )
    p.add_argument(
        "--config",
        help="Optional training/inference JSON; inference_dtype controls model and VQ compute precision",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()
    configure(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_info = Path(args.checkpoint) / "EXPORT_INFO.json"
    precision_config = (
        json.loads(Path(args.config).read_text())
        if args.config
        else (json.loads(export_info.read_text()) if export_info.is_file() else {})
    )
    dtype = inference_dtype(precision_config, args.precision)
    args.precision = str(dtype).removeprefix("torch.")
    model = (
        LLaDAForMultiModalGeneration.from_pretrained(
            args.checkpoint,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    vae = load_vae(args.vq_vae, device, dtype)
    height, width = args.height or args.resolution, args.width or args.resolution
    batch, source, mask = prepare_source(
        args.source, args.mask, args.instruction, vae, height, width
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = edit_tokens(
        model,
        tokenizer,
        batch,
        phase=args.phase,
        steps=args.steps,
        text_cfg=args.text_cfg,
        image_cfg=args.image_cfg,
        halo=args.halo,
        cache=args.cache == "on",
        generator=generator,
    )
    save_images(batch, result, vae, args.output, height, width, source, mask)
    config = dict(
        vars(args),
        height=height,
        width=width,
        steps=1 if args.phase == "dimo" else (args.steps or 18),
        guidance="conditional-only"
        if args.phase == "dimo"
        else "Lumina text + image CFG",
    )
    Path(args.output).with_suffix(".json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    print(f"Saved {args.output}", flush=True)


def load_vae(path, device, dtype=torch.float32):
    kw = {"subfolder": "vqvae"} if (Path(path) / "vqvae").is_dir() else {}
    vae = (
        VQModel.from_pretrained(path, torch_dtype=dtype, local_files_only=True, **kw)
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    return vae


if __name__ == "__main__":
    cli()
