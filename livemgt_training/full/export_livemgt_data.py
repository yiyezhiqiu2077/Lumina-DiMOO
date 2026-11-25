"""Bounded explicit bridge from upstream LiveMGT raw readers to portable files.

Uses source-specific filters/mask decoding and aligned crop from upstream; no
Flux tokenizer or text encoder is constructed. This is data preparation, not
training. Each source is exported separately; runtime mixture probabilities
remain independent of the number materialized from each source.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image


def tensor_image(t):
    x = (t.detach().cpu().clamp(0, 1) * 255).round().byte()
    if x.shape[0] == 1:
        return Image.fromarray(x[0].numpy(), "L")
    return Image.fromarray(x.permute(1, 2, 0).numpy(), "RGB")


def main(a):
    root = Path(a.upstream_root).resolve()
    sys.path.insert(0, str(root))
    torch.manual_seed(a.seed)
    common = dict(
        root_dir=a.data_root,
        tokenizer=None,
        size=a.resolution,
        rank=0,
        world_size=1,
        seed=a.seed,
        cache_dir=a.cache_dir,
        repeat=False,
    )
    if a.source == "interedit":
        from src.inter_edit_dataset import InterEditDataset

        dataset = InterEditDataset(
            **common,
            split="train",
            language=a.language,
            exclude_text_editing=a.exclude_text_editing,
            translations_dir=a.translations_dir,
        )
    elif a.source == "magicbrush":
        from src.magicbrush_dataset import MagicBrushDataset

        dataset = MagicBrushDataset(**common, split="train", language=a.language)
    else:
        from src.maskedit_dataset import MaskEditDataset

        dataset = MaskEditDataset(
            **common, configs=tuple(a.maskedit_configs.split(","))
        )
    # Capture stable raw-source identity before crop/resize, without modifying upstream files.
    original_process = dataset._process_sample

    def process(*items):
        result = original_process(*items)
        if result is None:
            return result
        row = items[0]
        if a.source == "magicbrush":
            group = str(row["img_id"])
        elif a.source == "interedit":
            group = (
                str(row.get("source_archive", ""))
                + "/"
                + str(row.get("source_file", row.get("source_path", row["sample_id"])))
            )
        else:
            import io

            raw = dataset._image(row["source"], items[2]).convert("RGB")
            buffer = io.BytesIO()
            raw.save(buffer, format="PNG")
            group = hashlib.sha256(buffer.getvalue()).hexdigest()
        instruction = result["instruction"]
        # Upstream InterEditTranslations overlays instruction before this call.
        # Never relabel a surviving untranslated/mixed Han prompt as English.
        if (
            not isinstance(instruction, str)
            or not re.search(r"[A-Za-z]", instruction)
            or re.search(
                r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]",
                instruction,
            )
        ):
            return None
        if row.get("instruction_original"):
            result["instruction_original"] = row["instruction_original"]
        result["raw_source_group"] = group
        return result

    dataset._process_sample = process
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    seen = set()
    for sample in dataset:
        sample_id = str(sample["sample_id"])
        if sample_id in seen:
            continue
        seen.add(sample_id)
        folder = out / str(len(rows))
        folder.mkdir()
        row = dict(
            dataset_name=a.source,
            group_id=sample["raw_source_group"],
            sample_uid=hashlib.sha256(
                (a.source + ":" + sample_id).encode()
            ).hexdigest(),
            instruction_en=sample["instruction"].strip(),
            instruction_original=sample.get("instruction_original"),
            upstream_split="train",
        )
        for key, field in [
            ("source", "reference_image"),
            ("target", "image"),
            ("mask", "edit_mask"),
        ]:
            p = folder / f"{key}.png"
            tensor_image(sample[field]).save(p)
            path_root = (
                Path(a.manifest_root).resolve() if a.manifest_root else out.resolve()
            )
            try:
                row[key] = str(p.resolve().relative_to(path_root))
            except ValueError as exc:
                raise ValueError(
                    "exported assets must lie within --manifest-root"
                ) from exc
            if key != "mask":
                row[f"{key}_content_sha256"] = hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
        rows.append(row)
        if len(rows) >= a.max_samples:
            break
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    import subprocess

    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    (out / "DATA_PROVENANCE.json").write_text(
        json.dumps(
            dict(
                upstream_commit=commit,
                source=a.source,
                split="train",
                resolution=a.resolution,
                seed=a.seed,
                rows=len(rows),
                repeat=False,
                exclude_text_editing=a.exclude_text_editing,
                language=a.language,
                translated_instructions=bool(a.translations_dir),
                asset_root="."
                if not a.manifest_root
                else str(out.resolve().relative_to(Path(a.manifest_root).resolve())),
                validation="must be a separate audited manifest; never split turns/images randomly",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument(
        "--source", choices=["interedit", "magicbrush", "maskedit"], required=True
    )
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", required=True)
    p.add_argument(
        "--language",
        choices=["en"],
        default="en",
        help="English only; --translations-dir overlays existing Inter-Edit translations before filtering",
    )
    p.add_argument(
        "--exclude-text-editing", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--manifest-root",
        help="Portable path root for exported files; defaults to output directory. Set asset_roots[source] to this root on each server.",
    )
    p.add_argument("--translations-dir")
    p.add_argument("--maskedit-configs", default="scene,infographics_en")
    args = p.parse_args()
    if args.max_samples < 1:
        p.error("max-samples must be positive")
    main(args)
