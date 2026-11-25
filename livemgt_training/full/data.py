"""Aligned triplet manifests, reproducible shuffled streams and source mixtures."""

import hashlib
import random
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from livemgt_training.data import read_manifest
from utils.edit_mask import pixel_mask_to_vq
from utils.image_utils import encode_img_with_breaks


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def locate(row, key, roots):
    if key in row and isinstance(row[key], str):
        path = Path(row[key])
        if not path.is_absolute():
            path = Path(roots.get(row.get("dataset_name", ""), ".")) / path
    else:
        loc = row[
            {
                "source": "source_locator",
                "target": "target_locator",
                "mask": "region_locator",
            }[key]
        ]
        if loc["backend"] != "file":
            raise ValueError("export raw datasets to file-backed manifests first")
        path = Path(roots[row["dataset_name"]]) / loc["relative_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@torch.no_grad()
def encode(row, vae, resolution, roots):
    images = {
        k: Image.open(locate(row, k, roots)).convert("L" if k == "mask" else "RGB")
        for k in ("source", "target", "mask")
    }
    if images["mask"].size not in {images["source"].size, images["target"].size}:
        raise ValueError(
            "mask must align with source or registered target before resize"
        )
    images = {
        k: x.resize(
            (resolution, resolution),
            Image.Resampling.NEAREST if k == "mask" else Image.Resampling.LANCZOS,
        )
        for k, x in images.items()
    }
    size = resolution // 16

    def codes(image):
        raw = encode_img_with_breaks(image, vae)
        return torch.tensor(
            [v for v in raw if 126356 <= v < 134548], device=vae.device
        ).reshape(1, size, size)

    return dict(
        source=codes(images["source"]),
        target=codes(images["target"]),
        original=pixel_mask_to_vq(images["mask"], (size, size)).to(vae.device)[None],
        instruction=row.get("instruction_en", row.get("instruction")),
        uid=row["sample_uid"],
        source_image=images["source"],
        target_image=images["target"],
        mask_image=images["mask"],
    )


class Stream:
    """Cursor-based replay. Dataset auditing belongs to data preparation.

    Training ranks own disjoint slices; each local source is reshuffled per epoch.
    Mixture source choices are independent draws with the configured probabilities.
    Validation always iterates the same first samples on every rank.
    """

    def __init__(
        self,
        config,
        vae,
        rank,
        world,
        cursor=0,
        validation=False,
        state=None,
        reset_counts=False,
    ):
        self.c = config
        self.vae = vae
        self.rank = rank
        self.world = world
        self.cursor = cursor
        self.validation = validation
        self.cache = {}
        self.orders = {}
        self.device = vae.device if vae is not None else torch.device("cpu")
        manifest = config["validation_manifest" if validation else "train_manifest"]
        self.rows = read_manifest(manifest)
        if not self.rows:
            raise ValueError(f"empty manifest: {manifest}")
        self.sources = {}
        for row in self.rows:
            self.sources.setdefault(row["dataset_name"], []).append(row)
        weights = (
            config.get("data_mixture", {})
            if config["data_mode"] == "mixture" and not validation
            else {}
        )
        if weights:
            if (
                set(weights) - set(self.sources)
                or any(v <= 0 for v in weights.values())
                or abs(sum(weights.values()) - 1) > 1e-8
            ):
                raise ValueError(
                    "mixture probabilities must name available sources and sum to 1"
                )
            self.sources = {k: self.sources[k] for k in weights}
        elif not validation:
            self.sources = {"manifest": self.rows}
        self.keys = sorted(self.sources)
        self.weights = [weights.get(k, 1.0 / len(self.keys)) for k in self.keys]
        if not validation and any(len(rows) < world for rows in self.sources.values()):
            raise ValueError("each training source needs at least world_size samples")

        self.owned_sources = {
            key: rows[self.rank :: self.world] for key, rows in self.sources.items()
        }
        self.source_counts = {key: 0 for key in self.keys}
        if state is not None:
            self.load_state_dict(state)
        elif cursor and not validation and not reset_counts:
            # Compatibility for direct cursor-only callers. Checkpoints save counts,
            # so normal resume does not replay the source-choice prefix.
            for turn in range(cursor // world):
                self.source_counts[self._source(turn)] += 1

    def _source(self, turn):
        return random.Random(
            self.c["seed"] + 104729 + 1000003 * self.rank + turn
        ).choices(self.keys, self.weights)[0]

    def state_dict(self):
        return dict(
            version=2,
            rank=self.rank,
            world_size=self.world,
            cursor=self.cursor,
            source_counts=dict(self.source_counts),
        )

    def load_state_dict(self, state):
        if (
            state.get("version") != 2
            or state["rank"] != self.rank
            or state["world_size"] != self.world
            or state["cursor"] != self.cursor
            or set(state["source_counts"]) != set(self.keys)
        ):
            raise ValueError(
                "sampler identity/cursor differs; world-size migration requires explicit count reset"
            )
        counts = state["source_counts"]
        if any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError("invalid per-source sampler counts")
        self.source_counts = dict(counts)

    def _row(self):
        turn = self.cursor // self.world
        self.cursor += self.world
        if self.validation:
            return self.rows[turn % len(self.rows)]
        key = self._source(turn)
        owned = self.owned_sources[key]
        count = self.source_counts[key]
        self.source_counts[key] += 1
        epoch, index = divmod(count, len(owned))
        identity = (key, epoch)
        if identity not in self.orders:
            order = list(range(len(owned)))
            key_seed = int.from_bytes(
                hashlib.sha256(key.encode()).digest()[:4], "little"
            )
            random.Random(
                self.c["seed"] + key_seed + 1000003 * self.rank + epoch
            ).shuffle(order)
            # Only the current epoch per source is retained.
            self.orders = {k: v for k, v in self.orders.items() if k[0] != key}
            self.orders[identity] = order
        return owned[self.orders[identity][index]]

    def next(self):
        error = None
        batch = None
        try:
            row = self._row()
            uid = row["sample_uid"]
            resolution = self.c[
                "validation_resolution" if self.validation else "resolution"
            ]
            if uid not in self.cache:
                value = encode(row, self.vae, resolution, self.c["asset_roots"])
                if not value["original"].any():
                    raise ValueError(
                        f"empty original mask {uid}; curate manifest before training"
                    )
                if len(self.cache) >= 2:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[uid] = {
                    k: v.cpu() if torch.is_tensor(v) else v for k, v in value.items()
                }
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in self.cache[uid].items()
            }
        except Exception as exc:
            error = exc
        if dist.is_initialized():
            failed = torch.tensor(int(error is not None), device=self.device)
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            if failed.item():
                raise RuntimeError(
                    f"Invalid training/validation data on a rank; no rank skipped a step. Local error: {error!r}"
                ) from error
        elif error is not None:
            raise error
        return batch
