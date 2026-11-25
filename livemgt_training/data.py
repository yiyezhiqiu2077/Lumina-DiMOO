"""Existing file-backed triplets, deterministic stream, group/content split audit."""

import json, hashlib
from pathlib import Path
import torch
from PIL import Image
from utils.image_utils import encode_img_with_breaks
from utils.edit_mask import pixel_mask_to_vq


def read_manifest(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def audit_splits(train, validation):
    tg = {(r["dataset_name"], r["group_id"]) for r in train}
    vg = {(r["dataset_name"], r["group_id"]) for r in validation}

    def hashes(rows):
        return {
            r[k]
            for r in rows
            for k in [
                "source_content_sha256",
                "target_content_sha256",
                "source_sha256",
                "target_sha256",
            ]
            if r.get(k)
        }

    overlap = tg & vg
    shared = hashes(train) & hashes(validation)
    if overlap or shared:
        raise ValueError(
            f"train/validation leakage: {len(overlap)} groups, {len(shared)} image hashes"
        )
    return dict(
        train_rows=len(train),
        validation_rows=len(validation),
        group_overlap=0,
        image_hash_overlap=0,
        train_datasets=sorted(set(r["dataset_name"] for r in train)),
        validation_usage="evaluation only, never optimizer data",
    )


def encode_triplet(row, vae, resolution=512, roots=None):
    from .full.data import encode

    return encode(row, vae, resolution, roots or {})


class ReplayStream:
    def __init__(self, rows, vae, resolution=512, offset=0, roots=None):
        if not rows:
            raise ValueError("empty training manifest")
        self.roots = roots or {}
        self.rows = rows
        self.vae = vae
        self.resolution = resolution
        self.offset = offset
        self.cache = {}

    def next(self):
        index = self.offset % len(self.rows)
        self.offset += 1
        if index not in self.cache:
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[index] = encode_triplet(
                self.rows[index], self.vae, self.resolution, self.roots
            )
        return self.cache[index]


def audit_official_eval(train, train_manifest, dev_manifest, prior_report):
    """Recheck raw MagicBrush groups; pin earlier expensive official image audit."""
    dev = read_manifest(dev_manifest)
    train_groups = {
        str(r["source_group_id"]) for r in train if r["dataset_name"] == "magicbrush"
    }
    dev_groups = {
        str(r.get("source_group_id", r.get("cluster_id", r.get("group_id"))))
        for r in dev
    }
    if train_groups & dev_groups:
        raise ValueError("MagicBrush official dev group leakage")
    report = json.loads(Path(prior_report).read_text())
    current = hashlib.sha256(Path(train_manifest).read_bytes()).hexdigest()
    if (
        report["manifest_sha256"] != current
        or report["official_eval_exact_image_overlap"]
        or report["official_eval_quarantined_source_groups_retained"]
    ):
        raise ValueError("official evaluation image audit missing/mismatched/failed")
    return dict(
        magicbrush_dev_rows=len(dev),
        magicbrush_dev_groups=len(dev_groups),
        raw_source_group_overlap=0,
        pinned_official_image_overlap=0,
        prior_report=str(prior_report),
        prior_report_sha256=hashlib.sha256(Path(prior_report).read_bytes()).hexdigest(),
        image_audit_scope="reuse verified dataset-preparation audit tied to exact train manifest; no expensive rescan of test images",
    )
