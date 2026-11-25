"""Small CPU tests for export termination, sampler replay and checkpoint edges."""

import copy
import json
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

from livemgt_training.full import checkpoint, data
from livemgt_training.full.config import load
from livemgt_training.full.runtime import CPUEMA, check_finite
from livemgt_training.precision import inference_dtype

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load(ROOT / "configs/formal/sft.json")


def make_stream(tmp_path, rank=0, world=2, **kwargs):
    manifest = tmp_path / "rows.jsonl"
    rows = [
        dict(dataset_name=k, sample_uid=f"{k}:{i}")
        for k in ("interedit", "magicbrush", "maskedit")
        for i in range(12)
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = config()
    c.update(
        data_mode="mixture",
        data_mixture={"interedit": 0.9, "magicbrush": 0.05, "maskedit": 0.05},
        train_manifest=str(manifest),
        validation_manifest=str(manifest),
    )
    return data.Stream(c, None, rank, world, **kwargs)


def test_independent_source_epochs_and_resume(tmp_path):
    stream = make_stream(tmp_path)
    for key in stream.keys:
        assert (
            stream.owned_sources[key]
            == stream.sources[key][stream.rank :: stream.world]
        )

    class NoReslice(list):
        def __getitem__(self, index):
            if isinstance(index, slice):
                raise AssertionError("source shard was reconstructed during reading")
            return super().__getitem__(index)

    stream.sources = {key: NoReslice(rows) for key, rows in stream.sources.items()}
    selected = {k: [] for k in stream.keys}
    for _ in range(450):
        row = stream._row()
        selected[row["dataset_name"]].append(row["sample_uid"])
    for key, values in selected.items():
        assert len(values) >= 6
        for offset in range(0, len(values) - 5, 6):
            assert len(set(values[offset : offset + 6])) == 6
        assert stream.source_counts[key] == len(values)
    state = stream.state_dict()
    resumed = make_stream(tmp_path, cursor=stream.cursor, state=state)
    assert [stream._row() for _ in range(60)] == [resumed._row() for _ in range(60)]
    migrated = make_stream(
        tmp_path, rank=0, world=3, cursor=state["cursor"], reset_counts=True
    )
    assert not any(migrated.source_counts.values())
    with pytest.raises(ValueError, match="sampler"):
        make_stream(tmp_path, rank=0, world=3, cursor=state["cursor"], state=state)


@pytest.mark.parametrize("source", ["interedit", "magicbrush", "maskedit"])
def test_export_finite_and_relative(tmp_path, monkeypatch, source):
    from livemgt_training.full import export_livemgt_data as exporter

    observed = {}

    class Dataset:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def _image(self, *args):
            return Image.new("RGB", (2, 2))

        def _process_sample(self, row, *args):
            image = torch.ones(3, 2, 2)
            return dict(
                sample_id=row["sample_id"],
                instruction=row["instruction"],
                reference_image=image,
                image=image,
                edit_mask=image[:1],
            )

        def __iter__(self):
            assert observed["repeat"] is False
            for i, text in [(0, "Make it blue"), (0, "Make it blue"), (1, "删除物体")]:
                row = dict(
                    sample_id=i,
                    instruction=text,
                    source="bytes",
                    img_id="image",
                    source_file="src",
                )
                args = (row,) if source == "interedit" else (row, i, "file")
                sample = self._process_sample(*args)
                if sample is not None:
                    yield sample

    modules = {
        "interedit": ("src.inter_edit_dataset", "InterEditDataset"),
        "magicbrush": ("src.magicbrush_dataset", "MagicBrushDataset"),
        "maskedit": ("src.maskedit_dataset", "MaskEditDataset"),
    }
    module_name, class_name = modules[source]
    parent = ModuleType("src")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "src", parent)
    module = ModuleType(module_name)
    setattr(module, class_name, Dataset)
    monkeypatch.setitem(sys.modules, module_name, module)
    import subprocess

    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **kw: "reference-commit\n"
    )
    args = SimpleNamespace(
        upstream_root=str(tmp_path),
        data_root="relative/raw",
        resolution=512,
        seed=42,
        cache_dir=str(tmp_path / "cache"),
        source=source,
        language="en",
        exclude_text_editing=True,
        translations_dir="existing-translations",
        maskedit_configs="scene",
        output=str(tmp_path / source),
        max_samples=99,
        manifest_root=None,
    )
    exporter.main(args)
    rows = [
        json.loads(x)
        for x in (Path(args.output) / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1 and rows[0]["instruction_en"] == "Make it blue"
    for key in ("source", "target", "mask"):
        assert not Path(rows[0][key]).is_absolute()
        assert data.locate(rows[0], key, {source: args.output}).is_file()
    if source == "interedit":
        assert (
            observed["exclude_text_editing"]
            and observed["translations_dir"] == "existing-translations"
        )


def origin():
    return dict(
        base={"files": {"weights": "base"}},
        vq_vae={"files": {"weights": "vq"}},
        teacher={"files": {"weights": "teacher"}},
        data_versions={"release": "v1"},
        train="train",
        validation="val",
        phase="sft",
        semantics=dict(
            gradient_accumulation=8,
            resolved_global_batch_size=16,
            micro_batch_size=1,
            backend="ddp",
            learning_rate=1e-6,
        ),
    )


def test_resume_rules_and_asset_identity(tmp_path):
    o = origin()
    meta = dict(origin=copy.deepcopy(o), world_size=2)
    assert checkpoint.resume_rules(meta, o, 2)["data_order_exact"]
    for key in ("gradient_accumulation", "resolved_global_batch_size"):
        changed = copy.deepcopy(o)
        changed["semantics"][key] *= 2
        with pytest.raises(ValueError, match="semantics"):
            checkpoint.resume_rules(meta, changed, 2)
        assert checkpoint.resume_rules(meta, changed, 2, allow_batch_change=True)[
            "changes"
        ]
    changed = copy.deepcopy(o)
    changed["semantics"]["backend"] = "fsdp"
    with pytest.raises(ValueError):
        checkpoint.resume_rules(meta, changed, 2)
    assert not checkpoint.resume_rules(meta, changed, 2, allow_backend_change=True)[
        "rng_exact_replay"
    ]
    with pytest.raises(ValueError):
        checkpoint.resume_rules(meta, o, 3)
    migrated = checkpoint.resume_rules(meta, o, 3, allow_world_change=True)
    assert not migrated["data_order_exact"] and not migrated["rng_exact_replay"]
    for key in ("vq_vae", "teacher", "data_versions"):
        changed = copy.deepcopy(o)
        changed[key] = {"changed": True}
        with pytest.raises(ValueError, match="identity"):
            checkpoint.resume_rules(meta, changed, 2)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.safetensors").write_bytes(b"first")
    before = checkpoint.asset_identity(tmp_path)
    (tmp_path / "weights.safetensors").write_bytes(b"second")
    assert before != checkpoint.asset_identity(tmp_path)


def test_ema_first_update_and_formal_teacher_logic():
    model = torch.nn.Linear(2, 2, bias=False)
    c = config()
    ema = CPUEMA(model, c)
    previous = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1)
    assert ema.update(model, 1) == 0.1
    torch.testing.assert_close(ema.state["weight"], previous + 0.9)
    o = origin()
    assert not checkpoint.training_provenance(c, {"student_updates": 19999}, o)[
        "formal_sft"
    ]
    assert checkpoint.training_provenance(c, {"student_updates": 20000}, o)[
        "formal_sft"
    ]
    o["phase"] = "dimo"
    assert not checkpoint.training_provenance(c, {"student_updates": 20000}, o)[
        "formal_sft"
    ]
    assert inference_dtype({"inference_dtype": "bfloat16"}) == torch.bfloat16
    assert inference_dtype({"inference_dtype": "bfloat16"}, "float32") == torch.float32


def test_export_precision_metadata(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"amp_bf16": True, "torch_dtype": "bfloat16", "precision": "amp_bf16"}
        )
    )
    checkpoint.export_metadata(tmp_path, torch.float32)
    c = json.loads((tmp_path / "config.json").read_text())
    assert c["amp_bf16"] is False and c["torch_dtype"] == "float32"
    assert "precision" not in c


@pytest.fixture
def gloo(tmp_path):
    dist.init_process_group(
        "gloo", init_method="file://" + str(tmp_path / "pg"), rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_checkpoint_interruption_sampler_and_teacher(tmp_path, monkeypatch, gloo):
    # Small CPU state files, no model training/GPU/VQ execution.
    from transformers import PretrainedConfig

    model = torch.nn.Linear(2, 2)
    model.config = PretrainedConfig()
    c = config()
    c["resolved_global_batch_size"] = 8
    ema = CPUEMA(model, c)
    stream = make_stream(tmp_path, world=1)
    for _ in range(8):
        stream._row()
    o = origin()
    progress = dict(
        student_updates=1, global_sample_cursor=stream.cursor, auxiliary_updates=0
    )
    monkeypatch.setattr(checkpoint, "rng_state", lambda: {"test": True})
    monkeypatch.setattr(checkpoint, "set_rng", lambda s: None)
    cp = tmp_path / "checkpoint-1"
    cp.mkdir()
    (cp / "partial").write_text("interrupted old writer")
    checkpoint.save(
        cp, {"student": model}, {}, {}, ema, progress, c, o, stream=stream, final=True
    )
    assert (cp / "COMPLETE").is_file() and json.loads(
        (tmp_path / "latest.json").read_text()
    )["checkpoint"] == cp.name
    assert list(tmp_path.glob(".checkpoint-1.interrupted-*"))
    restored, recovery = checkpoint.load(
        cp, {"student": model}, {}, {}, ema, o, stream_required=True
    )
    assert restored == progress and recovery["sampler_state"] == stream.state_dict()
    # A new interrupted save leaves no published final directory or latest change.
    original = torch.save

    def fail(*args):
        raise OSError("injected interrupted writer")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match="injected interrupted writer"):
        checkpoint.save(
            tmp_path / "checkpoint-2",
            {"student": model},
            {},
            {},
            ema,
            progress,
            c,
            o,
            stream=stream,
        )
    assert not (tmp_path / "checkpoint-2").exists()
    monkeypatch.setattr(torch, "save", original)
    checkpoint.save(
        tmp_path / "checkpoint-2",
        {"student": model},
        {},
        {},
        ema,
        progress,
        c,
        o,
        stream=stream,
    )
    reference = checkpoint.save_teacher(tmp_path, model, c, o["teacher"])
    assert reference["model_identity"] == o["teacher"]
    with pytest.raises(RuntimeError, match="different Teacher"):
        checkpoint.save_teacher(tmp_path, model, c, {"files": {"weights": "different"}})


def _nan_rank(rank, init, output):
    dist.init_process_group(
        "gloo", init_method=init, rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:
        try:
            check_finite(
                torch.tensor(float("nan") if rank == 1 else 1.0), "rank-local test"
            )
        except RuntimeError as exc:
            Path(output, f"rank-{rank}").write_text(str(exc))
        else:
            raise AssertionError("all ranks must reject NaN before backward")
    finally:
        dist.destroy_process_group()


def test_nan_stops_all_cpu_ranks(tmp_path):
    mp.spawn(
        _nan_rank,
        args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert all(
        "nonfinite rank-local test" in (tmp_path / f"rank-{r}").read_text()
        for r in range(2)
    )


def test_real_small_model_export_metadata_and_provenance(tmp_path, monkeypatch, gloo):
    from model import LLaDAForMultiModalGeneration
    from model.configuration_llada import LLaDAConfig
    from transformers import AutoTokenizer

    architecture = LLaDAConfig(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        mlp_hidden_size=64,
        vocab_size=256,
        embedding_size=256,
        block_type="llama",
        rope=True,
        weight_tying=False,
        activation_type="silu",
        layer_norm_type="rms",
        include_bias=False,
    )
    model = LLaDAForMultiModalGeneration(architecture)
    model.model.reset_parameters()
    model.config.amp_bf16 = True
    c = config()
    ema = CPUEMA(model, c)
    progress = {
        "student_updates": 20000,
        "auxiliary_updates": 0,
        "global_sample_cursor": 0,
    }
    o = origin()
    monkeypatch.setattr(checkpoint, "rng_state", lambda: {"test": True})
    cp = tmp_path / "checkpoint-20000"
    checkpoint.save(cp, {"student": model}, {}, {}, ema, progress, c, o, final=True)
    checkpoint.export(tmp_path / "direct", model, dtype=torch.float32)

    # Standalone export uses a tokenizer stub only; all model tensor IO is real.
    class Tokenizer:
        def save_pretrained(self, path):
            return None

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    checkpoint.export_checkpoint(cp, tmp_path / "standalone", dtype=torch.float32)
    direct = json.loads((tmp_path / "direct/config.json").read_text())
    standalone = json.loads((tmp_path / "standalone/config.json").read_text())
    assert direct["amp_bf16"] == standalone["amp_bf16"] == False
    assert direct["torch_dtype"] == standalone["torch_dtype"] == "float32"
    provenance = json.loads(
        (tmp_path / "standalone/TRAINING_PROVENANCE.json").read_text()
    )
    assert (
        provenance["formal_sft"]
        == checkpoint.training_provenance(c, progress, o)["formal_sft"]
    )


def _log_rank(rank, init, output):
    from livemgt_training.full.train import log

    dist.init_process_group(
        "gloo", init_method=init, rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:
        if rank == 0:
            (Path(output) / "train.jsonl").mkdir()
        dist.barrier()
        try:
            log(Path(output), "write_error")
        except RuntimeError as exc:
            Path(output, f"log-rank-{rank}").write_text(str(exc))
        else:
            raise AssertionError("disk write error must reach every rank")
    finally:
        dist.destroy_process_group()


def test_log_write_failure_reaches_all_ranks(tmp_path):
    mp.spawn(
        _log_rank,
        args=("file://" + str(tmp_path / "log-rendezvous"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert all(
        "IsADirectoryError" in (tmp_path / f"log-rank-{r}").read_text()
        for r in range(2)
    )


@pytest.mark.parametrize(
    "teacher_vq", [None, {"files": {"weight": "other"}}, {"files": {"weight": "same"}}]
)
def test_formal_dimo_checks_teacher_vq(tmp_path, monkeypatch, teacher_vq):
    from livemgt_training.full import train

    c = config()
    c["teacher_status"] = "FORMAL_SFT_TRAINED"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(c))
    current = {"files": {"weight": "same"}}
    provenance = {
        "formal_sft": True,
        "origin": {} if teacher_vq is None else {"vq_vae": teacher_vq},
    }
    monkeypatch.setattr(train, "initialize", lambda: (0, 1, torch.device("cpu")))
    monkeypatch.setattr(
        train,
        "content_origin",
        lambda *a: {"teacher": {"provenance": provenance}, "vq_vae": current},
    )
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)

    class ReachedTokenizer(Exception):
        pass

    def stop(*a, **kw):
        raise ReachedTokenizer()

    monkeypatch.setattr(train.AutoTokenizer, "from_pretrained", stop)
    args = SimpleNamespace(
        config=str(path),
        set=[],
        phase="dimo",
        resume=None,
        output=str(tmp_path / "run"),
        max_updates=1,
        allow_untrained_teacher=False,
    )
    if teacher_vq == current:
        with pytest.raises(ReachedTokenizer):
            train.main(args)
    else:
        with pytest.raises(ValueError, match="VQ-VAE identity"):
            train.main(args)
