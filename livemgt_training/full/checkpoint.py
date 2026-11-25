"""Portable full model + named optimizer checkpoint, collective FSDP gather.

CPU files are streamed role-by-role; no rank keeps four full model states at once.
Every rank participates in FSDP/ZeRO collectives. Full exports support world-size
changes; exact per-rank RNG replay applies only at the original world size.
"""

import contextlib
import gc
import json
import os
import random
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from .data import digest
from .runtime import collective_call, named, unwrap


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rank_zero(function):
    return collective_call(lambda: function() if dist.get_rank() == 0 else None)


def asset_identity(path, subfolder=None):
    """Content identity, computed once at startup, independent of server paths."""
    root = Path(path)
    if subfolder and (root / subfolder).is_dir():
        root = root / subfolder
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError(f"model configuration missing: {root}")
    weights = sorted(set(root.glob("*.safetensors")) | set(root.glob("*.bin")))
    if not weights:
        raise ValueError(f"model weight files missing: {root}")
    files = [root / "config.json", *weights]
    files += sorted(root.glob("*.index.json"))
    # Provenance binds Teacher eligibility and dataset/weight lineage, too.
    if (root / "TRAINING_PROVENANCE.json").is_file():
        files.append(root / "TRAINING_PROVENANCE.json")
    identity = dict(kind="pretrained_model", files={p.name: digest(p) for p in files})
    provenance = root / "TRAINING_PROVENANCE.json"
    if provenance.is_file():
        identity["provenance"] = json.loads(provenance.read_text())
    return identity


def training_provenance(config, progress, origin, role="student"):
    base = origin.get("base")
    known_source = isinstance(base, dict) and bool(
        base.get("files") or base.get("config")
    )
    formal = (
        config.get("mode") == "full"
        and origin.get("phase") == "sft"
        and known_source
        and progress["student_updates"] >= config["sft_updates"]
    )
    return dict(
        formal_sft=formal,
        phase=origin.get("phase"),
        updates=progress["student_updates"],
        origin=origin,
        role=role,
    )


def resume_rules(
    meta,
    origin,
    world,
    *,
    allow_world_change=False,
    allow_batch_change=False,
    allow_backend_change=False,
):
    saved = meta["origin"]
    if {k: v for k, v in saved.items() if k != "semantics"} != {
        k: v for k, v in origin.items() if k != "semantics"
    }:
        raise ValueError(
            "checkpoint model/data identity mismatch; use model initialization for different assets or legacy identity schemas"
        )
    previous, current = saved["semantics"], origin["semantics"]
    if previous.keys() != current.keys():
        raise ValueError(
            "checkpoint semantic schema differs; initialize from exported weights instead of claiming exact resume"
        )
    changes = {
        k: {"saved": previous[k], "requested": current[k]}
        for k in previous
        if previous[k] != current[k]
    }
    batch = {"micro_batch_size", "gradient_accumulation", "resolved_global_batch_size"}
    allowed = (batch if allow_batch_change else set()) | (
        {"backend"} if allow_backend_change else set()
    )
    if set(changes) - allowed:
        raise ValueError(
            f"checkpoint training semantics changed: {changes}; batch/backend changes require their explicit resume flags"
        )
    world_changed = meta["world_size"] != world
    if world_changed and not allow_world_change:
        raise ValueError(
            "world size differs; explicitly enable --allow-world-size-change"
        )
    return dict(
        world_size_changed=world_changed,
        changes=changes,
        rng_exact_replay=not (world_changed or changes),
        data_order_exact=not world_changed,
        sampler_policy="reset per-source counts on new rank slices; no exact replay"
        if world_changed
        else "restore per-rank source counts and cursor",
    )


def export_metadata(path, dtype):
    config_path = Path(path) / "config.json"
    config = json.loads(config_path.read_text())
    config["torch_dtype"] = str(dtype).removeprefix("torch.")
    config["amp_bf16"] = dtype == torch.bfloat16
    config.pop("precision", None)
    atomic_text(config_path, json.dumps(config, indent=2) + "\n")


def full_context(model):
    return (
        FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )
        if isinstance(model, FSDP)
        else contextlib.nullcontext()
    )


def model_state(model):
    with full_context(model):
        state = (
            model.state_dict()
            if isinstance(model, FSDP)
            else unwrap(model).state_dict()
        )
    if dist.get_rank() != 0:
        return {}
    return {k: v.detach().cpu() for k, v in state.items()}


def named_optimizer(model, opt):
    if isinstance(model, FSDP):
        with full_context(model):
            return FSDP.optim_state_dict(model, opt)
    if hasattr(opt, "consolidate_state_dict"):
        opt.consolidate_state_dict(to=0)
    if dist.get_rank() != 0:
        return {}
    raw = opt.state_dict()
    names = [n for n, p in named(model).items() if p.requires_grad]
    ids = [i for g in raw["param_groups"] for i in g["params"]]
    mapping = dict(zip(ids, names))
    return dict(
        state={mapping[k]: v for k, v in raw["state"].items()},
        param_groups=[
            dict(g, params=[mapping[i] for i in g["params"]])
            for g in raw["param_groups"]
        ],
    )


def restore_optimizer(model, opt, state):
    if isinstance(model, FSDP):
        local = FSDP.scatter_full_optim_state_dict(
            state if dist.get_rank() == 0 else None, model, optim=opt
        )
    else:
        # ZeRO.load_state_dict accepts the global parameter order, not its local optimizer order.
        ids = list(range(sum(len(g["params"]) for g in opt.param_groups)))
        names = [n for n, p in named(model).items() if p.requires_grad]
        mapping = dict(zip(names, ids))
        local = dict(
            state={mapping[k]: v for k, v in state["state"].items()},
            param_groups=[
                dict(g, params=[mapping[n] for n in g["params"]])
                for g in state["param_groups"]
            ],
        )
    opt.load_state_dict(local)


def load_model(model, path, mmap_weights=False):
    # FSDP full load requires full tensors on each rank. CPU RAM requirement is
    # explicit in report; different world sizes are repartitioned by FSDP.
    state = collective_call(
        lambda: torch.load(
            path, map_location="cpu", weights_only=True, mmap=mmap_weights
        )
    )
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(state, strict=True)
    else:
        unwrap(model).load_state_dict(state, strict=True)
    del state
    gc.collect()


def rng_state():
    return dict(
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(),
        python=random.getstate(),
        numpy=np.random.get_state(),
    )


def set_rng(s):
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state(s["cuda"])
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])


def write(path, state):
    def writer():
        torch.save(state, path)
        atomic_text(path.with_suffix(path.suffix + ".sha256"), digest(path) + "\n")

    rank_zero(writer)


def complete_directory(path):
    """Recognize published state without scanning tensor contents."""
    path = Path(path)
    try:
        marker = (path / "COMPLETE").read_text()
        if marker.startswith("{"):
            json.loads(marker)
        if (path / "metadata.json").is_file():
            meta = json.loads((path / "metadata.json").read_text())
            required = [path / "ema.pt"]
            for role in meta["roles"]:
                if role == "teacher" and meta.get("teacher_reference"):
                    required.append(path / meta["teacher_reference"]["path"])
                else:
                    required.append(path / f"{role}.pt")
                if role in meta["schedulers"]:
                    required.append(path / f"{role}_optimizer.pt")
            required += [path / f"rng-rank{r}.pt" for r in range(meta["world_size"])]
            return all(p.is_file() and p.stat().st_size for p in required)
        identity = json.loads(marker)
        return bool(
            identity.get("model_identity")
            and (path / "teacher.pt").is_file()
            and (path / "teacher.pt.sha256").is_file()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def temporary_directory(path):
    """Fresh staging directory; leave interrupted data intact under a hidden name."""
    path = Path(path)

    def begin():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if complete_directory(path):
                raise FileExistsError(f"complete checkpoint already exists: {path}")
            path.rename(
                path.with_name("." + path.name + ".interrupted-" + uuid.uuid4().hex)
            )
        name = "." + path.name + ".tmp-" + uuid.uuid4().hex
        (path.parent / name).mkdir()
        return name

    name = rank_zero(begin)
    names = [name]
    dist.broadcast_object_list(names, src=0)
    return path.parent / names[0]


def save(
    path,
    models,
    opts,
    schedulers,
    ema,
    progress,
    c,
    origin,
    tokenizer=None,
    stream=None,
    final=False,
):
    path = Path(path)
    staged = temporary_directory(path)

    def configuration():
        unwrap(models["student"]).config.to_json_file(staged / "model_config.json")
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(staged / "tokenizer")

    rank_zero(configuration)
    teacher_reference = None
    for role, model in models.items():
        if role == "teacher":
            teacher_reference = save_teacher(
                path.parent, model, c, origin.get("teacher")
            )
            continue
        state = model_state(model)
        write(staged / f"{role}.pt", state)
        del state
        gc.collect()
        if role in opts:
            state = named_optimizer(model, opts[role])
            write(staged / f"{role}_optimizer.pt", state)
            del state
            gc.collect()
    with ema_weights(models["student"], ema):
        state = model_state(models["student"])
        write(staged / "ema.pt", state)
        del state
    collective_call(
        lambda: torch.save(rng_state(), staged / f"rng-rank{dist.get_rank()}.pt")
    )
    sampler_states = [None] * dist.get_world_size()
    dist.all_gather_object(sampler_states, stream.state_dict() if stream else None)

    def finish():
        meta = dict(
            version=4,
            final=final,
            formal_sft=training_provenance(c, progress, origin)["formal_sft"],
            teacher_reference=teacher_reference,
            world_size=dist.get_world_size(),
            config=c,
            origin=origin,
            progress=progress,
            sampler_states=sampler_states,
            schedulers={k: v.state_dict() for k, v in schedulers.items()},
            roles=list(models),
        )
        atomic_text(staged / "metadata.json", json.dumps(meta, indent=2) + "\n")
        files = {
            str(p.relative_to(staged)): digest(p)
            for p in staged.rglob("*")
            if p.is_file() and p.suffix not in (".pt", ".sha256")
        }
        files.update({p.name: digest(p) for p in staged.glob("rng-rank*.pt")})
        atomic_text(staged / "COMPLETE", json.dumps(files, indent=2) + "\n")
        os.replace(staged, path)
        atomic_text(
            path.parent / "latest.json",
            json.dumps({"checkpoint": path.name, "update": progress["student_updates"]})
            + "\n",
        )

    rank_zero(finish)


def load(
    path,
    models,
    opts,
    schedulers,
    ema,
    origin,
    allow_world_change=False,
    verify_files=True,
    mmap_weights=False,
    allow_batch_change=False,
    allow_backend_change=False,
    stream_required=False,
):
    path = Path(path)

    def metadata():
        if not (path / "COMPLETE").is_file():
            raise ValueError("incomplete checkpoint")
        if verify_files:
            marker = (path / "COMPLETE").read_text()
            if marker.startswith("{"):
                for name, sha in json.loads(marker).items():
                    if digest(path / name) != sha:
                        raise ValueError(f"checkpoint SHA256 mismatch: {name}")
        return json.loads((path / "metadata.json").read_text())

    meta = collective_call(metadata)
    recovery = collective_call(
        lambda: resume_rules(
            meta,
            origin,
            dist.get_world_size(),
            allow_world_change=allow_world_change,
            allow_batch_change=allow_batch_change,
            allow_backend_change=allow_backend_change,
        )
    )
    changed = recovery["world_size_changed"]
    states = meta.get("sampler_states", [])
    if (
        stream_required
        and not changed
        and (len(states) != dist.get_world_size() or not all(states))
    ):
        raise ValueError(
            "checkpoint lacks per-source sampler state; initialize from model weights instead"
        )
    for role, model in models.items():
        role_path = path / f"{role}.pt"
        if role == "teacher" and meta.get("teacher_reference"):
            ref = meta["teacher_reference"]
            role_path = path / ref["path"]

            def check_teacher():
                if role_path.with_suffix(".pt.sha256").read_text().strip() != ref[
                    "sha256"
                ] or ref.get("model_identity") != origin.get("teacher"):
                    raise ValueError("frozen Teacher identity differs from checkpoint")

            collective_call(check_teacher)
        prepare_read(role_path, verify_files)
        load_model(model, role_path, mmap_weights)
        if role in opts:
            prepare_read(path / f"{role}_optimizer.pt", verify_files)
            state = collective_call(
                lambda: (
                    torch.load(
                        path / f"{role}_optimizer.pt",
                        map_location="cpu",
                        weights_only=False,
                        mmap=mmap_weights,
                    )
                    if dist.get_rank() == 0 or not isinstance(model, FSDP)
                    else None
                )
            )
            restore_optimizer(model, opts[role], state)
            del state
            gc.collect()
            schedulers[role].load_state_dict(meta["schedulers"][role])
    # Load EMA into Student's shard layout, then restore Student parameters.
    student = models["student"]
    backup = {n: p.detach().cpu().clone() for n, p in named(student).items()}
    prepare_read(path / "ema.pt", verify_files)
    load_model(student, path / "ema.pt", mmap_weights)
    for n, p in named(student).items():
        ema.state[n].copy_(p.detach().to(ema.state[n].device))
    with torch.no_grad():
        for n, p in named(student).items():
            p.copy_(backup[n].to(p.device))
    if not changed:
        set_rng(
            collective_call(
                lambda: torch.load(
                    path / f"rng-rank{dist.get_rank()}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            )
        )
    else:
        seed = (
            meta["config"]["seed"]
            + dist.get_rank()
            + meta["progress"]["student_updates"]
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    recovery["sampler_state"] = (
        None
        if changed
        else meta.get("sampler_states", [None] * dist.get_world_size())[dist.get_rank()]
    )
    recovery["portable_named_optimizer"] = True
    recovery["file_sha256_verified"] = verify_files
    return meta["progress"], recovery


def export(path, student, dtype=torch.float32, ema=None, inference_dtype="float32"):
    path = Path(path)
    with ema_weights(student, ema):
        state = model_state(student)

    def write_export():
        path.mkdir(parents=True, exist_ok=False)
        exported_state = {
            n: v.to(dtype) if v.is_floating_point() and "live_type_rope" not in n else v
            for n, v in state.items()
        }
        unwrap(student).save_pretrained(
            path,
            state_dict=exported_state,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        export_metadata(path, dtype)
        (path / "EXPORT_INFO.json").write_text(
            json.dumps(
                dict(
                    purpose="inference weights only; no optimizer/RNG",
                    world_size=dist.get_world_size(),
                    dtype=str(dtype),
                    role="ema" if ema else "student",
                    inference_dtype=inference_dtype,
                    rope="pretrained_type_phase",
                )
            )
        )

    rank_zero(write_export)
    state.clear()
    gc.collect()
    dist.barrier()


def prepare_read(path, verify):
    # Warm just the NEXT file, not all role files at once. Concurrent mmap
    # page faults on a cold multi-GB archive caused measured shared-disk stalls.
    # Sequential SHA verification also serves as prefetch, without holding a
    # full tensor buffer in Python. Page cache remains OS-reclaimable.
    def read():
        import hashlib

        h = hashlib.sha256() if verify else None
        with path.open("rb", buffering=0) as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            for chunk in iter(lambda: f.read(64 * 1024**2), b""):
                if h is not None:
                    h.update(chunk)
        if (
            verify
            and h.hexdigest()
            != path.with_suffix(path.suffix + ".sha256").read_text().strip()
        ):
            raise ValueError(f"checkpoint SHA256 mismatch: {path.name}")

    rank_zero(read)


@contextlib.contextmanager
def ema_weights(student, ema=None):
    """Temporarily swap local shards; all ranks must enter for FSDP forwards."""
    if ema is None:
        yield
        return
    backup = {n: p.detach().cpu().clone() for n, p in named(student).items()}
    try:
        with torch.no_grad():
            for n, p in named(student).items():
                p.copy_(ema.state[n].to(p.device))
        yield
    finally:
        with torch.no_grad():
            for n, p in named(student).items():
                p.copy_(backup[n].to(p.device))


def save_teacher(root, teacher, config, source_identity=None):
    """One immutable, content-bound Teacher per run."""
    folder = Path(root) / "teacher"
    if not source_identity:
        raise ValueError(
            "Teacher source identity is required; refusing unbound teacher/ reuse"
        )

    def existing():
        if not (folder / "COMPLETE").is_file():
            return None
        identity = json.loads((folder / "COMPLETE").read_text())
        if identity.get("model_identity") != source_identity:
            raise ValueError("shared teacher/ belongs to a different Teacher model")
        if identity["sha256"] != (folder / "teacher.pt.sha256").read_text().strip():
            raise ValueError("shared Teacher weight identity differs")
        return identity

    identity = rank_zero(existing)
    objects = [identity]
    dist.broadcast_object_list(objects, src=0)
    identity = objects[0]
    if identity is None:
        staged = temporary_directory(folder)
        state = model_state(teacher)
        write(staged / "teacher.pt", state)
        del state
        gc.collect()

        def finish():
            value = dict(
                source=config.get("teacher_checkpoint"),
                model_identity=source_identity,
                sha256=(staged / "teacher.pt.sha256").read_text().strip(),
            )
            atomic_text(staged / "COMPLETE", json.dumps(value, indent=2) + "\n")
            os.replace(staged, folder)

        rank_zero(finish)
        identity = collective_call(
            lambda: json.loads((folder / "COMPLETE").read_text())
        )
    return {"path": "../teacher/teacher.pt", **identity}


def prune(root, keep=2, milestones=()):
    """Call only after save() completed on every rank. Retain latest + milestones."""

    def cleanup():
        complete = sorted(
            (
                p
                for p in Path(root).glob("checkpoint-*")
                if p.name.split("-")[-1].isdigit() and (p / "COMPLETE").is_file()
            ),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        retained = {p.name for p in complete[-keep:]}
        retained.update(f"checkpoint-{s}" for s in milestones)
        final = [
            p
            for p in complete
            if (p / "metadata.json").is_file()
            and json.loads((p / "metadata.json").read_text()).get("final")
        ]
        if final:
            retained.add(final[-1].name)
        for p in complete:
            if p.name not in retained:
                shutil.rmtree(p)

    rank_zero(cleanup)


def export_checkpoint(
    source, output, role="student", dtype=torch.float32, mmap=False, architecture=None
):
    """Export original master weights directly; no BF16 round trip or GPU needed."""
    from transformers import AutoTokenizer

    from model import LLaDAForMultiModalGeneration
    from model.configuration_llada import LLaDAConfig

    source, output = Path(source), Path(output)
    if not (source / "COMPLETE").is_file():
        raise ValueError("incomplete training checkpoint")
    if role not in ("student", "ema"):
        raise ValueError("export role must be student or ema")
    architecture = Path(architecture) if architecture else source / "model_config.json"
    if not architecture.exists():
        raise ValueError(
            "checkpoint lacks model_config.json; supply --architecture with matching model config/tokenizer"
        )
    config = LLaDAConfig.from_pretrained(architecture, local_files_only=True)
    weights = source / f"{role}.pt"
    if digest(weights) != weights.with_suffix(".pt.sha256").read_text().strip():
        raise ValueError("checkpoint SHA256 mismatch")
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = LLaDAForMultiModalGeneration(config)
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=mmap)
    state = {
        n: v.to(dtype) if v.is_floating_point() and "live_type_rope" not in n else v
        for n, v in state.items()
    }
    # assign materializes all registered modules without allocating another full model.
    model.load_state_dict(state, strict=True, assign=True)
    model.config.init_device = "cpu"
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(
        output, state_dict=state, safe_serialization=True, max_shard_size="4GB"
    )
    export_metadata(output, dtype)
    tokenizer_source = (
        source / "tokenizer" if (source / "tokenizer").is_dir() else architecture
    )
    AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=True
    ).save_pretrained(output)
    meta = json.loads((source / "metadata.json").read_text())
    atomic_text(
        output / "EXPORT_INFO.json",
        json.dumps(
            {
                "purpose": "inference weights only; no optimizer/RNG",
                "role": role,
                "dtype": str(dtype),
                "inference_dtype": meta["config"].get("inference_dtype", "float32"),
            },
            indent=2,
        )
        + "\n",
    )
    (output / "TRAINING_PROVENANCE.json").write_text(
        json.dumps(
            {
                "architecture_source": str(architecture),
                "source_checkpoint": str(source),
                "source_sha256": weights.with_suffix(".pt.sha256").read_text().strip(),
                "role": role,
                "dtype": str(dtype),
                "updates": meta["progress"]["student_updates"],
                "formal_sft": training_provenance(
                    meta["config"], meta["progress"], meta["origin"], role
                )["formal_sft"],
                "origin": meta["origin"],
                "precision_source": "original training master weights, not BF16 export",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Export Student or EMA from a complete training checkpoint"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--role", choices=["student", "ema"], default="student")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--mmap", action="store_true")
    p.add_argument(
        "--architecture",
        help="Explicit config/tokenizer fallback for old checkpoints lacking them",
    )
    a = p.parse_args()
    export_checkpoint(
        a.checkpoint, a.output, a.role, getattr(torch, a.dtype), a.mmap, a.architecture
    )
