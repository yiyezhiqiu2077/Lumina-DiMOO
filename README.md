# Lumina-DiMOO × LiveMGT

## Introduction

This project adapts LiveMGT image editing to Lumina-DiMOO 8B. Sparse Target
Modeling processes editable target tokens alongside a complete text/reference
prefix, using Prefix Attention throughout sampling. Condition KV Cache reuses
prefix keys and values between inference steps. Region-size Conditioning and
pretrained-compatible Token-Type RoPE (C) provide region and token-role signals
while retaining Lumina's absolute positions. Region-DiMO trains a one-step editor
with CBC weighting, regional distillation, GT cross-entropy and an auxiliary model.

## Installation

Use Python 3.10+ and PyTorch 2.5.1. Install the PyTorch wheel appropriate for your
CUDA environment, then the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The main dependencies include Transformers 4.46.2 and Diffusers 0.34.0.
Optional DDP 8-bit AdamW requires `bitsandbytes==0.45.5`.

## Model Preparation

Download [Lumina-DiMOO](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO), including
its tokenizer and `vqvae/`, or reuse an existing local copy:

```bash
huggingface-cli download Alpha-VLLM/Lumina-DiMOO --local-dir checkpoints/Lumina-DiMOO
```

Set `base_checkpoint` to this directory. Set `vq_checkpoint` to the same directory
or to a separate VQ-VAE directory. Model loading uses local files. Keep the same
VQ-VAE for SFT, Teacher initialization, distillation and inference.

## Data Preparation

Use JSONL manifests containing aligned source/target images, binary edit masks
and English instructions. White mask pixels indicate the edit region. Paths are
relative to `asset_roots[dataset_name]`, or may be absolute.

```json
{"sample_uid":"pair-1","dataset_name":"interedit","group_id":"source-1","source":"source.png","target":"target.png","mask":"mask.png","instruction_en":"Make the jacket blue"}
```

Prepare separate train and validation manifests without shared source groups.
Remove invalid images and empty masks before training. Source, target and mask
are resized together; masks use nearest-neighbor resizing. Existing file-backed
locator manifests are also supported.

For optional raw dataset export, use a local
[LiveMGT checkout](https://github.com/Shi-qingyu/EditMGT) with its dataset
loading dependencies:

```bash
python -m livemgt_training.full.export_livemgt_data \
  --upstream-root EditMGT --data-root datasets/raw/interedit --source interedit \
  --cache-dir .cache/interedit --output datasets/interedit --max-samples 50000
```

Use `--translations-dir` to apply existing Inter-Edit English translations.
Export defaults to 512 resolution, excludes Inter-Edit text editing and traverses
each source once. MagicBrush uses its original mask decoder. Exported paths are
relative to the output directory unless `--manifest-root` specifies another root.

Copy the training templates and configure the model, manifests and asset roots:

```bash
cp configs/formal/sft.json config.local.sft.json
cp configs/formal/dimo.json config.local.dimo.json
```

Set these fields in both files, retaining the remaining template settings:

```json
{
  "base_checkpoint": "checkpoints/Lumina-DiMOO",
  "vq_checkpoint": "checkpoints/Lumina-DiMOO",
  "train_manifest": "datasets/train.jsonl",
  "validation_manifest": "datasets/validation.jsonl",
  "asset_roots": {
    "interedit": "datasets/interedit",
    "magicbrush": "datasets/magicbrush",
    "maskedit": "datasets/maskedit"
  }
}
```

`data_mode=manifest` shuffles all training rows. For source-weighted sampling, use
`configs/formal/livemgt_mixture.json` as the template: Inter-Edit 90%, MagicBrush
5%, MaskEdit 5%. Each rank owns a disjoint shard and each source advances its own
shuffle epoch. Treat assets as immutable; update dataset provenance or
`data_versions` when replacing them.

## Training

### SFT

```bash
TRAIN_CONFIG=config.local.sft.json bash scripts/train.sh --output runs/sft --export student
```

### Region-DiMO

In `config.local.dimo.json`, set `teacher_checkpoint` to
`runs/sft/student-export-20000` and `teacher_status` to `FORMAL_SFT_TRAINED`.
The Teacher must have SFT provenance recording the same VQ-VAE identity.

```bash
TRAIN_CONFIG=config.local.dimo.json bash scripts/dimo/train.sh --output runs/dimo --export ema
```

Training defaults to full fine-tuning, DDP, FP32 with TF32 disabled, and 512×512
images. SFT uses 20000 updates; Region-DiMO uses 6000 Student updates. The
microbatch is 1 with gradient accumulation 8. AdamW uses LR 1e-6, betas
(0.9, 0.999), epsilon 1e-8, weight decay 0.01, 100 warmup updates and a constant
schedule afterward. Gradient clipping is 1.0, gradient checkpointing is enabled,
and CPU EMA uses decay 0.9999 with warmup.

The VQ-VAE and Teacher are frozen. Student and Auxiliary have independent full
parameters, optimizers and schedulers. Region conditioning and Type RoPE C are
enabled; ordinary type embeddings are disabled. Halo 1 expands the original mask
M by one VQ cell in all eight directions to form R. SFT supervises corrupted
positions in R; GT CE uses M with weight 0.1. CBC distillation and Auxiliary CE
use the sampled pseudo-mask Q within R. Source tokens outside R stay fixed.

### GPUs and distributed backends

By default, the launchers use one process per visible GPU. For example:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 TRAIN_CONFIG=config.local.sft.json \
  bash scripts/train.sh --output runs/sft-fsdp --set backend=fsdp
```

`GPU_IDS` overrides `CUDA_VISIBLE_DEVICES`. Backends are `ddp`, `ddp_zero` and
`fsdp` (FULL_SHARD); optimizers are configured independently as `adamw` or
`adamw8bit`. The 8-bit optimizer supports DDP only. Backends and precision never
change automatically. Multi-node launch uses `NNODES`, `NODE_RANK`, `MASTER_ADDR`
and `MASTER_PORT`; existing torchrun workers use their rank environment directly.

Global batch size is world size × microbatch × accumulation. Set
`global_batch_size` to fix it; it must be divisible by world size × microbatch.
Learning rate is not scaled automatically. To select BF16 compute and validation
explicitly, set `mixed_precision=bf16` and `inference_dtype=bfloat16`.

### Checkpoints and resume

```bash
TRAIN_CONFIG=config.local.sft.json bash scripts/train.sh --output runs/sft \
  --resume runs/sft/checkpoint-500
```

Complete checkpoints contain model, optimizer, scheduler, EMA, RNG and data-stream
state. The latest two checkpoints, configured milestones and the latest final
checkpoint are retained. Checkpoints are saved every 500 updates and at training
completion; fixed validation examples are saved every 200 updates. When moving a
DiMO run, copy its shared `teacher/` directory together with the checkpoint.

Resume checks model/data identities and training settings. World-size changes
require `--allow-world-size-change` and reset per-source counts on new rank shards;
they do not reproduce the previous sample order. Batch or backend changes require
`--allow-batch-size-change` or `--allow-backend-change`, respectively. Checkpoints
without the required identities or sampler state can initialize model weights
through an inference export instead of a full-state resume.

Interrupted saves can leave hidden `.checkpoint-*.tmp-*`,
`.checkpoint-*.interrupted-*` or corresponding `.teacher.*` directories. After all
workers using that run directory have stopped, inspect and manually remove only
confirmed abandoned directories. Active staging directories are never removed
automatically; keep complete checkpoints and the shared `teacher/` directory.

## Inference

SFT editing uses the original 18-step MaskGit sampler with Text CFG 2.5 and Image
CFG 4:

```bash
bash scripts/infer.sh --checkpoint runs/sft/student-export-20000 \
  --vq-vae checkpoints/Lumina-DiMOO --source source.png --mask mask.png \
  --instruction 'Make the jacket blue' --phase sft --steps 18 \
  --text-cfg 2.5 --image-cfg 4 --cache on --output outputs/sft.png
```

Region-DiMO uses one conditional-only step with no additional CFG:

```bash
bash scripts/infer.sh --checkpoint runs/dimo/ema-export-6000 \
  --vq-vae checkpoints/Lumina-DiMOO --source source.png --mask mask.png \
  --instruction 'Make the jacket blue' --phase dimo --cache on --output outputs/dimo.png
```

Use `--cache off` to disable Condition KV Cache. Defaults are resolution 512,
halo 1 and seed 42; override them with `--resolution`, `--halo` and `--seed`.
Omitting `--mask` edits the full image. `--precision float32|bfloat16` overrides
`inference_dtype` from an optional `--config` file or the model export.

Load Student or EMA by selecting the corresponding exported directory. To export
either role from a complete checkpoint, use `--role student` or `--role ema`:

```bash
python -m livemgt_training.full.checkpoint --checkpoint runs/dimo/checkpoint-6000 \
  --role student --output weights/dimo-student
```

Exports default to FP32 and contain inference weights and tokenizer files rather
than optimizer or RNG state. The editor saves the result, source/mask comparison
and inference settings beside `--output`.

## Acknowledgements & License

This work builds on [Lumina-DiMOO](https://github.com/Alpha-VLLM/Lumina-DiMOO)
by Alpha-VLLM and [EditMGT / LiveMGT](https://github.com/Shi-qingyu/EditMGT)
by Shi-qingyu and collaborators. Preserve their attribution when redistributing
adaptations. See [LICENSE](LICENSE),
[the upstream EditMGT license](licenses/EditMGT-LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights and datasets retain
their respective licenses.
