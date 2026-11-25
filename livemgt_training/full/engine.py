"""Full-parameter SFT and Region-DiMO updates using the shared CBC kernels."""

import random

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from livemgt_training.core import (
    auxiliary_ce,
    combined_evidence,
    fkl_surrogate,
    generation_region,
    gt_ce,
    lumina_cfg,
    one_step_initial,
    pseudo_mask,
    region_weight_map,
    sample_hard_copy,
    sample_mask_ratio,
    signed_mask_distance,
)
from livemgt_training.precision import autocast
from livemgt_training.predict import predict_grid

from .runtime import check_finite, no_sync


def prediction(
    model, tokenizer, batch, tokens, region, config, branch="cond", grad=False
):
    with autocast(config):
        logits = predict_grid(
            model,
            tokenizer,
            batch,
            tokens,
            region,
            branch,
            gradient_checkpointing=grad and config["gradient_checkpointing"],
            condition_cache=None,
            region_conditioning=True,
            token_type_embedding=False,
            token_type_rope=True,
            rope_mode="pretrained_type_phase",
        )

    check_finite(logits, "prediction logits")
    return logits


class Engine:
    def __init__(self, models, optimizers, schedulers, ema, tokenizer, config):
        self.models = models
        self.opts = optimizers
        self.sch = schedulers
        self.ema = ema
        self.tokenizer = tokenizer
        self.c = config

    def pred(self, role, batch, tokens, region, branch="cond", grad=False):
        return prediction(
            self.models[role],
            self.tokenizer,
            batch,
            tokens,
            region,
            self.c,
            branch,
            grad,
        )

    def region(self, batch):
        return generation_region(batch["original"], True, self.c["halo"])

    def step(self, role):
        model, opt = self.models[role], self.opts[role]
        if isinstance(model, FSDP):
            norm = model.clip_grad_norm_(self.c["max_grad_norm"])
        else:
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.c["max_grad_norm"]
            )
        check_finite(norm, f"{role} gradient norm")
        lr = opt.param_groups[0]["lr"]
        opt.step()
        self.sch[role].step()
        opt.zero_grad(set_to_none=True)
        return {"gradient_norm": float(norm), "learning_rate": lr}

    def sft(self, stream, step):
        c, model = self.c, self.models["student"]
        model.train()
        total = c["gradient_accumulation"] * c["micro_batch_size"]
        losses = []
        for micro in range(total):
            batch = stream.next()
            region = self.region(batch)
            q = pseudo_mask(region, sample_mask_ratio(1, region.device))
            supervised = q if c["supervise_target_halo"] else q & batch["original"]
            if not supervised.any():
                supervised = pseudo_mask(
                    batch["original"], torch.ones(1, device=region.device)
                )
                q = q | supervised
            clean = torch.where(region, batch["target"], batch["source"])
            tokens = torch.where(q, 126336, clean)
            draw, drop = random.random(), c["cfg_dropout"]
            branch = (
                "uncond_image"
                if draw < drop["image"]
                else "uncond_text"
                if draw < sum(drop.values())
                else "cond"
            )
            with no_sync(model, c, micro == total - 1):
                logits = self.pred("student", batch, tokens, region, branch, grad=True)
                loss = gt_ce(logits, batch["target"], supervised)
                check_finite(loss, "SFT loss")
                (loss / total).backward()
            losses.append(float(loss.detach()))
        stats = self.step("student")
        self.ema.update(model, step)
        return dict(loss=sum(losses) / total, **stats)

    @torch.no_grad()
    def generated(self, batch):
        region = self.region(batch)
        initial = one_step_initial(
            batch["source"], region, self.c["student_initial_mask_ratio"]
        )
        logits = self.pred("student", batch, initial, region)
        return region, initial, sample_hard_copy(logits, batch["source"], region)

    def dimo(self, stream, step):
        c = self.c
        student, aux, teacher = (
            self.models[k] for k in ("student", "auxiliary", "teacher")
        )
        total = c["gradient_accumulation"] * c["micro_batch_size"]
        fkl_sum = gt_sum = aux_sum = 0.0
        for micro in range(total):
            batch = stream.next()
            student.eval()
            aux.eval()
            teacher.eval()
            with torch.no_grad():
                region, initial, sample = self.generated(batch)
                distance = signed_mask_distance(batch["original"], region)
                geometry = region_weight_map(
                    batch["original"], region, c["cbc"], distance=distance
                )
                q = pseudo_mask(
                    region,
                    sample_mask_ratio(1, region.device),
                    geometry if c["cbc"]["weighted_pseudo_mask"] else None,
                )
                corrupt = torch.where(q, 126336, sample)
                cond = self.pred("teacher", batch, corrupt, region)
                ut = self.pred("teacher", batch, corrupt, region, "uncond_text")
                ui = self.pred("teacher", batch, corrupt, region, "uncond_image")
                target = lumina_cfg(
                    cond, ut, ui, c["teacher_text_cfg"], c["teacher_image_cfg"]
                )
                evidence = (
                    combined_evidence(
                        cond,
                        ut,
                        ui,
                        c["evidence_mode"],
                        c["cbc"]["evidence_chunk_size"],
                    )
                    if c["cbc"]["enabled"] and c["cbc"]["evidence_weight"]
                    else None
                )
                weights = region_weight_map(
                    batch["original"], region, c["cbc"], evidence, distance
                )
                fake = self.pred("auxiliary", batch, corrupt, region)
            student.train()
            with no_sync(student, c, micro == total - 1):
                logits = self.pred("student", batch, initial, region, grad=True)
                fkl = fkl_surrogate(logits, target, fake, q, weights)
                gt = (
                    gt_ce(logits, batch["target"], batch["original"])
                    * c["gt_ce"]["weight"]
                    if c["gt_ce"]["enabled"]
                    else logits.sum() * 0
                )
                loss = fkl + gt
                check_finite(loss, "DiMO Student loss")
                (loss / total).backward()
            fkl_sum += float(fkl.detach())
            gt_sum += float(gt.detach())
        stats = self.step("student")
        for micro in range(total):
            # New stream samples and the updated Student, after its optimizer step.
            batch = stream.next()
            student.eval()
            with torch.no_grad():
                region, _, labels = self.generated(batch)
                geometry = region_weight_map(batch["original"], region, c["cbc"])
                q = pseudo_mask(
                    region,
                    sample_mask_ratio(1, region.device),
                    geometry if c["cbc"]["weighted_pseudo_mask"] else None,
                )
                corrupt = torch.where(q, 126336, labels)
            aux.train()
            with no_sync(aux, c, micro == total - 1):
                logits = self.pred("auxiliary", batch, corrupt, region, grad=True)
                loss = auxiliary_ce(
                    logits, labels - 126356, q, c["auxiliary_label_smoothing"]
                )
                check_finite(loss, "Auxiliary loss")
                (loss / total).backward()
            aux_sum += float(loss.detach())
        astats = self.step("auxiliary")
        self.ema.update(student, step)
        return dict(
            loss=(fkl_sum + gt_sum) / total,
            fkl=fkl_sum / total,
            gt_ce=gt_sum / total,
            auxiliary_ce=aux_sum / total,
            auxiliary_gradient_norm=astats["gradient_norm"],
            **stats,
        )
