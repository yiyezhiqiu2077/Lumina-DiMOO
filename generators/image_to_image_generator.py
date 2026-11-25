# -*- coding: utf-8 -*-
"""
Image-to-image generator (supports DDP)
"""

import torch
import math
from typing import Callable, Optional
from utils.generation_utils import (
    cosine_schedule,
    gumbel_max_sample,
    mask_by_random_topk,
)


@torch.no_grad()
def _generate_i2i(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    cfg_img: float = 0.0,
    uncon_text: torch.LongTensor,
    uncon_image: torch.LongTensor,
    code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    edit_mask: Optional[torch.Tensor] = None,
    source_vq: Optional[torch.Tensor] = None,
    halo: int = 0,
    prefix_mode: Optional[str] = None,
    condition_cache: bool = False,
    _condition_kv_cache=None,
    region_conditioning=None,
    token_type_embedding=None,
    token_type_rope=None,
    rope_mode=None,
) -> torch.LongTensor:
    """
    Image-to-image MaskGit generation (supports CFG for text and image)

    Args:
        model: Model
        prompt: Prompt tensor
        seq_len: Sequence length
        newline_every: Newline interval per row
        timesteps: Number of timesteps
        mask_token_id: Mask token id
        newline_id: Newline token id
        temperature: Temperature
        cfg_scale: Text CFG scale
        cfg_img: Image CFG scale
        code_start: Prediction image token satrt index
        uncon_text: Unconditional text input
        uncon_image: Unconditional image input
        codebook_size: Codebook size
        noise_schedule: Noise schedule function
        text_vocab_size: Text vocabulary size
        generator: Random number generator

    Returns:
        Final VQ codes (1, seq_len)
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    device = (
        prompt.device if isinstance(model, FSDP) else next(model.parameters()).device
    )
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt.clone()
    allowed = None
    if edit_mask is not None:
        from utils.edit_mask import pixel_mask_to_vq, vq_positions

        if code_start is None or source_vq is None or seq_len % newline_every:
            raise ValueError(
                "mask editing requires source_vq, code_start and valid grid"
            )
        grid = (seq_len // newline_every, newline_every)
        expected_end = code_start + seq_len + grid[0]
        if code_start < 2 or x.shape[1] != expected_end + 2:
            raise ValueError("target image layout length/boundaries mismatch")
        newline_positions = (
            torch.arange(grid[0], device=device) * (newline_every + 1)
            + code_start
            + newline_every
        )
        if not (x[:, newline_positions] == newline_id).all():
            raise ValueError("target image newline positions mismatch")
        active = pixel_mask_to_vq(edit_mask, grid, halo).flatten().to(device)
        positions = vq_positions(*grid, code_start, device)
        source_vq = torch.as_tensor(source_vq, device=device).reshape(1, seq_len)
        if not ((source_vq >= 126356) & (source_vq < 126356 + codebook_size)).all():
            raise ValueError("source_vq must contain offset VQ IDs only")
        x[:, positions] = torch.where(active[None], mask_token_id, source_vq)
        allowed = torch.zeros_like(x, dtype=torch.bool)
        allowed[:, positions] = active

    vq_mask = x == mask_token_id
    if allowed is not None:
        vq_mask &= allowed
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    # Infer text vocabulary size
    if text_vocab_size is None:
        config = getattr(model, "config", None)
        if config is not None:
            text_vocab_size = (
                config.embedding_size or config.vocab_size
            ) - codebook_size
        else:
            vocab_total = model(
                torch.zeros(1, 1, dtype=torch.long, device=device), infer=True
            ).logits.size(-1)
            text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size
    if prefix_mode is not None and allowed is None:
        raise ValueError("prefix editing requires explicit edit_mask and source_vq")
    if condition_cache and prefix_mode != "sparse":
        raise ValueError("condition cache requires sparse prefix mode")

    branch_layouts = {}

    def predict(branch_ids, branch_mask, prefix_length, branch):
        if prefix_mode is None:
            return model(branch_ids, infer=True).logits[
                :, branch_mask[0], vocab_offset : vocab_offset + codebook_size
            ]
        if branch not in branch_layouts:
            from model.livemgt import PrefixLayout

            branch_positions = vq_positions(*grid, prefix_length + 2, device)[active]
            from model.livemgt_conditioning import bbox_size
            from utils.edit_mask import sequence_position_metadata

            types, coordinates, version = sequence_position_metadata(
                branch_ids,
                prefix_length,
                grid,
                reference_enabled=branch != "uncond_image",
            )
            original = pixel_mask_to_vq(edit_mask, grid, 0)
            layout = PrefixLayout(
                prefix_length,
                branch_positions,
                bbox_size(original),
                types,
                coordinates,
                version,
            )
            branch_layouts[branch] = layout
        layout = branch_layouts[branch]
        logits = model(
            branch_ids,
            infer=True,
            live_layout=layout,
            mode=prefix_mode,
            condition_cache=_condition_kv_cache,
            branch=branch,
            vocab_slice=(vocab_offset, vocab_offset + codebook_size),
            region_conditioning=region_conditioning,
            token_type_embedding=token_type_embedding,
            token_type_rope=token_type_rope,
            rope_mode=rope_mode,
        ).logits
        return logits[:, branch_mask[0, layout.active_positions]]

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        # Forward pass (with/without CFG)
        if cfg_scale > 0 or cfg_img > 0:
            # CFG text
            uncond_text = torch.cat(
                (uncon_text.to(x.device), x[:, code_start - 2 :]), dim=1
            )
            uncond_text_vq_mask = torch.cat(
                (
                    torch.zeros(
                        (1, uncon_text.size(1)), dtype=torch.bool, device=x.device
                    ),
                    vq_mask[:, code_start - 2 :],
                ),
                dim=1,
            )
            # CFG image
            uncond_img = torch.cat(
                (uncon_image.to(x.device), x[:, code_start - 2 :]), dim=1
            )
            uncond_img_vq_mask = torch.cat(
                (
                    torch.zeros(
                        (1, uncon_image.size(1)), dtype=torch.bool, device=x.device
                    ),
                    vq_mask[:, code_start - 2 :],
                ),
                dim=1,
            )

            cond_logits = predict(x, vq_mask, code_start - 2, "cond")
            uncond_logits_text = predict(
                uncond_text, uncond_text_vq_mask, uncon_text.shape[1], "uncond_text"
            )
            uncond_logits_img = predict(
                uncond_img, uncond_img_vq_mask, uncon_image.shape[1], "uncond_image"
            )
            logits = (
                cond_logits
                + cfg_scale * (cond_logits - uncond_logits_text)
                + cfg_img * (cond_logits - uncond_logits_img)
            )
        else:
            logits = predict(x, vq_mask, code_start - 2, "cond")

        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        mask_sel = mask_by_random_topk(
            keep_n.squeeze(1), conf, temperature=temperature, generator=generator
        )
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        if allowed is not None:
            vq_mask &= allowed
            assert torch.equal(x[:, positions[~active]], source_vq[:, ~active])
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

    # Remove newline tokens
    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids


from functools import wraps


@wraps(_generate_i2i)
@torch.no_grad()
def generate_i2i(*args, **kwargs):
    """Original interface plus mask/prefix options; generation owns cache lifetime."""
    from model.livemgt import ConditionKVCache

    cache = ConditionKVCache() if kwargs.get("condition_cache", False) else None
    try:
        return _generate_i2i(*args, **kwargs, _condition_kv_cache=cache)
    finally:
        if cache is not None:
            cache.clear()
