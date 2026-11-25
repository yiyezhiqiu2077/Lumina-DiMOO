"""DiMO/CBC kernels ported from Shi-qingyu/EditMGT, b0f7ce3.
Original license and copyright: see licenses/EditMGT-LICENSE.
Lumina adapters below explicitly use 8192 visual classes and its two CFG scales.
"""
import math
import torch
import torch.nn.functional as F

def pseudo_mask(region, ratio, weights=None):
    """Sample an exact-size pseudo mask, optionally without replacement by weight."""
    flat = region.flatten(1)
    if (~flat.any(1)).any():
        raise ValueError('Empty generation region')
    if ratio.shape != (len(region),) or not torch.isfinite(ratio).all() or ((ratio < 0) | (ratio > 1)).any():
        raise ValueError('Expected one finite mask ratio in [0,1] per sample')
    count = (flat.sum(1) * ratio).round().long().clamp_min(1)
    if weights is None:
        # Uniform exact-count sampling, matching SFT/upstream DiMO's mask counts.
        order = torch.rand_like(flat.float()).masked_fill(~flat, 2).argsort(1).argsort(1)
    else:
        if weights.shape != region.shape or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Pseudo-mask weights must be finite, nonnegative and match the region')
        flat_weights = weights.float().flatten(1).masked_fill(~flat, 0)
        if ((flat_weights > 0).sum(1) < count).any():
            raise ValueError('Pseudo-mask weights must be positive for enough generation tokens')
        # Gumbel top-k is weighted sampling without replacement. Ranking supports
        # a different exact mask count per sample without a Python batch loop.
        uniform = torch.rand_like(flat_weights).clamp_(min=torch.finfo(torch.float32).tiny,
                                                        max=1 - torch.finfo(torch.float32).eps)
        score = flat_weights.clamp_min(torch.finfo(torch.float32).tiny).log()
        score = score - (-uniform.log()).log()
        order = score.masked_fill(~flat, -torch.inf).argsort(1, descending=True).argsort(1)
    q = (order < count[:, None]) & flat
    return q.reshape_as(region)


def signed_mask_distance(mask, region):
    """Discrete signed distance to a mask boundary on the token grid.

    Positive values are inside ``mask`` and negative values are in the exterior
    part of ``region``. Distances use an 8-connected (chessboard) metric, which
    is inexpensive on GPU and matches the square token receptive field. The two
    token bands adjacent to the boundary have magnitudes 0.5.
    """
    if mask.shape != region.shape or mask.ndim != 3:
        raise ValueError('Mask and generation region must have matching [B,H,W] shapes')
    mask, region = mask.bool(), region.bool()
    if (mask & ~region).any():
        raise ValueError('The original mask must be a subset of the generation region')
    distance = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
    current = mask
    # A fixed bound avoids a device-to-host synchronization at every erosion.
    for depth in range(max(mask.shape[-2:])):
        eroded = -F.max_pool2d(-F.pad(current[:, None].float(), (1, 1, 1, 1)), 3, stride=1)
        eroded = eroded[:, 0].bool()
        distance.masked_fill_(current & ~eroded, depth + 0.5)
        current = eroded
    covered = mask
    for depth in range(max(mask.shape[-2:])):
        expanded = F.max_pool2d(covered[:, None].float(), 3, stride=1, padding=1)[:, 0].bool()
        band = expanded & ~covered & region
        distance.masked_fill_(band, -(depth + 0.5))
        covered = expanded
    if (region & ~mask & distance.eq(0)).any():
        raise ValueError('Generation region contains tokens unreachable from the edit mask')
    return distance


def region_weight_map(original, region, config, evidence=None, distance=None):
    """Build normalized Core--Boundary--Context weights on the token grid."""
    if not config or not config.get('enabled', False):
        return None
    original, region = original.bool(), region.bool()
    distance = signed_mask_distance(original, region) if distance is None else distance.float()
    if distance.shape != region.shape:
        raise ValueError('Region distance must match the generation region')
    core = original & (distance >= config['core_margin'])
    weights = (config['core_weight'] * core.float()
               + config['boundary_weight'] * torch.exp(-distance.abs() / config['boundary_sigma']))
    if evidence is not None:
        if evidence.shape != region.shape or not torch.isfinite(evidence).all() or (evidence < 0).any():
            raise ValueError('Teacher edit evidence must be finite, nonnegative and match the region')
        weights = weights + config['evidence_weight'] * evidence.float()
    weights = weights.masked_fill(~region, 0)
    # Preserve the effective loss scale: each sample has mean weight one in G.
    denominator = weights.flatten(1).sum(1)
    if (denominator <= 0).any() or not torch.isfinite(denominator).all():
        raise ValueError('Region weights must have positive finite mass in every sample')
    scale = region.flatten(1).sum(1).float() / denominator
    return weights * scale[:, None, None]


@torch.no_grad()
def teacher_edit_evidence(conditional_logits, unconditional_logits, chunk_size=1024):
    """Normalized JS divergence between conditional/unconditional Teacher distributions."""
    if conditional_logits.shape != unconditional_logits.shape or conditional_logits.ndim != 4:
        raise ValueError('Teacher evidence expects matching [B,H,W,V] logits')
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Evidence chunk_size must be a positive integer')
    conditional_logits, unconditional_logits = conditional_logits.detach(), unconditional_logits.detach()
    log_z_c = torch.logsumexp(conditional_logits.float(), dim=-1)
    log_z_u = torch.logsumexp(unconditional_logits.float(), dim=-1)
    evidence = torch.zeros_like(log_z_c)
    log_two = math.log(2.0)
    for start in range(0, conditional_logits.shape[-1], chunk_size):
        end = min(start + chunk_size, conditional_logits.shape[-1])
        log_c = conditional_logits[..., start:end].float() - log_z_c[..., None]
        log_u = unconditional_logits[..., start:end].float() - log_z_u[..., None]
        log_m = torch.logaddexp(log_c, log_u) - log_two
        evidence.add_(0.5 * ((log_c.exp() * (log_c - log_m)).sum(-1)
                             + (log_u.exp() * (log_u - log_m)).sum(-1)))
    return (evidence / log_two).clamp_(0, 1)


def masked_mean(values, mask):
    return ((values * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).mean()




def ema_decay_at_step(decay, global_step=None):
    """Official DiMO rampup; step is completed updates BEFORE this update."""
    if global_step is None:
        return decay
    if type(global_step) is not int or global_step < 0:
        raise ValueError('EMA global_step must be a nonnegative integer')
    return min(decay, (1 + global_step) / (10 + global_step))


@torch.no_grad()
def update_ema(ema, student, decay, global_step=None):
    rate = ema_decay_at_step(decay, global_step)
    for e, s in zip(ema.parameters(), student.parameters()):
        if global_step is None:
            e.lerp_(s, 1 - rate)
        else:
            # Match official train_DIMO_Meissonic.py, including operation order.
            e.detach().mul_(rate).add_(s.to(e.device), alpha=1 - rate)
    for e, s in zip(ema.buffers(), student.buffers()):
        e.copy_(s)
    return rate




def fkl_surrogate(student_logits, teacher_logits, auxiliary_logits, q, weights=None):
    """DiMO FKL logit gradient p_aux - p_teacher; normalized per edited sample.

    Follows upstream train_DIMO_Meissonic.py FKL/MSE gradient injection,
    with explicit per-sample Q normalization for regional training.
    All logits use [B,H,W,V].
    """
    direction = (auxiliary_logits.float().softmax(-1) - teacher_logits.float().softmax(-1)).detach()
    z = student_logits.float()
    # Algebraically identical upstream MSE gradient injection, avoiding the
    # cancellation in z - (z-direction).detach() for small probability deltas.
    loss = 0.5 * ((z - z.detach()) + direction).square().sum(-1)
    if weights is not None:
        if weights.shape != q.shape or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('FKL weights must be finite, nonnegative and match Q')
        loss = loss * weights
    return masked_mean(loss, q)


def auxiliary_ce(logits, sampled_tokens, q, label_smoothing=0.1):
    loss = F.cross_entropy(logits.movedim(-1, 1).float(), sampled_tokens.detach(),
                           label_smoothing=label_smoothing, reduction='none')
    return masked_mean(loss, q)



VQ_OFFSET=126356
CODEBOOK_SIZE=8192
MASK_ID=126336

def visual_logits(logits):
    if logits.shape[-1]==CODEBOOK_SIZE:return logits
    if logits.shape[-1] < VQ_OFFSET+CODEBOOK_SIZE:raise ValueError('invalid unified visual vocabulary')
    return logits[...,VQ_OFFSET:VQ_OFFSET+CODEBOOK_SIZE]


def lumina_cfg(conditional,unconditional_text,unconditional_image,text_scale,image_scale):
    if any(not math.isfinite(v) or v<0 for v in (text_scale,image_scale)):raise ValueError('invalid CFG')
    return conditional+text_scale*(conditional-unconditional_text)+image_scale*(conditional-unconditional_image)


def generation_region(original,regional=True,halo=0):
    if type(halo) is not int or halo<0:raise ValueError('invalid halo')
    region=original.bool() if regional else torch.ones_like(original,dtype=torch.bool)
    if halo and regional:
        region=F.max_pool2d(region[:,None].float(),2*halo+1,stride=1,padding=halo)[:,0].bool()
    if not region.any():raise ValueError('empty generation region; skip sample explicitly')
    return region


def one_step_initial(source,region,r_init=1.):
    if isinstance(r_init,bool) or not math.isfinite(r_init) or not 0<r_init<=1:raise ValueError('invalid r_init')
    if r_init==1:return torch.where(region,MASK_ID,source)
    random_codes=torch.randint(VQ_OFFSET,VQ_OFFSET+CODEBOOK_SIZE,source.shape,device=source.device)
    initial=torch.where(region,random_codes,source)
    q=pseudo_mask(region,source.new_full((len(source),),r_init,dtype=torch.float32))
    return torch.where(q,MASK_ID,initial)


def sample_hard_copy(logits,source,region):
    logits=visual_logits(logits)
    with torch.no_grad():
        sampled=torch.distributions.Categorical(logits=logits.float()).sample()+VQ_OFFSET
        return torch.where(region,sampled,source)


def gt_ce(logits,target,valid):
    labels=target-VQ_OFFSET
    if ((labels<0)|(labels>=CODEBOOK_SIZE)).any():raise ValueError('GT must be offset visual IDs')
    return auxiliary_ce(visual_logits(logits),labels,valid,label_smoothing=0.)


def combined_evidence(cond,ut,ui,mode='text',chunk_size=1024):
    # Text-only is the faithful conditional/null-text JS analogue from EditMGT.
    # Mean JS of text/image ablations is an explicit Lumina extension.
    text=teacher_edit_evidence(cond,ut,chunk_size)
    if mode=='text':return text
    if mode=='mean_text_image':return .5*(text+teacher_edit_evidence(cond,ui,chunk_size))
    raise ValueError('unknown evidence mode')


def sample_mask_ratio(batch_size,device):
    """EditMGT MaskRatioSampler: pure unshifted cosine, NOT uniform ratio."""
    return torch.cos(torch.rand(batch_size,device=device,dtype=torch.float32)*math.pi*.5)
