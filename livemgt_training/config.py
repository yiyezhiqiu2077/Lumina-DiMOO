import json
from pathlib import Path


def load_config(path):
    c=json.loads(Path(path).read_text())
    # New explicit flags cannot silently contradict legacy embedding settings.
    if 'token_type_embedding' in c and c['token_type_embedding']!=c.get('p3',{}).get('token_type',c['token_type_embedding']):
        raise ValueError('token_type_embedding conflicts with legacy p3.token_type (embedding)')
    if 'region_conditioning' in c and c['region_conditioning']!=c.get('p3',{}).get('region',c['region_conditioning']):
        raise ValueError('region_conditioning conflicts with p3.region')
    if c.get('type_rope') is not None:
        spec=c['type_rope']
        if 'token_type_rope' in c and c['token_type_rope']!=spec.get('enabled',True):raise ValueError('rotary enable flags disagree')
        if 'rope_mode' in c and c['rope_mode']!=spec.get('mode','pretrained_type_phase'):raise ValueError('rotary mode flags disagree')
    if c['student_initial_mask_ratio']<=0 or c['student_initial_mask_ratio']>1:raise ValueError('invalid initial ratio')
    if c.get('freeze_embedding',True) is not True or c.get('freeze_head',True) is not True:
        raise ValueError('shared immutable-base PEFT requires frozen embedding/head; use a separate FSDP implementation for full-parameter training')
    drop=c['cfg_dropout']
    if any(not 0<=drop[k]<=1 for k in ['text','image']) or drop['text']+drop['image']>1:
        raise ValueError('mutually exclusive Text/Image CFG dropout probabilities must sum to <=1')
    if c['cbc']['enabled'] and c['cbc']['boundary_sigma']<=0:raise ValueError('invalid CBC sigma')
    if c['gan']['enabled']:raise ValueError('No validated production image discriminator registered; GAN training is intentionally blocked')
    return c
