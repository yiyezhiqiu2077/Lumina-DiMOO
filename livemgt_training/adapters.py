"""Four logical LoRA/P3 models over one immutable, possibly SFT-merged 8B base.

No graph may survive a role switch. Student's differentiable forward is replayed
AFTER no-grad teacher/auxiliary evaluation. Independent optimizers own separate
moments over the physical adapter slots. Role snapshots, EMA and optimizer states
are distinct; frozen base weights are never copied four times or optimized.
"""
import copy
import torch
from torch import nn
from model.livemgt_conditioning import enable_conditioning
from .core import ema_decay_at_step
from model.livemgt_rope import enable_type_rope,upgrade_roles

class LoRALinear(nn.Module):
    def __init__(self,base,rank=4,alpha=8):
        super().__init__();self.base=base;self.scale=alpha/rank
        self.A=nn.Linear(base.in_features,rank,bias=False,device=base.weight.device,dtype=torch.float32)
        self.B=nn.Linear(rank,base.out_features,bias=False,device=base.weight.device,dtype=torch.float32)
        nn.init.kaiming_uniform_(self.A.weight,a=5**.5);nn.init.zeros_(self.B.weight)
    def forward(self,x):return self.base(x)+(self.B(self.A(x.float()))*self.scale).to(x.dtype)


def install_trainables(model,c):
    model.requires_grad_(False)
    for block in model.model.transformer.blocks:
        for name in c.get('lora_targets',['q_proj','v_proj']):
            setattr(block,name,LoRALinear(getattr(block,name),c.get('lora_rank',4),c.get('lora_alpha',8)))
    spec=c.get('p3',{})
    enable_conditioning(model,region=c.get('region_conditioning',spec.get('region',False)),token_type=c.get('token_type_embedding',spec.get('token_type',False))).float().requires_grad_(True)
    rope_spec=c.get('type_rope',getattr(model.config,'live_mgt_rope',None))
    if rope_spec is not None:
        rope_spec=dict(rope_spec)
        rope_spec['enabled']=c.get('token_type_rope',rope_spec.get('enabled',True))
        rope_spec['mode']=c.get('rope_mode',rope_spec.get('mode','pretrained_type_phase'))
        enable_type_rope(model,**rope_spec).float().requires_grad_(True)
    return {name:p for name,p in model.named_parameters() if p.requires_grad}

class RoleBank:
    def __init__(self,model,params,teacher_state=None,freeze_policy=None):
        self.model=model;self.params=params;self.current=None;self.graph_live=False;self.freeze_policy=freeze_policy or {}
        initial={n:p.detach().cpu().clone() for n,p in params.items()}
        if teacher_state is not None:
            upgraded,migrations=upgrade_roles({'teacher':teacher_state},params)
            initial={n:p.cpu().clone() for n,p in upgraded['teacher'].items()}
        self.states={role:copy.deepcopy(initial) for role in ['teacher','student','auxiliary','ema']}
    def activate(self,role,train=False):
        if self.graph_live:raise RuntimeError('backward must finish before role switching')
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n,p in self.params.items():p.copy_(self.states[role][n])
        for name,p in self.params.items():
            frozen=any(name.startswith(prefix) for prefix in self.freeze_policy.get(role,[]))
            p.requires_grad_(train and role in ['student','auxiliary'] and not frozen)
        self.model.train(train);self.current=role
        return self.model
    def save_role(self):
        if self.graph_live:raise RuntimeError('cannot save role with outstanding graph')
        self.states[self.current]={n:p.detach().cpu().clone() for n,p in self.params.items()}
    @torch.no_grad()
    def ema_update(self,decay,step,warmup=True):
        rate=ema_decay_at_step(decay,step if warmup else None)
        for n,e in self.states['ema'].items():e.mul_(rate).add_(self.states['student'][n],alpha=1-rate)
        return rate
