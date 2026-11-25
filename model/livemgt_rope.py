"""Q/K-only rotary modes: compatible Lumina phase extension and Flux 3-axis.

Compatible mode keeps half-split Lumina pairing and original absolute frequency.
Full three-axis mode uses Flux adjacent-pair rotation and per-axis frequencies;
it is NOT pretrained compatible. Neither mode rotates V.
"""
import torch
from torch import nn

MAPPING_VERSION='lumina_roles_v1'


def legal_axes(head_dim):
    if head_dim<6 or head_dim%2:raise ValueError('three-axis RoPE needs even head_dim>=6')
    type_dim=max(2,2*round(head_dim/16))
    row_dim=2*((head_dim-type_dim)//4)
    return (type_dim,row_dim,head_dim-type_dim-row_dim)

class TokenTypeRotary(nn.Module):
    def __init__(self,head_dim,axes_dims=None,theta=10000.):
        super().__init__()
        if head_dim<2 or head_dim%2:raise ValueError('1D rotary head_dim must be positive and even')
        self.phase=nn.Parameter(torch.zeros(3,head_dim//2,dtype=torch.float32))
        self.axes_dims=tuple(axes_dims) if axes_dims is not None else (legal_axes(head_dim) if head_dim>=6 else ())
        self.theta=float(theta)
        if self.axes_dims and (len(self.axes_dims)!=3 or sum(self.axes_dims)!=head_dim or any(d<2 or d%2 for d in self.axes_dims)):
            raise ValueError('axes must be three positive even dimensions summing to head_dim')

    def compatible_trig(self,sin,cos,types):
        # Compose sin/cos rather than reconstructing base angles (zero phase
        # preserves the existing cached trig bits and still has gradients).
        delta=self.phase[types].to(dtype=sin.dtype)
        delta=torch.cat((delta,delta),dim=-1)[None,None]
        ds,dc=delta.sin(),delta.cos()
        return sin*dc+cos*ds,cos*dc-sin*ds

    def three_axis_trig(self,ids):
        if not self.axes_dims:raise ValueError('head_dim too small for three nonempty rotary axes')
        cosines=[];sines=[]
        for axis,dim in enumerate(self.axes_dims):
            frequency=1./self.theta**(torch.arange(0,dim,2,device=ids.device,dtype=torch.float64)/dim)
            angles=ids[:,axis].float().double()[:,None]*frequency[None]
            cosines.append(angles.cos().repeat_interleave(2,dim=-1).float())
            sines.append(angles.sin().repeat_interleave(2,dim=-1).float())
        return torch.cat(sines,dim=-1)[None,None],torch.cat(cosines,dim=-1)[None,None]

    def three_axis_rotate(self,x,sin,cos):
        real,imag=x.reshape(*x.shape[:-1],-1,2).unbind(-1)
        rotated=torch.stack((-imag,real),dim=-1).flatten(-2)
        # Match upstream Flux apply_branch_rotary_emb arithmetic exactly.
        return (x.float()*cos+rotated.float()*sin).to(x.dtype)


def enable_type_rope(model,enabled=True,mode='pretrained_type_phase',axes_dims=None,theta=10000.,mapping_version=MAPPING_VERSION):
    if mode not in ('original','pretrained_type_phase','three_axis'):raise ValueError('unknown rope mode')
    if mapping_version!=MAPPING_VERSION:raise ValueError('unknown rotary role mapping version')
    head_dim=model.config.d_model//model.config.n_heads
    if not hasattr(model,'live_type_rope'):
        model.live_type_rope=TokenTypeRotary(head_dim,axes_dims,theta).to(model.device)
    module=model.live_type_rope
    axes=tuple(axes_dims) if axes_dims is not None else module.axes_dims
    if mode=='three_axis' and not axes:raise ValueError('three-axis mode needs head_dim>=6')
    if axes!=module.axes_dims or theta!=module.theta:raise ValueError('existing rotary module architecture differs')
    model.config.live_mgt_rope=dict(enabled=enabled,mode=mode,axes_dims=list(axes),theta=theta,mapping_version=mapping_version)
    return module


def upgrade_roles(states,params):
    """Only zero-initialize the new phase key; never drop unknown/trained keys."""
    result={};migrated=[]
    for role,state in states.items():
        unexpected=set(state)-set(params);missing=set(params)-set(state)
        if unexpected or missing-{'live_type_rope.phase'}:raise ValueError(f'{role}: adapter schema differs; missing={missing}, unexpected={unexpected}')
        result[role]=dict(state)
        for name in missing:
            if params[name].detach().count_nonzero():raise ValueError('cannot replace an already trained phase with legacy zero state')
            result[role][name]=torch.zeros_like(params[name],device='cpu');migrated.append((role,name))
        for name,value in result[role].items():
            if value.shape!=params[name].shape:raise ValueError(f'{role}: shape mismatch at {name}')
    return result,migrated
