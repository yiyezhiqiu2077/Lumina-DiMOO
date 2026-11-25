"""Explicit white=edit masks on Lumina's row-delimited VQ grid."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pixel_mask_to_vq(mask, grid_size, halo=0):
    if isinstance(halo, bool) or not isinstance(halo, int) or halo < 0:
        raise ValueError('halo must be a nonnegative integer in VQ cells')
    if isinstance(mask, Image.Image):
        mask = torch.from_numpy(np.array(mask.convert('L'), copy=True)) > 127
    mask = torch.as_tensor(mask)
    if mask.ndim != 2:
        raise ValueError('one binary [H,W] mask required')
    active = F.adaptive_max_pool2d(mask.bool()[None, None].float(), grid_size)
    if halo:
        active = F.max_pool2d(active, 2 * halo + 1, stride=1, padding=halo)
    return active[0, 0].bool()


def vq_positions(grid_height, grid_width, code_start=0, device=None):
    """code_start is AFTER answer_start and BOI; each row ends in newline."""
    r = torch.arange(grid_height, device=device)[:, None]
    c = torch.arange(grid_width, device=device)[None, :]
    return (code_start + r * (grid_width + 1) + c).flatten()


ROLE_MAPPING_VERSION='lumina_roles_v1'


def sequence_position_metadata(input_ids,prefix_length,target_grid,reference_enabled=True):
    """Unified training/sampling role v1 and Flux [type,row,col] coordinates.

    Text (including text specials): [0,0,0], matching upstream text IDs.
    Image BOI:[type,-1,-1], VQ:[type,row,col], newline:[type,row,W],
    EOI:[type,H,W]; target BOA:[2,-2,-2], EOA:[2,H+1,W].
    Reference roles are inferred only from actual BOI..EOI spans in prefix;
    image-null CFG cannot fabricate one. Absolute 1D positions are separate.
    """
    if input_ids.ndim!=2 or input_ids.shape[0]!=1:raise ValueError('single sample metadata required')
    values=input_ids[0].tolist();n=len(values);h,w=target_grid
    # Parse the already materialized CPU token list, then transfer two compact
    # metadata tensors once. Avoid one tiny CUDA allocation per reference token.
    types=[0]*n;ids=[[0,0,0] for _ in range(n)]
    index=0
    while reference_enabled and index<prefix_length:
        if values[index]!=126349:index+=1;continue
        start=index;index+=1;row=0;col=0
        ids[start]=[1,-1,-1]
        while index<prefix_length and values[index]!=126350:
            ids[index]=[1,row,col]
            if values[index]==126084:row+=1;col=0
            else:col+=1
            index+=1
        if index>=prefix_length:raise ValueError('unterminated reference image')
        first_newline=next((j for j in range(start+1,index) if values[j]==126084),None)
        width=first_newline-start-1 if first_newline is not None else col
        ids[index]=[1,row,width];types[start:index+1]=[1]*(index-start+1);index+=1
    expected=prefix_length+2+h*(w+1)+2
    if n!=expected:raise ValueError('target metadata length mismatch')
    types[prefix_length:]=[2]*(n-prefix_length)
    ids[prefix_length]=[2,-2,-2];ids[prefix_length+1]=[2,-1,-1]
    for row in range(h):
        start=prefix_length+2+row*(w+1)
        ids[start:start+w+1]=[[2,row,col] for col in range(w+1)]
        if values[start+w]!=126084:raise ValueError('target newline mismatch')
    ids[-2]=[2,h,w];ids[-1]=[2,h+1,w]
    return (torch.tensor(types,device=input_ids.device,dtype=torch.long),
            torch.tensor(ids,device=input_ids.device,dtype=torch.long),ROLE_MAPPING_VERSION)
