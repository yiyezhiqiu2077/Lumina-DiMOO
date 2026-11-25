"""Lumina triplets -> branch-specific full prefix and sparse visual logits."""
import torch
from model.livemgt import PrefixLayout
from model.livemgt_conditioning import bbox_size
from utils.image_utils import add_break_line
from utils.edit_mask import vq_positions,sequence_position_metadata
from utils.prompt_utils import generate_image_to_image_prompt
from .core import VQ_OFFSET,CODEBOOK_SIZE


def sequence_inputs(tokenizer,batch,tokens,region,branch='cond'):
    con,un,_=generate_image_to_image_prompt(batch['instruction'],'edit')
    p=tokenizer(con).input_ids;u=tokenizer(un).input_ids
    h,w=tokens.shape[-2:]
    ref=[126349]+add_break_line(batch['source'].flatten().tolist(),h,w,126084)+[126350]
    if branch=='cond':prefix=p[:-1]+ref+p[-1:];ref_start=len(p)-1
    elif branch=='uncond_text':prefix=u[:-1]+ref+u[-1:];ref_start=len(u)-1
    elif branch=='uncond_image':prefix=p;ref_start=None
    else:raise ValueError('invalid CFG branch')
    tail=[126354,126349]+add_break_line(tokens.flatten().tolist(),h,w,126084)+[126350,126355]
    ids=torch.tensor([prefix+tail],device=tokens.device)
    types,coordinates,version=sequence_position_metadata(ids,len(prefix),(h,w),reference_enabled=branch!='uncond_image')
    positions=vq_positions(h,w,len(prefix)+2,tokens.device)[region.flatten()]
    layout=PrefixLayout(len(prefix),positions,bbox_size(batch['original'][0]),types,coordinates,version)
    return ids,layout


def predict_grid(model,tokenizer,batch,tokens,region,branch='cond',gradient_checkpointing=False,
                 condition_cache=None,region_conditioning=None,token_type_encoding=None,token_type_embedding=None,token_type_rope=None,rope_mode=None):
    ids,layout=sequence_inputs(tokenizer,batch,tokens,region,branch)
    logits=model(ids,infer=True,live_layout=layout,mode='sparse',branch=branch,
                 vocab_slice=(VQ_OFFSET,VQ_OFFSET+CODEBOOK_SIZE),gradient_checkpointing=gradient_checkpointing,
                 condition_cache=condition_cache,region_conditioning=region_conditioning,
                 token_type_encoding=token_type_encoding,token_type_embedding=token_type_embedding,token_type_rope=token_type_rope,rope_mode=rope_mode).logits
    # Differentiable scatter; no inactive token entered the Transformer body.
    flat=logits.new_zeros((1,region.numel(),CODEBOOK_SIZE))
    flat=flat.index_copy(1,region.flatten().nonzero().flatten(),logits)
    return flat.reshape(1,*tokens.shape[-2:],CODEBOOK_SIZE)
