"""Sparse + cache one-step EMA inference; no claim of per-step cache speedup."""
import argparse,json,hashlib,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from diffusers import VQModel
from model import LLaDAForMultiModalGeneration
from model.livemgt import ConditionKVCache
from .adapters import install_trainables,RoleBank
from .data import read_manifest,encode_triplet
from .predict import predict_grid
from .core import generation_region,one_step_initial,sample_hard_copy
from utils.image_utils import decode_vq_to_image

@torch.no_grad()
def main(args):
    path=Path(args.checkpoint)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=path.with_suffix('.sha256').read_text().strip():raise ValueError('bad checkpoint checksum')
    state=torch.load(path,map_location='cpu',weights_only=False);c=state['config']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(c['seed'])
    from livemgt_training.precision import configure
    configure(False)
    model=LLaDAForMultiModalGeneration.from_pretrained(c['base_checkpoint'],torch_dtype=torch.float32 if c.get('mixed_precision','no')=='no' else torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True).to('cuda').eval()
    params=install_trainables(model,c);bank=RoleBank(model,params);from model.livemgt_rope import upgrade_roles
    bank.states,_=upgrade_roles(state['roles'],params);bank.activate(args.role)
    tok=AutoTokenizer.from_pretrained(c['base_checkpoint'],local_files_only=True)
    vae=VQModel.from_pretrained(c['base_checkpoint'],subfolder='vqvae',local_files_only=True).to('cuda').eval()
    batch=encode_triplet(read_manifest(args.manifest or c['validation_manifest'])[args.index],vae,c['resolution'],c.get('asset_roots',{}))
    region=generation_region(batch['original'],c['regional'],c['distillation_halo']);initial=one_step_initial(batch['source'],region,c['student_initial_mask_ratio'])
    cache=ConditionKVCache() if args.cache else None
    try:
        torch.cuda.synchronize();start=time.perf_counter()
        logits=predict_grid(model,tok,batch,initial,region,condition_cache=cache)
        result=sample_hard_copy(logits,batch['source'],region)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
    finally:
        if cache is not None:cache.clear()
    assert torch.equal(result[~region],batch['source'][~region])
    torch.save(result.cpu(),out/'vq_tokens.pt')
    image=decode_vq_to_image(result.flatten(1),'',c['base_checkpoint'],c['resolution'],c['resolution'],vae);image.save(out/'result.png')
    (out/'summary.json').write_text(json.dumps(dict(role=args.role,uid=batch['uid'],source_hard_copy=True,steps=1,cache=args.cache,
       forward_sampling_seconds=seconds,quality_evaluated=False,
       cache_note='One step only prefills KV; no intra-generation reuse speedup claimed',teacher_validity='smoke checkpoint, no final-quality Teacher'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--role',choices=['ema','student'],default='ema');p.add_argument('--cache',action='store_true')
    p.add_argument('--manifest');p.add_argument('--index',type=int,default=0);main(p.parse_args())
