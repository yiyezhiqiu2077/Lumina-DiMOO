"""Role adapters, independent moments/schedules, replay position and all RNGs."""
import os,random,hashlib,json
from pathlib import Path
import numpy as np
import torch
from model.livemgt_rope import upgrade_roles


def save_checkpoint(path,bank,optimizers,schedulers,progress,config,origin):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    payload=dict(roles=bank.states,optimizers={k:v.state_dict() for k,v in optimizers.items()},
                 schedulers={k:v.state_dict() for k,v in schedulers.items()},progress=progress,config=config,origin=origin,
                 parameter_names=list(bank.params),
                 rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),python=random.getstate(),numpy=np.random.get_state()))
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary);os.replace(temporary,path)
    path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')


def load_checkpoint(path,bank,optimizers,schedulers,origin):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=path.with_suffix('.sha256').read_text().strip():raise ValueError('checkpoint checksum mismatch')
    # Only read trusted local checkpoints produced by this pipeline (NumPy RNG).
    state=torch.load(path,map_location='cpu',weights_only=False)
    saved_origin=state['origin']
    # Early smoke files from this experiment predate the explicit run_mode
    # field. They may resume as smoke only and can never qualify as Teachers.
    compared=dict(origin)
    if 'run_mode' not in saved_origin and compared.get('run_mode','').startswith('smoke_'):
        compared.pop('run_mode')
    if 'type_rope' not in saved_origin and 'type_rope' in compared:
        if compared['type_rope'].get('mode')=='three_axis':raise ValueError('legacy resume cannot silently replace 1D RoPE with three-axis; initialize a new experiment')
        compared.pop('type_rope')
    if saved_origin!=compared:raise ValueError('base/teacher/data origin mismatch')
    bank.states,migrations=upgrade_roles(state['roles'],bank.params)
    old_names=state.get('parameter_names',list(state['roles']['student']))
    new_names=list(bank.params)
    for role,optimizer in optimizers.items():
        saved=state['optimizers'][role]
        if old_names!=new_names:
            if set(new_names)-set(old_names)!={'live_type_rope.phase'} or set(old_names)-set(new_names):
                raise ValueError('unsupported optimizer parameter schema migration')
            if len(saved['param_groups'])!=1:raise ValueError('legacy optimizer migration requires one group')
            current=optimizer.state_dict()
            old_ids=dict(zip(old_names,saved['param_groups'][0]['params']))
            new_ids=dict(zip(new_names,current['param_groups'][0]['params']))
            migrated={new_ids[n]:saved['state'][old_ids[n]] for n in old_names if old_ids[n] in saved['state']}
            group=dict(saved['param_groups'][0],params=list(new_ids.values()))
            saved=dict(state=migrated,param_groups=[group])
        optimizer.load_state_dict(saved)
    bank.last_migration=migrations
    for k,v in schedulers.items():v.load_state_dict(state['schedulers'][k])
    rng=state['rng'];torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])
    random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    return state['progress']
