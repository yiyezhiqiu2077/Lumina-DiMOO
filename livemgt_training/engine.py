"""DiMO Student FKL -> updated-Student fresh-data Aux CE -> EMA.

Upstream loss, CBC and phase ordering are retained. Shared immutable Lumina base
is a PEFT memory strategy, not a different distillation objective.
"""
import torch
from .core import (generation_region,one_step_initial,sample_hard_copy,pseudo_mask,
                   signed_mask_distance,region_weight_map,combined_evidence,lumina_cfg,
                   fkl_surrogate,auxiliary_ce,gt_ce,VQ_OFFSET,sample_mask_ratio)
from .predict import predict_grid
from .gan import student_gan_loss


def grad_stats(params):
    grads=[p.grad for p in params if p.grad is not None]
    return dict(tensors=len(grads),finite=bool(grads) and all(torch.isfinite(g).all().item() for g in grads),
                norm=sum(g.float().square().sum().item() for g in grads)**.5,
                nonzero=sum(int(g.count_nonzero()) for g in grads))


def state_delta(before,after):
    return sum((after[n].float()-p.float()).square().sum().item() for n,p in before.items())**.5

class DiMOEngine:
    def __init__(self,bank,tokenizer,config,optimizers,schedulers,gan_plugin=None):
        self.bank=bank;self.tokenizer=tokenizer;self.c=config;self.opt=optimizers;self.sch=schedulers;self.gan=gan_plugin
    def predict(self,batch,tokens,region,branch='cond',grad=False):
        return predict_grid(self.bank.model,self.tokenizer,batch,tokens,region,branch,
                            gradient_checkpointing=grad and self.c.get('gradient_checkpointing',True))
    def region(self,batch):
        return generation_region(batch['original'],self.c['regional'],self.c['distillation_halo'])
    def one_step(self,batch):
        region=self.region(batch);initial=one_step_initial(batch['source'],region,self.c['student_initial_mask_ratio'])
        logits=self.predict(batch,initial,region)
        return region,initial,logits,sample_hard_copy(logits,batch['source'],region)
    def backward_step(self,loss,role):
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        self.bank.graph_live=True
        try:loss.backward()
        finally:self.bank.graph_live=False
        stats=grad_stats(self.bank.params.values())
        phase=self.bank.params.get('live_type_rope.phase')
        stats['type_rope_norm']=float(phase.grad.norm()) if phase is not None and phase.grad is not None else None
        if not stats['finite'] or stats['norm']==0:raise RuntimeError(f'{role} missing/nonfinite/zero gradient: {stats}')
        if role=='student':
            torch.nn.utils.clip_grad_norm_(list(self.bank.params.values()),self.c.get('max_grad_norm',1.))
        self.opt[role].step();self.sch[role].step();self.opt[role].zero_grad(set_to_none=True);self.bank.save_role()
        return stats
    def update(self,student_batch,auxiliary_batch_fn,student_step):
        c=self.c;bank=self.bank
        # No live Student graph is retained across role switches. Sampling is
        # categorical/no-grad upstream too; replay same initial inputs for FKL.
        bank.activate('student')
        with torch.no_grad():
            region,initial,_,sampled=self.one_step(student_batch)
            distance=signed_mask_distance(student_batch['original'],region) if c['cbc']['enabled'] else None
            geometry=region_weight_map(student_batch['original'],region,c['cbc'],distance=distance)
            ratio=sample_mask_ratio(1,region.device)
            q=pseudo_mask(region,ratio,geometry if c['cbc']['weighted_pseudo_mask'] else None)
            corrupted=torch.where(q,126336,sampled)
            bank.activate('teacher')
            conditional=self.predict(student_batch,corrupted,region)
            ut=self.predict(student_batch,corrupted,region,'uncond_text')
            ui=self.predict(student_batch,corrupted,region,'uncond_image')
            teacher_logits=lumina_cfg(conditional,ut,ui,c['teacher_text_cfg'],c['teacher_image_cfg'])
            evidence=combined_evidence(conditional,ut,ui,c.get('evidence_mode','text'),c['cbc']['evidence_chunk_size']) if c['cbc']['enabled'] and c['cbc']['evidence_weight'] else None
            weights=region_weight_map(student_batch['original'],region,c['cbc'],evidence,distance)
            bank.activate('auxiliary')
            auxiliary_logits=self.predict(student_batch,corrupted,region)
        before_s={n:p.clone() for n,p in bank.states['student'].items()}
        before_a={n:p.clone() for n,p in bank.states['auxiliary'].items()}
        bank.activate('student',train=True)
        logits=self.predict(student_batch,initial,region,grad=True)
        fkl=fkl_surrogate(logits,teacher_logits,auxiliary_logits,q,weights)
        gt=gt_ce(logits,student_batch['target'],student_batch['original'])*c['gt_ce']['weight'] if c['gt_ce']['enabled'] else logits.sum()*0
        gan=student_gan_loss(logits,student_batch['source'],region,student_step,c['gan'],self.gan)
        stats_s=self.backward_step(fkl+gt+gan,'student')
        assert state_delta(before_a,bank.states['auxiliary'])==0, 'Student phase changed Auxiliary'
        aux_records=[]
        for _ in range(c.get('fake_rounds',1)):
            # This callback is called AFTER Student optimizer.step().
            fresh=auxiliary_batch_fn()
            if fresh['uid']==student_batch['uid']:raise ValueError('Auxiliary must use fresh data, not Student-phase sample')
            bank.activate('student')
            with torch.no_grad():
                ar,_,_,labels=self.one_step(fresh)
                geom=region_weight_map(fresh['original'],ar,c['cbc'])
                aq=pseudo_mask(ar,sample_mask_ratio(1,ar.device),geom if c['cbc']['weighted_pseudo_mask'] else None)
                corrupted_aux=torch.where(aq,126336,labels)
            student_snapshot={n:p.clone() for n,p in bank.states['student'].items()}
            bank.activate('auxiliary',train=True)
            aux_logits=self.predict(fresh,corrupted_aux,ar,grad=True)
            aux_loss=auxiliary_ce(aux_logits,labels-VQ_OFFSET,aq,c['auxiliary_label_smoothing'])
            stats_a=self.backward_step(aux_loss,'auxiliary')
            assert state_delta(student_snapshot,bank.states['student'])==0, 'Auxiliary phase changed Student'
            assert torch.equal(labels[~ar],fresh['source'][~ar])
            aux_records.append(dict(uid=fresh['uid'],loss=aux_loss.item(),gradient=stats_a,
                                    original_tokens=int(fresh['original'].sum()),region_tokens=int(ar.sum()),pseudo_tokens=int(aq.sum()),
                                    label_source='fresh updated Student',teacher_grad=False,student_grad_during_aux=False))
        ema_before={n:p.clone() for n,p in bank.states['ema'].items()}
        rate=bank.ema_update(c['ema_decay'],student_step,c['ema_warmup'])
        assert torch.equal(sampled[~region],student_batch['source'][~region])
        return dict(student_uid=student_batch['uid'],fkl=fkl.item(),gt_ce=gt.item(),gan=gan.item(),student_gradient=stats_s,
                    original_tokens=int(student_batch['original'].sum()),region_tokens=int(region.sum()),pseudo_tokens=int(q.sum()),
                    cbc_enabled=c['cbc']['enabled'],geometry_weight_sum=float(geometry.sum()) if geometry is not None else None,
                    evidence_mean=float(evidence[region].mean()) if evidence is not None else None,
                    loss_weight_sum=float(weights.sum()) if weights is not None else None,
                    student_delta=state_delta(before_s,bank.states['student']),auxiliary_delta=state_delta(before_a,bank.states['auxiliary']),
                    ema_delta=state_delta(ema_before,bank.states['ema']),ema_decay=rate,auxiliary=aux_records,
                    type_rope_deltas={role:float(bank.states[role]['live_type_rope.phase'].norm()) for role in bank.states} if 'live_type_rope.phase' in bank.params else None,
                    source_hard_copy=True,phase_isolation=True,teacher_frozen=True)
