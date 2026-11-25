"""Experimental Student GAN interface; no unvalidated discriminator selected.

Proposed Lumina input: probabilities -> expected VQ embeddings -> differentiable
VQ decoder -> validated image discriminator. This avoids treating unified text
IDs as image features. A real discriminator needs an external validation record;
there is no built-in production critic. Toy tests verify gradients only.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class ValidatedImageCritic:
    decoder: object
    critic: object
    validation_record: str
    test_fixture_only: bool=False
    def __post_init__(self):
        if not self.validation_record:raise ValueError('a validated discriminator record is required')
        if isinstance(self.critic,torch.nn.Module):self.critic.eval().requires_grad_(False)
    def loss(self,visual_probs,source,region):
        images=self.decoder(visual_probs,source,region)
        return F.softplus(-self.critic(images)).mean()


def student_gan_loss(logits,source,region,step,config,plugin=None,allow_test_fixture=False):
    if not config.get('enabled',False) or step<config.get('warmup',200):return logits.sum()*0
    if plugin is None:raise RuntimeError('GAN enabled but no validated image discriminator installed')
    if plugin.test_fixture_only and not allow_test_fixture:raise RuntimeError('toy critic forbidden in actual training')
    probabilities=logits.float().softmax(-1)
    hard_source=F.one_hot((source-126356).long(),logits.shape[-1]).to(probabilities.dtype)
    probabilities=torch.where(region[...,None],probabilities,hard_source)
    return plugin.loss(probabilities,source,region)*config.get('weight',0.01)
