import functools
import logging
import math
from typing import List
import torch.nn.functional as F
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from .modeling_llada import LLaDAModelLM
from .configuration_llada import LLaDAConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
__all__ = ["LLaDAForMultiModalGeneration"]

def create_attention_mask(original_lengths, max_tokens, device):
    batch_size = len(original_lengths)
    attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
    for i, length in enumerate(original_lengths):
        attention_mask[i, :length] = 1  # 有效位置设为1
    return attention_mask

class LLaDAForMultiModalGeneration(LLaDAModelLM):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    _keep_in_fp32_modules = ["live_type_rope"]
    def __init__(self, config: LLaDAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if getattr(config, 'live_mgt_conditioning', None):
            from .livemgt_conditioning import enable_conditioning
            enable_conditioning(self, **config.live_mgt_conditioning)
        if getattr(config,'live_mgt_rope',None):
            from .livemgt_rope import enable_type_rope
            enable_type_rope(self,**config.live_mgt_rope)
    
    def _init_weights(self, module):
        super()._init_weights(module)
        from .livemgt_rope import TokenTypeRotary
        if isinstance(module,TokenTypeRotary):
            torch.nn.init.zeros_(module.phase)

    @classmethod
    def from_pretrained(cls,*args,**kwargs):
        requested=kwargs.pop('output_loading_info',False)
        model,info=super().from_pretrained(*args,output_loading_info=True,**kwargs)
        if any('live_type_rope.' in name for name in info['unexpected_keys']):
            raise ValueError('trained rotary parameters present without matching model config; refusing to discard')
        return (model,info) if requested else model

    def forward(self, input_ids=None, labels=None, infer=False, use_cache=False, to_compute_mask=None, cat='', **kwargs):
        live_layout = kwargs.pop("live_layout", None)
        if live_layout is not None:
            from .livemgt import forward_prefix
            return forward_prefix(self, input_ids, live_layout, **kwargs)
        explicit_bias = kwargs.pop("attention_bias", None)
        if infer:
            input_ids = input_ids.tolist()
        # ========================================================
        # padding input batch len & attention bias for attention mask
        # ========================================================
        max_tokens = max([len(_) for _ in input_ids])
        original_lengths = [len(example) for example in input_ids] # every sample len --> record for attention mask
        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids] # padding 0 to right --> max length
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device) 
        # attn mask
        attention_mask = create_attention_mask(original_lengths, max_tokens, self.device)
        # Equal-length inputs have no padding: retain the original mask-free
        # bidirectional fast path. Explicit isolation masks still reach SDPA.
        attention_bias = None
        if min(original_lengths) != max_tokens:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).unsqueeze(1)
        if explicit_bias is not None:
            if attention_bias is None:
                attention_bias = explicit_bias
            elif explicit_bias.dtype == torch.bool:
                attention_bias = attention_bias & explicit_bias
            else:
                attention_bias = torch.zeros_like(attention_bias, dtype=explicit_bias.dtype).masked_fill(
                    ~attention_bias, float('-inf')) + explicit_bias
        # ========================================================
        # model output 
        # ========================================================
        output = LLaDAModelLM.forward(self, input_ids=input_ids, attention_bias=attention_bias, use_cache=use_cache, to_compute_mask=to_compute_mask, cat=cat)
        if infer:
            return output
        
        # ========================================================
        # padding label batch len & loss
        # ========================================================
        labels = [label + [-100] * (max_tokens - len(label)) for label in labels] # padding -100 to right --> max length
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)
        logits = output.logits
        loss = F.cross_entropy(logits.contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1), ignore_index=-100,)
        return loss
    
    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.transformer.blocks), self.model.transformer.ff_out]
        return modules