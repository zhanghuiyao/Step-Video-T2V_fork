# Copyright 2025 StepFun Inc. All Rights Reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from stepvideo.text_encoder.flashattention import FlashSelfAttention
from stepvideo.modules.normalization import RMSNorm
from stepvideo.text_encoder.tokenizer import LLaMaEmbedding, Wrapped_StepChatTokenizer
from stepvideo.utils import with_empty_init
from safetensors.torch import load_file
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from einops import rearrange
import json


    
def safediv(n, d):
    q, r = divmod(n, d)
    assert r == 0
    return q


class MultiQueryAttention(nn.Module):
    def __init__(self, cfg, layer_id=None):
        super().__init__()

        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.max_seq_len = cfg.seq_length
        self.use_flash_attention = cfg.use_flash_attn
        assert self.use_flash_attention, 'FlashAttention is required!'

        self.n_groups = cfg.num_attention_groups
        self.tp_size = 1
        self.n_local_heads = cfg.num_attention_heads
        self.n_local_groups = self.n_groups

        self.wqkv = nn.Linear(
            cfg.hidden_size,
            cfg.hidden_size + self.head_dim * 2 * self.n_groups,
            bias=False,
        )
        self.wo = nn.Linear(
            cfg.hidden_size,
            cfg.hidden_size,
            bias=False,
        )

        assert self.use_flash_attention, 'non-Flash attention not supported yet.'
        self.core_attention = FlashSelfAttention(attention_dropout=cfg.attention_dropout)
        
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seq_len: Optional[torch.Tensor],
    ):
        # import pdb;pdb.set_trace()

        seqlen, bsz, dim = x.shape
        xqkv = self.wqkv(x)

        xq, xkv = torch.split(
            xqkv,
            (dim // self.tp_size,
             self.head_dim*2*self.n_groups // self.tp_size
            ),
            dim=-1,
        )

        # gather on 1st dimension
        xq = xq.view(seqlen, bsz, self.n_local_heads, self.head_dim)
        xkv = xkv.view(seqlen, bsz, self.n_local_groups, 2 * self.head_dim)
        xk, xv = xkv.chunk(2, -1)

        # rotary embedding + flash attn
        xq = rearrange(xq, "s b h d -> b s h d")
        xk = rearrange(xk, "s b h d -> b s h d")
        xv = rearrange(xv, "s b h d -> b s h d")

        q_per_kv = self.n_local_heads // self.n_local_groups
        if q_per_kv > 1:
            b, s, h, d = xk.size()
            if h == 1:
                xk = xk.expand(b, s, q_per_kv, d)
                xv = xv.expand(b, s, q_per_kv, d)
            else:
                ''' To cover the cases where h > 1, we have
                    the following implementation, which is equivalent to:
                        xk = xk.repeat_interleave(q_per_kv, dim=-2)
                        xv = xv.repeat_interleave(q_per_kv, dim=-2)
                    but can avoid calling aten::item() that involves cpu.
                '''
                idx = torch.arange(q_per_kv * h, device=xk.device).reshape(q_per_kv, -1).permute(1, 0).flatten()
                xk = torch.index_select(xk.repeat(1, 1, q_per_kv, 1), 2, idx).contiguous()
                xv = torch.index_select(xv.repeat(1, 1, q_per_kv, 1), 2, idx).contiguous()

        if self.use_flash_attention:
            output = self.core_attention(xq, xk, xv,
                                      cu_seqlens=cu_seqlens,
                                      max_seq_len=max_seq_len)
            # reduce-scatter only support first dimention now
            output = rearrange(output, "b s h d -> s b (h d)").contiguous()
        


            # # debug
            # https://huggingface.co/stepfun-ai/Step-Audio-Chat/commit/aa82b184aa5ec627ef94545daa7a661711e83596#d2h-542184
            # https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B/blob/main/modeling_step1.py
            #
            # _mask = self.build_alibi_cache(xk.shape[1], xq.shape[2], xq.dtype, xq.device)[:, :, -xq.shape[1] :, :]
            # xq = xq.transpose(1, 2)   # b s h d -> b h s d
            # xk = xk.transpose(1, 2)
            # xv = xv.transpose(1, 2)
            # attn_output = torch.nn.functional.scaled_dot_product_attention(
            #     xq, xk, xv, attn_mask=_mask
            # )
            # attn_output = attn_output.transpose(1, 2)
            #
            # import pdb;pdb.set_trace()

        else:
            xq, xk, xv = [
                rearrange(x, "b s ... -> s b ...").contiguous()
                for x in (xq, xk, xv)
            ]
            output = self.core_attention(xq, xk, xv, mask)
        output = self.wo(output)
        return output


    def build_alibi_cache(self, block_size, n_heads, dtype, device):
        
        import math

        # get slopes
        n = 2 ** math.floor(math.log2(n_heads))  # nearest 2**n to n_heads
        m0 = 2.0 ** (-8.0 / n)
        # 2^(-8/n), 2^(-8*2/n), 2^(-8*3/n), ...
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** (-4.0 / n)
            # 2^(-8/(2n)), 2^(-8*3/(2n)), 2^(-8*5/(2n)), ...
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        slopes = slopes.to(device)

        tril = torch.tril(torch.ones(1, 1, block_size, block_size, device=device))

        bias_rows = torch.arange(block_size, device=device).view(1, -1)
        bias_cols = torch.arange(block_size, device=device).view(-1, 1)
        bias = -torch.sqrt(bias_cols - bias_rows)
        bias = bias.view(1, block_size, block_size) * slopes.view(-1, 1, 1)
        bias = bias.masked_fill(tril == 0, float("-inf"))

        return bias.type(dtype)



class FeedForward(nn.Module):
    def __init__(
        self,
        cfg,
        dim: int,
        hidden_dim: int,
        layer_id: int,
        multiple_of: int=256,
    ):
        super().__init__()

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]
        self.swiglu = swiglu
            
        self.w1 = nn.Linear(
            dim,
            2 * hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x):
        x = self.swiglu(self.w1(x))
        output = self.w2(x)
        return output



class TransformerBlock(nn.Module):
    def __init__(
        self, cfg, layer_id: int
    ):
        super().__init__()

        self.n_heads = cfg.num_attention_heads
        self.dim = cfg.hidden_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.attention = MultiQueryAttention(
            cfg,
            layer_id=layer_id,
        )

        self.feed_forward = FeedForward(
            cfg,
            dim=cfg.hidden_size,
            hidden_dim=cfg.ffn_hidden_size,
            layer_id=layer_id,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
        )
        self.ffn_norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seq_len: Optional[torch.Tensor],
    ):
        residual = self.attention.forward(
            self.attention_norm(x), mask,
            cu_seqlens, max_seq_len
        )
        h = x + residual
        ffn_res = self.feed_forward.forward(self.ffn_norm(h))
        out = h + ffn_res

        # import pdb;pdb.set_trace()

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        config,
        max_seq_size=8192,
    ):
        super().__init__()
        self.num_layers = config.num_layers
        self.layers = self._build_layers(config)

    def _build_layers(self, config):
        layers = torch.nn.ModuleList()
        for layer_id in range(self.num_layers):
            layers.append(
                TransformerBlock(
                    config,
                    layer_id=layer_id + 1 ,
                )
            )
        return layers

    def forward(
        self,
        hidden_states,
        attention_mask,
        cu_seqlens=None,
        max_seq_len=None,
    ):

        if max_seq_len is not None and not isinstance(max_seq_len, torch.Tensor):
            max_seq_len = torch.tensor(max_seq_len, dtype=torch.int32, device="cpu")

        for lid, layer in enumerate(self.layers):
            hidden_states = layer(
                                    hidden_states,
                                    attention_mask,
                                    cu_seqlens,
                                    max_seq_len,
                                )
        return hidden_states


class Step1Model(PreTrainedModel):
    config_class=PretrainedConfig
    @with_empty_init
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.tok_embeddings = LLaMaEmbedding(config)
        self.transformer = Transformer(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):

        hidden_states = self.tok_embeddings(input_ids)

        hidden_states = self.transformer(
            hidden_states,
            attention_mask,
        )
        return hidden_states
    
    

class STEP1TextEncoder(torch.nn.Module):
    def __init__(self, model_dir, max_length=320):
        super(STEP1TextEncoder, self).__init__()
        self.max_length = max_length
        self.text_tokenizer = Wrapped_StepChatTokenizer(os.path.join(model_dir, 'step1_chat_tokenizer.model'))
        text_encoder = Step1Model.from_pretrained(model_dir)
        self.text_encoder = text_encoder.eval().to(torch.bfloat16)
        
    @torch.no_grad
    def forward(self, prompts, with_mask=True, max_length=None):
        self.device = next(self.text_encoder.parameters()).device
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if type(prompts) is str:
                prompts = [prompts]
            
            txt_tokens = self.text_tokenizer(
                prompts, max_length=max_length or self.max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            y = self.text_encoder(
                txt_tokens.input_ids.to(self.device), 
                attention_mask=txt_tokens.attention_mask.to(self.device) if with_mask else None
            )
            y_mask = txt_tokens.attention_mask
        return y.transpose(0,1), y_mask

