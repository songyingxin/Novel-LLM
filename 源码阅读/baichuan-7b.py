
from .configuration_baichuan import BaichuanConfig
from .generation_utils import build_chat_input, TextIterStreamer

import math
from typing import List, Optional, Tuple, Union
from threading import Thread

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerationConfig
from transformers.utils import logging, ContextManagers

import os
from contextlib import contextmanager
logger = logging.get_logger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm 实现，比 LayerNorm 快
        Args:
            hidden_size: baichuan 的 hidden size，可以参见模型文件 config.json
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [bs, seq_len, hidden_size]
        Return:
            [bs, seq_len, hidden_size]
        """

        # 计算方差
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)

        # 用标准差进行归一化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 权重矩阵
        return self.weight * hidden_states


class MLP(nn.Module):
    """ SwiGLU 实现 """
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        """
        Args:
            intermediate_size: SwiGLU的中间维度，可以灵活配置，具体可以参见 config.json
            hidden_act:激活函数名，swish 是 silu
        """
        super().__init__()

        # SwiGLU 的两个 Linear
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # swish 的 torch 实现
        self.act_fn = ACT2FN[hidden_act]

        # 激活函数值计算后接 Linear
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        """
        Args: 
            x: [bs, seq_len, hidden_size]
        Return: [bs, seq_len, hidden_size]
        """
        # SwiGLU + Linear 的计算
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """多头注意力机制实现"""

    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads # 头数
        self.head_dim = self.hidden_size // self.num_heads # 每个头的维度
        self.max_position_embeddings = config.max_position_embeddings

        # x 变换成 QKV 的权重矩阵
        self.W_pack = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # 旋转位置编码
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None
    ):
        """
        Args:
            hidden_states: [bs,seq_len,hidden_size]
            attention_mask: [bs,1,seq_len,seq_len]
            position_ids: [bs, seq_len]
        Return: [bs, seq_len, hidden_size]
        """

        # 输入 x 的线性变换, [bs,seq_len, 3*hidden_size]
        proj = self.W_pack(hidden_states)

        # 拆分成 Q,K,V
        proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)

        # q,k,v: [bs, num_heads, seq_len, hidden_size]
        query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        
        # 融入旋转位置编码
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # 计算 self-attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = attention_mask)

        # 合并多个头的attention
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 计算完 attention后的 Linear 变换，用于融合多头信息
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class DecoderLayer(nn.Module):
    """GPT style 的 Transformer"""

    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config) # 注意力层

        # Linear 层
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        # 归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None
    ):
        """
        Args:
            hidden_states: [bs,seq_len,hidden_size]
            attention_mask: [bs,1,seq_len,seq_len]
            position_ids: [bs, seq_len]
        Return: [bs, seq_len, hidden_size]
        """

        residual = hidden_states

        # 对 Transformer 的每一层输入进行归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 计算该层的 Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # 残差连接
        hidden_states = residual + hidden_states
        residual = hidden_states

        # 残差后进行归一化
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 全连接
        hidden_states = self.mlp(hidden_states)

        # 再次残差连接
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs

class BaichuanModel(BaichuanPreTrainedModel):
    def __init__(self, config: BaichuanConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # word 转化为 embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Transformer 层
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None
    ):

        batch_size, seq_length = input_ids.shape
    
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        inputs_embeds = self.embed_tokens(input_ids)

        # attention mask 
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        # attention_mask 转换
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Transformer 多层计算
        for idx, decoder_layer in enumerate(self.layers):

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

        # 再次归一化
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class NormHead(nn.Module):
    """ 归一化输出Linear，baichuan1 中无"""

    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.first_flag = True

    def forward(self, hidden_states):
        if self.training:
            norm_weight = nn.functional.normalize(self.weight)
        elif self.first_flag:
            self.first_flag = False
            self.weight.data = nn.functional.normalize(self.weight)
            norm_weight = self.weight
        else:
            norm_weight = self.weight
        return nn.functional.linear(hidden_states, norm_weight)


class BaichuanForCausalLM(BaichuanPreTrainedModel):
    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config, *model_args, **model_kwargs)
        self.model = BaichuanModel(config)

        self.lm_head = NormHead(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None
    ):

        """
        Args:
            input_ids: [bs,seq_len]
            attention_mask: [bs,seq_len]
            position_ids: [bs, seq_len]
        """

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # [bs,seq_len,hidden_size]
        logits = self.lm_head(hidden_states) # [bs,seq_len,vocab_size]

        loss = None
        if labels is not None:
            # 通过shift 来计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 计算损失，采用 z-loss
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            softmax_normalizer = shift_logits.max(-1).values ** 2
            z_loss = self.config.z_loss_weight * softmax_normalizer.mean()

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels) + z_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )