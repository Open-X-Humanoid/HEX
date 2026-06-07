import os
import json
import math
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from typing import Dict, Any, Optional
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from einops import rearrange
from collections import defaultdict
from typing import Dict, Optional, List
from transformers.activations import ACT2FN


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))
    

class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x
    

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # Supported options include: 'layer_norm', 'ada_norm', etc.
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        # Optional positional embedding module applied before attention.
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Block 1: normalization before attention.
        # When `cross_attention_dim` is provided, `attn1` can operate as cross-attention;
        # otherwise, it defaults to self-attention.
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # Block 2: feed-forward network with pre-normalization.
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # Optional dropout applied to the attention output before the residual connection.
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        # 1. Pre-normalization before attention.
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # Apply positional encoding if enabled.
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 2. Attention block.
        # This behaves as self-attention when `encoder_hidden_states` is None,
        # and as cross-attention otherwise.
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            # encoder_attention_mask=encoder_attention_mask,
        )

        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        # Residual connection after attention.
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Feed-forward block with residual connection.
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class StateL2Head(nn.Module):
    """
    用当前 state + VLM 特征，一次性预测未来 H 个 state（L2 回归）：
      - 输入:
          state_curr: [B, state_dim]         当前时刻 s_t
          vl_feats  : [B, L_vl, cross_dim]   VLM 特征
      - 输出:
          pred_future: [B, H, state_dim]     预测的未来 state 序列
    """
    def __init__(self, full_config, interleave_self_attention=False):
        super().__init__()
        config = full_config.framework.state_model
        self.state_dim = config.transformer_block.state_dim
        self.input_dim = config.transformer_block.input_dim
        self.cross_attention_dim = config.transformer_block.cross_attention_dim
        self.num_layers = config.transformer_block.num_layers
        self.interleave_self_attention = interleave_self_attention
        self.horizon = config.state_horizon

        # 1) 状态映射到 transformer dim
        self.state_encoder = MLP(
            input_dim=self.state_dim,
            hidden_dim=config.transformer_block.hidden_dim,
            output_dim=self.input_dim,
        )

        # 2) 未来 token 的可学习初始值 (H 个)
        self.future_token_embed = nn.Parameter(
            torch.randn(config.state_horizon-1, self.input_dim) * 0.02
        )

        # 3) 序列位置编码：长度 = 1 + H (第0个是当前 state，后面是未来 token)
        self.pos_embed = nn.Embedding(config.state_horizon, self.input_dim)

        # 4) Transformer blocks（交替 self / cross）
        blocks = []
        for idx in range(config.transformer_block.num_layers):
            use_self_attn = (idx % 2 == 1) and interleave_self_attention
            curr_cross_attention_dim = config.transformer_block.cross_attention_dim if not use_self_attn else None

            block = BasicTransformerBlock(
                dim=self.input_dim,
                num_attention_heads=config.transformer_block.num_heads,
                attention_head_dim=config.transformer_block.head_dim,
                dropout=config.transformer_block.dropout,
                cross_attention_dim=curr_cross_attention_dim,           
                activation_fn="geglu",
                attention_bias=False,
                upcast_attention=False,
                norm_elementwise_affine=True,
                norm_type="layer_norm",
                norm_eps=1e-5,
                final_dropout=False,
                attention_type="default",
                positional_embeddings=None,
                num_positional_embeddings=None,
                ff_inner_dim=None,
                ff_bias=True,
                attention_out_bias=True,
            )
            blocks.append(block)
        self.transformer_blocks = nn.ModuleList(blocks)

        # 5) 输出 head: hidden -> flow (velocity)
        self.output_head = nn.Linear(self.input_dim, self.state_dim)

    def forward(
        self,
        state: torch.Tensor,        
        vl_feats: torch.Tensor, 
        temb: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        B, T, D = state.shape
        device = state.device
        assert D == self.state_dim

         # 1) 把当前 state 编码成一个 token: [B, 1, input_dim]
        curr_token = self.state_encoder(state[:, 0])            # [B, input_dim]
        curr_token = curr_token.unsqueeze(1)                    # [B, 1, input_dim]

        time_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embed(time_ids)     # [B, T, input_dim]

        # 2) 构造 H 个未来 token：可学习参数，复制到 batch
        # future_token_embed: [H, input_dim] -> [1, H, input_dim] -> [B, H, input_dim]
        future_tokens = self.future_token_embed.unsqueeze(0).expand(B, self.horizon-1, -1)

        # 3) 拼成完整序列: [B, 1+H, input_dim]
        hidden_states = torch.cat([curr_token, future_tokens], dim=1)

        # 4) 加位置编码
        pos_ids = torch.arange(self.horizon, device=device).unsqueeze(0).expand(B, self.horizon)  # [B, 1+H]
        pos_emb = self.pos_embed(pos_ids)                                               # [B, 1+H, input_dim]
        hidden_states = hidden_states + pos_emb
        encoder_hidden_states = vl_feats

       # 3) 通过若干 transformer block（无 causal mask）
        for idx, block in enumerate(self.transformer_blocks):
            if (idx % 2 == 1) and self.interleave_self_attention:
                # 奇数层: 只 self-attention，不看 encoder
                hidden_states = block(
                    hidden_states,                  # [B, T, input_dim]
                    attention_mask=None,            # ✅ Flow Matching 不需要 causal mask
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # 偶数层: self-attention + cross-attention 到 VL
                hidden_states = block(
                    hidden_states,                  # [B, T, input_dim]
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )

        state_all = self.output_head(hidden_states)         # [B, 1+H, state_dim]
        pred_future = state_all[:, 1:, :]

        if return_loss:
            loss = F.mse_loss(pred_future, state[:, 1:])
            return hidden_states, loss

        return hidden_states
    
    @torch.no_grad()
    def predict_state(
        self,
        state_curr: torch.Tensor,           # 可为 [B, state_dim] 或 [B, T, state_dim]
        vl_feats: torch.Tensor,        # [B, L_vl, cross_dim]
        temb: Optional[torch.Tensor] = None,
    ):
        """
        推理：使用当前 state + VLM 特征预测未来 H 帧。

        输入：
            state:
                [B, state_dim]        或
                [B, T, state_dim]（自动取 state[:,0]）
            vl_feats: [B, L_vl, cross_dim]

        输出：
            pred_future: [B, H, state_dim]
        """
        # ---- 1) 兼容两种输入格式 ----
        B, D = state_curr.shape
        device = state_curr.device
        T = self.horizon  # = state_horizon

        # ---- 2) encode 当前 state token ----
        curr_token = self.state_encoder(state_curr)          # [B, input_dim]
        curr_token = curr_token.unsqueeze(1)                 # [B, 1, input_dim]

        # ---- 3) 扩展可学习 future tokens ----
        # self.future_token_embed: [H, input_dim]
        future_tokens = self.future_token_embed.unsqueeze(0).expand(B, T-1, -1)
        # 拼接序列: [B, 1 + (T-1), input_dim] = [B, T, input_dim]
        hidden_states = torch.cat([curr_token, future_tokens], dim=1)

        # ---- 4) 加位置编码 ----
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embed(pos_ids)                    # [B, T, input_dim]
        hidden_states = hidden_states + pos_emb

        encoder_hidden_states = vl_feats

        # ---- 5) Transformer blocks ----
        for idx, block in enumerate(self.transformer_blocks):
            if (idx % 2 == 1) and self.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )

        # ---- 6) Head 输出 ----
        state_all = self.output_head(hidden_states)           # [B, T, state_dim]

        # ---- 7) 取未来部分 ----
        pred_future = state_all[:, 1:, :]                     # [B, T-1, D]

        return pred_future
    

class MoEMLP(nn.Module):
    def __init__(self, config, hidden_size = None, intermediate_size = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[self.config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss
    

class MoEGate_load_bal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape        
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.mean(0)    # .float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight.to(dtype=hidden_states.dtype), aux_loss


class HBMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([MoEMLP(config, intermediate_size = config.moe_intermediate_size) for i in range(config.n_routed_experts)])
        self.gate = MoEGate_load_bal(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MoEMLP(config=config, intermediate_size = intermediate_size)
    
    def forward(self, hidden_states, return_gate_info=False):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)     # .float()
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])       # .float()
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1).to(dtype=hidden_states.dtype)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        if return_gate_info:
            gate_info = {
                "topk_idx": topk_idx,          # [B, N, K]
                "topk_weight": topk_weight,    # [B, N, K]
                "aux_loss": aux_loss,
            }
            return y, gate_info
        
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)      # .float()
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)      # .float()
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
    

class CrossEmnodiedStateMoEL2Head(nn.Module):
    def __init__(self, full_config, state_registry, interleave_self_attention=True):
        super().__init__()
        config = full_config.framework.state_model
        self.config = config
        self.interleave_self_attention = interleave_self_attention
        self.hidden_dim = config.transformer_block.hidden_dim  # Unified latent dimensionality
        self.horizon = config.state_horizon   # Prediction horizon H
        self.state_registry = state_registry
        
        # 1. Dynamic projection layers (input encoders and output decoders)
        # Automatically scan the registry and instantiate dedicated projection
        # layers for every observed [body-part, state-dimension] specification.
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.canonical_parts = [
            "left_arm", "right_arm", "left_hand", "right_hand", 
            "left_leg", "right_leg",
            "head", "waist",
            "others",
        ]
        
        self._batch_register(state_registry)

        # 2. Learnable placeholder token for missing body-part states
        self.missing_part_token = nn.Parameter(torch.zeros(1, self.hidden_dim))

        # 3. Sequence tokens and positional embeddings: Query tokens corresponding to future steps 1 ... H-1 for each body part
        self.future_tokens = nn.Parameter(torch.randn(len(self.canonical_parts), self.horizon - 1, self.hidden_dim) * 0.02)
        self.temp_pos_embed = nn.Parameter(torch.randn(1, 1, self.horizon, self.hidden_dim) * 0.02)
        self.part_pos_embed = nn.Parameter(torch.randn(1, len(self.canonical_parts), 1, self.hidden_dim) * 0.02)

        # 4. Transformer backbone
        blocks = []
        for idx in range(config.transformer_block.num_layers):
            use_self_attn = (idx % 2 == 1) and interleave_self_attention
            curr_cross_attention_dim = config.transformer_block.cross_attention_dim if not use_self_attn else None

            block = BasicTransformerBlock(
                dim=config.transformer_block.input_dim,
                num_attention_heads=config.transformer_block.num_heads,
                attention_head_dim=config.transformer_block.head_dim,
                dropout=config.transformer_block.dropout,
                cross_attention_dim=curr_cross_attention_dim,           
                activation_fn="geglu",
                attention_bias=False,
                upcast_attention=False,
                norm_elementwise_affine=True,
                norm_type="layer_norm",
                norm_eps=1e-5,
                final_dropout=False,
                attention_type="default",
                positional_embeddings=None,
                num_positional_embeddings=None,
                ff_inner_dim=None,
                ff_bias=True,
                attention_out_bias=True,
            )
            blocks.append(block)
        self.transformer_blocks = nn.ModuleList(blocks)

        self.moe_in = HBMoE(config.MoE_block)
        self.moe_out = HBMoE(config.MoE_block)

    def _batch_register(self, state_registry):
        for robot_tag, parts in state_registry.items():
            base_name = robot_tag.rsplit("_v", 1)[0]
            for p_name, info in parts.items():
                p_dim = info["dim"]
                embodiment_key = f"{base_name}_{p_name}_dim{p_dim}"
                if embodiment_key not in self.encoders:
                    self.encoders[embodiment_key] = nn.Linear(p_dim, self.hidden_dim)
                    self.decoders[embodiment_key] = nn.Linear(self.hidden_dim, p_dim)

    def forward(
        self, 
        state,
        vl_feats,
        tags: str,
        temb: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        return_moe_info: bool = False,
    ):
        B = state.shape[0]
        # --- 1. 异构输入编码 ---
        group = defaultdict(list)  # (tag, p_name) -> list[(i, p_idx)]
        for i, tag in enumerate(tags):
            parts_cfg = self.state_registry[tag]
            for p_idx, p_name in enumerate(self.canonical_parts):
                if p_name in parts_cfg:
                    group[(tag, p_name)].append((i, p_idx))

        part_latents = torch.zeros(
            (B, len(self.canonical_parts), self.hidden_dim),
            device=state.device,
            dtype=state.dtype,
        )

        for (tag, p_name), idxs in group.items():
            info = self.state_registry[tag][p_name]
            start, end, p_dim = info["start"], info["end"], info["dim"]
            base = tag.split("_v", 1)[0]
            key = f"{base}_{p_name}_dim{p_dim}"
            raw = torch.stack([state[i, 0, start:end] for i, _ in idxs], dim=0)  # [N, p_dim]
            lat = self.encoders[key](raw)                                       # [N, hidden]
            for j, (i, p_idx) in enumerate(idxs):
                part_latents[i, p_idx] = lat[j]
        curr_token = part_latents.unsqueeze(2)

        future_queries = self.future_tokens.unsqueeze(0).expand(B, -1, -1, -1)  
        x = torch.cat([curr_token, future_queries], dim=2)
        x = x + self.temp_pos_embed 
        x = x + self.part_pos_embed

        B, P, T, D = x.shape

        if return_moe_info:
            moe_info = {}
            hidden_states, moe_in_gate = self.moe_in(x.reshape(B, P * T, D), return_gate_info=True)
            moe_info["moe_in"] = moe_in_gate
        else:
            hidden_states = self.moe_in(x.reshape(B, P * T, D))
        encoder_hidden_states = vl_feats

       # 3) 通过若干 transformer block（无 causal mask）
        for idx, block in enumerate(self.transformer_blocks):
            if (idx % 2 == 1) and self.interleave_self_attention:
                # 奇数层: 只 self-attention，不看 encoder
                hidden_states = block(
                    hidden_states,                  # [B, T, input_dim]
                    attention_mask=None,           
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # 偶数层: self-attention + cross-attention 到 VL
                hidden_states = block(
                    hidden_states,       # [B, T, input_dim]
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )

        if return_moe_info:
            hidden_states, moe_out_gate = self.moe_out(hidden_states, return_gate_info=True)
            moe_info["moe_out"] = moe_out_gate
        else:
            hidden_states = self.moe_out(hidden_states)

        hidden_states = rearrange(hidden_states, 'b (p t) d -> b p t d', p=P, t=T)

        if return_moe_info:
            moe_info = self._format_moe_gate_info(moe_info, B=B, P=P, T=T)
            self._dump_moe_routing(moe_info)

        if return_loss:
            total = 0.0
            denom = 0.0

            for (tag, p_name), idxs in group.items():
                info = self.state_registry[tag][p_name]
                start, end, p_dim = info["start"], info["end"], info["dim"]
                base = tag.split("_v", 1)[0]
                key = f"{base}_{p_name}_dim{p_dim}"
                lat = torch.stack([hidden_states[i, p_idx] for i, p_idx in idxs], dim=0)  # [N, T, D]
                pred = self.decoders[key](lat)                                            # [N, T, p_dim]
                gt = torch.stack([state[i, 1:, start:end] for i, _ in idxs], dim=0)       # [N, T-1, p_dim]
                total += (pred[:, 1:] - gt).pow(2).sum()
                denom += gt.numel()
            loss = total / (denom + 1e-8)

            return rearrange(hidden_states, 'b p t d -> b (p t) d'), loss

        return rearrange(hidden_states, 'b p t d -> b (p t) d')

    def _format_moe_gate_info(
        self,
        moe_info: Dict[str, Dict[str, Any]],
        B: int,
        P: int,
        T: int,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Reshape MoE routing outputs into a part- and time-aligned format.

        For each MoE block (e.g., ``moe_in`` and ``moe_out``), this function
        normalizes the gate outputs so that ``topk_idx`` and ``topk_weight`` are
        represented as tensors of shape [B, P, T, K], where:
            - B: batch size
            - P: number of canonical body parts
            - T: number of temporal slots
            - K: top-k routed experts

        Supported input layouts:
            1. [B * P * T, K]
            2. [B, P * T, K]

        In addition, body-part names and temporal indices are attached for
        downstream analysis and visualization.
        """
        for key in ["moe_in", "moe_out"]:
            if key not in moe_info:
                continue

            topk_idx = moe_info[key]["topk_idx"]
            topk_weight = moe_info[key]["topk_weight"]

            # Case 1:
            # Gate outputs are flattened over batch, part, and time: [B*P*T, K].
            if topk_idx.dim() == 2:
                BN, K = topk_idx.shape
                assert BN == B * P * T, f"{key}: BN={BN}, but expected {B*P*T}"

                topk_idx = rearrange(topk_idx, "(b p t) k -> b p t k", b=B, p=P, t=T)
                topk_weight = rearrange(topk_weight, "(b p t) k -> b p t k", b=B, p=P, t=T)

            # Case 2:
            # Gate outputs are flattened only over part and time: [B, P*T, K].
            elif topk_idx.dim() == 3:
                topk_idx = rearrange(topk_idx, "b (p t) k -> b p t k", p=P, t=T)
                topk_weight = rearrange(topk_weight, "b (p t) k -> b p t k", p=P, t=T)

            else:
                raise ValueError(f"{key}: unexpected topk_idx shape {topk_idx.shape}")

            moe_info[key]["topk_idx"] = topk_idx
            moe_info[key]["topk_weight"] = topk_weight
            moe_info[key]["part_names"] = self.canonical_parts
            moe_info[key]["time_slots"] = list(range(T))

        return moe_info

    def _build_moe_routing_record(
        self,
        moe_info: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Convert formatted MoE routing information into a JSON-serializable record.

        Tensor fields are moved to CPU and converted to Python lists so that the
        result can be directly dumped to a JSONL file.
        """
        def _pack_block(block_name: str) -> Optional[Dict[str, Any]]:
            if block_name not in moe_info:
                return None

            block = moe_info[block_name]
            return {
                "topk_idx": block["topk_idx"].detach().cpu().tolist(),
                "topk_weight": block["topk_weight"].detach().cpu().tolist(),
                "part_names": block["part_names"],
                "time_slots": block["time_slots"],
            }

        return {
            "moe_in": _pack_block("moe_in"),
            "moe_out": _pack_block("moe_out"),
        }

    def _dump_moe_routing(
        self,
        moe_info: Dict[str, Dict[str, Any]],
        save_path: str = "./moe_info/moe_routing.jsonl",
    ) -> None:
        """
        Append one MoE routing record to a JSONL file.

        Each line corresponds to one forward pass (or one sampled batch, depending
        on how the method is invoked), and contains the formatted routing decisions
        for later inspection or visualization.
        """
        final_path = self._get_incremental_path(save_path)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        record = self._build_moe_routing_record(moe_info)

        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _get_incremental_path(self, save_path: str) -> str:
        """
        Return a non-conflicting file path by appending an incremental suffix
        such as `_1`, `_2`, ... if the target file already exists.

        Example:
            ./moe_info/moe_routing.jsonl
            -> ./moe_info/moe_routing_1.jsonl
            -> ./moe_info/moe_routing_2.jsonl
        """
        path = Path(save_path)

        if not path.exists():
            return str(path)

        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        idx = 1
        while True:
            candidate = parent / f"{stem}_{idx}{suffix}"
            if not candidate.exists():
                return str(candidate)
            idx += 1
    

if __name__ == "__main__":
    torch.manual_seed(0)

    # ===== 1. 构造一个假的 config =====
    from omegaconf import OmegaConf
    full_config = OmegaConf.load("hex/config/training/hex_cotrain_eai_ball.yaml")
    state_model_config = full_config.framework.state_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2                                   # batch size
    T = state_model_config.state_horizon    # state 序列长度，要和 config 对齐
    state_dim = state_model_config.state_dim
    cross_dim = state_model_config.cross_attention_dim

    # 随便设一个 VLM token 长度
    vl_len = 64

    print(f"Using device: {device}")
    print(f"state_horizon={T}, state_dim={state_dim}, vl_len={vl_len}, cross_dim={cross_dim}")

    # ===== 2. 实例化模型 =====
    model = StateL2Head(full_config, interleave_self_attention=True)
    model.to(device)
    model.train()

    # ===== 3. 构造随机输入 =====
    # 整段 state 轨迹 [B, T, D]
    state = torch.randn(B, T, state_dim, device=device)

    # VLM 特征 [B, L_vl, cross_dim]
    vl_feats = torch.randn(B, vl_len, cross_dim, device=device)

    # 一个简单的 optimizer，测试反向传播
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ===== 4. 前向 + loss + 反向 =====
    loss, pred_future, state_all = model(
        state=state,
        vl_feats=vl_feats,
        temb=None,
        return_loss=True,
    )

    print("=== Train forward ===")
    print(f"loss: {loss.item():.6f}")
    print(f"state shape      : {state.shape}")        # [B, T, D]
    print(f"state_all shape  : {state_all.shape}")    # [B, T, D]
    print(f"pred_future shape: {pred_future.shape}")  # [B, T-1, D]

    # 反向传播测试
    optimizer.zero_grad()
    loss.backward()
    total_grad = 0.0
    n_params = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.data.norm().item() ** 2
            n_params += 1
    total_grad = total_grad ** 0.5 if n_params > 0 else 0.0
    print(f"grad norm: {total_grad:.6f}")
    optimizer.step()

    # ===== 5. 测试 predict_state =====
    model.eval()
    with torch.no_grad():
        # 这里直接把整段 state 丢进去，内部会取 state[:,0]
        pred_future_eval = model.predict_state(
            state_curr=state[:, 0],           # [B, T, D] or 也可以只给 [B, D]
            vl_feats=vl_feats,
            temb=None,
        )  # [B, T-1, D]

    print("\n=== Inference predict_state ===")
    print(f"pred_future_eval shape: {pred_future_eval.shape}")  # [B, T-1, D]
    print(
        f"pred_future_eval mean/std: "
        f"{pred_future_eval.mean().item():.4f}, {pred_future_eval.std().item():.4f}"
    )

    # 简单检查数值是否正常
    if torch.isfinite(pred_future_eval).all():
        print("check: pred_future_eval is finite ✅")
    else:
        print("check: pred_future_eval contains NaN/Inf ❌")

