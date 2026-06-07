# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)


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
        dropout: float = 0.0,
        cross_attention_dim_vl: Optional[int] = None,
        cross_attention_dim_state: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm' or 'ada_norm'
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

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm_attn = AdaLayerNorm(dim)
        else:
            self.norm_attn = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
            
        self.norm_self = nn.LayerNorm(dim)

        self.self_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
        )
        self.attn_vl = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim_vl,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )
        self.attn_state = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim_state,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.gate_proj = nn.Linear(3 * dim, 1)
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

        self.norm_ff = nn.LayerNorm(
            dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states_vl: Optional[torch.Tensor] = None,
        encoder_hidden_states_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        inference: bool = False,
    ) -> torch.Tensor:
        """
        Dual Cross-Attention + Gating Fusion:
          1) norm + (optional) pos-embed
          2) cross-attn to VLM & State
          3) gate = sigmoid(W [q_norm, h_vl, h_state])
          4) h = gate * h_state + (1 - gate) * h_vl + residual
          5) FFN + residual
        """
        # 1. Norm + Positional Embedding
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm_attn(hidden_states, temb)
        else:
            norm_hidden_states = self.norm_attn(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 2. 双路 Cross-Attention
        h_vl = self.attn_vl(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states_vl,
            attention_mask=attention_mask,
            # encoder_attention_mask=encoder_attention_mask, 
        )
        h_state = self.attn_state(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states_state,
            attention_mask=attention_mask,
            # encoder_attention_mask=encoder_attention_mask,
        )
        if self.final_dropout is not None:
            h_vl = self.final_dropout(h_vl)
            h_state = self.final_dropout(h_state)

        # 3. Gating Fusion
        fused_input = torch.cat([h_vl, h_state, norm_hidden_states], dim=-1)
        raw_gate = torch.sigmoid(self.gate_proj(fused_input))
        if raw_gate.dim() == 3:
            B, T, _ = raw_gate.shape
            time_weight = torch.linspace(
                0.0, 1.0, T, device=raw_gate.device
            ).view(1, T, 1)
            gate = raw_gate * time_weight
        else:
            gate = raw_gate
        fused = h_vl + gate * h_state
        hidden_states = hidden_states + fused

        # ----- Self-Attention -----
        h_self = self.self_attn(self.norm_self(hidden_states))
        hidden_states = hidden_states + h_self

        # 4. FFN + Residual
        norm_ff = self.norm_ff(hidden_states)
        ff_out = self.ff(norm_ff)
        hidden_states = hidden_states + ff_out
        return hidden_states, gate


class HLM_DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype: torch.dtype = torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention: bool = False,  
        cross_attention_dim_vl: Optional[int] = None,
        cross_attention_dim_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim   # 12*64
        self.gradient_checkpointing = False

        # Timestep encoder
        # self.config.compute_dtype 可能不存在，要提前处理
        compute_dtype = getattr(self.config, 'compute_dtype', torch.float32)
        self.timestep_encoder = TimestepEncoder( # TODO BUG, train 的时候 self.config.compute_dtype 不会报错， 但是 eval 的时候会
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )

        all_blocks, all_blocks_state = [], []
        for idx in range(self.config.num_layers // 2):
            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=None,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim_vl=cross_attention_dim_vl,
                    cross_attention_dim_state=cross_attention_dim_state,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)
        self.transformer_blocks_state = nn.ModuleList(all_blocks_state)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states_vl: torch.Tensor,  # Shape: (B, S, D)
        encoder_hidden_states_state: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        inference=False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states_vl = encoder_hidden_states_vl.contiguous()
        encoder_hidden_states_state = encoder_hidden_states_state.contiguous()

        all_hidden_states = [hidden_states]
        gates = []
        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states, gate = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states_vl=encoder_hidden_states_vl,
                    encoder_hidden_states_state=encoder_hidden_states_state,
                    attention_mask=None,
                    encoder_attention_mask=None,
                    temb=temb,
                    inference=inference,
                )
                
            all_hidden_states.append(hidden_states)
            gates.append(gate)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states, gates
        else:
            return self.proj_out_2(hidden_states), gates
        