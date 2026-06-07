import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)


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
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
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

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
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

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
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

        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,    
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            # encoder_attention_mask=encoder_attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class StateFlowMatchingHead(nn.Module):
    """
    Flow Matching 版的 state 模型：
      - 输入: 当前插值后的状态 x_tau [B, state_dim]，时间 tau [B]，VL 特征 [B, L_vl, cross_dim]
      - 输出: 对应的速度/流 v(x_tau, tau, cond) [B, state_dim]
    """
    def __init__(self, full_config, interleave_self_attention=False):
        super().__init__()
        config = full_config.framework.state_model
        self.state_dim = config.state_dim
        self.input_dim = config.input_dim
        self.cross_attention_dim = config.cross_attention_dim
        self.num_layers = config.num_layers
        self.interleave_self_attention = interleave_self_attention

        # 1) 状态映射到 transformer dim
        self.state_encoder = MLP(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.input_dim,
        )

        # 2) 时间 embedding（连续时间 τ）
        self.pos_embed = nn.Embedding(config.state_horizon, config.input_dim)
        self.time_mlp = MLP(
            input_dim=1,
            hidden_dim=config.hidden_dim,
            output_dim=config.input_dim,
        )

        # 3) transformer blocks（这里默认每层都可以做 self + cross，
        # 3) 堆叠多层 BasicTransformerBlock
        blocks = []
        for idx in range(config.num_layers):
            use_self_attn = (idx % 2 == 1) and interleave_self_attention
            curr_cross_attention_dim = config.cross_attention_dim if not use_self_attn else None

            block = BasicTransformerBlock(
                dim=config.input_dim,
                num_attention_heads=config.num_heads,
                attention_head_dim=config.head_dim,
                dropout=config.dropout,
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

        # 4) 输出 head: hidden -> flow (velocity)
        self.flow_head = nn.Linear(config.input_dim, config.state_dim)

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

        # 1) 采噪声 + 连续时间
        noise = torch.randn_like(state)                           # [B, T, D]
        tau = torch.rand(B, 1, 1, device=device, dtype=state.dtype)  # [B, 1, 1]
        x_tau = (1.0 - tau) * noise + tau * state                 # [B, T, D]
        v_target = state - noise                                  # [B, T, D]

        # 2) encode state token + time embedding + pos embedding
        h = self.state_encoder(x_tau)     # 实际 shape 是 [B, T, input_dim]

        time_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embed(time_ids)     # [B, T, input_dim]

        tau_flat = tau.view(B, 1)                                            # [B, 1]
        t_emb = self.time_mlp(tau_flat)                                      # [B, input_dim]
        t_emb = t_emb.unsqueeze(1).expand(B, T, self.input_dim)  

        hidden_states = h + pos_emb + t_emb             # [B, T, input_dim]
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

        v_pred = self.flow_head(hidden_states)                              # [B, T, D]

        if return_loss:
            loss = F.mse_loss(v_pred, v_target)
            return v_pred, hidden_states, loss

        return v_pred, hidden_states
    
    @torch.no_grad()
    def predict_state(
        self,
        vl_feats: torch.Tensor,          # [B, L_vl, cross_attention_dim]
        T: int,                          # 生成的 state 序列长度 (state_horizon)
        num_steps: int = 32,             # ODE Euler 积分步数
        temb: Optional[torch.Tensor] = None,
        init_noise: Optional[torch.Tensor] = None,  # [B, T, state_dim]，不给就自己采
    ) -> torch.Tensor:
        """
        使用学到的 flow v(x_tau, tau, cond) 从噪声积分到数据，生成 state 轨迹：
          - 输入: VLM 特征 vl_feats, 目标 horizon T, (可选) 初始噪声 init_noise
          - 输出: 预测的 state 轨迹 [B, T, state_dim]
        """
        device = vl_feats.device
        dtype = vl_feats.dtype
        B = vl_feats.shape[0]

        # 1) 初始化 x_0：噪声状态
        if init_noise is None:
            x = torch.randn(B, T, self.state_dim, device=device, dtype=dtype)  # [B, T, D]
        else:
            x = init_noise.to(device=device, dtype=dtype)
            assert x.shape == (B, T, self.state_dim), \
                f"init_noise.shape={x.shape}, expected {(B, T, self.state_dim)}"

        dt = 1.0 / num_steps
        encoder_hidden_states = vl_feats  # [B, L_vl, cross_dim]

        # 2) 从 tau=0 → 1 做 Euler ODE integration
        for k in range(num_steps):
            # 当前 step 的 tau（用中点 τ_{k+1/2} 会更平滑一点）
            tau_val = (k + 0.5) * dt
            tau = torch.full((B, 1, 1), tau_val, device=device, dtype=dtype)  # [B,1,1]

            # --- 下面 basically 是你 forward 里的 "2) encode + transformer + flow_head"，
            #     只是把 x_tau 换成当前 x，而不再采新的 noise/state ---

            # (a) encode state token
            h = self.state_encoder(x)  # [B, T, input_dim]

            # (b) 序列位置编码
            time_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
            pos_emb = self.pos_embed(time_ids)                                   # [B, T, input_dim]

            # (c) 连续时间 τ embedding，并广播到每个 time step
            tau_flat = tau.view(B, 1)                                            # [B,1]
            t_emb = self.time_mlp(tau_flat)                                      # [B, input_dim]
            t_emb = t_emb.unsqueeze(1).expand(B, T, self.input_dim)              # [B, T, input_dim]

            # (d) 合成 hidden_states
            hidden_states = h + pos_emb + t_emb                                  # [B, T, input_dim]

            # (e) 过 transformer blocks（无 causal mask）
            for idx, block in enumerate(self.transformer_blocks):
                if (idx % 2 == 1) and self.interleave_self_attention:
                    # 奇数层: 只 self-attention
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
                        hidden_states,                  # [B, T, input_dim]
                        attention_mask=None,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=None,
                        temb=temb,
                    )

            # (f) 预测 flow v(x_tau, tau, cond)
            v = self.flow_head(hidden_states)   # [B, T, state_dim]

            # (g) Euler 积分更新 x: x_{τ+dt} = x_τ + v * dt
            x = x + v * dt

        # 积分结束的 x ≈ 数据分布下的 state 轨迹
        return x  # [B, T, state_dim]
    

if __name__ == "__main__":
    torch.manual_seed(0)

    # ===== 1. 构造一个假的 config =====
    from omegaconf import OmegaConf
    full_config = OmegaConf.load("reflex_vla/config/training/reflex_vla_cotrain_eai_ball.yaml")
    state_model_config = full_config.framework.state_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 2                 # batch size
    T = 10                # state 序列长度 (state_horizon)

    # ===== 2. 实例化模型 =====
    model = StateFlowMatchingHead(full_config, interleave_self_attention=True)
    model.to(device)
    model.train()

    # ===== 3. 构造随机输入 =====
    # 真实轨迹上的 state（训练时的 "data"）
    state = torch.randn(B, T, state_model_config.state_dim, device=device)

    # VLM 特征
    vl_feats = torch.randn(B, 117, 2048, device=device)

    # 一个简单的 optimizer，测试反向传播
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ===== 4. 前向 + loss + 反向 =====
    loss, v_pred, v_target = model(
        state=state,
        vl_feats=vl_feats,
        temb=None,
        return_loss=True,
    )

    print(f"[train] loss: {loss.item():.6f}")
    print(f"[train] v_pred shape   : {v_pred.shape}")    # [B, T, state_dim]
    print(f"[train] v_target shape : {v_target.shape}")  # [B, T, state_dim]

    # 反向传播测试
    optimizer.zero_grad()
    loss.backward()
    grad_norm = 0.0
    n_params = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm().item() ** 2
            n_params += 1
    grad_norm = grad_norm ** 0.5 if n_params > 0 else 0.0
    print(f"[train] grad_norm: {grad_norm:.6f}")
    optimizer.step()

    # ===== 5. 测试 predict_state 采样 =====
    model.eval()
    with torch.no_grad():
        pred_state = model.predict_state(
            vl_feats=vl_feats,
            T=T,
            num_steps=32,
            temb=None,
            init_noise=None,
        )  # [B, T, state_dim]

    print(f"[sample] pred_state shape: {pred_state.shape}")  # 期望 [B, T, state_dim]
    print(f"[sample] pred_state mean/std: {pred_state.mean().item():.4f}, {pred_state.std().item():.4f}")

    # 简单检查数值是否正常
    if torch.isfinite(pred_state).all():
        print("[check] pred_state is finite ✅")
    else:
        print("[check] pred_state contains NaN/Inf ❌")
