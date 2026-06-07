import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


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


class StateCausalTransformerDecoder(nn.Module):
    """
    自回归的 state Transformer:
      - 输入: states 序列 [B, T, dim]，VL 特征 [B, L_vl, cross_dim]
      - 输出: 对每个时间步的下一步 state 预测 [B, T, dim]
      - 内部用 BasicTransformerBlock + causal mask + cross-attention 到 VL
    """
    def __init__(
        self,
        full_config,
        interleave_self_attention=False,
    ):
        super().__init__()
        config = full_config.framework.state_model
        self.state_dim = config.state_dim
        self.dim = config.input_dim
        self.state_horizon = config.state_horizon
        self.interleave_self_attention = interleave_self_attention

        # 1) 状态本映射到 dim
        self.state_encoder = MLP(
            input_dim=config.state_dim, 
            hidden_dim=config.hidden_dim,
            output_dim=config.input_dim,
        ) if config.state_dim else None

        # 2) 简单位置编码 (时间步)
        self.pos_embed = nn.Embedding(config.state_horizon, config.input_dim)

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

        # 5) 输出 head，把 hidden dim 映射回 state dim
        self.output_proj = nn.Linear(config.input_dim, config.state_dim)

    def _build_causal_mask(self, batch_size: int, seq_len: int, device: torch.device):
        """
        生成 [B, 1, T, T] 的 causal mask：
        - mask[i, 0, t1, t2] = 0 (可看过去)
        - mask[i, 0, t1, t2] = -inf (不能看未来)
        """
        # [T, T]
        base_mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        base_mask = torch.triu(base_mask, diagonal=1)  # 上三角 -inf, 下三角 0

        # [1, 1, T, T]
        base_mask = base_mask.unsqueeze(0).unsqueeze(0)

        # 扩展到 batch 维度 [B, 1, T, T]
        mask = base_mask.expand(batch_size, -1, -1, -1).contiguous()
        return mask

    def forward(
        self,
        states: torch.Tensor,          # [B, T, dim]  s_0..s_T
        vl_feats: torch.Tensor,        # [B, L_vl, cross_attention_dim]
        temb: Optional[torch.Tensor] = None,
        return_loss=False,
    ) -> torch.Tensor:
        """
        自回归训练用：
          输入: 整段 states（长度 T），VL 条件
          输出: 对每个时间步的“下一步 state”预测 [B, T, dim]
        在训练时一般用 pred[:, :-1] 对齐 target[:, 1:].
        """
        B, T, D = states.shape
        device = states.device
        assert D == self.state_dim, f"states 最后一维 {D} != dim {self.dim}"
        assert T <= self.state_horizon, "序列太长，调大 max_seq_len"

        # 1) 加位置编码
        # 时间步: [T] -> [1, T] -> [B, T]
        states = self.state_encoder(states)
        time_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embed(time_ids)          # [B, T, dim]
        hidden_states = states + pos_emb            # [B, T, dim]
        encoder_hidden_states = vl_feats           

        # 2) 构造 causal mask，用在 self-attention 上
        causal_mask = self._build_causal_mask(B, T, device=device)  # [B, 1, T, T]

        # 3) 过多层 Transformer Block
        for idx, block in enumerate(self.transformer_blocks):
            if (idx % 2 == 1) and self.interleave_self_attention:
                # 奇数层: 只 self-attention，不看 encoder
                hidden_states = block(
                    hidden_states,                  # [B, T, dim]
                    attention_mask=causal_mask,     # 只看过去/当前
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                # 偶数层: self-attention + cross-attention 到 VL
                hidden_states = block(
                    hidden_states,                  # [B, T, dim]
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )
                
        # 4) 输出每个时间步的“下一步状态预测” [B, T, dim]
        pred_next = self.output_proj(hidden_states) 
        # pred_next = states + self.output_proj(hidden_states)

        if return_loss:
            loss_state = F.mse_loss(pred_next[:, :-1], states[:, 1:])
            return pred_next, hidden_states, loss_state
        
        return pred_next, hidden_states

    @torch.no_grad()
    def generate(
        self,
        prefix_states: torch.Tensor,   # [B, T0, dim] 已知序列 s_0..s_t
        vl_feats: torch.Tensor,        # [B, L_vl, cross_attention_dim]
        num_steps: int = 8,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        自回归生成:
          输入: prefix_states = [B, T0, dim]
          输出: future = [B, num_steps, dim]，对应 s_{t+1..t+num_steps}
        """
        device = prefix_states.device
        B, T0, D = prefix_states.shape
        assert D == self.state_dim

        cur_states = prefix_states  # [B, cur_T, dim]
        generated = []

        for k in range(num_steps):
            cur_T = cur_states.shape[1]
            assert cur_T <= self.state_horizon, "当前序列超过 state_horizon"

            # 用当前所有状态做一次自回归预测
            pred_all, state_hidden = self.forward(cur_states, vl_feats, temb=temb)  # [B, cur_T, dim]

            # 取最后一个时间步的预测，作为 s_{当前最后时刻+1}
            s_next = pred_all[:, -1, :]  # [B, dim]
            generated.append(s_next)

            # 把 s_next 拼回序列，作为下一轮的上下文
            cur_states = torch.cat([cur_states, s_next.unsqueeze(1)], dim=1)  # [B, cur_T+1, dim]

        # [B, num_steps, dim]
        future = torch.stack(generated, dim=1)
        return future


if __name__ == "__main__":
    torch.manual_seed(0)

    # ===== 1. 构造一个假的 config =====
    from omegaconf import OmegaConf
    full_config = OmegaConf.load("reflex_vla/config/training/reflex_vla_cotrain_eai_ball.yaml")
    state_model_config = full_config.framework.state_model

    # ===== 2. 构造模型 =====
    model = StateCausalTransformerDecoder(
        full_config=full_config,
        interleave_self_attention=True,  # 随便测一个配置
    )

    model.eval()  # 我们只做前向测试

    # ===== 3. 构造假输入 =====
    B = 2
    T = 5
    L_vl = 10

    states = torch.randn(B, T, state_model_config.state_dim)           # [B, T, state_dim]
    vl_feats = torch.randn(B, L_vl, state_model_config.cross_attention_dim)  # [B, L_vl, cross_dim]

    # ===== 4. 测试 forward 形状是否正确 =====
    with torch.no_grad():
        pred_next, hidden_state = model(states, vl_feats)  # [B, T, state_dim]

    print("forward 输出形状:", pred_next.shape)
    assert pred_next.shape == (B, T, state_model_config.state_dim), "forward 输出形状不对！"

    # ===== 5. 测试 generate 基本功能和形状 =====
    T0 = 3
    prefix_states = states[:, :T0, :]  # 已知前 T0 个状态
    num_steps = 4

    with torch.no_grad():
        future = model.generate(
            prefix_states=prefix_states,
            vl_feats=vl_feats,
            num_steps=num_steps,
        )

    print("generate 输出形状:", future.shape)
    assert future.shape == (B, num_steps, state_model_config.state_dim), "generate 输出形状不对！"

    # ===== 6. 一步一致性测试：num_steps=1 时应该等价于 forward 最后一帧 =====
    with torch.no_grad():
        # 直接 self.forward，取最后一个时间步的预测
        pred_all, _ = model(prefix_states, vl_feats)      # [B, T0, state_dim]
        last_step_pred = pred_all[:, -1, :]            # [B, state_dim]

        # 用 generate 只生成 1 步
        future_1 = model.generate(
            prefix_states=prefix_states,
            vl_feats=vl_feats,
            num_steps=1,
        )  # [B, 1, state_dim]
        future_1_step = future_1[:, 0, :]              # [B, state_dim]

    print("last_step_pred  (from forward):", last_step_pred[0, :5])
    print("future_1_step (from generate):", future_1_step[0, :5])

    # 数值上应该完全相等（同一次前向）
    max_diff = (last_step_pred - future_1_step).abs().max().item()
    print("forward 最后一帧 vs generate 一步 的最大差值:", max_diff)
    assert max_diff < 1e-6, "generate 的一步结果和 forward 最后一帧预测不一致！"

    print("✅ generate 基本逻辑和 self.forward 一致，测试通过。")
