# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with HEX, e.g., "rm "].
# Action repeat is inspired by CogACT

from dataclasses import dataclass, field
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from hex.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from hex.model.modules.action_model.flow_matching_head.cross_attention_dit_hex import HLM_DiT
from hex.utils.mee import simple_mee_loss_hex


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
        action_registry,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size 
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = HLM_DiT(**diffusion_model_cfg)
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps

        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        self.action_dim_by_tag = dict(action_registry)
        for tag, total_dim in self.action_dim_by_tag.items():
            key = f"{tag}_dim{total_dim}"
            self.action_encoders[key] = ActionEncoder(
                action_dim=total_dim,
                hidden_size=self.input_embedding_dim,
            )
            self.action_decoders[key] = MLP(
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=total_dim,
            )
        self.max_action_dim = 128

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

        self.enable_mee = full_config.enable_mee if hasattr(full_config, "enable_mee") else False

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, state_embs: torch.Tensor = None, tags: list[str] = None):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = vl_embs.device
        B, H, _ = actions.shape

        # Embed noised action trajectory.
        t = self.sample_time(B, device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        tag2idx = defaultdict(list)
        for i, tag in enumerate(tags):
            tag2idx[tag].append(i)
        action_features = actions.new_zeros((B, H, self.input_embedding_dim))
        vel_list = [None] * B
        # t_dis = (t * self.num_timestep_buckets).long().clamp(0, self.num_timestep_buckets-1)

        for tag, idxs in tag2idx.items():
            act_dim = self.action_dim_by_tag[tag]
            key = f"{tag}_dim{act_dim}"
            act = actions[idxs, :, :act_dim]          # [b_k, H, act_dim]
            noise = torch.randn_like(act)
            noisy = (1 - t[idxs]) * noise + t[idxs] * act        # [b_k, H, act_dim]
            vel = act - noise                          # [b_k, H, act_dim]
            feat = self.action_encoders[key](noisy, t_discretized[idxs])  # [b_k, H, embed]
            action_features[idxs] = feat.to(action_features.dtype)
            for j, i in enumerate(idxs):
                vel_list[i] = vel[j]

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join VLM features with state and action embedding along sequence dimension.
        model_output, gates = self.model(
            hidden_states=action_features,
            encoder_hidden_states_vl=vl_embs,
            encoder_hidden_states_state=state_embs,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
    
        # Slice out only the action portion of pred and target.
        total, denom = 0.0, 0.0
        all_pred, all_vel = [], []
        for tag, idxs in tag2idx.items():
            act_dim = self.action_dim_by_tag[tag]
            key = f"{tag}_dim{act_dim}"
            pred = self.action_decoders[key](model_output[idxs])   # [b_k, H, act_dim]
            vel = torch.stack([vel_list[i] for i in idxs])        # [b_k, H, act_dim]
            total += (pred - vel).pow(2).sum()
            denom += vel.numel()
            all_pred.append(pred)
            all_vel.append(vel)
        loss = total / (denom + 1e-8)

        if self.enable_mee:
            mee_loss = simple_mee_loss_hex(all_pred, all_vel, sigma=0.5)
            output_dict = {
                "action_loss": loss,
                "mee_loss": mee_loss,
            }
        else:
            output_dict = {
                "action_loss": loss,
            }
        
        return output_dict

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state_embs: torch.Tensor = None,
        tags: list[str] = None,
    ) -> torch.Tensor:
        """
        Inference-time action generation with Euler integration.

        The method starts from Gaussian noise in the padded action space and
        iteratively updates the valid action dimensions for each embodiment
        using the predicted velocity field.

        Args:
            vl_embs: Visual-language embeddings with shape [B, L, C].
            state_embs: Optional state embeddings for state-conditioned action generation.
            tags: Embodiment tag for each sample. Required for selecting the
                correct action dimensions, encoders, and decoders.

        Returns:
            Predicted padded action tensor with shape [B, H, D_max].
            For each sample, only the first action_dim_by_tag[tag] dimensions
            are valid.
        """
        device = vl_embs.device
        B = vl_embs.shape[0]

        # Use the configured action horizon.
        H = self.action_horizon

        if tags is None:
            # Tags are required because different embodiments may use different
            # action dimensions and different encoder/decoder heads.
            raise ValueError("HEX predict_action requires `tags` to select per-tag action dims/enc/dec.")

        # Initialize the full padded action tensor from Gaussian noise.
        # Each embodiment will only update its own valid action dimensions.
        actions = torch.randn(
            (B, H, self.max_action_dim),
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / float(num_steps)

        # Precompute batch indices for each embodiment tag.
        tag2idx = defaultdict(list)
        for i, tag in enumerate(tags):
            tag2idx[tag].append(i)

        for step in range(num_steps):
            # Convert the current integration step to a discrete timestep bucket.
            t_cont = step / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full(
                (B,),
                t_discretized,
                device=device,
                dtype=torch.long,
            )

            # Encode the current action estimate with embodiment-specific encoders.
            action_features = actions.new_zeros((B, H, self.input_embedding_dim))

            for tag, idxs in tag2idx.items():
                act_dim = self.action_dim_by_tag[tag]
                key = f"{tag}_dim{act_dim}"

                act = actions[idxs, :, :act_dim]  # [b_k, H, act_dim]
                feat = self.action_encoders[key](
                    act,
                    timesteps[idxs],
                )  # [b_k, H, input_embedding_dim]

                action_features[idxs] = feat.to(action_features.dtype)

            # Add temporal position embeddings if enabled.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(H, dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Predict the velocity field for the current action estimate.
            model_output, _gates = self.model(
                hidden_states=action_features,
                encoder_hidden_states_vl=vl_embs,
                encoder_hidden_states_state=state_embs,
                timestep=timesteps,
                return_all_hidden_states=False,
                inference=True,
            )

            # Decode and integrate velocity for each embodiment group.
            for tag, idxs in tag2idx.items():
                act_dim = self.action_dim_by_tag[tag]
                key = f"{tag}_dim{act_dim}"

                pred_vel = self.action_decoders[key](model_output[idxs])  # [b_k, H, act_dim]

                # Euler update: x_{t+dt} = x_t + dt * v_theta(x_t, t).
                actions[idxs, :, :act_dim] = actions[idxs, :, :act_dim] + dt * pred_vel

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None, action_registry=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(config, action_registry)


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently
    pass
