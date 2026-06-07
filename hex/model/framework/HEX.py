# Copyright 2026 HEX community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shuanghao / Xi'an Jiaotong University] in [2026]. 
"""
HEX Framework
A lightweight implementation that integrates Qwen-VL, UPP, and a flow-matching action head to directly predict continuous actions.
The overall architecture is adapted from StarVLA. The flow-matching action head is adapted from GR00T N1.5.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from collections import deque
from typing import List, Optional, Tuple

from hex.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

from hex.model.modules.vlm import get_vlm_model
from hex.model.modules.state_model.HEX_L2_StateDecoder import CrossEmnodiedStateMoEL2Head
from hex.model.modules.action_model.HEX_ActionHeader import get_action_model, FlowmatchingActionHead
from hex.model.tools import FRAMEWORK_REGISTRY
from hex.model.framework.base_framework import baseframework
from hex.utils.cross_embodiments import get_embodiment_registry
from hex.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("HEX")
class HEX(baseframework):
    """
    HEX: a humanoid vision-language-action framework.

    The model combines:
      - a Qwen-VL backbone for visual-language token encoding,
      - a cross-embodiment state MoE module for proprioceptive dynamics modeling,
      - a flow-matching action head for future continuous action prediction.

    The model predicts future action chunks conditioned on visual observations,
    language instructions, embodiment tags, and proprioceptive states.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the HEX model and cache key configuration values.

        Args:
            config: Hierarchical configuration that defines the VLM backbone,
                state model, action model, dataset settings, and trainer options.
            **kwargs: Reserved for future extensions.
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)  # Qwen-VL backbone
        
        # Match the action head cross-attention dimension to the VLM hidden size
        dim = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = dim

        # Load embodiment-specific state and action registries
        self.state_registry, self.action_registry = get_embodiment_registry(config, is_train=True)

        # State MoE module for cross-embodiment proprioceptive modeling
        self.state_model = CrossEmnodiedStateMoEL2Head(
            self.config,
            self.state_registry,
            interleave_self_attention=True,
        ).to(torch.bfloat16)

        self.max_state_dim = 128

        # Flow-matching action head for continuous action chunk prediction
        self.action_model: FlowmatchingActionHead = get_action_model(
            self.config,
            self.action_registry,
        )

        self.max_action_dim = 128
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        if config.framework.qwenvl.add_query:
            self.query_proj = nn.Linear(dim, dim)
            self.query_queue = deque(maxlen=config.datasets.vla_data.vision_history_length+1)
    
    def forward(
        self,
        examples: List[dict] = None,
        cotrain: bool = True,
        **kwargs,
    ) -> Tuple:
        """
        Forward pass for joint state prediction and action prediction.

        The input batch contains images, language instructions, actions,
        proprioceptive states, and embodiment tags. The model first
        encodes image-language observations with Qwen-VL, then predicts a
        state-conditioned action chunk through the flow-matching action head.

        Args:
            examples: A list of training samples. Each sample contains:
                - image: visual observations, possibly with temporal history
                - lang: language instruction
                - action: target action trajectory
                - state: proprioceptive state sequence
                - tag: embodiment tag
            cotrain: If False, only return the state prediction loss.
                If True, compute both state and action losses.
            **kwargs: Reserved for future extensions.

        Returns:
            A dictionary containing action-model outputs and state loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, action_dim]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        tags = [example["tag"] for example in examples] if "tag" in examples[0] else None  # [B, x]
        B, chunk_size, a_chunk_size = len(state), state[0].shape[0], actions[0].shape[0]
        
        # Step 1: Encode each temporal observation step with Qwen-VL
        all_query_tokens = []
        history_len = self.config.datasets.vla_data.vision_history_length + 1

        num_total_imgs = len(batch_images[0])
        assert num_total_imgs % history_len == 0, \
            f"num_total_imgs={num_total_imgs} is not divisible by history_len={history_len}"
        num_cams = num_total_imgs // history_len
        for t in range(history_len):    
            images_t = [
                [
                    sample[cam_id * history_len + t]
                    for cam_id in range(num_cams)
                ]
                for sample in batch_images
            ]

            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=images_t,
                instructions=instructions,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]
            
            if self.config.framework.qwenvl.add_query:
                query_token = last_hidden[:, -8:, :]            # [B, 8, H]
                all_query_tokens.append(query_token)

        if self.config.framework.qwenvl.add_query:
            vision_query_tokens = torch.cat(all_query_tokens, dim=1)
            vision_query_tokens = self.query_proj(vision_query_tokens)
            last_hidden = torch.cat([last_hidden[:, :-8, :], vision_query_tokens], dim=1)

        # step 2: state prediction with qwen last_hidden
        if state is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                padded_state = torch.zeros((B, chunk_size, self.max_state_dim), device=last_hidden.device, dtype=last_hidden.dtype)
                
                for i, s in enumerate(state):
                    s_tensor = torch.from_numpy(np.array(s)).to(last_hidden.device)
                    curr_dim = s_tensor.shape[-1]
                    padded_state[i, :, :curr_dim] = s_tensor
                    
                state = padded_state
                state_hidden, loss_state = self.state_model(state, last_hidden, tags, return_loss=True)

        with torch.autocast("cuda", dtype=torch.float32):
            if not cotrain:
                return {"state_loss": loss_state}
            else:
                # Step 3: Action Expert Forward and Loss
                padded_actions = torch.zeros((B, a_chunk_size, self.max_action_dim), device=last_hidden.device, dtype=last_hidden.dtype)
                for i, a in enumerate(actions):
                    a_tensor = torch.from_numpy(np.array(a)).to(last_hidden.device)
                    curr_dim = a_tensor.shape[-1]
                    padded_actions[i, :, :curr_dim] = a_tensor
                actions_target = padded_actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

                repeated_diffusion_steps = (self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4)
                actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                tags_repeated = []
                for _ in range(repeated_diffusion_steps):
                    tags_repeated.extend(tags)
                
                state_repeated = None
                if state is not None:
                    state_repeated = state_hidden.repeat(repeated_diffusion_steps, 1, 1).to(last_hidden.dtype)

                output_dict = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated, tags_repeated)  # (B, chunk_len, action_dim)
                output_dict['state_loss'] = loss_state

                return output_dict

    @torch.inference_mode()
    def predict_action_batch(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: np.ndarray = None,
        tags: list[str] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Batched inference for future action prediction.

        This function encodes a batch of image-history observations with Qwen-VL,
        predicts state-aware hidden representations, and directly outputs
        normalized future action chunks through the action head.

        Args:
            batch_images: A batch of visual observations. Each sample is a list
                of PIL images arranged by temporal history and camera view.
            instructions: Natural language task instructions.
            state: Optional proprioceptive state sequence with shape [B, T, state_dim].
            tags: Optional embodiment tags for selecting embodiment-specific heads.
            **kwargs: Reserved for future extensions.

        Returns:
            A dictionary containing:
                normalized_actions: Predicted normalized actions with shape
                    [B, T, action_dim].
        """
        B, chunk_size = len(state), state[0].shape[0]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: Encode each temporal observation step with Qwen-VL
        all_query_tokens = []
        for i in range(self.config.datasets.vla_data.vision_history_length+1):
            images_i = [[sample[i]] for sample in batch_images]
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=images_i,
                instructions=instructions,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

            if self.config.framework.qwenvl.add_query:
                query_token = last_hidden[:, -8:, :]            # [B, 8, H]
                all_query_tokens.append(query_token)

        if self.config.framework.qwenvl.add_query:
            vision_query_tokens = torch.cat(all_query_tokens, dim=1)
            vision_query_tokens = self.query_proj(vision_query_tokens)
            last_hidden = torch.cat([last_hidden[:, :-8, :], vision_query_tokens], dim=1)
        
        # step 2: state prediction with qwen last_hidden
        if state is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                padded_state = torch.zeros((B, chunk_size, self.max_state_dim), device=last_hidden.device, dtype=last_hidden.dtype)
                
                for i, s in enumerate(state):
                    s_tensor = torch.from_numpy(np.array(s)).to(last_hidden.device)
                    curr_dim = s_tensor.shape[-1]
                    padded_state[i, :, :curr_dim] = s_tensor
                    
                state = padded_state
                state_hidden = self.state_model(state, last_hidden, tags, return_loss=False)

        # Step 3: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state_hidden, tags)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
    
    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: np.ndarray = None,
        tags: list[str] = None,
        return_moe_info: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Online inference for future action prediction.

        This function is designed for step-by-step deployment. It maintains a
        queue of recent Qwen-VL query tokens when visual history is enabled,
        predicts state-aware hidden representations, and outputs normalized
        future action chunks.

        Args:
            batch_images: A batch of current visual observations.
            instructions: Natural language task instructions.
            state: Optional proprioceptive state input.
            tags: Optional embodiment tags.
            return_moe_info: Whether to return MoE routing information from the
                state model for analysis or visualization.
            **kwargs: Reserved for future extensions.

        Returns:
            A dictionary containing:
                normalized_actions: Predicted normalized actions with shape
                    [B, T, action_dim].
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: Encode each temporal observation step with Qwen-VL
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        if self.config.framework.qwenvl.add_query:
            query_token = last_hidden[:, -8:, :]            # [B, 8, H]
            self.query_queue.append(query_token)
            current_queue = list(self.query_queue)
            if len(current_queue) < self.config.datasets.vla_data.vision_history_length + 1:
                gap = self.config.datasets.vla_data.vision_history_length + 1 - len(current_queue)
                fillers = [current_queue[-1]] * gap
                vision_query_tokens = torch.cat(current_queue + fillers, dim=1)
            else:
                vision_query_tokens = torch.cat(current_queue, dim=1)
            vision_query_tokens = self.query_proj(vision_query_tokens)
            last_hidden = torch.cat([last_hidden[:, :-8, :], vision_query_tokens], dim=1)
        
        # step 2: state prediction with qwen last_hidden
        if state is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
                if return_moe_info:
                    state_hidden = self.state_model(state, last_hidden, tags, return_loss=False, return_moe_info=True)
                else:
                    state_hidden = self.state_model(state, last_hidden, tags, return_loss=False)
                
        # Step 3: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state_hidden, tags)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def reset(self):
        if self.config.framework.qwenvl.add_query:
            self.query_queue.clear()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./hex/config/training/hex_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: HEX = HEX(cfg)
    print(model)

    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from hex.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
