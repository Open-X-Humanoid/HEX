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

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
# from gr00t.model.transforms import GR00TTransform

from hex.dataloader.gr00t_lerobot.datasets import ModalityConfig
from hex.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform, ModalityTransform
from hex.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from hex.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from hex.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


##################################### Oxe: Droid ######################################################

class OxeDroidDataConfig:
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_position": "binary",
                },
                target_rotations={"action.eef_rotation_delta": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


##################################### Oxe: Bridge ######################################################

class OxeBridgeDataConfig:
    video_keys = [
        "video.image_0",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.roll": "q99",
                    "state.pitch": "q99",
                    "state.yaw": "q99",
                    "state.pad": "q99",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "binary",
                },
            ),
            # concat transforms
            # ConcatTransform(
            #     # video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return ComposedModalityTransform(transforms=transforms)


##################################### Oxe: RT1 ######################################################

class OxeRT1DataConfig:
    video_keys = [
        "video.image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.rx",
        "state.ry",
        "state.rz",
        "state.rw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "q99",
                    "state.y": "q99",
                    "state.z": "q99",
                    "state.rx": "q99",
                    "state.ry": "q99",
                    "state.rz": "q99",
                    "state.rw": "q99",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "q99",
                    "action.y": "q99",
                    "action.z": "q99",
                    "action.roll": "q99",
                    "action.pitch": "q99",
                    "action.yaw": "q99",
                    "action.gripper": "binary",
                },
            ),
            # concat transforms
            # ConcatTransform(
            #     # video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]

        return ComposedModalityTransform(transforms=transforms)


##################################### Single Frank aRobotiq ######################################################

class SingleFrankaRobotiqDeltaEefDataConfig:
    video_keys = [
        "video.base_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
    ]
    action_keys = [
        "action.delta_eef_position",
        "action.delta_eef_rotation",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.eef_rotation": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_eef_position": "min_max",
                    "action.delta_eef_rotation": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


##################################### Libero ######################################################

class Libero4in1DataConfig:
    video_keys = [
        "video.image",
        "video.wrist_image",
    ]
    
    state_keys = [
        "state.right_arm",
        "state.right_hand",
    ]
    action_keys = [
        "action.right_arm",
        "action.right_hand",
    ]
    
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    

class Libero4in1HEXDataConfig:
    video_keys = [
        "video.image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.right_arm",
        "state.right_hand",
    ]
    action_keys = [
        "action.right_arm",
        "action.right_hand",
    ]
    
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(8))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.right_arm": "min_max",
                    "state.right_hand": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
            apply_to=self.action_keys,
            normalization_modes={
                    "action.right_arm": "min_max",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


##################################### Single Franka Robotiq ######################################################

class SingleFrankaRobotiqDeltaJointsDataConfig:
    video_keys = [
        "video.base_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.joints",
    ]
    action_keys = [
        "action.delta_joints",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joints": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.delta_joints": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###################################### Unitree G1 #####################################################

class UnitreeG1DataConfig(BaseDataConfig):
    video_keys = ["video.rs_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            # VideoToTensor(apply_to=self.video_keys),
            # VideoCrop(apply_to=self.video_keys, scale=0.95),
            # VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            # VideoColorJitter(
            #     apply_to=self.video_keys,
            #     brightness=0.3,
            #     contrast=0.4,
            #     saturation=0.5,
            #     hue=0.08,
            # ),
            # VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            # ConcatTransform(
            #     video_concat_order=self.video_keys,
            #     state_concat_order=self.state_keys,
            #     action_concat_order=self.action_keys,
            # ),
            # model-specific transform
            # GR00TTransform(
            #     state_horizon=len(self.observation_indices),
            #     action_horizon=len(self.action_indices),
            #     max_state_dim=64,
            #     max_action_dim=32,
            # ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class UnitreeG1FullBodyDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.rs_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))


class Agibot2UnitreeG1DataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.left_leg", 
        "state.right_leg", 
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.others",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    horizon = 100
    state_horizon = 50
    action_indices = list(range(horizon))
    state_indices = list(range(state_horizon))
    norm_mode = "q99"    # "min_max", "mean_std"

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: self.norm_mode for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: self.norm_mode for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class UnitreeG1HEDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.left_leg", 
        "state.right_leg", 
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    horizon = 100
    state_horizon = 50
    action_indices = list(range(horizon))
    state_indices = list(range(state_horizon))
    norm_mode = "q99"    # "min_max", "mean_std"

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: self.norm_mode for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: self.norm_mode for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class UnitreeG1SonicDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.left_hand",
        "state.right_arm",  
        "state.right_hand",
    ]
    action_keys = [
        "action.motion_token",
        "action.hand_binary_action",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    horizon = 50
    state_horizon = 25
    action_indices = list(range(horizon))
    state_indices = list(range(state_horizon))
    norm_mode = "q99"    # "min_max", "mean_std"

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: self.norm_mode for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: self.norm_mode for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

###################################### Unitree H1 #####################################################

class UnitreeH1HEDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.left_leg", 
        "state.right_leg",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    horizon = 100
    state_horizon = 50
    action_indices = list(range(horizon))
    state_indices = list(range(state_horizon))
    norm_mode = "q99"    # "min_max", "mean_std"

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: self.norm_mode for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: self.norm_mode for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###################################### Leju #####################################################

class LejuRoboCOINDataConfig(BaseDataConfig):
    video_keys = ["video.image"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.left_leg", 
        "state.right_leg",
        "state.head",
        "state.others",  
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.left_leg",
        "action.right_leg",
        "action.head",
        "action.others",  
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    horizon = 100
    state_horizon = 50
    action_indices = list(range(horizon))
    state_indices = list(range(state_horizon))
    norm_mode = "q99"    # "min_max", "mean_std"

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: self.norm_mode for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: self.norm_mode for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###################################### TianGong 2.0 #####################################################

class TianGong2DataConfig(BaseDataConfig):
    def __init__(
        self, 
        modality_file_path: str,
        observation_horizon: int = 1,
        state_horizon: int = 1,
        action_horizon: int = 50,
        norm_mode: str = "q99", # "min_max", "mean_std"
    ):
        super().__init__()
        self.norm_mode = norm_mode
        
        # Automatically extract all available state and action components.
        try:
            with open(modality_file_path, 'r') as f:
                modality_data = json.load(f)

            state_parts = modality_data.get("state", {}).keys()
            action_parts = modality_data.get("action", {}).keys()
            
            self.state_keys = [f"state.{p}" for p in state_parts]
            self.action_keys = [f"action.{p}" for p in action_parts]
            
            self.video_keys = ["video.image"]
            self.language_keys = ["annotation.human.action.task_description"]

            self.observation_indices = list(range(observation_horizon))
            self.state_indices = list(range(state_horizon))
            self.action_indices = list(range(action_horizon))
        except:
            print('The modality file path is wrong.')

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: self.norm_mode for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: self.norm_mode for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
    

###################################### TianGong 3.0 #####################################################

class TianGong3DataConfig(BaseDataConfig):
    def __init__(
        self, 
        modality_file_path: str,
        observation_horizon: int = 1,
        state_horizon: int = 1,
        action_horizon: int = 50,
        norm_mode: str = "q99", # "min_max", "mean_std"
    ):
        super().__init__()
        self.norm_mode = norm_mode
        
        with open(modality_file_path, 'r') as f:
            modality_data = json.load(f)
        
        # Automatically extract all available state and action components.
        try:
            state_parts = modality_data.get("state", {}).keys()
            action_parts = modality_data.get("action", {}).keys()
            
            self.state_keys = [f"state.{p}" for p in state_parts]
            self.action_keys = [f"action.{p}" for p in action_parts]
            
            self.video_keys = ["video.image"]
            self.language_keys = ["annotation.human.action.task_description"]

            self.observation_indices = list(range(observation_horizon))
            self.state_indices = list(range(state_horizon))
            self.action_indices = list(range(action_horizon))
        except:
            print('The modality file path is wrong.')

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: self.norm_mode for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: self.norm_mode for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
    

###################################### TianYi 2 #####################################################

class TianYiDataConfig(BaseDataConfig):
    def __init__(
        self, 
        modality_file_path: str,
        observation_horizon: int = 1,
        state_horizon: int = 1,
        action_horizon: int = 50,
        norm_mode: str = "q99", # "min_max", "mean_std"
    ):
        super().__init__()
        self.norm_mode = norm_mode
        
        with open(modality_file_path, 'r') as f:
            modality_data = json.load(f)
        
        # Automatically extract all available state and action components.
        try:
            state_parts = modality_data.get("state", {}).keys()
            action_parts = modality_data.get("action", {}).keys()
            
            self.state_keys = [f"state.{p}" for p in state_parts]
            self.action_keys = [f"action.{p}" for p in action_parts]
            
            self.video_keys = ["video.image"]
            self.language_keys = ["annotation.human.action.task_description"]

            self.observation_indices = list(range(observation_horizon))
            self.state_indices = list(range(state_horizon))
            self.action_indices = list(range(action_horizon))
        except:
            print('The modality file path is wrong.')

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.state_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: self.norm_mode for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: self.norm_mode for key in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
    

###########################################################################################
###########################################################################################
###########################################################################################

ACTION_HORIZON = 100
STATE_HORIZON = ACTION_HORIZON // 2
DATA_ROOT = os.environ.get("HEX_DATA_ROOT", "/media/bsh/data/eval")
PRETRAIN_DATA_ROOT = os.environ.get("HEX_PRETRAIN_DATA_ROOT", "/media/bsh/data/pretrain")
META = "meta/modality.json"

# 1) Two profiles: baseline vs. hex
PROFILE_KWARGS = {
    "baseline": dict(action_horizon=ACTION_HORIZON),
    "hex": dict(state_horizon=STATE_HORIZON, action_horizon=ACTION_HORIZON),
}

# 2) A DataConfig constructor for each robot family
CTORS = {
    "tiangong2": TianGong2DataConfig,
    "tiangong3": TianGong3DataConfig,
    "tianyi": TianYiDataConfig,
}

# 3) You only need to maintain this task list
TASKS = [
    ("tiangong2", "v1", "dvt217_react_to_ball_251112_3_direction"),
    ("tiangong2", "v2", "dvt217_whack_a_mole"),
    ("tiangong2", "v3", "dvt217_pour_wine_follow_the_finger"),
    ("tiangong2", "v3", "dvt217_stack_cube"),

    ("tiangong3", "v1", "dex7_block_ball"),
    ("tiangong3", "v2", "dex7_catch_ball"),
    ("tiangong3", "v3", "evt12_put_tennis"),
    ("tiangong3", "v4", "evt12_tidy_table"),
    ("tiangong3", "v5", "evt2_40_pick_up_toy"),
    ("tiangong3", "v6", "dex7_grab_hat"),
    ("tiangong3", "v7", "dex11_follow_hat"),
    ("tiangong3", "v8", "dex11_grab_hat_put_on_head"),
    ("tiangong3", "v9", "evt12_fold_towel"),

    ("tianyi", "v1", "tianyi_29_pour_wine_and_handover"),
]


PRETRAIN_FAMILY_DIRS = {
    "tiangong2": "tiangong2",
    "tiangong3": "tiangong3",
    "tianyi": "tianyi",
}


def _pretrain_robot_type(family: str, dataset_name: str) -> str:
    dataset_name = dataset_name.replace("tienkung", "tiangong")
    if dataset_name.startswith(f"{family}_"):
        return dataset_name
    return f"{family}_{dataset_name}"


def _first_existing_modality_path(family: str, dataset_name: str) -> Path | None:
    candidates = [
        Path(DATA_ROOT) / dataset_name / META,
        Path(PRETRAIN_DATA_ROOT) / PRETRAIN_FAMILY_DIRS.get(family, family) / dataset_name / META,
        Path("/media/bsh/data/eai_real_world") / dataset_name / META,
    ]
    return next((path for path in candidates if path.exists()), None)


def build_map():
    m = {
        "libero_franka": Libero4in1DataConfig(),
        "libero_franka_hex": Libero4in1HEXDataConfig(),
        "oxe_droid": OxeDroidDataConfig(),
        "oxe_bridge": OxeBridgeDataConfig(),
        "oxe_rt1": OxeRT1DataConfig(),
        "demo_sim_franka_delta_joints": SingleFrankaRobotiqDeltaJointsDataConfig(),
        "custom_robot_config": SingleFrankaRobotiqDeltaEefDataConfig(),
        "g1_a2ug1": Agibot2UnitreeG1DataConfig(),
        "g1_he": UnitreeG1HEDataConfig(),
        "g1_sonic": UnitreeG1SonicDataConfig(),
        "h1_he": UnitreeH1HEDataConfig(),
        "leju_robocoin": LejuRoboCOINDataConfig(),
    }
    for family, task, d in TASKS:
        path = _first_existing_modality_path(family, d)
        if path is None:
            continue
        ctor = CTORS[family]
        m[f"{family}_{task}_baseline"] = ctor(str(path), action_horizon=ACTION_HORIZON)
        m[f"{family}_{task}"] = ctor(str(path), state_horizon=STATE_HORIZON, action_horizon=ACTION_HORIZON)

    pretrain_root = Path(PRETRAIN_DATA_ROOT)
    for family, subdir in PRETRAIN_FAMILY_DIRS.items():
        ctor = CTORS[family]
        family_root = pretrain_root / subdir
        if not family_root.exists():
            continue
        for modality_path in sorted(family_root.glob(f"*/{META}")):
            dataset_name = modality_path.parent.parent.name
            m[_pretrain_robot_type(family, dataset_name)] = ctor(
                str(modality_path),
                state_horizon=STATE_HORIZON,
                action_horizon=ACTION_HORIZON,
            )
    return m


def get_data_config(robot_type: str, dataset_path: Path | str | None = None):
    if dataset_path is not None:
        dataset_path = Path(dataset_path)
        modality_path = dataset_path / META
        if modality_path.exists():
            if robot_type.startswith("tiangong2_"):
                return TianGong2DataConfig(
                    str(modality_path),
                    state_horizon=STATE_HORIZON,
                    action_horizon=ACTION_HORIZON,
                )
            if robot_type.startswith("tiangong3_"):
                return TianGong3DataConfig(
                    str(modality_path),
                    state_horizon=STATE_HORIZON,
                    action_horizon=ACTION_HORIZON,
                )
            if robot_type.startswith("tianyi_"):
                return TianYiDataConfig(
                    str(modality_path),
                    state_horizon=STATE_HORIZON,
                    action_horizon=ACTION_HORIZON,
                )

    return ROBOT_TYPE_CONFIG_MAP[robot_type]


ROBOT_TYPE_CONFIG_MAP = build_map()
