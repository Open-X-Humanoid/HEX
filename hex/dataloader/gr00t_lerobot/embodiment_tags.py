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

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    OXE_DROID = "oxe_droid"
    """
    The OxE Droid dataset.
    """

    OXE_BRIDGE = "oxe_bridge"
    """
    The OxE Bridge dataset.
    """

    OXE_RT1 = "oxe_rt1"
    """
    The OxE RT-1 dataset.
    """

    AGIBOT_GENIE1 = "agibot_genie1"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    FRANKA = 'franka'
    """
    The Franka Emika Panda robot.
    """

    UNITREE_G1_V1 = 'unitree_g1_v1'
    """
    The Unitree G1 robot (Humanoid Everyday).
    """

    UNITREE_G1_V2 = 'unitree_g1_v2'
    """
    The Unitree G1 robot (Agibot2G1).
    """

    UNITREE_G1_Sonic = 'unitree_g1_sonic'
    """
    The Unitree G1 robot with Sonic.
    """

    UNITREE_H1_V1 = 'unitree_h1_v1'
    """
    The Unitree H1 robot (Humanoid Everyday).
    """

    LEJU_KUAVO_V1 = 'leju_kuavo_v1'
    """
    The Leju Kuavo robot (RoboCOIN).
    """

    TIANGONG2_V1 = 'tiangong2_v1'
    """
    The TianGong2 robot. state: arm+hand+waist+head (state: 33, action: 23)
    """

    TIANGONG2_V2 = 'tiangong2_v2'
    """
    The TianGong2 robot. state: arm+hand+waist (state: 30, action: 20)
    """

    TIANGONG2_V3 = 'tiangong2_v3'
    """
    The TianGong2 robot. state: arm+hand+waist (state: 36, action: 32)
    """

    TIANGONG2_V4 = 'tiangong2_v4'
    """
    The TianGong2 robot. state: arm+hand+waist (state: 26, action: 16)
    """

    TIANGONG3_V1 = 'tiangong3_v1'
    """
    The TianGong3 robot. state: arm+hand+waist+leg+others. (state: 51, action: 20)
    """

    TIANGONG3_V2 = 'tiangong3_v2'
    """
    The TianGong3 robot. state: arm+hand+waist+leg+others, left gripper. (state: 46 action: 20)
    """

    TIANGONG3_V3 = 'tiangong3_v3'
    """
    The TianGong3 robot. (state: 46, action: 8)
    """

    TIANGONG3_V4 = 'tiangong3_v4'
    """
    The TianGong3 robot. (state: 100, action: 28)
    """

    TIANGONG3_V5 = 'tiangong3_v5'
    """
    The TianGong3 robot. (state: 126, action: 34)
    """

    TIANGONG3_V6 = 'tiangong3_v6'
    """
    The TianGong3 robot. (state: 45, action: 19)
    """

    TIANGONG3_V7 = 'tiangong3_v7'
    """
    The TianGong3 robot. (state: 66, action: 23)
    """

    TIANGONG3_V8 = 'tiangong3_v8'
    """
    The TianGong3 robot. (state: 54, action: 23)
    """

    TIANGONG3_V9 = 'tiangong3_v9'
    """
    The TianGong3 robot. (state: 68, action: 8)
    """

    TIANYI_V1 = 'tianyi_v1'
    """
    The TianYi robot. state: arm+hand, action: arm+hand. (state: 16, action: 16)
    """


# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.OXE_DROID.value: 30,
    EmbodimentTag.OXE_BRIDGE.value: 29,
    EmbodimentTag.OXE_RT1.value: 28,
    EmbodimentTag.AGIBOT_GENIE1.value: 27,
    EmbodimentTag.GR1.value: 26,
    EmbodimentTag.FRANKA.value: 25,
    # Unitree G1
    EmbodimentTag.UNITREE_G1_V1.value: 0,
    EmbodimentTag.UNITREE_G1_V2.value: 1,
    EmbodimentTag.UNITREE_G1_Sonic.value: 2,
    # Unitree H1
    EmbodimentTag.UNITREE_H1_V1.value: 5,
    # Leju Kuavo
    EmbodimentTag.LEJU_KUAVO_V1.value: 9,
    # TianGong2
    EmbodimentTag.TIANGONG2_V1.value: 10,
    EmbodimentTag.TIANGONG2_V2.value: 11,
    EmbodimentTag.TIANGONG2_V3.value: 12,
    EmbodimentTag.TIANGONG2_V4.value: 13,
    # TianGong3
    EmbodimentTag.TIANGONG3_V1.value: 15,
    EmbodimentTag.TIANGONG3_V2.value: 16,
    EmbodimentTag.TIANGONG3_V3.value: 17,
    EmbodimentTag.TIANGONG3_V4.value: 18,
    EmbodimentTag.TIANGONG3_V5.value: 19,
    EmbodimentTag.TIANGONG3_V6.value: 21,
    EmbodimentTag.TIANGONG3_V7.value: 22,
    EmbodimentTag.TIANGONG3_V8.value: 23,
    EmbodimentTag.TIANGONG3_V9.value: 24,
    # TianYi
    EmbodimentTag.TIANYI_V1.value: 20,

}

# Robot type to embodiment tag mapping
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka": EmbodimentTag.FRANKA,
    "libero_franka_hex": EmbodimentTag.FRANKA,
    "oxe_droid": EmbodimentTag.OXE_DROID,
    "oxe_bridge": EmbodimentTag.OXE_BRIDGE,
    "oxe_rt1": EmbodimentTag.OXE_RT1,
    "demo_sim_franka_delta_joints": EmbodimentTag.FRANKA,
    "custom_robot_config": EmbodimentTag.NEW_EMBODIMENT,

    # unitree g1
    "g1_he": EmbodimentTag.UNITREE_G1_V1,
    "g1_a2ug1": EmbodimentTag.UNITREE_G1_V2,        # agibot2g1
    "g1_sonic": EmbodimentTag.UNITREE_G1_Sonic,     # unitree g1 with sonic

    # unitree h1
    "h1_he": EmbodimentTag.UNITREE_H1_V1,

    # leju kuavo
    "leju_robocoin": EmbodimentTag.LEJU_KUAVO_V1,

    # tiangong2: baseline
    "tiangong2_v1_baseline": EmbodimentTag.TIANGONG2_V1,
    "tiangong2_v2_baseline": EmbodimentTag.TIANGONG2_V2,
    "tiangong2_v3_baseline": EmbodimentTag.TIANGONG2_V3,

    # tiangong2: hex
    "tiangong2_v1": EmbodimentTag.TIANGONG2_V1,
    "tiangong2_v2": EmbodimentTag.TIANGONG2_V2,
    "tiangong2_v3": EmbodimentTag.TIANGONG2_V3,
    "tiangong2_v4": EmbodimentTag.TIANGONG2_V4,

    # tiangong3: hex
    "tiangong3_v1": EmbodimentTag.TIANGONG3_V1,
    "tiangong3_v2": EmbodimentTag.TIANGONG3_V2,
    "tiangong3_v3": EmbodimentTag.TIANGONG3_V3,
    "tiangong3_v4": EmbodimentTag.TIANGONG3_V4,
    "tiangong3_v5": EmbodimentTag.TIANGONG3_V5,
    "tiangong3_v6": EmbodimentTag.TIANGONG3_V6,
    "tiangong3_v7": EmbodimentTag.TIANGONG3_V7,
    "tiangong3_v8": EmbodimentTag.TIANGONG3_V8,
    "tiangong3_v9": EmbodimentTag.TIANGONG3_V9,

    # tianyi: hex
    "tianyi_v1": EmbodimentTag.TIANYI_V1,
}


def get_embodiment_tag(robot_type: str) -> EmbodimentTag:
    if robot_type in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        return ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    return EmbodimentTag.NEW_EMBODIMENT
