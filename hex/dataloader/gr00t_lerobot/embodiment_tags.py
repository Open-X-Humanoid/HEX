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

    TIENKUNG2_V1 = 'tienkung2_v1'
    """
    The Tienkung2 robot. state: arm+hand+waist+head (state: 33, action: 23)
    """

    TIENKUNG2_V2 = 'tienkung2_v2'
    """
    The Tienkung2 robot. state: arm+hand+waist (state: 30, action: 20)
    """

    TIENKUNG2_V3 = 'tienkung2_v3'
    """
    The Tienkung2 robot. state: arm+hand+waist (state: 36, action: 32)
    """

    TIENKUNG3_V1 = 'tienkung3_v1'
    """
    The Tienkung3 robot. state: arm+hand+waist+leg+others. (state: 51, action: 20)
    """

    TIENKUNG3_V2 = 'tienkung3_v2'
    """
    The Tienkung3 robot. state: arm+hand+waist+leg+others, left gripper. (state: 46 action: 20)
    """

    TIENKUNG3_V3 = 'tienkung3_v3'
    """
    The Tienkung3 robot. (state: 46, action: 8)
    """

    TIENKUNG3_V4 = 'tienkung3_v4'
    """
    The Tienkung3 robot. (state: 100, action: 28)
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
    # Tienkung2
    EmbodimentTag.TIENKUNG2_V1.value: 10,
    EmbodimentTag.TIENKUNG2_V2.value: 11,
    EmbodimentTag.TIENKUNG2_V3.value: 12,
    # Tienkung3
    EmbodimentTag.TIENKUNG3_V1.value: 15,
    EmbodimentTag.TIENKUNG3_V2.value: 16,
    EmbodimentTag.TIENKUNG3_V3.value: 17,
    EmbodimentTag.TIENKUNG3_V4.value: 18,
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

    # tienkung2: baseline
    "tienkung2_v1_baseline": EmbodimentTag.TIENKUNG2_V1,
    "tienkung2_v2_baseline": EmbodimentTag.TIENKUNG2_V2,
    "tienkung2_v3_baseline": EmbodimentTag.TIENKUNG2_V3,

    # tienkung2: hex
    "tienkung2_v1": EmbodimentTag.TIENKUNG2_V1,
    "tienkung2_v2": EmbodimentTag.TIENKUNG2_V2,
    "tienkung2_v3": EmbodimentTag.TIENKUNG2_V3,

    # tienkung3: hex
    "tienkung3_v1": EmbodimentTag.TIENKUNG3_V1,
    "tienkung3_v2": EmbodimentTag.TIENKUNG3_V2,
    "tienkung3_v3": EmbodimentTag.TIENKUNG3_V3,
    "tienkung3_v4": EmbodimentTag.TIENKUNG3_V4,

    # tianyi: hex
    "tianyi_v1": EmbodimentTag.TIANYI_V1,
}
