"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""
from pathlib import Path

root_path = "/media/bsh/data/eai_real_world"       # Change this to your dataset directory

# example
EAI_TIENKUNG2 = [
    ("dvt217_react_to_ball_251112_3_direction_lerobot", "tienkung2_v1"),
    ("dvt217_whack_a_mole_251227_lerobot", "tienkung2_v2"),
    ("dvt217_imitate_posture_260126_lerobot", "tienkung2_v3"),
    ("dvt217_pour_wine_follow_the_finger_260126_lerobot", "tienkung2_v3"),
    # ...
]

EAI_TIENKUNG3 = [
    ("dex7_block_ball_251205_lerobot", "tienkung3_v1"),
    ("dex7_catch_ball_251215_lerobot", "tienkung3_v2"),
    ("dex7_play_tennis_260104_lerobot", "tienkung3_v2"),
    ("evt12_put_cube_in_box_260113_lerobot", "tienkung3_v3"),
    ("evt12_put_tennis_ball_in_box_260110_lerobot", "tienkung3_v3"),
    # ...
]

EAI_TIENYI = [
    ("tienkung_29_pour_wine_and_handover_251129_am_master", "tianyi_v1"),
]


def build_agibot_to_g1_mix(root=root_path):
    '''
    Import all tasks of Agibot dataset
    '''
    root = Path(root)
    tasks = sorted([
        p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("g1_") and p.name != "g1_humanoid_everyday"
    ])
    return [(t, 1.0, "g1_a2ug1") for t in tasks]


def build_robocoin_leju_mix(root=root_path):
    '''
    Import Leju tasks of RoboCOIN dataset
    '''
    root = Path(root)
    tasks = sorted([
        p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("leju_")
    ])
    return [(t, 1.0, "leju_robocoin") for t in tasks]


def build_mix_with_type_budget(
    tienkung2_budget=0.10,
    tienkung3_budget=0.10,
    tienyi_budget=0.03,
    g1_a2ug1_budget=0.25,
    g1_he_budget=0.20,
    h1_he_budget=0.20,
    leju_budget=0.12,
):
    mix = []

    # Tiekung: evenly split the budget within the tienkung groups
    mix += [(name, tienkung2_budget / len(EAI_TIENKUNG2), cfg) for name, cfg in EAI_TIENKUNG2]
    mix += [(name, tienkung3_budget / len(EAI_TIENKUNG3), cfg) for name, cfg in EAI_TIENKUNG3]
    mix += [(name, tienyi_budget / len(EAI_TIENYI), cfg) for name, cfg in EAI_TIENYI]

    # Agibot (g1_a2ug1): evenly split the budget across tasks
    g1_tasks = build_agibot_to_g1_mix()  # [(task_name, 1.0, "g1_a2ug1"), ...]
    w_g1 = g1_a2ug1_budget / max(1, len(g1_tasks))
    mix += [(name, w_g1, cfg) for (name, _, cfg) in g1_tasks]

    # Humanoid Evertyday (g1_he / h1_he): each treated as one large dataset group
    mix += [("g1_humanoid_everyday", g1_he_budget, "g1_he")]
    mix += [("h1_humanoid_everyday", h1_he_budget, "h1_he")]

    # RoboCOIN - Leju
    leju_tasks = build_robocoin_leju_mix()
    w_leju = leju_budget / max(1, len(leju_tasks))
    mix += [(name, w_leju, cfg) for (name, _, cfg) in leju_tasks]

    return mix


def build_mix_with_type_budget_wo_tienkung(
    g1_a2ug1_budget=0.30,
    g1_he_budget=0.25,
    h1_he_budget=0.25,
    leju_budget=0.20,
):
    mix = []

    # Agibot (g1_a2ug1): evenly split the budget across tasks
    g1_tasks = build_agibot_to_g1_mix()  # [(task_name, 1.0, "g1_a2ug1"), ...]
    w_g1 = g1_a2ug1_budget / max(1, len(g1_tasks))
    mix += [(name, w_g1, cfg) for (name, _, cfg) in g1_tasks]

    # Humanoid Evertyday (g1_he / h1_he): each treated as one large dataset group
    mix += [("g1_humanoid_everyday", g1_he_budget, "g1_he")]
    mix += [("h1_humanoid_everyday", h1_he_budget, "h1_he")]

    # RoboCOIN - Leju
    leju_tasks = build_robocoin_leju_mix()
    w_leju = leju_budget / max(1, len(leju_tasks))
    mix += [(name, w_leju, cfg) for (name, _, cfg) in leju_tasks]

    return mix


# Dataset mixture name mapped to a list of tuples containing:
# {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {
    "libero_all_baseline": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal_baseline": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_object_baseline": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_spatial_baseline": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_10_baseline": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_90_baseline": [
        ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],

    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_hex"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_hex"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_hex"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_hex"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_hex"),
    ],
    "libero_object": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_hex"),
    ],
    "libero_spatial": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_hex"),
    ],
    "libero_10": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_hex"),
    ],
    "libero_90": [
        ("libero_90_no_noops_lerobot", 1.0, "libero_franka_hex"),
    ],

    "bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_rt_1": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],

    "demo_sim_pick_place": [
        ("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),
    ],

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],

    # tienkung2: hex
    "EAI_real_world_react_to_ball": [
        ("dvt217_react_to_ball_251112_3_direction_lerobot", 1.0, "tienkung2_v1"),
    ],
    "EAI_real_world_whack_a_mole": [
        ("dvt217_whack_a_mole_251227_lerobot", 1.0, "tienkung2_v2"),
    ],
    "EAI_real_world_imitate_gesture_old": [
        ("dvt217_imitate_posture_260104_lerobot", 1.0, "tienkung2_v2"),
    ],
    "EAI_real_world_pour_wine_follow_finger_old": [
        ("dvt217_pour_wine_follow_the_finger_251227_lerobot", 1.0, "tienkung2_v2"),
    ],
    "EAI_real_world_imitate_gesture": [
        ("dvt217_imitate_posture_260126_lerobot", 1.0, "tienkung2_v3"),
    ],
    "EAI_real_world_pour_wine_follow_finger": [
        ("dvt217_pour_wine_follow_the_finger_260126_lerobot", 1.0, "tienkung2_v3"),
    ],
    "EAI_real_world_carry_boxes_avoid_obstacles": [
        ("dvt217_carry_boxes_and_avoid_obstacles_260113_lerobot", 1.0, "tienkung2_v3"),
    ],
    "EAI_real_world_turn_around_and_carry_boxes": [
        ("dvt217_turn_around_and_carry_boxes_260113_lerobot", 1.0, "tienkung2_v3"),
    ],
    "EAI_real_world_carry_boxes_follow_human": [
        ("dvt217_carry_boxes_follow_human_260126_lerobot", 1.0, "tienkung2_v3"),
    ],
    
    # tienkung3: hex
    "EAI_real_world_block_ball": [
        ("dex7_block_ball_251205_lerobot", 1.0, "tienkung3_v1"),
    ],
    "EAI_real_world_catch_ball": [
        ("dex7_catch_ball_251215_lerobot", 1.0, "tienkung3_v2"),
    ],
    "EAI_real_world_put_cube_in_box": [
        ("evt12_put_cube_in_box_260330_lerobot", 1.0, "tienkung3_v4"),
    ],
    "EAI_real_world_carry_box_and_tidy_table": [
        ("evt12_carry_box_and_tidy_table_260318_lerobot",1.0, "tienkung3_v4"),
    ],
    "EAI_real_world_tidy_table": [
        ("evt12_tidy_table_260318_lerobot",1.0, "tienkung3_v4"),
    ],

    # tianyi: hex
    "EAI_real_world_pour_wine": [
        ("tienkung_29_pour_wine_and_handover_251130_master", 1.0, "tianyi_v1"),
        ("tienkung_29_pour_wine_and_handover_251201_master", 1.0, "tianyi_v1"),
        ("tienkung_29_pour_wine_and_handover_251202_master", 1.0, "tianyi_v1"),
        ("tienkung_29_pour_wine_and_handover_251203_master", 1.0, "tianyi_v1"),
        ("tienkung_29_pour_wine_and_handover_251204_master", 1.0, "tianyi_v1"),   
    ],

    # g1
    "g1_he_real_world": [
        ("g1_humanoid_everyday", 1.0, "g1_he"),
    ],
    "g1_a2ug1_real_world": build_agibot_to_g1_mix(),
    "g1_sonic_real_world_pick_cola": [
        ("cola_0530", 1.0, "g1_sonic"),
        ("cola_0604", 0.3, "g1_sonic"),
    ],

    # h1
    "h1_he_real_world": [
        ("h1_humanoid_everyday", 1.0, "h1_he"),
    ],

    # corss-embodiment pretraining without Tienkung series
    "EAI_real_world_wo_tienkung": build_mix_with_type_budget_wo_tienkung(),

    # corss-embodiment pretraining
    "EAI_real_world": build_mix_with_type_budget(),
}


