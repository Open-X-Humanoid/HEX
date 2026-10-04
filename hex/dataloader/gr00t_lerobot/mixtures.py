"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""
from pathlib import Path
import os

root_path = os.environ.get("HEX_PRETRAIN_DATA_ROOT", "/media/bsh/data/pretrain")


PRETRAIN_ROBOT_TYPES = {
    # tiangong2
    "dvt217_react_to_ball_3_direction": "tiangong2_v1",
    "dvt217_moving_objects_manipulation": "tiangong2_v1",
    "dvt217_carrot_or_paper": "tiangong2_v2",
    "dvt217_place_parts_in_box_put_box_on_chair": "tiangong2_v2",
    "dvt217_whack_the_mole": "tiangong2_v2",
    "dvt217_put_ball_into_box": "tiangong2_v3",
    "dvt217_put_basket": "tiangong2_v3",
    "dvt426_catch_car": "tiangong2_v3",
    "dvt426_speed_stack": "tiangong2_v3",
    "dvt426_stack_cube": "tiangong2_v3",
    "dvt217_stack_cube": "tiangong2_v4",
    # tiangong3
    "dex7_block_ball": "tiangong3_v1",
    "dex7_react_flag": "tiangong3_v1",
    "dex7_catch_ball": "tiangong3_v2",
    "dex7_play_tennis": "tiangong3_v2",
    "evt12_put_tennis_ball_in_box": "tiangong3_v3",
    "evt12_move_box_stage": "tiangong3_v4",
    "evt2_40_carry_ball_to_box": "tiangong3_v5",
    "evt2_40_close_computer": "tiangong3_v5",
    "evt2_40_pick_up_clothes_from_overhead_rack": "tiangong3_v5",
    "evt2_40_pick_up_clothes_from_standing_rack": "tiangong3_v5",
    "evt2_40_water_the_flower": "tiangong3_v5",
    "dex7_grab_hat": "tiangong3_v6",
    "dex7_pick_up_toy_while_walking": "tiangong3_v6",
    "dex11_follow_hat": "tiangong3_v7",
    "dex11_pick_hat": "tiangong3_v7",
    "dex11_grab_hat_put_on_head": "tiangong3_v8",
    "evt12_fold_towel": "tiangong3_v9",
    # tianyi
    "tianyi_29_pour_wine_and_handover": "tianyi_v1",
}


def _dataset_dirs(root: Path, subdir: str | None = None, prefix: str | None = None):
    base = root / subdir if subdir else root
    if not base.exists():
        return []
    return sorted([
        p for p in base.iterdir()
        if p.is_dir() and (prefix is None or p.name.startswith(prefix))
    ])


def _relative_name(root: Path, dataset_dir: Path) -> str:
    return dataset_dir.relative_to(root).as_posix()


def build_agibot_to_g1_mix(root=root_path):
    '''
    Import all tasks of Agibot dataset
    '''
    root = Path(root)
    dataset_dirs = _dataset_dirs(root, "agibot", prefix="g1_") or _dataset_dirs(root, prefix="g1_")
    tasks = [
        _relative_name(root, p)
        for p in dataset_dirs
        if p.name != "g1_humanoid_everyday"
    ]
    return [(t, 1.0, "g1_a2ug1") for t in tasks]


def build_robocoin_leju_mix(root=root_path):
    '''
    Import Leju tasks of RoboCOIN dataset
    '''
    root = Path(root)
    dataset_dirs = _dataset_dirs(root, "leju", prefix="leju_") or _dataset_dirs(root, prefix="leju_")
    tasks = [_relative_name(root, p) for p in dataset_dirs]
    return [(t, 1.0, "leju_robocoin") for t in tasks]


def _with_total_budget(mix, total_budget):
    if len(mix) == 0:
        return []
    per_dataset_weight = total_budget / len(mix)
    return [(name, per_dataset_weight, cfg) for name, _, cfg in mix]


def _build_pretrain_tiangong_family_mix(root, subdir, family, budget):
    root = Path(root)
    family_root = root / subdir
    dataset_dirs = [
        p.parent.parent
        for p in sorted(family_root.glob("*/meta/modality.json"))
        if p.parent.parent.name in PRETRAIN_ROBOT_TYPES
    ]
    if not dataset_dirs:
        return []
    weight = budget / len(dataset_dirs)
    return [
        (
            _relative_name(root, p),
            weight,
            PRETRAIN_ROBOT_TYPES[p.name],
        )
        for p in dataset_dirs
    ]


def build_tiangong_mix(
    tiangong2_budget=0.45,
    tiangong3_budget=0.45,
    tianyi_budget=0.1,
    root=root_path,
):
    root = Path(root)
    pretrain_mix = []
    pretrain_mix += _build_pretrain_tiangong_family_mix(root, "tiangong2", "tiangong2", tiangong2_budget)
    pretrain_mix += _build_pretrain_tiangong_family_mix(root, "tiangong3", "tiangong3", tiangong3_budget)
    pretrain_mix += _build_pretrain_tiangong_family_mix(root, "tianyi", "tianyi", tianyi_budget)
    return pretrain_mix


def build_tiangong3_mix(root=root_path):
    root = Path(root)
    return _build_pretrain_tiangong_family_mix(root, "tiangong3", "tiangong3", budget=1.0)


def _pretrain_or_legacy_name(root, subdir, dataset_name):
    root = Path(root)
    pretrain_path = root / subdir / dataset_name
    if pretrain_path.exists():
        return pretrain_path.relative_to(root).as_posix()
    return dataset_name


def build_he_mix(g1_he_budget=0.5, h1_he_budget=0.5, root=root_path):
    return [
        (_pretrain_or_legacy_name(root, "g1", "g1_humanoid_everyday"), g1_he_budget, "g1_he"),
        (_pretrain_or_legacy_name(root, "h1", "h1_humanoid_everyday"), h1_he_budget, "h1_he"),
    ]


def build_g1_he_mix(g1_he_budget=1.0, root=root_path):
    return [
        (_pretrain_or_legacy_name(root, "g1", "g1_humanoid_everyday"), g1_he_budget, "g1_he"),
    ]


def build_h1_he_mix(h1_he_budget=1.0, root=root_path):
    return [
        (_pretrain_or_legacy_name(root, "h1", "h1_humanoid_everyday"), h1_he_budget, "h1_he"),
    ]


def build_tiangong_g1_a2ug1_mix(
    tiangong_budget=0.5,
    g1_a2ug1_budget=0.5,
):
    mix = []
    mix += build_tiangong_mix(
        tiangong2_budget=tiangong_budget / 3,
        tiangong3_budget=tiangong_budget / 3,
        tianyi_budget=tiangong_budget / 3,
    )
    mix += _with_total_budget(build_agibot_to_g1_mix(), g1_a2ug1_budget)
    return mix


def build_tiangong_g1_he_mix(
    tiangong_budget=0.6,
    g1_he_budget=0.4,
):
    mix = []
    mix += build_tiangong_mix(
        tiangong2_budget=tiangong_budget*0.45,
        tiangong3_budget=tiangong_budget*0.45,
        tianyi_budget=tiangong_budget*0.1,
    )
    mix += build_g1_he_mix(g1_he_budget=g1_he_budget)
    return mix


def build_tiangong_h1_he_mix(
    tiangong_budget=0.6,
    h1_he_budget=0.4,
):
    mix = []
    mix += build_tiangong_mix(
        tiangong2_budget=tiangong_budget*0.45,
        tiangong3_budget=tiangong_budget*0.45,
        tianyi_budget=tiangong_budget*0.1,
    )
    mix += build_h1_he_mix(h1_he_budget=h1_he_budget)
    return mix


def build_tiangong_he_mix(
    tiangong_budget=0.5,
    he_budget=0.5,
):
    mix = []
    mix += build_tiangong_mix(
        tiangong2_budget=tiangong_budget*0.45,
        tiangong3_budget=tiangong_budget*0.45,
        tianyi_budget=tiangong_budget*0.1,
    )
    mix += build_he_mix(
        g1_he_budget=he_budget / 2,
        h1_he_budget=he_budget / 2,
    )
    return mix


def build_mix_with_type_budget(
    tiangong2_budget=0.20,
    tiangong3_budget=0.24,
    tianyi_budget=0.02,
    g1_a2ug1_budget=0.05,
    g1_he_budget=0.30,
    h1_he_budget=0.10,
    leju_budget=0.05,
):
    mix = []

    # TianGong: evenly split the budget within the TianGong groups
    mix += build_tiangong_mix(tiangong2_budget, tiangong3_budget, tianyi_budget)

    # Agibot (g1_a2ug1): evenly split the budget across tasks
    g1_tasks = build_agibot_to_g1_mix()  # [(task_name, 1.0, "g1_a2ug1"), ...]
    w_g1 = g1_a2ug1_budget / max(1, len(g1_tasks))
    mix += [(name, w_g1, cfg) for (name, _, cfg) in g1_tasks]

    # Humanoid Evertyday (g1_he / h1_he): each treated as one large dataset group
    mix += build_he_mix(g1_he_budget=g1_he_budget, h1_he_budget=h1_he_budget)

    # RoboCOIN - Leju
    leju_tasks = build_robocoin_leju_mix()
    w_leju = leju_budget / max(1, len(leju_tasks))
    mix += [(name, w_leju, cfg) for (name, _, cfg) in leju_tasks]

    return mix


def build_mix_with_type_budget_wo_tiangong(
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
    mix += build_he_mix(g1_he_budget=g1_he_budget, h1_he_budget=h1_he_budget)

    # RoboCOIN - Leju
    leju_tasks = build_robocoin_leju_mix()
    w_leju = leju_budget / max(1, len(leju_tasks))
    mix += [(name, w_leju, cfg) for (name, _, cfg) in leju_tasks]

    return mix


# Dataset mixture name mapped to a list of tuples containing:
# {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {
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

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],

    # tiangong2: hex
    "EAI_real_world_imitate_gesture": [
        ("dvt217_imitate_posture", 1.0, "tiangong2_v3"),
    ],
    "EAI_real_world_pour_wine_follow_finger": [
        ("dvt217_pour_wine_follow_the_finger", 1.0, "tiangong2_v3"),
    ],
    "EAI_real_world_carry_boxes_avoid_obstacles": [
        ("dvt217_carry_boxes_and_avoid_obstacles", 1.0, "tiangong2_v3"),
    ],
    "EAI_real_world_turn_around_and_carry_boxes": [
        ("dvt217_turn_around_and_carry_boxes", 1.0, "tiangong2_v3"),
    ],
    "EAI_real_world_carry_boxes_follow_human": [
        ("dvt217_carry_boxes_follow_human", 1.0, "tiangong2_v3"),
    ],
    
    # tiangong3: hex
    "EAI_real_world_put_cube_in_box": [
        ("evt12_put_cube_in_box", 1.0, "tiangong3_v4"),
    ],
    "EAI_real_world_carry_box_and_tidy_table": [
        ("evt12_carry_box_and_tidy_table",1.0, "tiangong3_v4"),
    ],
    "EAI_real_world_tidy_table": [
        ("evt12_tidy_table",1.0, "tiangong3_v4"),
    ],
    "EAI_real_pick_up_toy": [
        ("evt2_40_pick_up_toy", 1.0, "tiangong3_v5"),
    ],
    "EAI_real_pick_up_box": [
        ("evt2_40_pick_up_box", 1.0, "tiangong3_v5"),
    ],

    # tianyi: hex
    "EAI_real_world_pour_wine": [
        ("tianyi_29_pour_wine_and_handover", 1.0, "tianyi_v1"),
    ],

    # g1
    "g1_he_real_world": [
        ("g1_humanoid_everyday", 1.0, "g1_he"),
    ],
    "g1_a2ug1_real_world": build_agibot_to_g1_mix(),

    # h1
    "h1_he_real_world": [
        ("h1_humanoid_everyday", 1.0, "h1_he"),
    ],

    # source ablations for cross-embodiment pretraining
    "pretrain_g1_a2ug1": build_agibot_to_g1_mix(),
    "pretrain_g1_he": build_g1_he_mix(),
    "pretrain_h1_he": build_h1_he_mix(),
    "pretrain_he": build_he_mix(),
    "pretrain_leju": build_robocoin_leju_mix(),
    "pretrain_tiangong": build_tiangong_mix(),
    "pretrain_tiangong3": build_tiangong3_mix(),
    "pretrain_tiangong_g1_a2ug1": build_tiangong_g1_a2ug1_mix(),
    "pretrain_tiangong_g1_he": build_tiangong_g1_he_mix(),
    "pretrain_tiangong_h1_he": build_tiangong_h1_he_mix(),
    "pretrain_tiangong_he": build_tiangong_he_mix(),

    # corss-embodiment pretraining without TianGong series
    "EAI_real_world_wo_tiangong": build_mix_with_type_budget_wo_tiangong(),

    # corss-embodiment pretraining
    "EAI_real_world": build_mix_with_type_budget(),
}
