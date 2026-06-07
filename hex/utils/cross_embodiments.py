import os
import json
import copy
from pathlib import Path
import torch.distributed as dist

from hex.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from hex.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG


def parse_modality_config(modality_path):
    with open(modality_path, "r") as f:
        data = json.load(f)

    def parse(block):
        return {
            part: {
                "start": info["start"],
                "end": info["end"],
                "dim": info["end"] - info["start"],
            }
            for part, info in block.items()
        }

    return parse(data.get("state", {})), parse(data.get("action", {}))


def merge_embodiment_registries(
    base_state_reg: dict,
    base_action_reg: dict,
    extra_state_reg: dict,
    extra_action_reg: dict,
    *,
    strict: bool = True,
):
    """
    Merge two embodiment registries.

    base_*      : Usually the registry loaded from the pretrained checkpoint.
    extra_*     : Usually the registry obtained by scanning the current dataset.
    strict=True : Raise an error if the same tag has inconsistent definitions;
                otherwise, keep the base definition and print a warning.
    """
    
    merged_state = copy.deepcopy(base_state_reg)
    merged_action = copy.deepcopy(base_action_reg)

    # merge state_registry: tag -> part -> {start, end, dim}
    for tag, state_reg in extra_state_reg.items():
        if tag in merged_state:
            if merged_state[tag] != state_reg:
                msg = f"[EmbodimentRegistry] Conflict on state_registry[{tag}]: " \
                      f"base={merged_state[tag]} vs extra={state_reg}"
                if strict:
                    raise ValueError(msg)
                else:
                    print("WARNING:", msg, "-> keeping base version.")
        else:
            merged_state[tag] = state_reg

    # merge action_registry: tag -> max_action_dim
    for tag, dim in extra_action_reg.items():
        if tag in merged_action:
            if merged_action[tag] != dim:
                msg = f"[EmbodimentRegistry] Conflict on action_registry[{tag}]: " \
                      f"base={merged_action[tag]} vs extra={dim}"
                if strict:
                    raise ValueError(msg)
                else:
                    # 可以选 max，也可以保留 base，这里我用 max 比较安全一点
                    merged_action[tag] = max(merged_action[tag], dim)
                    print("WARNING:", msg, "-> using max =", merged_action[tag])
        else:
            merged_action[tag] = dim

    return merged_state, merged_action


def load_registry(path):
    """Safely load registry json"""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[EmbodimentRegistry] WARNING: corrupted registry at {path}")
        return {}
    
    
def build_registry_from_dataset(config):
    """Scan dataset to build registry"""
    data_cfg = config.datasets.vla_data
    data_root_dir = Path(data_cfg.data_root_dir)
    data_mix = data_cfg.data_mix
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]

    state_registry = {}
    action_registry = {}

    for d_name, _, robot_type in mixture_spec:

        tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type].value
        meta = data_root_dir / d_name / "meta/modality.json"

        state_reg, action_reg = parse_modality_config(meta)

        state_registry[tag] = state_reg
        action_registry[tag] = max(info["end"] for info in action_reg.values())

    return state_registry, action_registry


def get_embodiment_registry(config, is_train=True):
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    run_dir = Path(config.run_root_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    registry_path = run_dir / "embodiment_registry.json"

    # --------------------------------------------------
    # 1. TEST / INFERENCE MODE
    # --------------------------------------------------
    if not is_train:
        if not registry_path.exists():
            raise RuntimeError(
                f"[EmbodimentRegistry] registry not found at {registry_path}"
            )

        reg = load_registry(registry_path)

        print(f"[EmbodimentRegistry] Loaded registry from {registry_path}")

        return reg["state_registry"], reg["action_registry"]

    # --------------------------------------------------
    # 2. TRAIN MODE
    # --------------------------------------------------

    # build registry from dataset
    cur_state_registry, cur_action_registry = build_registry_from_dataset(config)

    # load pretrained registry
    pretrained_state_registry = {}
    pretrained_action_registry = {}

    pretrained_root = getattr(config.framework, "pretrained_run_root_dir", None)
    pretrained_run_id = getattr(config.framework, "pretrained_run_id", None)

    if pretrained_root and pretrained_run_id:

        pretrained_path = Path(pretrained_root) / pretrained_run_id / "embodiment_registry.json"

        if pretrained_path.exists():

            pre_reg = load_registry(pretrained_path)

            pretrained_state_registry = pre_reg.get("state_registry", {})
            pretrained_action_registry = pre_reg.get("action_registry", {})

            print(f"[EmbodimentRegistry] Loaded pretrained registry from {pretrained_path}")

    # merge pretrained + dataset
    if pretrained_state_registry:
        merged_state_registry, merged_action_registry = merge_embodiment_registries(
            pretrained_state_registry,
            pretrained_action_registry,
            cur_state_registry,
            cur_action_registry,
            strict=False,
        )

    else:
        merged_state_registry = cur_state_registry
        merged_action_registry = cur_action_registry

    final_reg = {
        "state_registry": merged_state_registry,
        "action_registry": merged_action_registry,
    }

    # --------------------------------------------------
    # 3. SAVE (rank0 only)
    # --------------------------------------------------

    if rank == 0:

        tmp = registry_path.with_suffix(".tmp")

        with open(tmp, "w") as f:
            json.dump(final_reg, f, indent=2)

        os.replace(tmp, registry_path)

        print(f"[EmbodimentRegistry] Saved registry to {registry_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return merged_state_registry, merged_action_registry
