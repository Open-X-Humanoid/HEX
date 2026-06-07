<h1 align="center">HEX: Humanoid-Aligned Experts for Cross-Embodiment Whole-Body Manipulation</h1>

<div align="center">

<a href="https://arxiv.org/abs/2604.07993">
  <img src="https://img.shields.io/badge/arXiv-2604.07993-b31b1b.svg" alt="arXiv">
</a>
<a href="https://hex-humanoid.github.io/">
  <img src="https://img.shields.io/badge/Project-Page-2f80ed.svg" alt="Project Page">
</a>
<a href="https://huggingface.co/Cognition2ActionLab/HEX-model">
  <img src="https://img.shields.io/badge/Hugging%20Face-Model-ffcc4d.svg?logo=huggingface&logoColor=black" alt="Model">
</a>
<a href="https://huggingface.co/datasets/Cognition2ActionLab/eai_real_world">
  <img src="https://img.shields.io/badge/Hugging%20Face-Data-ffcc4d.svg?logo=huggingface&logoColor=black" alt="Data">
</a>

</div>

<br>

<p align="center">
  <img src="assets/teaser.png" alt="HEX teaser image" />
</p>

HEX is a whole-body vision-language-action framework for full-sized humanoid robots. It combines a Qwen-VL backbone, a Unified Proprioceptive Predictor (UPP), and a flow-matching action head to predict continuous future actions.
The key idea of HEX is to align heterogeneous humanoid states into shared body-part slots and learn predictive body dynamics from cross-embodiment humanoid data. This enables the policy to transfer across different humanoid platforms and perform long-horizon whole-body manipulation.
During deployment, HEX directly predicts arm, hand, and waist actions, while providing high-level commands to a low-level RL-based whole-body controller for generating leg actions. This design enables coordinated and stable humanoid manipulation.

## News

- ✅ **2026/05/17**: Pretraining and fine-tuning code of VLA has been released.

## Installation

First, git clone this repo and `cd` into it.

```bash
# clone project
git clone https://github.com/Open-X-Humanoid/HEX.git
cd HEX
```

Then create python/pytorch env.

```bash
# crerate conda environment
conda create -n hex python=3.10 -y
conda activate hex

# Install env dependencies
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa

# Install requirements
pip install -r requirements.txt

# Install FlashAttention2
pip install flash-attn --no-build-isolation

# Install HEX
pip install -e .
```

If `flash-attn` fails to install correctly, you can run

```bash
python hex/utils/test_flash_attn.py
```

to check the versions of PyTorch, CUDA, and the libstdc++ ABI.
Then, manually download a compatible wheel from the [flash-attn release](https://github.com/Dao-AILab/flash-attention/releases).
We use version 2.7.3. However, for newer GPUs (e.g., NVIDIA RTX 5090), you should install the latest available release (e.g., version 2.8.3) to ensure compatibility.
Example:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```


## Quick Start

We release the pretrained HEX checkpoint on [Hugging Face](https://huggingface.co/Cognition2ActionLab/HEX-model).

| Description | Params | Link |
|:-----------:|:------:|:----:|
| HEX | 2.4B | 🤗 [HEX-model](https://huggingface.co/Cognition2ActionLab/HEX-model) |

### Download HEX Checkpoints

To download the HEX checkpoint, first modify the target download path in [`hex/utils/download_model_hex.py`](hex/utils/download_model_hex.py), and then run:

```bash
python hex/utils/download_model_hex.py
```

### Download the Base VLM

Before running inference, please also download the Qwen3-VL base model:

```bash
python hex/utils/download_model_qwen.py
```

After downloading Qwen3-VL, update the `framework.qwenvl.base_vlm` field in the `config.yaml` file of the downloaded HEX checkpoint to your local Qwen3-VL path.

### Run Inference

Once both the HEX checkpoint and the Qwen3-VL model are prepared, follow [`notebooks/eval_model.ipynb`](notebooks/eval_model.ipynb) to run model inference.



## Data

### Data Source

We open-source the 8 real-world evaluation task datasets collected in HEX, which can be directly used for fine-tuning.
The full training data used in this project consists of the following sources:

| Embodiment / Platform | Source | Dataset |
|:----------------------|:-------|:--------|
| Tienkung Series | HEX | 🤗 [HF Link](https://huggingface.co/datasets/Cognition2ActionLab/eai_real_world) |
| Unitree G1 | [Humanoid Everyday](https://arxiv.org/abs/2510.08807) | 🤗 [HF Link](https://huggingface.co/datasets/USC-PSI-Lab/Humanoid-Everyday-G1) | 
| AgiBot-to-Unitree G1 | [AgiBot World Colosseo](https://arxiv.org/abs/2503.06669) & [TrajBooster](https://arxiv.org/abs/2509.11839) | 🤗 [HF Link](https://huggingface.co/datasets/l2aggle/Agibot2UnitreeG1Retarget) |
| Unitree H1 | [Humanoid Everyday](https://arxiv.org/abs/2510.08807) | 🤗 [HF Link](https://huggingface.co/datasets/USC-PSI-Lab/Humanoid-Everyday-H1) |
| Leju Kuavo | [RoboCOIN](https://arxiv.org/abs/2511.17441) | 🤗 [HF Link](https://huggingface.co/collections/RoboCOIN/robocoin) |

To download all datasets, run:

```bash
bash scripts/download_datasets.sh
```

Since HEX still follows the LeRobot v2.1 data format, each dataset should contain a corresponding `modality.json`.  
For each Leju Kuavo dataset, please copy `examples/real_world/modality_leju/modality.json` to `<leju_dataset>/meta/modality.json`.

The overall data structure is as follows:

```text
eai_real_world/
├── dvt217_carry_boxes_and_avoid_obstacles_260113_lerobot
├── ...
├── evt12_carry_box_and_tidy_table_260318_lerobot
├── ...
├── g1_add_the_seasoning_to_the_pot
├── ...
├── g1_humanoid_everyday
├── h1_humanoid_everyday
├── leju_robot_box_storage_parcel
└── ...
```


### Data Collection

Due to commercial restrictions, we are unable to release the data collection pipeline used for the Tienkung series robots.

For users interested in collecting data on Unitree G1, we recommend referring to the following open-source data collection pipelines:

- [OpenTrajBooster](https://github.com/OpenHelix-Team/OpenTrajBooster), which uses a VR headset and handheld joysticks for full-body teleoperation.
- [Psi0](https://github.com/physical-superintelligence-lab/Psi0/tree/main/real): uses a PICO VR headset with controllers, along with a waist tracker and foot trackers for full-body teleoperation.



## Pretraining

You can download our [pretrained HEX model](https://huggingface.co/Cognition2ActionLab/HEX-model) and skip this step if you only want to run inference or evaluation.

Before pretraining, please download the Qwen3-VL backbone:

```bash
bash scripts/download_models.sh
```

Then, update the dataset paths in the following files to match your local directory structure:

- [`hex/dataloader/gr00t_lerobot/mixtures.py`](hex/dataloader/gr00t_lerobot/mixtures.py), Line 9
- [`hex/dataloader/gr00t_lerobot/data_config.py`](hex/dataloader/gr00t_lerobot/data_config.py), Line 1299

Next, modify the following fields in [`scripts/pretrain_hex.sh`](scripts/pretrain_hex.sh):

- `base_vlm`: path to your downloaded Qwen3-VL model
- `data_root_dir`: path to your local dataset directory
- `dataset_name`: the dataset mixture name, which should be consistent with the settings in [`hex/dataloader/gr00t_lerobot/mixtures.py`](hex/dataloader/gr00t_lerobot/mixtures.py)

Finally, start pretraining with:

```bash
bash scripts/pretrain_hex.sh
```


## Fine-tuning

After obtaining the [pretrained HEX model](https://huggingface.co/Cognition2ActionLab/HEX-model), you can further fine-tune HEX on downstream datasets.

Before fine-tuning, please modify the following fields in [`scripts/fine_tune_hex.sh`](scripts/fine_tune_hex.sh):

- `base_vlm`: path to your Qwen3-VL backbone
- `data_root_dir`: path to your local dataset directory
- `dataset_name`: name of the downstream dataset mixture, which should be consistent with the settings in [`hex/dataloader/gr00t_lerobot/mixtures.py`](hex/dataloader/gr00t_lerobot/mixtures.py)
- `pretrained_models_path`: path to the pretrained HEX checkpoint

Then, start fine-tuning with:

```bash
bash scripts/fine_tune_hex.sh
```

## Depolyment

Due to commercial restrictions, the low-level RL-based whole-body controller used for the Tienkung series robots is not open-sourced. However, we provide a sample deployment interface in [`examples/real_world`](examples/real_world).

If you want to deploy your own model on Unitree G1, you may refer to the following open-source projects:

- [OpenTrajBooster](https://github.com/OpenHelix-Team/OpenTrajBooster): uses [HOMIE](https://github.com/InternRobotics/OpenHomie) as the low-level RL-based whole-body controller.
- [Psi0](https://github.com/physical-superintelligence-lab/Psi0/tree/main/real): uses [AMO](https://github.com/OpenTeleVision/AMO) as the low-level RL-based whole-body controller.

When training your own low-level controller, please make sure that the command space output by the high-level VLA policy matches the input space expected by the low-level controller. The dataset construction process should also follow the same interface for consistent training and deployment.


## Simulation

Thanks to the cross-embodiment capability of VLA models, HEX can also be evaluated in simulation environments such as LIBERO.

First, download the LIBERO datasets:

```bash
python hex/utils/download_dataset_libero.py --base_dir /your/dataset/path
```

Then, replace the `modality.json` file for each LIBERO suite with the provided template in [examples/LIBERO/modality.json](examples/LIBERO/modality.json).

Next, modify the following fields in [`scripts/libero/train_hex_libero.sh`](scripts/libero/train_hex_libero.sh):

- `base_vlm`: path to your Qwen3-VL backbone
- `dataset_name`: name of the LIBERO dataset mixture
- `data_root_dir`: path to your local LIBERO dataset directory

Then start training with:

```bash
bash scripts/libero/train_hex_libero.sh
```

For evaluation, modify the following fields in [`scripts/libero/eval_libero.sh`](scripts/libero/eval_libero.sh):

- `ckpt_root`: root directory of the trained checkpoint
- `ckpt_path`: relative path to the checkpoint file

Then run:

```bash
bash scripts/libero/eval_libero.sh
```


## Citation

```
@article{bai2026hex,
  title={HEX: Humanoid-Aligned Experts for Cross-Embodiment Whole-Body Manipulation},
  author={Bai, Shuanghao and Li, Meng and Lv, Xinyuan and Wang, Jiawei and Wang, Xinhua and Liao, Fei and Hou, Chengkai and Gu, Langzhe and Zhou, Wanqi and Wu, Kun and others},
  journal={arXiv preprint arXiv:2604.07993},
  year={2026}
}
```

## Ackwnledgemments

This project draws inspiration from and builds upon several notable open-source projects, including: [StarVLA](https://github.com/starVLA/starVLA), [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), [HiMoE-VLA](https://github.com/ZhiyingDu/HiMoE-VLA), [LeRobot](https://github.com/huggingface/lerobot), [Humanoid Everyday](https://github.com/physical-superintelligence-lab/Humanoid-Everyday), [RoboCOIN](https://github.com/FlagOpen/RoboCOIN), [AgiBot-World](https://github.com/OpenDriveLab/AgiBot-World), and [OpenTrajBooster](https://github.com/OpenHelix-Team/OpenTrajBooster).
