#!/bin/bash

# cd /media/bsh/HEX && ./scripts/fine_tune_hex.sh

# Activate the conda environment
source /media/bsh/miniconda3/etc/profile.d/conda.sh
conda activate hex

# Set distributed training environment variables
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export action_input_dim=2
# export WANDB_MODE=disabled
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3
# export PYTHONNOUSERSITE=1

para_type=2B
base_vlm=pretrained_models/Qwen3-VL-2B-Instruct

dataset_name=EAI_real_pick_up_toy   # EAI_real_pick_up_box ...
data_root_dir=/media/bsh/data/eval
export HEX_DATA_ROOT=${data_root_dir}

vision_history_length=2
enable_mee=false
max_train_steps=30000
save_interval=10000
train_steps_k="$((max_train_steps / 1000))k"
run_id=hex_ac100_${train_steps_k}_state_query_history${vision_history_length}_ft
pretrained_models_path=/media/bsh/code/HEX/pretrained_models_new/hex_pretrained/EAI_real_world_2B/hex_ac100_200k_8gpu_state_query_history2_v2/checkpoints/steps_200000_pytorch_model.pt
visible_devices=0,1,2,3,4,5,6,7
num_processes=1

# ✅ Launch fine-tuning with Accelerate
CUDA_VISIBLE_DEVICES=${visible_devices} /media/bsh/miniconda3/envs/hex/bin/accelerate launch \
  --config_file hex/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  hex/training/fine_tune_hex.py \
  --config_yaml ./hex/config/training/hex_cotrain_eai_ft.yaml \
  --framework.name HEX \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.pretrained_run_root_path ${pretrained_models_path} \
  --framework.action_model.action_hidden_dim 2 \
  --framework.action_model.action_model_type DiT-B \
  --framework.qwenvl.add_query True \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${dataset_name} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.need_state True \
  --datasets.vla_data.need_tag True \
  --datasets.vla_data.vision_history_length ${vision_history_length} \
  --trainer.freeze_modules "" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 10000 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.learning_rate.state_model 4e-5 \
  --trainer.learning_rate.action_model 4e-5 \
  --run_root_dir ./pretrained_models/hex/${dataset_name}_${para_type} \
  --run_id "${run_id}" \
  --wandb_project hex \
  --enable_mee ${enable_mee}
  