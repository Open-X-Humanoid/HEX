#!/bin/bash

# cd /media/bsh/code/HEX && ./scripts/pretrain_hex.sh

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

dataset_name=g1_sonic_real_world_pick_cola
data_root_dir=/media/bsh/data/hex_data

vision_history_length=0
enable_mee=false
run_id=hex_ac100_30k_8gpu_state_query_history${vision_history_length}

# ✅ Launch training with Accelerate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /media/bsh/miniconda3/envs/hex/bin/accelerate launch \
  --config_file hex/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  hex/training/pretrain_hex.py \
  --config_yaml ./hex/config/training/hex_cotrain_eai_pretrain.yaml \
  --framework.name HEX \
  --framework.qwenvl.base_vlm ${base_vlm} \
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
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100000 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.learning_rate.state_model 2e-5 \
  --trainer.learning_rate.action_model 2e-5 \
  --run_root_dir ./pretrained_models/hex/${dataset_name}_${para_type} \
  --run_id ${run_id} \
  --wandb_project hex \
  --enable_mee ${enable_mee}