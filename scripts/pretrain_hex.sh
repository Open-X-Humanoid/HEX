#!/bin/bash

# cd /media/bsh/HEX && ./scripts/pretrain_hex.sh

source /media/bsh/miniconda3/etc/profile.d/conda.sh
conda activate hex

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export action_input_dim=2

para_type=2B
base_vlm=pretrained_models/Qwen3-VL-2B-Instruct

dataset_name=EAI_real_world
data_root_dir=/media/bsh/data/pretrain
export HEX_PRETRAIN_DATA_ROOT=${data_root_dir}

vision_history_length=2
enable_mee=false
max_train_steps=200000
save_interval=100000
train_steps_k="$((max_train_steps / 1000))k"
run_id="hex_ac100_${train_steps_k}_state_query_history${vision_history_length}"
visible_devices=0,1,2,3,4,5,6,7
num_processes=8

CUDA_VISIBLE_DEVICES=${visible_devices} /media/bsh/miniconda3/envs/hex/bin/accelerate launch \
  --config_file hex/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
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
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100000 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.learning_rate.state_model 2e-5 \
  --trainer.learning_rate.action_model 2e-5 \
  --run_root_dir ./pretrained_models/hex_pretrained/${dataset_name}_${para_type} \
  --run_id ${run_id} \
  --wandb_project hex \
  --enable_mee ${enable_mee}
