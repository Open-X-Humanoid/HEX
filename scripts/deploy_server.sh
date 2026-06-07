#!/bin/bash

# cd /media/bsh/code/HEX && ./scripts/deploy_server.sh

# Activate the conda environment
source /media/bsh/miniconda3/etc/profile.d/conda.sh
conda activate hex

python deployment/model_server/run_hex_server.py \
      --model_path /media/bsh/code/HEX/pretrained_models/hex/g1_sonic_real_world_pick_cola_2B/hex_ac100_3w_8gpu_state_query_history0_ft/checkpoints/steps_30000_pytorch_model.pt \
      --port 10093 \
      --device cuda \
      --dtype bfloat16