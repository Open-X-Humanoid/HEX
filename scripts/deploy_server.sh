#!/bin/bash

# cd /media/eai/Ark3/bsh/HEX && ./scripts/deploy_server.sh box 10093
# cd /media/eai/Ark3/bsh/HEX && ./scripts/deploy_server.sh toy 10093

set -e

# Usage: ./scripts/deploy_server.sh [box|toy] [port]
task_name="${1:-box}"
server_port="${2:-10093}"

case "$task_name" in
  box)
    model_path="/media/eai/Ark3/bsh/HEX/pretrained_models/hex/EAI_real_pick_up_box_2B/hex_ac100_30k_state_query_history2_ft/checkpoints/steps_30000_pytorch_model.pt"
    ;;
  toy)
    model_path="/media/eai/Ark3/bsh/HEX/pretrained_models/hex/EAI_real_pick_up_toy_2B/hex_ac100_30k_state_query_history2_ft/checkpoints/steps_30000_pytorch_model.pt"
    ;;
  *)
    echo "Usage: $0 [box|toy] [port]" >&2
    exit 2
    ;;
esac

source /home/eai/miniconda3/etc/profile.d/conda.sh
conda activate hex

exec python deployment/run_hex_server.py \
  --model_path "$model_path" \
  --port "$server_port" \
  --device cuda \
  --dtype bfloat16
