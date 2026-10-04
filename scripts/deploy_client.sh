#!/bin/bash

# cd /media/eai/Ark3/bsh/HEX && ./scripts/deploy_client.sh box 10093
# cd /media/eai/Ark3/bsh/HEX && ./scripts/deploy_client.sh toy 10093

set -e

# Usage: ./scripts/deploy_client.sh [box|toy] [port]
task_name="${1:-box}"
server_port="${2:-10093}"

case "$task_name" in
  box)
    client_script="examples/real_world/xrocs_infer_hex_tienkung3_pick_up_box.py"
    ;;
  toy)
    client_script="examples/real_world/xrocs_infer_hex_tienkung3_pick_up_toy.py"
    ;;
  *)
    echo "Usage: $0 [box|toy] [port]" >&2
    exit 2
    ;;
esac

source /home/eai/miniconda3/etc/profile.d/conda.sh
conda activate eai_env
source /opt/ros/jazzy/setup.bash

exec python "$client_script" \
  --host 127.0.0.1 \
  --port "$server_port" \
  --execute-step-count 20 \
  --arm-interpolation