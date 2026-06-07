#!/bin/bash

# cd /mnt/dataset/vnwy44/code/HEX && ./scripts/libero/eval_libero.sh

source /mnt/dataset/vnwy44/miniconda3/etc/profile.d/conda.sh
conda activate hex

export CUDA_VISIBLE_DEVICES=0
ckpt_root=./pretrained_models/hex/libero_all_2B
ckpt_path=hex_ac8_3w_8gpu_state_history0_all_camera_mee/checkpoints/steps_30000_pytorch_model.pt
run_id=$(echo "$ckpt_path" | cut -d'/' -f1)
your_ckpt="$ckpt_root/$ckpt_path"
log_path="$ckpt_root/$run_id"
task_suite_names=$(basename "$ckpt_root" | sed 's/_2B.*//')
echo $task_suite_names
if [[ "$task_suite_names" == "libero_all" ]]; then
    task_suite_names=("libero_goal" "libero_spatial" "libero_object" "libero_10")
else
    task_suite_names=("$task_suite_names")
fi
base_port=10093

export LIBERO_HOME=/root/workspace/code/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"
host="127.0.0.1"
base_port=10093
unnorm_key="franka"

for task_suite_name in "${task_suite_names[@]}"; do
    python ./examples/LIBERO/eval_libero_direct.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port $base_port \
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_out_path" \
        --args.log_path ${log_path}
done
