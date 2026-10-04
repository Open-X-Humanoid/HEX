#!/bin/bash

# cd /mnt/dataset/vnwy44/code/HEX && ./scripts/download_models.sh

source /media/bsh/miniconda3/etc/profile.d/conda.sh
conda activate hex

base_dir=/media/bsh/HEX/pretrained_models
python hex/utils/download_model_qwen.py --base_dir ${base_dir}
