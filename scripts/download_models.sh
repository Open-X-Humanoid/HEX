#!/bin/bash

# cd /mnt/dataset/vnwy44/code/HEX && ./scripts/download_models.sh

source /mnt/dataset/vnwy44/miniconda3/etc/profile.d/conda.sh
conda activate hex

base_dir=/mnt/dataset/vnwy44/model/bsh
python hex/utils/download_model_qwen.py --base_dir ${base_dir}
