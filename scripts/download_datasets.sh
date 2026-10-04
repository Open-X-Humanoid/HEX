#!/bin/bash

# cd /media/bsh/HEX && ./scripts/download_datasets.sh

source /media/bsh/miniconda3/etc/profile.d/conda.sh
conda activate hex

export HF_TOKEN=

# download all data
base_dir=/media/bsh/data
python hex/utils/download_dataset.py --base_dir ${base_dir}