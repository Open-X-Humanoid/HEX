#!/bin/bash

# cd /mnt/dataset/vnwy44/code/HEX && ./scripts/download_datasets.sh

source /mnt/dataset/vnwy44/miniconda3/etc/profile.d/conda.sh
conda activate hex

export HF_TOKEN=

# download HEX, HE, RoboCOIN datasets, and process HEX and HE
base_dir=/mnt/dataset/vnwy44/data/bsh/eai_real_world
python hex/utils/download_dataset.py --base_dir ${base_dir}
python hex/utils/process_hex_and_he_dataset.py --parent_dir ${base_dir}

# download agibot dataset, and process it
base_dir=/mnt/dataset/vnwy44/data/bsh/eai_real_world_tmp
dst_root=/mnt/dataset/vnwy44/data/bsh/eai_real_world
python hex/utils/download_dataset_agibot2g1.py --base_dir ${base_dir}
cd ${base_dir}/Agibot2UnitreeG1Retarget
cat A2UG1_dataset.tar.gz.* | tar -xzf -

cd /mnt/dataset/vnwy44/code/HEX
python hex/utils/process_agibot_to_g1_dataset.py \
  --base_dir ${base_dir} \
  --dst_root ${dst_root}