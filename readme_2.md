
source /media/bsh/miniconda3/etc/profile.d/conda.sh
conda activate hex
python deployment/model_server/run_hex_server.py \
      --model_path /media/bsh/code/HEX/pretrained_models/hex/g1_sonic_real_world_pick_cola_2B/hex_ac100_3w_8gpu_state_query_history0_ft/checkpoints/steps_30000_pytorch_model.pt \
      --port 10093 \
      --device cuda \
      --dtype bfloat16

python deployment/run_starvla_inference.py \
  --stats-path /media/bsh/code/HEX/pretrained_models/hex/g1_sonic_real_world_pick_cola_2B/hex_ac100_3w_8gpu_state_query_history0_ft/dataset_statistics.json \
  --unnorm-key unitree_g1_sonic \
  --policy-port 5550 \
  --prompt "Walk forward, grab the cola and throw into the trash bin" \
  --action-publish-rate 50 \
  --camera-host 192.168.123.164 \
  --camera-port 5555