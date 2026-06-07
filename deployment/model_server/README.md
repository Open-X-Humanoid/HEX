
# start policy server


```bash

your_ckpt=./results/Checkpoints/1003_qwenfast/checkpoints/steps_50000_pytorch_model.pt

python deployment/model_server/run_hex_server.py \
    --model_path ${your_ckpt} \
    --port 10093 \
    --device cuda \
    --dtype bfloat16
```

`server_policy.py` is kept for compatibility with older commands.


# connect to policy server for debug

```bash
python deployment/model_server/debug_server_policy.py

# plus server_policy.py into your vla controler by ref to debug_server_policy.py
```

# HEX -> SONIC closed loop

Start the SONIC decoder / whole-body controller first:

```bash
cd /media/bsh/code/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh --input-type zmq_manager --zmq-host localhost --hand-type inspire real
```

Start the HEX model server:

```bash
python deployment/model_server/run_hex_server.py \
    --model_path ${your_ckpt} \
    --port 10093 \
    --device cuda \
    --dtype bfloat16
```

Start the HEX client loop that reads camera/state, calls the server, and publishes
motion tokens to SONIC:

```bash
python deployment/sonic/run_hex_sonic_closed_loop.py \
    --host 127.0.0.1 \
    --port 10093 \
    --checkpoint_path ${your_ckpt} \
    --prompt "pick up the cola"
```

Make sure the gear_sonic camera server is also running on `--camera_host/--camera_port`
(default `localhost:5555`), otherwise the client will wait for camera frames.

Use the keyboard publisher from GR00T-WholeBodyControl:

```bash
python /media/bsh/code/GR00T-WholeBodyControl/keyboard_pub.py
```

Keyboard controls:

- `k`: start / stop the C++ control loop.
- `i`: send the latent initial pose and switch from planner mode to pose mode.
- `p`: pause / resume HEX policy action publishing.
- `t <text>`: change the language prompt.
- `[` / `]`: toggle left / right hand closed for the initial pose.
