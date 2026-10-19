# Genie Sim + ACoT-VLA-PI05 New Instance Setup

This recreates the AMAX 4GPU setup used for Genie Sim G2 manipulation evaluation.

The split is:

- ACoT-VLA-PI05 policy server runs on the host in a `uv` Python venv.
- Genie Sim Isaac Sim runtime runs in Docker container `geniesim3`.
- Genie Sim host CLI runs from `~/genie_sim/.venv`.
- ModelScope downloads use separate `~/genie_sim/.venv-modelscope`.

Reference AMAX state:

- Host: `amax-ws-03`, `192.168.1.194`.
- ACoT repo: `/home/datamentors/ACoT-VLA`.
- Genie Sim repo: `/home/datamentors/genie_sim`.
- Assets: `/home/datamentors/assets`.
- Checkpoint: `/home/datamentors/genie_sim/checkpoints/manipulation_pi05`.
- Container image: `registry.agibot.com/genie-sim/geniesim3:latest`.

## 1. Host Prereqs

Use an NVIDIA Linux instance with Docker Engine and NVIDIA Container Toolkit.

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

Install `uv` if missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Install ModelScope download tooling in its own venv later; do not mix it with ACoT or Genie Sim runtime envs.

## 2. Clone ACoT-VLA-PI05

```bash
cd ~
git clone https://github.com/datamentors/ACoT-VLA-PI05.git ACoT-VLA
cd ~/ACoT-VLA
uv python install 3.11
uv sync --python 3.11 --no-dev
```

Expected important dependency:

```bash
uv pip list --python .venv/bin/python | grep -E 'jax|torch|nvidia-nccl-cu12|openpi'
```

AMAX used `nvidia-nccl-cu12==2.27.3` to avoid Blackwell multi-GPU NCCL crashes.

## 3. Clone Genie Sim

```bash
cd ~
git clone https://github.com/AgibotTech/genie_sim.git genie_sim
cd ~/genie_sim
git checkout main
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e source/geniesim_cli
```

AMAX had local Genie Sim patches for GUI/debug behavior:

- `source/geniesim_benchmark/src/geniesim_benchmark/app/workflow/app_launcher.py`
- `source/geniesim_benchmark/src/geniesim_benchmark/config/params.py`
- `source/geniesim_benchmark/src/geniesim_benchmark/benchmark/policy/corobotpolicy.py`
- `source/geniesim_benchmark/src/geniesim_benchmark/plugins/output_system/local_recorder.py`

To copy the exact AMAX patch stack:

```bash
ssh datamentors@192.168.1.194 'cd /home/datamentors/genie_sim && git diff' > /tmp/genie_sim_amax.patch
cd ~/genie_sim
git apply /tmp/genie_sim_amax.patch
```

Those patches are not required for headless inference, but they are required to match AMAX GUI and trace behavior exactly.

## 4. Install Assets

Create the ModelScope venv:

```bash
cd ~/genie_sim
uv venv --python 3.11 .venv-modelscope
uv pip install --python .venv-modelscope/bin/python modelscope
```

Download Genie Sim assets:

```bash
cd ~
mkdir -p ~/assets
~/genie_sim/.venv-modelscope/bin/modelscope download \
  --repo-type dataset agibot_world/GenieSimAssets \
  --local-dir ~/assets \
  --include "background/common/**" \
            "background/home/home_3/**" \
            "background/warehouse/warehouse_1/**" \
            "background/market/market_1/**" \
            "background/market/market_2/**" \
            "background/study_room/study_4/**" \
            "background/room/room_3/**" \
            "background/kitchen/kitchen_1/**" \
            "background/laboratory/laboratory_2/**" \
            "background/popcorn/popcorn_1/**" \
            "robot/G2_omnipicker/**" \
            "objects/benchmark/**"
```

Install assets into the Genie Sim host CLI venv so `geniesim docker up` can find them:

```bash
uv pip install --python ~/genie_sim/.venv/bin/python -e ~/assets
```

If `~/assets/pyproject.toml` is missing, the assets download is incomplete or wrong.

## 5. Download PI0.5 Manipulation Checkpoint

```bash
cd ~/genie_sim
mkdir -p checkpoints
~/genie_sim/.venv-modelscope/bin/modelscope download \
  --repo-type dataset agibot_world/GenieSim3.0-Dataset \
  --local-dir . \
  --include "checkpoints/manipulation_pi05/**"
```

Expected path:

```text
~/genie_sim/checkpoints/manipulation_pi05
```

Symlink checkpoint into ACoT:

```bash
mkdir -p ~/ACoT-VLA/checkpoints
ln -sfn ~/genie_sim/checkpoints/manipulation_pi05 ~/ACoT-VLA/checkpoints/manipulation_pi05
ln -sfn ~/genie_sim/checkpoints/manipulation_pi05 ~/ACoT-VLA/checkpoints/manipulation
```

## 6. Start ACoT Policy Server

Use host terminal 1:

```bash
cd ~/ACoT-VLA
export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_triton_gemm=false"

.venv/bin/python scripts/serve_policy.py \
  --host 0.0.0.0 \
  --port 8000 \
  --env PI05_GENIE_SIM_MANIP \
  policy:checkpoint \
  --policy.config pi05_genie_sim_manip_20260613 \
  --policy.dir checkpoints/manipulation_pi05
```

Expected:

```text
server listening on 0.0.0.0:8000
```

First load takes about 30 seconds. AMAX warm inference logs were about 460-480 ms.

## 7. Start Genie Sim Docker

Use host terminal 2:

```bash
cd ~/genie_sim
source .venv/bin/activate
geniesim docker up
```

This starts container `geniesim3`, mounts:

- `~/genie_sim` at `/workspace`
- `~/assets` at `/geniesim_assets`
- Isaac Sim cache dirs under `~/docker/isaac-sim`

Enter container:

```bash
geniesim docker into
```

Inside container:

```bash
geniesim status
```

The entrypoint installs tier-1 peer packages inside the container. Host `geniesim status` may only show `geniesim_cli` and assets installed; that is normal.

## 8. Smoke-Test Policy Protocol

Inside `geniesim3`:

```bash
geniesim benchmark check-inference --host 172.17.0.1 --port 8000 --iters 1
```

`172.17.0.1` is the Docker bridge gateway to the host. Confirm if needed:

```bash
docker network inspect bridge -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}'
```

## 9. Run Benchmarks

Headless:

```bash
geniesim benchmark run g2op_manip_open_door \
  --infer-host=172.17.0.1:8000 \
  --app.headless=true \
  --benchmark.num_episode=1
```

GUI with noVNC:

```bash
# Host terminal, before run:
DISPLAY=:1 xhost +local:

# Inside geniesim3:
export DISPLAY=:1
geniesim benchmark run g2op_manip_sorting_packages \
  --infer-host=172.17.0.1:8000 \
  --app.experience=isaacsim.exp.full.kit \
  --app.on_demand_render=true \
  --app.record_video=true \
  --benchmark.record=true \
  --benchmark.num_episode=1
```

Open noVNC:

```text
http://<INSTANCE_IP>:6080
```

If noVNC is not installed/running on a new instance, use headless first. The GUI path needs X/VNC/TigerVNC/websockify configured like AMAX.

## 10. Useful Debug Commands

Policy server:

```bash
ps -ef | grep serve_policy.py | grep -v grep
tail -100 ~/ACoT-VLA/server.log 2>/dev/null || true
```

Container:

```bash
docker ps -a | grep geniesim3
docker logs -f geniesim3
```

Score:

```bash
docker exec geniesim3 cat /workspace/output/benchmark/home_g2_op/open_door/evaluate_ret_00.json
```

Stop app process without removing container:

```bash
for P in $(docker exec geniesim3 pgrep -f app.py); do
  docker exec geniesim3 kill -9 "$P"
done
```

Stop/remove container:

```bash
cd ~/genie_sim
source .venv/bin/activate
geniesim docker down
```

## 11. Troubleshooting

`geniesim docker up` says assets are not installed:

```bash
uv pip install --python ~/genie_sim/.venv/bin/python -e ~/assets
```

Missing USD asset:

```bash
~/genie_sim/.venv-modelscope/bin/modelscope download \
  --repo-type dataset agibot_world/GenieSimAssets \
  --local-dir ~/assets \
  --include "objects/benchmark/<missing_object>/**"
```

`geniesim` not found inside direct `docker exec`:

```bash
docker exec -u $(id -u):$(id -g) -e HOME=/home/isaac-sim -w /workspace geniesim3 bash -lc "geniesim status"
```

Policy server reachable from host but not container:

```bash
docker exec geniesim3 bash -lc 'python3 - <<PY
import socket
s=socket.create_connection(("172.17.0.1", 8000), timeout=5)
print("TCP_OK")
s.close()
PY'
```

## 12. What To Commit In ACoT-VLA-PI05

Commit the ACoT changes in this repo. Genie Sim changes belong in a Genie Sim fork or patch file, not this ACoT repo.

Suggested commit message:

```text
Import AMAX PI0.5 Genie Sim training and validation setup

- add Blackwell-safe NCCL override and refreshed uv lock
- add G2 motor/waist PI0.5 fine-tune configs
- add TensorBoard logging and validation metric relay to training
- add held-out checkpoint/action validation scripts
- disable async checkpointing to avoid AMAX checkpoint deadlock
- fix live G2 state remapping for model-order inference
- document Genie Sim + ACoT-VLA-PI05 replication setup
```
