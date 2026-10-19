#!/usr/bin/env bash
# 隧道模式推理 Agent 启动脚本(替代旧的 server.sh / serve_policy.py 入口)。
#
# 用法:
#   ./scripts/tunnel.sh <CUDA_VISIBLE_DEVICES> <JOB_UUID> [GATEWAY_URL]
#
# 也支持环境变量:
#   CHALLENGE_TOKEN         登录凭据 (必填)
#   SIMUBOTIX_GATEWAY_URL   Gateway WS 地址 (与位置参数 GATEWAY_URL 二选一)
#   PI05_CONFIG             覆盖默认 openpi config (默认 pi05_genie_sim_10_mini_task_20260312)
#   PI05_CKPT_DIR           覆盖默认权重路径 (默认 checkpoints/30000)
#   PI05_PARALLELISM        单卡上并发的推理服务数 (默认 1),用于动态计算显存占比
#                           XLA_PYTHON_CLIENT_MEM_FRACTION = 0.9 / N (夹在 [0.10, 0.90])
#   XLA_PYTHON_CLIENT_MEM_FRACTION  显式设置则覆盖上面按并发数算出的值

set -euo pipefail

cart_num=${1:?"usage: $0 <CUDA_VISIBLE_DEVICES> <JOB_UUID> [GATEWAY_URL]"}
job_uuid=${2:?"usage: $0 <CUDA_VISIBLE_DEVICES> <JOB_UUID> [GATEWAY_URL]"}
gateway_url=${3:-${SIMUBOTIX_GATEWAY_URL:-}}

if [[ -z "${gateway_url}" ]]; then
  echo "error: 需要第三个位置参数 GATEWAY_URL 或设置 SIMUBOTIX_GATEWAY_URL 环境变量" >&2
  exit 1
fi
if [[ -z "${CHALLENGE_TOKEN:-}" ]]; then
  echo "error: 需要设置 CHALLENGE_TOKEN 环境变量" >&2
  exit 1
fi


# 按单卡并发数动态计算每个进程的显存上限:留 ~10% 余量后均分给 N 个服务。
# 单个推理服务运行时占用 < 8GB,24GB 卡 (如 4090) 可并发 3+ (N=3 → 0.30)。
parallelism=${PI05_PARALLELISM:-1}
mem_fraction=$(awk -v n="${parallelism}" 'BEGIN{f=0.9/n; if(f>0.9)f=0.9; if(f<0.1)f=0.1; printf "%.2f", f}')
# 按榜单选择权重: instruction/robust 共用同一权重, spatial / manip 各自独立。
board=${PI05_BOARD:-instruction}
case "${board}" in
  instruction|robust)
    board_config="pi05_genie_sim_instruction_and_robust_20260526"
    board_ckpt="checkpoints/instruction_and_robust_pi05" ;;
  spatial)
    board_config="pi05_genie_sim_spatial_20260528"
    board_ckpt="checkpoints/spatial_pi05" ;;
  manip)
    board_config="pi05_genie_sim_manip_20260613"
    board_ckpt="checkpoints/manipulation_pi05" ;;
  *)
    echo "error: 未知 PI05_BOARD='${board}' (取值: instruction|robust|spatial|manip)" >&2
    exit 1 ;;
esac

config=${PI05_CONFIG:-${board_config}}
ckpt_dir=${PI05_CKPT_DIR:-${board_ckpt}}
echo "[tunnel] board=${board} config=${config} ckpt_dir=${ckpt_dir}" >&2

export TF_NUM_INTRAOP_THREADS=16
export CUDA_VISIBLE_DEVICES=${cart_num}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-${mem_fraction}}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export PYTHONPATH=/root/openpi/src:${PYTHONPATH:-/app:/app/src}

GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/tunnel_agent.py \
  --access-token "${CHALLENGE_TOKEN}" \
  --job-uuid     "${job_uuid}" \
  --gateway-url  "${gateway_url}" \
  --config       "${config}" \
  --ckpt-dir     "${ckpt_dir}"
