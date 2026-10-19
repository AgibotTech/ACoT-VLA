cart_num=${1}
port=${2}
env=${3}

export TF_NUM_INTRAOP_THREADS=16
export CUDA_VISIBLE_DEVICES=${cart_num}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.48
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_autotune_level=0"

export PYTHONPATH=/root/openpi/src:${PYTHONPATH:-/app:/app/src}
GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/serve_policy.py --env ${env} --port ${port}
