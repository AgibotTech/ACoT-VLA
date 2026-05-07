FROM nvcr.io/nvidia/cuda@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
ENV GIT_LFS_SKIP_SMUDGE=1
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/
WORKDIR /app
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    git git-lfs build-essential clang \
    pkg-config \
    libgl1 libglib2.0-0 \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*
RUN git config --global http.postBuffer 524288000 && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999
ARG http_proxy
ARG https_proxy
ARG all_proxy
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/.venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_HTTP_TIMEOUT=600 \
    HF_ENDPOINT=https://hf-mirror.com
RUN uv venv --python python3.11 $UV_PROJECT_ENVIRONMENT
COPY pyproject.toml uv.lock LICENSE README.md ./
COPY packages ./packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
    --no-dev \
    --no-install-project \
    --no-build-package av \
    --verbose
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-build-package av
