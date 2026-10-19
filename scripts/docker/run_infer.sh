#!/bin/bash
CURRENT_DIR=$(pwd)

echo "Project directory: $CURRENT_DIR"
xhost +local:
docker run -it --name acot_vla_infer \
    --init \
    --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network=host \
    --privileged \
    -e DISPLAY \
    -e OPENPI_DATA_HOME=/openpi_assets \
    -e IS_DOCKER=true \
    -e https_proxy=http://127.0.0.1:7897 \
    -e http_proxy=http://127.0.0.1:7897 \
    -e all_proxy=socks5://127.0.0.1:7897 \
    -v /dev/input:/dev/input:rw \
    -v $CURRENT_DIR:/app:rw \
    -v ${OPENPI_DATA_HOME:-~/.cache/openpi}:/openpi_assets:rw \
    -w /app \
    openpi_server:latest \
    bash 
