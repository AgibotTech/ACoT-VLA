#!/bin/bash
# sim_rtc_stream.sh <run-name> <episodes> [run_rtc_stream flags...]
#   Genie Sim against the server-owned RTC stream (serve_rtc_stream.py --rtc.clock external).
#   e.g. sim_rtc_stream.sh stream_fold 1
#   CFG=<benchmark yaml in container> overrides the task (default fold_towels)
#   PORT=<server port> (default 8001)
# Copy this file to ~/g2sim/bin and run_rtc_stream.py + rtc_stream_policy.py to
# ~/g2sim/genie_sim/harness (mounted at /workspace/harness in geniesim3).
set -e
NAME=$1; EP=${2:-1}; shift 2 || true
CFG=${CFG:-/workspace/source/geniesim_benchmark/src/geniesim_benchmark/config/g2op_manip_fold_towels.yaml}
PORT=${PORT:-8001}
HOST_IP=$(hostname -I | awk "{print \$1}")
mkdir -p ~/g2sim/runs; LOG=~/g2sim/runs/$NAME.log
APP="--config $CFG --benchmark.infer_host=$HOST_IP:$PORT --benchmark.num_episode=$EP --benchmark.record=true ${APP_EXTRA}"
CMD="cd /workspace/harness && ${PY:-python3} run_rtc_stream.py $* --trace /workspace/runs/$NAME.jsonl -- $APP"
echo "$CMD" | tee $LOG
docker exec -e DISPLAY=:1 geniesim3 bash -lc "$CMD" 2>&1 | tee -a $LOG
