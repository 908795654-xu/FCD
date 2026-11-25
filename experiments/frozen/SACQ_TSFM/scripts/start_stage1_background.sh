#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/bkai/CG2/SACQ_TSFM
cd "$ROOT"

GPU="${GPU_ID:-2}"
mkdir -p logs

# 确保使用准备好的环境。
python - <<'PY'
import sys
from pathlib import Path
assert Path(sys.prefix).name == "sacq_tsfm", "Run conda activate sacq_tsfm first"
PY

bash -n scripts/run_stage1.sh

# 防止重复启动。
if [[ -f logs/stage1_background.pid ]]; then
    read -r old_pid < logs/stage1_background.pid
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "A previous process is still alive: PID=$old_pid"
        echo "Check it before starting another run."
        exit 1
    fi
fi

# 防止此前的前台批次仍在运行。
if pgrep -u "$(id -u)" -af \
    '(^|/|[[:space:]])(run_stage1\.sh|scripts/train\.py)([[:space:]]|$)'
then
    echo "Existing training process found. Stop it before restarting."
    exit 1
fi

JOBS=(
    "weather ttm 96 linear"
    "weather ttm 96 sacq"
    "weather ttm 720 linear"
    "weather ttm 720 sacq"
    "weather chronos 720 linear"
    "weather chronos 720 sacq"
    "weather timer 720 linear"
    "weather timer 720 sacq"
    "electricity ttm 96 linear"
    "electricity ttm 96 sacq"
    "electricity ttm 720 linear"
    "electricity ttm 720 sacq"
    "electricity chronos 720 linear"
    "electricity chronos 720 sacq"
    "electricity timer 720 linear"
    "electricity timer 720 sacq"
)

# 已完成的正式结果不能覆盖。
# 若有已完成组，停止，由我们核对后安排剩余任务。
for job in "${JOBS[@]}"; do
    read -r dataset backbone horizon head <<< "$job"
    name="${dataset}_p${horizon}_${backbone}_${head}_seed2026"

    for suffix in "" "_frozen_v1"; do
        result="results/formal/${name}${suffix}.json"
        if [[ -e "$result" ]]; then
            echo "Completed result exists: $result"
            echo "Stopped without changing it."
            exit 1
        fi
    done
done

# 没有正式结果的检查点属于未完成组，移动到独立目录保留。
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
ARCHIVE="checkpoints/interrupted/stage1_$STAMP"

for job in "${JOBS[@]}"; do
    read -r dataset backbone horizon head <<< "$job"
    name="${dataset}_p${horizon}_${backbone}_${head}_seed2026"

    for suffix in "" "_frozen_v1"; do
        checkpoint="checkpoints/formal/${name}${suffix}.pt"
        if [[ -e "$checkpoint" ]]; then
            mkdir -p "$ARCHIVE"
            mv -n -- "$checkpoint" "$ARCHIVE/"
            echo "Preserved: $ARCHIVE/$(basename "$checkpoint")"
        fi
    done
done

RUN_LOG="$ROOT/logs/stage1_background_$STAMP.log"

# nohup忽略SSH挂断信号；输入和输出均脱离终端。
nohup env GPU_ID="$GPU" \
    bash "$ROOT/scripts/run_stage1.sh" \
    > "$RUN_LOG" 2>&1 < /dev/null &

RUN_PID=$!
printf '%s\n' "$RUN_PID" > logs/stage1_background.pid
printf '%s\n' "$RUN_LOG" > logs/stage1_background.logpath

echo "Started background process: PID=$RUN_PID"
echo "Physical GPU: $GPU"
echo "Log: $RUN_LOG"
echo "Weather batch=128; Electricity batch=8"
echo "Check the log to confirm training starts successfully."