#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/bkai/CG2/SACQ_TSFM
cd "$ROOT"

BACKBONE="${1:-}"
MODE="${2:-}"

case "$BACKBONE" in
    chronos|timer) ;;
    *) echo "Usage: GPU_ID=N bash $0 chronos|timer"; exit 1 ;;
esac

: "${GPU_ID:?Set GPU_ID explicitly}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

mkdir -p logs
PREFIX="$ROOT/logs/electricity_${BACKBONE}_seeds"
PID_FILE="${PREFIX}.pid"
LOG_POINTER="${PREFIX}.logpath"

if [[ "$MODE" != "--worker" ]]; then
    python - <<'PY'
import sys
from pathlib import Path
assert Path(sys.prefix).name == "sacq_tsfm", "Activate sacq_tsfm first"
PY

    bash -n "$ROOT/scripts/run_electricity_backbone_seeds.sh"

    if [[ -f "$PID_FILE" ]]; then
        read -r old_pid < "$PID_FILE"
        if [[ "$old_pid" =~ ^[0-9]+$ ]] &&
           kill -0 "$old_pid" 2>/dev/null; then
            echo "Process still exists: PID=$old_pid"
            exit 1
        fi
    fi

    # 检查全部目标，避免覆盖结果或中断检查点。
    for seed in 2027 2028; do
        for head in linear sacq; do
            name="electricity_p720_${BACKBONE}_${head}_seed${seed}"
            for suffix in "" "_frozen_v1"; do
                for file in \
                    "results/formal/${name}${suffix}.json" \
                    "checkpoints/formal/${name}${suffix}.pt"
                do
                    if [[ -e "$file" ]]; then
                        echo "Existing output: $file"
                        echo "Stopped without overwriting."
                        exit 1
                    fi
                done
            done
        done
    done

    stamp="$(date +%Y%m%d_%H%M%S)_$$"
    run_log="${PREFIX}_${stamp}.log"

    nohup bash "$ROOT/scripts/run_electricity_backbone_seeds.sh" \
        "$BACKBONE" --worker \
        > "$run_log" 2>&1 < /dev/null &

    run_pid=$!
    printf '%s\n' "$run_pid" > "$PID_FILE"
    printf '%s\n' "$run_log" > "$LOG_POINTER"

    echo "Backbone: $BACKBONE"
    echo "Physical GPU: $GPU_ID"
    echo "Background PID: $run_pid"
    echo "Log: $run_log"
    exit 0
fi

CURRENT_JOB=preflight
trap 'code=$?; echo "FAILED: $CURRENT_JOB exit=$code"; exit "$code"' ERR
trap 'echo "INTERRUPTED: $CURRENT_JOB"; exit 130' INT
trap 'echo "TERMINATED: $CURRENT_JOB"; exit 143' TERM

python - <<'PY'
import sys
from pathlib import Path
import torch

assert Path(sys.prefix).name == "sacq_tsfm"
assert torch.cuda.is_available(), "CUDA unavailable"
print("Python:", sys.executable)
print("Logical cuda:0:", torch.cuda.get_device_name(0))
PY

echo "Electricity / $BACKBONE / horizon=720 / batch=8"
echo "Seeds=2027,2028; heads=linear,sacq"
echo "Protocol=frozen_v1; loss=MSE; optimizer=AdamW"
echo "lr=0.0001; weight_decay=0.01; epochs=50; patience=7"
date -Is

completed=0

for seed in 2027 2028; do
    for head in linear sacq; do
        CURRENT_JOB="electricity_p720_${BACKBONE}_${head}_seed${seed}"
        echo "START [$((completed + 1))/4]: $CURRENT_JOB"

        python -u scripts/train.py \
            --dataset electricity \
            --backbone "$BACKBONE" \
            --head "$head" \
            --prediction-length 720 \
            --device cuda:0 \
            --seed "$seed" \
            --batch-size 8 \
            --epochs 50 \
            --patience 7 \
            --learning-rate 0.0001 \
            --weight-decay 0.01 \
            --num-workers 0 \
            --training-protocol frozen_v1

        result="results/formal/${CURRENT_JOB}_frozen_v1.json"
        if [[ ! -s "$result" ]]; then
            echo "Missing result: $result"
            exit 1
        fi

        completed=$((completed + 1))
        echo "DONE [$completed/4]: $CURRENT_JOB"
        date -Is
    done
done

[[ "$completed" -eq 4 ]]
echo "ALL_${BACKBONE}_SEEDS_FINISHED: 4"
date -Is