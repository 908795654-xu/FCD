#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/bkai/CG2/SACQ_TSFM
cd "$ROOT"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-2}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

mkdir -p logs
PID_FILE="$ROOT/logs/electricity_ttm_seeds.pid"
LOG_POINTER="$ROOT/logs/electricity_ttm_seeds.logpath"

# 默认作为后台启动器；worker负责真正执行实验。
if [[ "${1:-}" != "--worker" ]]; then
    python - <<'PY'
import sys
from pathlib import Path
assert Path(sys.prefix).name == "sacq_tsfm", "Activate sacq_tsfm first"
PY

    bash -n "$ROOT/scripts/run_electricity_ttm_seeds.sh"

    if [[ -f "$PID_FILE" ]]; then
        read -r old_pid < "$PID_FILE"
        if [[ "$old_pid" =~ ^[0-9]+$ ]] &&
           kill -0 "$old_pid" 2>/dev/null; then
            echo "Process still exists: PID=$old_pid"
            exit 1
        fi
    fi

    # 全部检查通过后才启动，避免覆盖结果或中断检查点。
    for seed in 2027 2028; do
        for horizon in 96 720; do
            for head in linear sacq; do
                name="electricity_p${horizon}_ttm_${head}_seed${seed}"
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
    done

    stamp="$(date +%Y%m%d_%H%M%S)_$$"
    run_log="$ROOT/logs/electricity_ttm_seeds_${stamp}.log"

    nohup bash "$ROOT/scripts/run_electricity_ttm_seeds.sh" \
        --worker > "$run_log" 2>&1 < /dev/null &

    run_pid=$!
    printf '%s\n' "$run_pid" > "$PID_FILE"
    printf '%s\n' "$run_log" > "$LOG_POINTER"

    echo "Background PID: $run_pid"
    echo "Physical GPU: $CUDA_VISIBLE_DEVICES"
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
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Electricity / TTM / horizons=96,720 / heads=linear,sacq"
echo "Seeds=2027,2028; batch=8; loss=MSE; protocol=frozen_v1"
echo "lr=0.0001; weight_decay=0.01; epochs=50; patience=7"
date -Is

completed=0

for seed in 2027 2028; do
    for horizon in 96 720; do
        for head in linear sacq; do
            CURRENT_JOB="electricity_p${horizon}_ttm_${head}_seed${seed}"
            echo "START [$((completed + 1))/8]: $CURRENT_JOB"

            python -u scripts/train.py \
                --dataset electricity \
                --backbone ttm \
                --head "$head" \
                --prediction-length "$horizon" \
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
            echo "DONE [$completed/8]: $CURRENT_JOB"
            date -Is
        done
    done
done

[[ "$completed" -eq 8 ]]
echo "ALL_ELECTRICITY_TTM_SEEDS_FINISHED: 8"
date -Is