#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/bkai/CG2/SACQ_TSFM

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-2}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/home/bkai/CG2/SACQ_TSFM
export PYTHONUNBUFFERED=1

CURRENT_JOB=preflight
LOG_DIR="logs/stage1_resume_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"

trap 'code=$?; echo "FAILED: $CURRENT_JOB exit=$code logs=$LOG_DIR"; exit "$code"' ERR
trap 'echo "INTERRUPTED: $CURRENT_JOB"; exit 130' INT
trap 'echo "TERMINATED: $CURRENT_JOB"; exit 143' TERM

python - <<'PY'
import sys
from pathlib import Path
import torch

assert Path(sys.prefix).name == "sacq_tsfm", "Activate sacq_tsfm first"
assert torch.cuda.is_available(), "CUDA unavailable"
print("Python:", sys.executable)
print("GPU:", torch.cuda.get_device_name(0))
PY

JOBS=(
    "weather timer 720 sacq 128"
    "electricity ttm 96 linear 8"
    "electricity ttm 96 sacq 8"
    "electricity ttm 720 linear 8"
    "electricity ttm 720 sacq 8"
    "electricity chronos 720 linear 8"
    "electricity chronos 720 sacq 8"
    "electricity timer 720 linear 8"
    "electricity timer 720 sacq 8"
)

# 不覆盖任何已有结果或检查点。
for job in "${JOBS[@]}"; do
    read -r dataset backbone horizon head batch <<< "$job"
    name="${dataset}_p${horizon}_${backbone}_${head}_seed2026"

    for suffix in "" "_frozen_v1"; do
        for file in \
            "results/formal/${name}${suffix}.json" \
            "checkpoints/formal/${name}${suffix}.pt"
        do
            if [[ -e "$file" ]]; then
                echo "Existing output: $file"
                exit 1
            fi
        done
    done
done

printf '%s\n' "${JOBS[@]}" > "$LOG_DIR/jobs.txt"
echo "Logs: $LOG_DIR"
echo "Remaining jobs=9; Weather batch=128; Electricity batch=8"
echo "Seed=2026; protocol=frozen_v1; loss=MSE"
echo "lr=0.0001; weight_decay=0.01; epochs=50; patience=7"

completed=0

for job in "${JOBS[@]}"; do
    read -r dataset backbone horizon head batch <<< "$job"
    name="${dataset}_p${horizon}_${backbone}_${head}_seed2026"
    CURRENT_JOB="$name"

    echo "START [$((completed + 1))/9]: $name"
    date -Is

    python -u scripts/batch_memory_test.py \
        --dataset "$dataset" --backbone "$backbone" \
        --head "$head" --prediction-length "$horizon" \
        --batch-size "$batch" --device cuda:0 --seed 2026 \
        2>&1 | tee "$LOG_DIR/${name}_memory.log"

    grep -q '^BATCH_MEMORY_TEST_OK' "$LOG_DIR/${name}_memory.log"

    python -u scripts/train.py \
        --dataset "$dataset" --backbone "$backbone" \
        --head "$head" --prediction-length "$horizon" \
        --device cuda:0 --seed 2026 --batch-size "$batch" \
        --epochs 50 --patience 7 \
        --learning-rate 0.0001 --weight-decay 0.01 \
        --num-workers 0 --training-protocol frozen_v1 \
        2>&1 | tee "$LOG_DIR/${name}_train.log"

    grep -q '^FORMAL_TRAINING_OK' "$LOG_DIR/${name}_train.log"

    completed=$((completed + 1))
    echo "$name" >> "$LOG_DIR/completed.txt"
    echo "DONE [$completed/9]: $name"
done

[[ "$completed" -eq 9 ]]
echo "ALL_REMAINING_JOBS_FINISHED: $completed"
date -Is 