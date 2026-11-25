#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/bkai/CG2/SACQ_TSFM

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-2}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/home/bkai/CG2/SACQ_TSFM
export PYTHONUNBUFFERED=1

SEED=2026
CURRENT_JOB=preflight
LOG_DIR="logs/stage1_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"

trap 'code=$?; echo "FAILED: $CURRENT_JOB, exit=$code; logs=$LOG_DIR"; exit "$code"' ERR
trap 'echo "INTERRUPTED: $CURRENT_JOB"; exit 130' INT
trap 'echo "TERMINATED: $CURRENT_JOB"; exit 143' TERM

python - <<'PY'
import sys
from pathlib import Path
import torch

print("Python:", sys.executable)
assert Path(sys.prefix).name == "sacq_tsfm", "Activate sacq_tsfm first"
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
PY

JOBS=(
    "weather ttm 96 linear 128"
    "weather ttm 96 sacq 128"
    "weather ttm 720 linear 128"
    "weather ttm 720 sacq 128"
    "weather chronos 720 linear 128"
    "weather chronos 720 sacq 128"
    "weather timer 720 linear 128"
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

# 有同名输出时停止，避免覆盖旧实验。
for job in "${JOBS[@]}"; do
    read -r dataset backbone horizon head batch <<< "$job"
    name="${dataset}_p${horizon}_${backbone}_${head}_seed${SEED}"

    for suffix in "" "_frozen_v1"; do
        for file in \
            "results/formal/${name}${suffix}.json" \
            "checkpoints/formal/${name}${suffix}.pt"
        do
            if [[ -e "$file" ]]; then
                echo "Existing output: $file"
                echo "Stopped before training."
                exit 1
            fi
        done
    done
done

printf '%s\n' "${JOBS[@]}" > "$LOG_DIR/jobs.txt"
echo "Logs: $LOG_DIR"
echo "GPU=$CUDA_VISIBLE_DEVICES; seed=$SEED; protocol=frozen_v1"
echo "Weather batch=128; Electricity batch=8"
echo "AdamW; MSE; lr=0.0001; weight_decay=0.01"
echo "Epoch limit=50; patience=7"

completed=0

for job in "${JOBS[@]}"; do
    read -r dataset backbone horizon head batch <<< "$job"
    name="${dataset}_p${horizon}_${backbone}_${head}_seed${SEED}"
    CURRENT_JOB="$name"

    echo "START [$((completed + 1))/${#JOBS[@]}]: $name"
    date -Is

    python -u scripts/batch_memory_test.py \
        --dataset "$dataset" --backbone "$backbone" \
        --head "$head" --prediction-length "$horizon" \
        --batch-size "$batch" --device cuda:0 --seed "$SEED" \
        2>&1 | tee "$LOG_DIR/${name}_memory.log"

    grep -q '^BATCH_MEMORY_TEST_OK' "$LOG_DIR/${name}_memory.log"

    python -u scripts/train.py \
        --dataset "$dataset" --backbone "$backbone" \
        --head "$head" --prediction-length "$horizon" \
        --device cuda:0 --seed "$SEED" --batch-size "$batch" \
        --epochs 50 --patience 7 \
        --learning-rate 0.0001 --weight-decay 0.01 \
        --num-workers 0 --training-protocol frozen_v1 \
        2>&1 | tee "$LOG_DIR/${name}_train.log"

    grep -q '^FORMAL_TRAINING_OK' "$LOG_DIR/${name}_train.log"

    completed=$((completed + 1))
    echo "$name" >> "$LOG_DIR/completed.txt"
    echo "DONE [$completed/${#JOBS[@]}]: $name"
done

if [[ "$completed" -ne "${#JOBS[@]}" ]]; then
    echo "INCOMPLETE: $completed/${#JOBS[@]}"
    exit 1
fi

echo "ALL_STAGE1_JOBS_FINISHED: $completed"
echo "Logs: $LOG_DIR"
date -Is