#!/usr/bin/env bash
# Background: phase1 (96,720) then phase2 (192,336)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=logs/ett4_backbone_flatten_vs_sar
mkdir -p "$OUT"
chmod +x scripts/experiments/run_ett4_backbone_flatten_vs_sar.sh

nohup bash scripts/experiments/run_ett4_backbone_flatten_vs_sar.sh all \
  > "$OUT/nohup_launcher.log" 2>&1 &

echo "PID=$!"
echo "launcher: $OUT/nohup_launcher.log"
echo "master:   $OUT/master.log"
echo "summary:  $OUT/summary.csv"
echo "tasks/phase: 48 (4 ds × 2 pl × 3 models × 2 heads), total 96"
