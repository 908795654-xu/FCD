#!/usr/bin/env bash
export DATA=ETTh1
exec "$(dirname "$0")/dsacq_ablation_4gpu.sh" "$@"
