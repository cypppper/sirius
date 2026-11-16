#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
case_name="${1:-q14_sf5}"
experiment=experiments/fused_join_aggregate
mkdir -p "$experiment/profiles"
nsys profile --sample=none --cpuctxsw=none --trace=cuda --force-overwrite=true \
  -o "$experiment/profiles/$case_name" \
  "$experiment/bin/bench" "$experiment/data/$case_name.bin" 1 all \
  > "$experiment/results/nsys_capture.txt" 2>&1
nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv \
  "$experiment/profiles/$case_name.nsys-rep" > "$experiment/results/nsys_summary.txt" 2>&1
