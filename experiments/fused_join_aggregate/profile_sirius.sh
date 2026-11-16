#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
experiment=experiments/fused_join_aggregate
mkdir -p "$experiment/profiles"
for workload in q12_sf10 q14_sf1; do
  query="${workload%%_sf*}"
  query="${query#q}"
  scale="${workload##*_sf}"
  for mode in ordinary eager fused; do
    capture="sirius_${workload}_${mode}_pinned"
    nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx \
      --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
      -o "$experiment/profiles/$capture" \
      python "$experiment/run_sirius.py" --sf "$scale" --mode "$mode" --queries "$query" \
      --repetitions 1 --warmups 2 --pin --pin-queries 12 14 --capture --name "$capture" \
      > "$experiment/results/sirius/$capture.capture.txt" 2>&1
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
      "$experiment/profiles/$capture.nsys-rep" \
      > "$experiment/results/sirius/$capture.kernels.txt" 2>&1
  done
done
