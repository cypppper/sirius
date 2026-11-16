#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p experiments/fused_join_aggregate/bin
# Run under the repository CUDA 12 environment; native cubin for this Blackwell GPU.
"${CONDA_PREFIX}/bin/nvcc" -std=c++20 -O3 -lineinfo -arch=sm_120 \
  -I"${CONDA_PREFIX}/include/rapids" -I"${CONDA_PREFIX}/include" -I"${CONDA_PREFIX}/targets/x86_64-linux/include" \
  -L"${CONDA_PREFIX}/lib" -Xlinker -rpath -Xlinker "${CONDA_PREFIX}/lib" \
  experiments/fused_join_aggregate/bench.cu -lcudf -lrmm \
  -o experiments/fused_join_aggregate/bin/bench
