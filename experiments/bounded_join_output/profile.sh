#!/usr/bin/env bash
# Separate instrumented full-SQL runs; excluded from primary timing statistics.
set -euo pipefail
cd "$(dirname "$0")/../.."
root="$PWD/experiments/bounded_join_output"
mkdir -p "$root/profiles"
for mode in native range_full bounded; do
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true \
    --output "$root/profiles/$mode" \
    "$CONDA_PREFIX/bin/python" "$root/run.py" --case uniform_2048 --mode "$mode" \
    --warmups 1 --repetitions 1 --name "profile_$mode"
  nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$root/profiles/$mode.nsys-rep" \
    > "$root/results/profile_${mode}_kernels.csv"
  nsys stats --force-export=true --report cuda_api_sum --format csv "$root/profiles/$mode.nsys-rep" \
    > "$root/results/profile_${mode}_api.csv"
done
# Retain portable CSV tables without local SQLite/report-processing banners.
"$CONDA_PREFIX/bin/python" - "$root/results" <<'PY'
from pathlib import Path
import sys
for path in Path(sys.argv[1]).glob("profile_*_*.csv"):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Time (%),"))
    path.write_text("\n".join(lines[start:]).rstrip() + "\n")
PY
