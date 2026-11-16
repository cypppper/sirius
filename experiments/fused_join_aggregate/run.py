"""Run each case serially; bench randomizes method order within each repetition."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repetitions", type=int, default=15)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--memory", action="store_true")
    args = ap.parse_args()
    cases = args.cases or [p.stem for p in sorted((ROOT / "data").glob("*.bin"))]
    destination = ROOT / "results" / "memory" if args.memory else ROOT / "results"
    destination.mkdir(exist_ok=True)
    for name in cases:
        print(f"Running {name}", flush=True)
        start = time.monotonic()
        result = subprocess.run(
            [
                str(ROOT / "bin/bench"),
                str(ROOT / "data" / f"{name}.bin"),
                str(args.repetitions),
            ]
            + (["all", "memory"] if args.memory else []),
            capture_output=True,
            text=True,
            timeout=900,
        )
        (destination / f"{name}.stderr").write_text(result.stderr)
        result.check_returncode()
        (destination / f"{name}.csv").write_text(result.stdout)
        print(f"Validated all four methods ({time.monotonic()-start:.1f}s)", flush=True)

    def output(cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    manifest = dict(
        commit=output(["git", "rev-parse", "HEAD"]),
        branch=output(["git", "branch", "--show-current"]),
        repetitions=args.repetitions,
        memory_instrumented=args.memory,
        gpu=output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,power.limit,clocks.sm,temperature.gpu,utilization.gpu",
                "--format=csv",
            ]
        ),
        nvcc=output(["nvcc", "--version"]),
        host=output(["uname", "-a"]),
        sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / "bench.cu", ROOT / "generate.py", ROOT / "bin/bench"]
        },
    )
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        manifest["packages"] = {
            p.stem: json.loads(p.read_text())["version"]
            for p in (Path(prefix) / "conda-meta").glob("*.json")
            if p.name.startswith(("libcudf-", "librmm-", "cuda-nvcc-"))
        }
    (destination / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
