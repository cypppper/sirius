"""Serial fresh-process A/B trials using the actual Sirius engine."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from collect_sirius_evidence import collect

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument(
        "--all22", action="store_true", help="Repeat all 22 SF1 queries over Parquet"
    )
    args = ap.parse_args()
    queries = list(range(1, 23)) if args.all22 else [12, 14]
    input_modes = [False] if args.all22 else [False, True]
    scales = [1] if args.all22 else args.scales
    campaign = "all22" if args.all22 else "e2e"
    csv_name = "all22_end_to_end.csv" if args.all22 else "end_to_end.csv"
    metadata_name = "all22_campaign.json" if args.all22 else "campaign.json"
    output = ROOT / "results/sirius"
    output.mkdir(exist_ok=True)
    rows = []
    environments = []
    for sf in scales:
        for pinned in input_modes:
            for trial in range(args.trials):
                if args.all22:
                    modes = (
                        ["ordinary", "fused"]
                        if trial % 2 == 0
                        else ["fused", "ordinary"]
                    )
                else:
                    base = ["ordinary", "eager", "fused"]
                    modes = base[trial % 3 :] + base[: trial % 3]
                for mode in modes:
                    name = f"{campaign}_t{trial}_p{int(pinned)}"
                    dest = output / f"{name}_{mode}_sf{sf}"
                    cmd = [
                        sys.executable,
                        str(ROOT / "run_sirius.py"),
                        "--sf",
                        str(sf),
                        "--mode",
                        mode,
                        "--queries",
                        *map(str, queries),
                        "--repetitions",
                        str(args.repetitions),
                        "--warmups",
                        "2",
                        "--name",
                        name,
                    ]
                    if pinned:
                        cmd.append("--pin")
                    print(" ".join(cmd), flush=True)
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=600
                    )
                    dest.mkdir(exist_ok=True, parents=True)
                    (dest / "stdout.txt").write_text(result.stdout)
                    (dest / "stderr.txt").write_text(result.stderr)
                    if result.returncode:
                        raise RuntimeError(
                            f"{mode} SF{sf} pinned={pinned}: {result.stderr}"
                        )
                    measurements = json.loads((dest / "results.json").read_text())
                    metadata = json.loads((dest / "environment.json").read_text())
                    environments.append(metadata)
                    counts = collect(dest)
                    for q in ["q12", "q14"]:
                        if q in measurements:
                            expected = args.repetitions + 2 if mode != "ordinary" else 0
                            assert counts[q] == expected, (dest, q, counts[q], expected)
                    if mode == "ordinary" and any(counts.values()):
                        raise RuntimeError("Ordinary baseline unexpectedly used fusion")
                    if mode != "ordinary":
                        evidence = (dest / "operator_evidence.txt").read_text()
                        executions = [
                            line
                            for line in evidence.splitlines()
                            if "[FUSED_JOIN_AGGREGATE] build_rows=" in line
                        ]
                        assert executions and all(
                            "original_build_child=1" in line for line in executions
                        )
                        assert all(
                            f"lazy={'true' if mode == 'fused' else 'false'}" in line
                            for line in executions
                        )
                    for query, measured in measurements.items():
                        for repetition, ms in enumerate(measured["milliseconds"]):
                            rows.append(
                                dict(
                                    sf=sf,
                                    pinned=int(pinned),
                                    trial=trial,
                                    mode=mode,
                                    query=query,
                                    repetition=repetition,
                                    milliseconds=ms,
                                )
                            )
                    with (output / csv_name).open("w") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
    assert len({e["library_sha256"] for e in environments}) == 1
    assert all(
        e["gpu_execution"] and not e["enable_duckdb_fallback"] for e in environments
    )
    (output / metadata_name).write_text(
        json.dumps(
            dict(
                trials=args.trials,
                repetitions=args.repetitions,
                warmups=2,
                scales=scales,
                queries=queries,
                build_orientation="original HASH_JOIN child 1 preserved",
                library_sha256=environments[0]["library_sha256"],
                ordinary_fused_same_binary=True,
                fallback_disabled=True,
                execution="full Sirius SQL, serial fresh-process trials",
                process_count=len(environments),
                measured_queries=len(rows),
                modes=(
                    ["ordinary", "fused"]
                    if args.all22
                    else ["ordinary", "eager", "fused"]
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
