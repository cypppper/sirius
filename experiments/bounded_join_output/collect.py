"""Verify portable results and collect profiling/provenance without GPU execution."""

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESULTS = ROOT / "results"


def read_profile(path):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Time (%),"))
    return list(csv.DictReader(lines[start:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-binary", action="store_true")
    args = ap.parse_args()
    outcomes = json.loads((RESULTS / "outcomes.json").read_text())
    protocol = json.loads((RESULTS / "protocol.json").read_text())
    assert len(outcomes) == len(protocol) == 99
    assert all(
        all(row[k] == value for k, value in job.items())
        for row, job in zip(outcomes, protocol)
    )
    assert len([r for r in outcomes if r["section"] == "main"]) == 72
    assert len([r for r in outcomes if r["section"] == "tile_sweep"]) == 18
    assert all(r["status"] == "passed" for r in outcomes if r["section"] != "pressure")
    assert len(list(csv.DictReader((RESULTS / "timings.csv").open()))) == 273
    hashes = {r["library_sha256"] for r in outcomes}
    assert len(hashes) == 1
    library_hash = next(iter(hashes))
    if args.verify_binary:
        library = REPO / "build/release/install/lib/libsirius.so.0.0.0"
        assert hashlib.sha256(library.read_bytes()).hexdigest() == library_hash
    for row in outcomes:
        assert (
            row["environment"]["gpu_execution"]
            and not row["environment"]["enable_duckdb_fallback"]
        )
        if row["mode"] == "bounded" and row["status"] == "passed":
            assert row["consumer_records"]
            assert all(
                r["max_tile_rows"] * r["bytes_per_row"] <= r["tile_budget"]
                for r in row["consumer_records"]
            )
    pressure = [r for r in outcomes if r["section"] == "pressure"]
    assert all(r["status"] == "passed" for r in pressure if r["mode"] == "bounded")
    assert all(
        r["status"] == "error"
        and "maximum retry limit (100)" in r["measurement"]["error"]
        for r in pressure
        if r["mode"] != "bounded"
    )
    capacity = json.loads((RESULTS / "capacity_results.json").read_text())
    bounded = next(r for r in capacity if r["mode"] == "bounded")
    assert bounded["status"] == "passed" and bounded["measurement"]["matches_reference"]
    assert sum(r["pairs"] for r in bounded["consumer_records"]) == 2**32
    capacity_evidence = {}
    for row in capacity:
        assert row["environment"]["library_sha256"] == library_hash
        path = RESULTS / ("capacity_" + row["mode"]) / "execution_evidence.txt"
        if path.exists():
            capacity_evidence[row["mode"]] = path.read_text()
    if len(capacity_evidence) == 3:
        (RESULTS / "capacity_evidence.json.gz").write_bytes(
            gzip.compress(json.dumps(capacity_evidence).encode(), mtime=0)
        )
    semantic_dispatch = {}
    for mode in ["bounded", "range_full"]:
        folder = RESULTS / f"semantics_{mode}"
        semantics = json.loads((folder / "results.json").read_text())
        assert len(semantics) == 13 and all(v["passed"] for v in semantics.values())
        assert (
            json.loads((folder / "environment.json").read_text())["library_sha256"]
            == library_hash
        )
        assert (
            "ERROR SUMMARY: 0 errors" in (RESULTS / f"memcheck_{mode}.txt").read_text()
        )
        logs = "\n".join(
            p.read_text(errors="replace") for p in (folder / "logs").glob("*.log")
        )
        if logs:

            def normalize(sql):
                return " ".join(sql.strip().rstrip(";").lower().split())

            labels = {normalize(v["sql"]): k for k, v in semantics.items()}
            current = None
            dispatch = {name: [] for name in semantics}
            selected = []
            for line in logs.splitlines():
                if "QueryBegin:" in line and " SQL: " in line:
                    current = labels.get(normalize(line.split(" SQL: ", 1)[1]))
                if "[BOUNDED_JOIN]" in line and current:
                    record = {
                        k: int(v) if v.isdigit() else v
                        for k, v in re.findall(r"(\w+)=([\w]+)", line)
                    }
                    dispatch[current].append(record)
                    selected.append(line)
            assert all(not dispatch[k] for k in semantics if k.endswith("declined"))
            assert all(dispatch[k] for k in semantics if not k.endswith("declined"))
            if mode == "bounded":
                for k in ["one_row_resume", "one_row_decimal"]:
                    assert (
                        dispatch[k][0]["probe_rows"] == 1
                        and dispatch[k][0]["tiles"] == 32
                    )
            semantic_dispatch[mode] = dispatch
            (folder / "execution_evidence.txt").write_text("\n".join(selected) + "\n")
    if semantic_dispatch:
        (RESULTS / "semantic_dispatch.json").write_text(
            json.dumps(semantic_dispatch, indent=2) + "\n"
        )
    tpch = json.loads((RESULTS / "tpch22_validation.json").read_text())
    for mode, row in tpch.items():
        assert len(row["results"]) == 22 and all(
            v["matches_cpu"] for v in row["results"].values()
        )
        assert row["environment"]["library_sha256"] == library_hash
    profile = []
    for mode in ["native", "range_full", "bounded"]:
        kernels = read_profile(RESULTS / f"profile_{mode}_kernels.csv")
        api = read_profile(RESULTS / f"profile_{mode}_api.csv")
        counts = {r["Name"]: int(r["Num Calls"]) for r in api}
        folder = RESULTS / f"profile_{mode}"
        # A checkout can retain profile_summary.json while omitting raw process folders.
        if folder.exists():
            environment = json.loads((folder / "environment.json").read_text())
            measurement = json.loads((folder / "results.json").read_text())
        else:
            previous = json.loads((RESULTS / "profile_summary.json").read_text())[
                "runs"
            ]
            old = next(r for r in previous if r["mode"] == mode)
            environment, measurement = old["environment"], old["measurement"]
        assert environment["library_sha256"] == library_hash
        profile.append(
            dict(
                mode=mode,
                kernel_launches=sum(int(r["Instances"]) for r in kernels),
                summed_kernel_ms=sum(int(r["Total Time (ns)"]) for r in kernels) / 1e6,
                cuda_stream_synchronize_calls=counts.get("cudaStreamSynchronize", 0),
                cuda_launch_kernel_calls=counts.get("cudaLaunchKernel", 0),
                environment=environment,
                measurement=measurement,
            )
        )
    (RESULTS / "profile_summary.json").write_text(
        json.dumps(
            dict(
                scope="Whole process, including pinning, one warmup and one measured SQL execution. Instrumented timings excluded from primary statistics.",
                runs=profile,
            ),
            indent=2,
        )
        + "\n"
    )
    sources = [
        "src/cuda/bounded_join_output.cu",
        "src/op/bounded_join_output.hpp",
        "src/op/sirius_physical_hash_join.cpp",
        "src/op/sirius_physical_hash_join.hpp",
        "src/pipeline/gpu_pipeline_task.cpp",
    ]
    verification = dict(
        library_sha256=library_hash,
        primary_processes=90,
        primary_timings=270,
        primary_warmups=90,
        pressure_processes=9,
        pressure_passed=3,
        pressure_failed_with_oom=6,
        capacity_matches=2**32,
        semantic_checks=26,
        sanitizer_errors=0,
        tpch_comparisons=44,
        profiling_processes=3,
        source_sha256={
            p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in sources
        },
    )
    (RESULTS / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
