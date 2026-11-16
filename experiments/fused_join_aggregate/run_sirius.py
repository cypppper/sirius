"""Actual Sirius SQL execution, CPU references, strict fallback policy, raw trials."""

import argparse
import ctypes
from decimal import Decimal
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import duckdb

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO / "test/tpch_performance"))
from queries import queries_for_scale_factor
from tpch_pin_columns import QUERY_COLUMNS


def equal(actual, expected):
    if len(actual) != len(expected):
        return False
    for arow, brow in zip(actual, expected):
        if len(arow) != len(brow):
            return False
        for a, b in zip(arow, brow):
            if isinstance(a, float) or isinstance(b, float):
                if (
                    a is None
                    or b is None
                    or not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-7)
                ):
                    return False
            elif a != b:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=int, default=1)
    ap.add_argument(
        "--mode", choices=["ordinary", "eager", "fused"], default="ordinary"
    )
    ap.add_argument("--queries", nargs="+", type=int, default=list(range(1, 23)))
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--pin-queries", nargs="+", type=int)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--name", default="validation")
    ap.add_argument("--log-level", choices=["info", "debug"], default="info")
    ap.add_argument(
        "--audit",
        action="store_true",
        help="Count matched-pair expression evaluations; diagnostic only",
    )
    args = ap.parse_args()
    dest = ROOT / "results" / "sirius" / f"{args.name}_{args.mode}_sf{args.sf}"
    dest.mkdir(parents=True, exist_ok=True)
    os.environ["SIRIUS_CONFIG_FILE"] = str(ROOT / "sirius-kvikio.yaml")
    os.environ["KVIKIO_COMPAT_MODE"] = "ON"
    os.environ["SIRIUS_EXP_JOIN_AGGREGATION"] = "0" if args.mode == "ordinary" else "1"
    os.environ["SIRIUS_EXP_JOIN_AGGREGATION_LAZY"] = (
        "1" if args.mode == "fused" else "0"
    )
    os.environ["SIRIUS_EXP_JOIN_AGGREGATION_AUDIT"] = "1" if args.audit else "0"
    os.environ["SIRIUS_LOG_DIR"] = str(dest / "logs")
    os.environ["SIRIUS_LOG_LEVEL"] = args.log_level
    extension = (
        REPO / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
    )
    con = duckdb.connect(
        config={"allow_unsigned_extensions": True, "threads": 2, "memory_limit": "2GB"}
    )
    con.execute(f"LOAD '{extension}'")
    con.execute("SET enable_duckdb_fallback=false")
    con.execute("SET gpu_execution=true")
    settings = con.execute(
        "SELECT current_setting('gpu_execution'), current_setting('enable_duckdb_fallback')"
    ).fetchone()
    assert settings == (True, False), settings
    # Same admission/optimization settings in both arms of the experiment.
    cpu = duckdb.connect(config={"threads": 2, "memory_limit": "2GB"})
    for table in [
        "region",
        "nation",
        "supplier",
        "customer",
        "part",
        "partsupp",
        "orders",
        "lineitem",
    ]:
        source = ROOT / "data" / f"tpch_sf{args.sf}_parquet" / f"{table}.parquet"
        for connection in [con, cpu]:
            connection.execute(
                f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{source}')"
            )
    queries = queries_for_scale_factor(args.sf)
    pin_start = time.perf_counter()
    if args.pin:
        selected = {}
        for query in args.pin_queries or args.queries:
            for table, columns in QUERY_COLUMNS[query].items():
                selected.setdefault(table, set()).update(columns)
        for table, columns in selected.items():
            source = ROOT / "data" / f"tpch_sf{args.sf}_parquet" / f"{table}.parquet"
            names = ",".join("'" + c + "'" for c in sorted(columns))
            con.execute(
                f"CALL pin_table('{source}', name='{table}', tier='gpu', cols=[{names}])"
            )
    pin_seconds = time.perf_counter() - pin_start if args.pin else 0.0
    outputs = {}
    for number in args.queries:
        name = f"q{number}"
        query = queries[name]
        (dest / f"{name}.sql").write_text(query + ";\n")
        reference = cpu.execute(query).fetchall()
        reference_text = json.dumps(reference, default=str, indent=2) + "\n"
        if len(reference_text) > 256000:
            (dest / f"{name}_reference.json.gz").write_bytes(
                gzip.compress(reference_text.encode(), mtime=0)
            )
        else:
            (dest / f"{name}_reference.json").write_text(reference_text)
        elapsed = []
        for iteration in range(-args.warmups, args.repetitions):
            profiler = None
            if args.capture and iteration == 0:
                profiler = ctypes.CDLL(
                    str(Path(os.environ["CONDA_PREFIX"]) / "lib/libcudart.so")
                )
                assert profiler.cudaProfilerStart() == 0
            start = time.perf_counter()
            actual = con.execute(query).fetchall()
            duration = (time.perf_counter() - start) * 1000
            if profiler is not None:
                assert profiler.cudaProfilerStop() == 0
            if not equal(actual, reference):
                (dest / f"{name}_actual.json").write_text(
                    json.dumps(actual, default=str, indent=2) + "\n"
                )
                raise AssertionError(f"{name} result mismatch")
            if iteration >= 0:
                elapsed.append(duration)
        outputs[name] = {
            "milliseconds": elapsed,
            "rows": len(reference),
            "matches_cpu": True,
        }
        (dest / "results.json").write_text(json.dumps(outputs, indent=2) + "\n")
        print(args.mode, args.sf, name, elapsed, "validated", flush=True)
    info = dict(
        sf=args.sf,
        mode=args.mode,
        gpu_execution=True,
        enable_duckdb_fallback=False,
        pin=args.pin,
        pin_seconds=pin_seconds,
        log_level=args.log_level,
        build_orientation="original HASH_JOIN child 1 preserved",
        lazy_aggregation=args.mode == "fused",
        expression_audit=args.audit,
        kvikio_compat_mode="ON",
        duckdb_version=duckdb.__version__,
        gpu_snapshot=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,clocks.sm",
                "--format=csv",
            ],
            text=True,
        ).strip(),
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(
            (REPO / "build/release/install/lib/libsirius.so.0.0.0").read_bytes()
        ).hexdigest(),
    )
    (dest / "environment.json").write_text(json.dumps(info, indent=2) + "\n")
    con.close()
    cpu.close()


if __name__ == "__main__":
    main()
