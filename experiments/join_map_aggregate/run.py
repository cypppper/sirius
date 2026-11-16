"""Actual Sirius SQL trial for ordinary join versus cuDF-AST match-map consumption."""

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO / "experiments/fused_join_aggregate"
sys.path.insert(0, str(OLD))
from run_sirius import equal
import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=int, default=1)
    ap.add_argument("--mode", choices=["ordinary", "maps"], default="maps")
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--name", default="smoke")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--compression", choices=["on", "off"], default="off")
    args = ap.parse_args()
    dest = ROOT / "results" / f"{args.name}_{args.mode}_sf{args.sf}"
    dest.mkdir(parents=True, exist_ok=True)
    os.environ["SIRIUS_CONFIG_FILE"] = str(OLD / "sirius-kvikio.yaml")
    os.environ["KVIKIO_COMPAT_MODE"] = "ON"
    os.environ["SIRIUS_EXP_JOIN_AGGREGATION"] = "0"
    os.environ["SIRIUS_EXP_JOIN_MAP_SUM"] = "1" if args.mode == "maps" else "0"
    os.environ["SIRIUS_LOG_DIR"] = str(dest / "logs")
    os.environ["SIRIUS_LOG_LEVEL"] = "info"
    gpu = duckdb.connect(
        config={"allow_unsigned_extensions": True, "threads": 2, "memory_limit": "2GB"}
    )
    cpu = duckdb.connect(config={"threads": 2, "memory_limit": "2GB"})
    extension = (
        REPO / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
    )
    gpu.execute(f"LOAD '{extension}'")
    gpu.execute("SET gpu_execution=true")
    gpu.execute("SET enable_duckdb_fallback=false")
    gpu.execute(
        "SET enable_compressed_materialization="
        + ("true" if args.compression == "on" else "false")
    )
    gpu.execute("SET expression_evaluator_strategy='ast_interpret'")
    for table in ["part", "partsupp", "lineitem"]:
        path = OLD / "data" / f"tpch_sf{args.sf}_parquet" / f"{table}.parquet"
        for con in [cpu, gpu]:
            con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
    if args.pin:
        pin_columns = {
            "lineitem": ["l_partkey", "l_shipdate", "l_extendedprice", "l_discount"],
            "part": ["p_partkey"],
            "partsupp": ["ps_partkey"],
        }
        for table, columns in pin_columns.items():
            path = OLD / "data" / f"tpch_sf{args.sf}_parquet" / f"{table}.parquet"
            names = ",".join("'" + c + "'" for c in columns)
            gpu.execute(
                f"CALL pin_table('{path}', name='{table}', tier='gpu', cols=[{names}])"
            )
    base = "FROM lineitem l JOIN part p ON l.l_partkey=p.p_partkey WHERE l.l_shipdate>=DATE '1994-08-01' AND l.l_shipdate<DATE '1994-09-01'"
    queries = {
        "sum_price": "SELECT sum(l.l_extendedprice) " + base,
        "sum_product": "SELECT sum(l.l_extendedprice*l.l_discount) " + base,
        "two_sums": "SELECT sum(l.l_extendedprice),sum(l.l_discount) " + base,
        "duplicates": "SELECT sum(l.l_extendedprice) FROM lineitem l JOIN partsupp ps ON l.l_partkey=ps.ps_partkey WHERE l.l_shipdate>=DATE '1994-08-01' AND l.l_shipdate<DATE '1994-09-01'",
    }
    results = {}
    for name, sql in queries.items():
        expected = cpu.execute(sql).fetchall()
        (dest / f"{name}.sql").write_text(sql + ";\n")
        (dest / f"{name}_reference.json").write_text(
            json.dumps(expected, default=str, indent=2) + "\n"
        )
        times = []
        for i in range(-args.warmups, args.repetitions):
            start = time.perf_counter()
            actual = gpu.execute(sql).fetchall()
            elapsed = 1000 * (time.perf_counter() - start)
            assert equal(actual, expected), (name, actual, expected)
            if i >= 0:
                times.append(elapsed)
        results[name] = dict(milliseconds=times, matches_cpu=True, result=actual)
        print(args.mode, args.sf, name, times, "validated", flush=True)
        (dest / "results.json").write_text(
            json.dumps(results, default=str, indent=2) + "\n"
        )
    metadata = dict(
        sf=args.sf,
        mode=args.mode,
        pinned=args.pin,
        compressed_materialization=args.compression == "on",
        expression_strategy="ast_interpret",
        gpu_execution=True,
        enable_duckdb_fallback=False,
        library_sha256=hashlib.sha256(
            (REPO / "build/release/install/lib/libsirius.so.0.0.0").read_bytes()
        ).hexdigest(),
    )
    (dest / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    gpu.close()
    cpu.close()


if __name__ == "__main__":
    main()
