"""Complete Sirius SQL trial: native, range-full, or bounded join output."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import duckdb

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LIBRARY = REPO / "build/release/install/lib/libsirius.so.0.0.0"
EXTENSION = (
    REPO / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
)
TPCH_SQL = """SELECT count(*), sum(CASE WHEN a.l_orderkey < b.l_orderkey
THEN abs(a.l_quantity-b.l_quantity) ELSE 0 END)
FROM lineitem a JOIN lineitem b ON a.l_partkey=b.l_partkey"""


def configure(dest, mode, budget_mib, fraction):
    template = (
        REPO / "experiments/fused_join_aggregate/sirius-kvikio.yaml"
    ).read_text()
    template = template.replace(
        "usage_limit_fraction: 0.75", f"usage_limit_fraction: {fraction}"
    )
    template = template.replace("num_threads: 4", "num_threads: 1")
    (dest / "config.yaml").write_text(template)
    os.environ.update(
        SIRIUS_CONFIG_FILE=str(dest / "config.yaml"),
        KVIKIO_COMPAT_MODE="ON",
        SIRIUS_LOG_DIR=str(dest / "logs"),
        SIRIUS_LOG_LEVEL="info",
        SIRIUS_EXP_JOIN_AGGREGATION="0",
        SIRIUS_EXP_JOIN_MAP_SUM="0",
        SIRIUS_EXP_JOIN_OUTPUT="" if mode == "native" else mode,
        SIRIUS_EXP_JOIN_TILE_BYTES=str(int(budget_mib * 1024**2)),
    )


def connect():
    gpu = duckdb.connect(
        config={"allow_unsigned_extensions": True, "threads": 2, "memory_limit": "2GB"}
    )
    gpu.execute(f"LOAD '{EXTENSION}'")
    gpu.execute("SET gpu_execution=true")
    gpu.execute("SET enable_duckdb_fallback=false")
    gpu.execute("SET enable_compressed_materialization=false")
    gpu.execute("SET expression_evaluator_strategy='ast_interpret'")
    return gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="uniform_8")
    ap.add_argument(
        "--mode", choices=["native", "range_full", "bounded"], default="bounded"
    )
    ap.add_argument("--budget-mib", type=float, default=32)
    ap.add_argument("--fraction", type=float, default=0.75)
    ap.add_argument("--width", type=int, choices=[2, 8], default=2)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--name", default="pilot")
    args = ap.parse_args()
    dest = ROOT / "results" / args.name
    dest.mkdir(parents=True, exist_ok=True)
    configure(dest, args.mode, args.budget_mib, args.fraction)
    metadata = vars(args) | dict(
        library_sha256=hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
        gpu_execution=True,
        enable_duckdb_fallback=False,
        expression_strategy="ast_interpret",
        compression=False,
        pinned=True,
        pipeline_threads=1,
        kvikio_compat_mode="ON",
    )
    (dest / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    gpu = connect()
    if args.case == "tpch_pairs":
        path = ROOT / "data/tpch_sf1/lineitem.parquet"
        gpu.execute(f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{path}')")
        gpu.execute(
            f"CALL pin_table('{path}', name='lineitem', tier='gpu', cols=['l_partkey','l_orderkey','l_quantity'])"
        )
        sql = TPCH_SQL
        expected = json.loads((ROOT / "results/tpch_pair_reference.json").read_text())[
            "result"
        ]
    else:
        manifest = json.loads((ROOT / "results/datasets.json").read_text())[args.case]
        expected = [manifest["expected_narrow" if args.width == 2 else "expected_wide"]]
        for side in ["a", "b"]:
            path = ROOT / "data" / f"{args.case}_{side}.parquet"
            gpu.execute(f"CREATE VIEW {side} AS SELECT * FROM read_parquet('{path}')")
            cols = ",".join(
                "'" + c + "'" for c in ["k"] + [f"v{i}" for i in range(args.width)]
            )
            gpu.execute(
                f"CALL pin_table('{path}', name='{side}', tier='gpu', cols=[{cols}])"
            )
        sums = ",".join(f"sum(a.v{i}*b.v{i})" for i in range(args.width))
        sql = f"SELECT {sums},count(*) FROM a JOIN b ON a.k=b.k"
    (dest / "query.sql").write_text(sql + ";\n")
    times = []
    result = dict(status="started", milliseconds=times)
    (dest / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    try:
        for i in range(-args.warmups, args.repetitions):
            start = time.perf_counter()
            actual = gpu.execute(sql).fetchall()
            elapsed = 1000 * (time.perf_counter() - start)
            canonical = json.loads(json.dumps(actual, default=str))
            assert canonical == expected, (canonical, expected)
            if i >= 0:
                times.append(elapsed)
            print(args.case, args.mode, i, elapsed, "validated", flush=True)
        result.update(status="passed", matches_reference=True, result=actual)
    except Exception as error:
        result.update(
            status="error",
            error_type=type(error).__name__,
            error=str(error),
            matches_reference=False,
        )
        raise
    finally:
        (dest / "results.json").write_text(
            json.dumps(result, default=str, indent=2) + "\n"
        )
        gpu.close()


if __name__ == "__main__":
    main()
