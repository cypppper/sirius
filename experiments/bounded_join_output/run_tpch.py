"""All 22 TPC-H SQL checks on the built engine, CPU fallback disabled."""

import hashlib
import json
from pathlib import Path
import sys
import time
import duckdb
from run import ROOT, REPO, LIBRARY, configure, connect

sys.path.insert(0, str(REPO / "test/tpch_performance"))
from queries import queries_for_scale_factor

sys.path.insert(0, str(REPO / "experiments/fused_join_aggregate"))
from run_sirius import equal

mode = sys.argv[1]
dest = ROOT / "results" / f"tpch22_{mode}"
dest.mkdir(parents=True, exist_ok=True)
configure(dest, mode, 32, 0.75)
gpu = connect()
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
    for con in [cpu, gpu]:
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{ROOT}/data/tpch_sf1/{table}.parquet')"
        )
results = {}
for name, sql in queries_for_scale_factor(1).items():
    expected = cpu.execute(sql).fetchall()
    start = time.perf_counter()
    actual = gpu.execute(sql).fetchall()
    ms = 1000 * (time.perf_counter() - start)
    assert equal(actual, expected), name
    results[name] = dict(matches_cpu=True, rows=len(actual), milliseconds=ms)
    (dest / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(name, ms, "validated", flush=True)
(dest / "environment.json").write_text(
    json.dumps(
        dict(
            mode=mode,
            library_sha256=hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
            gpu_execution=True,
            enable_duckdb_fallback=False,
            compression=False,
        ),
        indent=2,
    )
    + "\n"
)
gpu.close()
cpu.close()
