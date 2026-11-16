"""Compute one independent CPU reference for the transaction-pair query."""

import json
from pathlib import Path
import time
import duckdb
from run import ROOT, TPCH_SQL

cpu = duckdb.connect(config={"threads": 2, "memory_limit": "2GB"})
cpu.execute(
    f"CREATE VIEW lineitem AS SELECT * FROM read_parquet('{ROOT}/data/tpch_sf1/lineitem.parquet')"
)
start = time.perf_counter()
result = cpu.execute(TPCH_SQL).fetchall()
(ROOT / "results/tpch_pair_reference.json").write_text(
    json.dumps(
        dict(sql=TPCH_SQL, result=result, cpu_seconds=time.perf_counter() - start),
        default=str,
        indent=2,
    )
    + "\n"
)
print(result, flush=True)
