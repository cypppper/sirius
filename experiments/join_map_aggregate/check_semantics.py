"""Full-Sirius correctness checks for the scalar SUM match-map consumer."""

import json
import os
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
OUT = ROOT / "results/semantics"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(
    SIRIUS_CONFIG_FILE=str(
        REPO / "experiments/fused_join_aggregate/sirius-kvikio.yaml"
    ),
    KVIKIO_COMPAT_MODE="ON",
    SIRIUS_EXP_JOIN_AGGREGATION="0",
    SIRIUS_EXP_JOIN_MAP_SUM="1",
    SIRIUS_LOG_DIR=str(OUT / "logs"),
)
cpu = duckdb.connect()
cpu.execute("CREATE TABLE a(k BIGINT,v INTEGER,x DECIMAL(18,2))")
cpu.execute(
    "INSERT INTO a VALUES (1,2,2.5),(1,-3,-1.25),(2,NULL,NULL),(3,5,4),(NULL,7,7)"
)
cpu.execute("CREATE TABLE b(k BIGINT)")
cpu.execute("INSERT INTO b VALUES (1),(1),(2),(4),(NULL)")
for table in ["a", "b"]:
    cpu.execute(
        f"COPY {table} TO '{DATA/table}.parquet' (FORMAT PARQUET, OVERWRITE true)"
    )
gpu = duckdb.connect(config={"allow_unsigned_extensions": True})
gpu.execute(
    f"LOAD '{REPO}/sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension'"
)
gpu.execute("SET enable_duckdb_fallback=false")
gpu.execute("SET gpu_execution=true")
gpu.execute("SET enable_compressed_materialization=false")
gpu.execute("SET expression_evaluator_strategy='ast_interpret'")
for table in ["a", "b"]:
    gpu.execute(
        f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{DATA/table}.parquet')"
    )
queries = {
    "duplicates_nulls_negative": "SELECT sum(a.v),sum(a.x),sum(b.k) FROM a JOIN b ON a.k=b.k",
    "integer_expression": "SELECT sum(a.v+1) FROM a JOIN b ON a.k=b.k",
    "all_null": "SELECT sum(a.v),sum(a.x) FROM a JOIN b ON a.k=b.k WHERE a.k=2",
    "empty_join": "SELECT sum(a.v) FROM a JOIN b ON a.k=b.k WHERE a.k=9",
    "reversed_sides": "SELECT sum(a.v),sum(a.x),sum(b.k) FROM b JOIN a ON a.k=b.k",
    "decimal_arithmetic_fallback": "SELECT sum(a.x*a.x) FROM a JOIN b ON a.k=b.k",
}
results = {}
for name, sql in queries.items():
    expected = cpu.execute(sql).fetchall()
    actual = gpu.execute(sql).fetchall()
    assert actual == expected, (name, actual, expected)
    results[name] = dict(passed=True, sql=sql, result=actual)
    (OUT / f"{name}.sql").write_text(sql + ";\n")
    print(name, actual, "passed", flush=True)
(OUT / "results.json").write_text(json.dumps(results, default=str, indent=2) + "\n")
gpu.close()
cpu.close()
