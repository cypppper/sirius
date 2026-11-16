"""Full engine correctness: small tiles, nulls, expressions and declined shapes."""

import hashlib
import json
from pathlib import Path
import sys

import duckdb
from run import ROOT, LIBRARY, configure, connect

mode = sys.argv[1] if len(sys.argv) > 1 else "bounded"
dest = ROOT / "results" / f"semantics_{mode}"
dest.mkdir(parents=True, exist_ok=True)
configure(dest, mode, 4096 / 1024**2, 0.75)
(dest / "environment.json").write_text(
    json.dumps(
        dict(
            mode=mode,
            library_sha256=hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
            tile_bytes=4096,
            forced_orientation_tests=["one_row_resume", "one_row_decimal"],
        ),
        indent=2,
    )
    + "\n"
)
(ROOT / "data").mkdir(exist_ok=True)
cpu = duckdb.connect()
cpu.execute("CREATE TABLE a(k BIGINT, v BIGINT, d DECIMAL(18,2))")
cpu.execute("CREATE TABLE b(k BIGINT, v BIGINT, d DECIMAL(18,2))")
cpu.execute(
    "INSERT INTO a VALUES (1,2,2.50),(1,-3,-1.25),(2,NULL,NULL),(3,5,4),(NULL,7,7),(9223372036854775807,1,0.5)"
)
cpu.execute(
    "INSERT INTO b VALUES (1,4,3.00),(1,-2,1.25),(2,3,2),(4,8,1),(NULL,9,3),(9223372036854775807,2,2)"
)
cpu.execute("CREATE TABLE x AS SELECT 1::BIGINT k,2::BIGINT v")
cpu.execute("CREATE TABLE y AS SELECT 1::BIGINT k,i::BIGINT v FROM range(5000) t(i)")
for table in ["a", "b", "x", "y"]:
    cpu.execute(
        f"COPY {table} TO '{ROOT}/data/semantics_{table}.parquet' (FORMAT PARQUET, OVERWRITE true)"
    )
gpu = connect()
for table in ["a", "b", "x", "y"]:
    gpu.execute(
        f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{ROOT}/data/semantics_{table}.parquet')"
    )
queries = {
    "duplicates_nulls": "SELECT sum(a.v*b.v),sum(a.d*b.d),count(*) FROM a JOIN b ON a.k=b.k",
    "case_decimal": "SELECT sum(CASE WHEN a.v < b.v THEN a.d*b.d ELSE 0 END) FROM a JOIN b ON a.k=b.k",
    "null_safe": "SELECT sum(a.v*b.v),count(*) FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k",
    "empty": "SELECT sum(a.v*b.v),count(*) FROM a JOIN b ON a.k=b.k WHERE a.k=123",
    "all_null": "SELECT sum(a.v),avg(a.v),count(a.v),count(*) FROM a JOIN b ON a.k=b.k WHERE a.k=2",
    "avg_min_max": "SELECT avg(a.v*b.v),min(a.v+b.v),max(a.v-b.v) FROM a JOIN b ON a.k=b.k",
    "reversed": "SELECT sum(a.v*b.v),count(*) FROM b JOIN a ON a.k=b.k",
    "distinct_declined": "SELECT count(DISTINCT a.v),sum(a.v*b.v) FROM a JOIN b ON a.k=b.k",
    "residual_declined": "SELECT sum(a.v*b.v),count(*) FROM a JOIN b ON a.k=b.k AND a.v<b.v",
    "multi_key_declined": "SELECT sum(a.v*b.v),count(*) FROM a JOIN b ON a.k=b.k AND a.v=b.v",
    "grouped_declined": "SELECT a.k,sum(a.v*b.v) FROM a JOIN b ON a.k=b.k GROUP BY a.k ORDER BY a.k",
    "one_row_resume": "SELECT sum(x.v*y.v),avg(x.v*y.v),min(x.v*y.v),max(x.v*y.v),count(*) FROM x JOIN y ON x.k=y.k",
    "one_row_decimal": "SELECT sum(CASE WHEN y.v%3=0 THEN CAST(x.v*y.v AS DECIMAL(18,2))*1.25 ELSE 0 END) FROM x JOIN y ON x.k=y.k",
}
results = {}
for name, sql in queries.items():
    if name == "one_row_resume":
        # Correctness stress only: ensure the single probe row sees >1 tile of matches.
        gpu.execute("SET disabled_optimizers='join_order,build_side_probe_side'")
    expected = cpu.execute(sql).fetchall()
    actual = gpu.execute(sql).fetchall()
    assert actual == expected, (name, actual, expected)
    results[name] = dict(passed=True, sql=sql, result=actual)
    print(name, actual, "passed", flush=True)
(dest / "results.json").write_text(json.dumps(results, default=str, indent=2) + "\n")
gpu.close()
cpu.close()
