"""SQL-level semantic checks for the experimental operator in the real engine."""

from pathlib import Path
import os
import json
import duckdb

root = Path(__file__).resolve().parent
repo = root.parent.parent
folder = root / "data/fusion_fixtures"
folder.mkdir(exist_ok=True)
os.environ["SIRIUS_CONFIG_FILE"] = str(root / "sirius-kvikio.yaml")
os.environ["KVIKIO_COMPAT_MODE"] = "ON"
os.environ["SIRIUS_EXP_JOIN_AGGREGATION"] = "1"
os.environ["SIRIUS_EXP_JOIN_AGGREGATION_LAZY"] = "1"
os.environ["SIRIUS_LOG_DIR"] = str(root / "results/sirius/semantics/logs")
cpu = duckdb.connect()
cpu.execute("CREATE TABLE b(k INT, flag INT)")
cpu.execute("INSERT INTO b VALUES (1,1),(1,0),(2,NULL),(2,1),(3,0),(4,1)")
cpu.execute("CREATE TABLE p(k INT,v DECIMAL(18,2),g VARCHAR)")
cpu.execute(
    "INSERT INTO p VALUES (1,2.5,'a'),(1,-0.5,'b'),(2,NULL,'a'),(3,4.25,'b'),(5,10,'z'),(NULL,7,'z'),(4,9223372036854775.00,'a')"
)
cpu.execute("INSERT INTO b SELECT 4,1 FROM range(31)")
cpu.execute(
    "CREATE TABLE p_large AS SELECT * FROM p UNION ALL SELECT 1, 0::DECIMAL(18,2), 'c' FROM range(128)"
)
cpu.execute("CREATE TABLE arithmetic(k INT, x BIGINT, y BIGINT)")
cpu.execute("INSERT INTO arithmetic VALUES (1,3,7),(2,9223372036854775807,2),(3,5,6)")
cpu.execute("CREATE TABLE selected(k INT, flag INT)")
cpu.execute("INSERT INTO selected VALUES (1,0),(3,1)")
cpu.execute("CREATE TABLE all_selected(k INT, flag INT)")
cpu.execute("INSERT INTO all_selected VALUES (1,0),(2,0),(3,1)")
cpu.execute("CREATE TABLE labels(k INT,s VARCHAR)")
cpu.execute(
    "INSERT INTO labels VALUES (1,'PROMO-LONG'),(1,'OTHER'),(2,NULL),(3,''),(4,'PROMO')"
)
tables = ["b", "p", "p_large", "arithmetic", "selected", "all_selected", "labels"]
for table in tables:
    cpu.execute(
        f"COPY {table} TO '{folder/table}.parquet' (FORMAT PARQUET, OVERWRITE true)"
    )
gpu = duckdb.connect(config={"allow_unsigned_extensions": True})
gpu.execute(
    f"LOAD '{repo}/sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension'"
)
gpu.execute("SET enable_duckdb_fallback=false")
gpu.execute("SET gpu_execution=true")
for table in tables:
    gpu.execute(
        f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{folder/table}.parquet')"
    )
queries = {
    "duplicates_nulls_negative": "SELECT sum(p.v),sum(CASE WHEN b.flag=1 THEN p.v ELSE 0 END) FROM p JOIN b ON p.k=b.k",
    "groups_with_unmatched_group": "SELECT p.g,sum(CASE WHEN b.flag=1 THEN p.v ELSE 0 END),sum(CASE WHEN b.flag=0 THEN p.v ELSE 0 END) FROM p JOIN b ON p.k=b.k GROUP BY p.g ORDER BY p.g",
    "all_null_values": "SELECT sum(p.v),sum(CASE WHEN b.flag=1 THEN p.v ELSE 0 END) FROM p JOIN b ON p.k=b.k WHERE p.k=2",
    "no_matches": "SELECT sum(p.v),sum(CASE WHEN b.flag=1 THEN p.v ELSE 0 END) FROM p JOIN b ON p.k=b.k WHERE p.k=5",
    "reversed_sql_sides": "SELECT sum(p.v),sum(CASE WHEN b.flag=1 THEN p.v ELSE 0 END) FROM b JOIN p ON b.k=p.k",
    "probe_side_groups": "SELECT p_large.g,sum(CASE WHEN b.flag=1 THEN p_large.v ELSE 0 END),sum(CASE WHEN b.flag=0 THEN p_large.v ELSE 0 END) FROM p_large JOIN b ON p_large.k=b.k GROUP BY p_large.g ORDER BY p_large.g",
    "unmatched_overflow_skipped": "SELECT sum(a.x*a.y) FROM arithmetic a JOIN selected s ON a.k=s.k",
    "case_false_overflow_skipped": "SELECT sum(CASE WHEN s.flag=1 THEN a.x*a.y ELSE 0 END) FROM arithmetic a JOIN all_selected s ON a.k=s.k",
    "string_prefix_and_null": "SELECT sum(CASE WHEN l.s LIKE 'PROMO%' THEN p.v ELSE 0 END),sum(p.v) FROM p JOIN labels l ON p.k=l.k",
    "string_in_three_valued": "SELECT sum(CASE WHEN l.s IN ('PROMO','PROMO-LONG',NULL) THEN p.v ELSE 0 END),sum(CASE WHEN l.s NOT IN ('PROMO','PROMO-LONG',NULL) THEN p.v ELSE 0 END) FROM p JOIN labels l ON p.k=l.k",
}
results = {}
output = root / "results/sirius/semantics"
output.mkdir(parents=True, exist_ok=True)
for index, (name, query) in enumerate(queries.items(), 1):
    (output / f"q{index}.sql").write_text(query + ";\n")
    expected = cpu.execute(query).fetchall()
    actual = gpu.execute(query).fetchall()
    assert actual == expected, (name, actual, expected)
    results[name] = {"passed": True, "result": actual, "sql": query}
    print(name, actual, "passed", flush=True)
overflow_sql = "SELECT sum(a.x*a.y) FROM arithmetic a JOIN all_selected s ON a.k=s.k"
for connection in [cpu, gpu]:
    try:
        connection.execute(overflow_sql).fetchall()
        raise AssertionError("Matched integer overflow was not reported")
    except duckdb.Error as error:
        assert "overflow" in str(error).lower(), error
        if connection is gpu:
            assert "lazy join aggregate expression overflow" in str(error), error
results["matched_overflow_reported"] = {"passed": True, "sql": overflow_sql}
print("matched_overflow_reported passed", flush=True)
(output / "results.json").write_text(json.dumps(results, default=str, indent=2) + "\n")
gpu.close()
cpu.close()
