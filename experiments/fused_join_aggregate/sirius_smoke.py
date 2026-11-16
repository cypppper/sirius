"""Full Sirius SQL smoke test; errors cannot silently fall back to DuckDB."""

import json
import os
from pathlib import Path
import time
import duckdb

root = Path(__file__).resolve().parent
repo = root.parent.parent
os.environ["SIRIUS_CONFIG_FILE"] = str(root / "sirius-kvikio.yaml")
os.environ["KVIKIO_COMPAT_MODE"] = "ON"
con = duckdb.connect(config={"allow_unsigned_extensions": True, "threads": 4})
con.execute(
    f"LOAD '{repo}/sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension'"
)
con.execute("SET enable_duckdb_fallback=false")
con.execute("SET gpu_execution=true")
path = root / "data/tpch_sf1_parquet/lineitem.parquet"
query = f"SELECT count(*), sum(l_quantity) FROM read_parquet('{path}')"
t = time.perf_counter()
result = con.execute(query).fetchall()
info = {
    "duckdb_version": duckdb.__version__,
    "query": query,
    "result": [list(map(str, row)) for row in result],
    "elapsed_seconds": time.perf_counter() - t,
    "gpu_execution": con.execute("SELECT current_setting('gpu_execution')").fetchone()[
        0
    ],
    "enable_duckdb_fallback": con.execute(
        "SELECT current_setting('enable_duckdb_fallback')"
    ).fetchone()[0],
    "kvikio_compat_mode": os.environ["KVIKIO_COMPAT_MODE"],
}
assert result[0][0] == 6001215, result
assert info["gpu_execution"] and not info["enable_duckdb_fallback"]
(root / "results/sirius_smoke.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info, indent=2), flush=True)
con.close()
