"""Export the existing TPC-H databases for full Sirius/KvikIO scan execution."""

import argparse
from pathlib import Path
import duckdb

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--scales", nargs="+", type=int, default=[1, 5, 10])
args = parser.parse_args()
for sf in args.scales:
    con = duckdb.connect(str(root / "data" / f"tpch_sf{sf}.duckdb"), read_only=True)
    con.execute("SET threads=4; SET memory_limit='2GB'")
    output = root / "data" / f"tpch_sf{sf}_parquet"
    output.mkdir(exist_ok=True)
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
        dest = output / f"{table}.parquet"
        if not dest.exists():
            con.execute(
                f"COPY {table} TO '{dest}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 122880)"
            )
        count = con.execute(f"SELECT count(*) FROM read_parquet('{dest}')").fetchone()[
            0
        ]
        reference = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert count == reference
        print(sf, table, count, dest.stat().st_size, flush=True)
    con.close()
