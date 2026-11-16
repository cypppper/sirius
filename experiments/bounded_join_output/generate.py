"""Generate controlled joins and TPC-H data; no engine input pre-aggregation."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = {
    "unique": dict(n=16384, keys=16384),
    "uniform_8": dict(n=16384, keys=2048),
    "uniform_128": dict(n=16384, keys=128),
    "uniform_1024": dict(n=16384, keys=16),
    "uniform_2048": dict(n=16384, keys=8),
    "skew_25pct": dict(n=16384, hot=4096),
    "one_probe": dict(n=32768, keys=1, probe=1),
    "one_hot_4b": dict(n=65536, keys=1),
    "one_hot_256m": dict(n=16384, keys=1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpch", action="store_true")
    ap.add_argument("--only", nargs="+", choices=list(CASES))
    args = ap.parse_args()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=2")
    manifest_path = ROOT / "results/datasets.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, config in CASES.items():
        if args.only and name not in args.only:
            continue
        arrays = {}
        start = time.perf_counter()
        for side, salt in [("a", 3), ("b", 11)]:
            n = config.get("probe", config["n"]) if side == "a" else config["n"]
            idx = np.arange(n, dtype=np.int64)
            if "hot" in config:
                keys = np.where(idx < config["hot"], 0, idx - config["hot"] + 1)
            else:
                keys = idx % config["keys"]
            # Shuffle independently: ranges must use original payload row IDs.
            rng = np.random.default_rng(105 + salt)
            permutation = rng.permutation(n)
            columns = {"k": keys.astype(np.int64)}
            for j in range(8):
                columns[f"v{j}"] = ((idx * (j * 2 + 1) + salt) % 97 - 48).astype(
                    np.int64
                )
            columns = {k: v[permutation] for k, v in columns.items()}
            arrays[side] = columns
            con.register("source", columns)
            path = data / f"{name}_{side}.parquet"
            con.execute(
                f"COPY (SELECT * FROM source) TO '{path}' (FORMAT PARQUET, OVERWRITE true)"
            )
            con.unregister("source")
        frequencies, sums = {}, {}
        for side in ["a", "b"]:
            counts = defaultdict(int)
            totals = defaultdict(lambda: [0] * 8)
            c = arrays[side]
            for i, key in enumerate(c["k"]):
                key = int(key)
                counts[key] += 1
                for j in range(8):
                    totals[key][j] += int(c[f"v{j}"][i])
            frequencies[side], sums[side] = counts, totals
        common = set(frequencies["a"]) & set(frequencies["b"])
        pairs = sum(frequencies["a"][k] * frequencies["b"][k] for k in common)
        products = [
            sum(sums["a"][k][j] * sums["b"][k][j] for k in common) for j in range(8)
        ]
        manifest[name] = dict(
            **config,
            input_rows_a=len(arrays["a"]["k"]),
            input_rows_b=len(arrays["b"]["k"]),
            pairs=pairs,
            expected_narrow=[products[0], products[1], pairs],
            expected_wide=products + [pairs],
            generation_seconds=time.perf_counter() - start,
        )
        print(name, pairs, flush=True)
    (ROOT / "results/datasets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.tpch:
        output = data / "tpch_sf1"
        output.mkdir(exist_ok=True)
        db = duckdb.connect(str(data / "tpch_sf1.duckdb"))
        db.execute("SET threads=2; SET memory_limit='2GB'; INSTALL tpch; LOAD tpch")
        if not db.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name='lineitem'"
        ).fetchone()[0]:
            db.execute("CALL dbgen(sf=1)")
        counts = {}
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
            path = output / f"{table}.parquet"
            if not path.exists():
                db.execute(
                    f"COPY {table} TO '{path}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 122880)"
                )
            counts[table] = db.execute(
                f"SELECT count(*) FROM read_parquet('{path}')"
            ).fetchone()[0]
            print(table, counts[table], flush=True)
        (ROOT / "results/tpch_generation.json").write_text(
            json.dumps(counts, indent=2) + "\n"
        )
        db.close()


if __name__ == "__main__":
    main()
