"""Generate genuine TPC-H input fragments and controlled operator inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def write_case(
    name, bk, bg, pk, pg, value, bg_count, pg_count, slots, expected, matches, metadata
):
    path = DATA / f"{name}.bin"
    arrays = [
        np.asarray(a, dtype=d)
        for a, d in [
            (bk, "<i4"),
            (bg, "<i4"),
            (pk, "<i4"),
            (pg, "<i4"),
            (value, "<i8"),
            (expected, "<i8"),
        ]
    ]
    assert len(arrays[0]) == len(arrays[1])
    assert len(arrays[2]) == len(arrays[3]) == len(arrays[4])
    assert len(arrays[5]) == bg_count * pg_count
    with path.open("wb") as f:
        f.write(
            struct.pack("<5iQ", len(bk), len(pk), bg_count, pg_count, slots, matches)
        )
        for a in arrays:
            f.write(a.tobytes())
    info = dict(
        name=name,
        build_rows=len(bk),
        probe_rows=len(pk),
        groups=bg_count * pg_count,
        matches=int(matches),
        **metadata,
    )
    if len(expected) <= 128:
        info["expected"] = [int(x) for x in expected]
    else:
        info["expected_sha256"] = hashlib.sha256(arrays[5].tobytes()).hexdigest()
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(name, len(bk), len(pk), matches, flush=True)


def tpch(sf):
    start = time.monotonic()
    db = DATA / f"tpch_sf{sf}.duckdb"
    con = duckdb.connect(str(db))
    con.execute("SET threads=4; SET memory_limit='8GB'")
    con.execute("INSTALL tpch; LOAD tpch")
    if not con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='lineitem'"
    ).fetchone()[0]:
        con.execute(f"CALL dbgen(sf={sf})")
        con.execute("CHECKPOINT")
    part = con.execute(
        "SELECT p_partkey::INT k, starts_with(p_type,'PROMO')::INT g FROM part"
    ).fetchnumpy()
    fact = con.execute(
        """SELECT l_partkey::INT k, 0::INT g,
        (CAST(l_extendedprice*100 AS BIGINT)*(100-CAST(l_discount*100 AS BIGINT)))::BIGINT v
        FROM lineitem WHERE l_shipdate>=DATE '1995-09-01' AND l_shipdate<DATE '1995-10-01'"""
    ).fetchnumpy()
    sql14 = """SELECT starts_with(p_type,'PROMO')::INT g,
        SUM(CAST(l_extendedprice*100 AS BIGINT)*(100-CAST(l_discount*100 AS BIGINT)))::BIGINT v
        FROM lineitem JOIN part ON l_partkey=p_partkey
        WHERE l_shipdate>=DATE '1995-09-01' AND l_shipdate<DATE '1995-10-01' GROUP BY 1"""
    expected = [0, 0]
    for g, v in con.execute(sql14).fetchall():
        expected[g] = v
    ratio = con.execute(
        """SELECT 100.0*sum(CASE WHEN p_type LIKE 'PROMO%' THEN
        l_extendedprice*(1-l_discount) ELSE 0 END)/sum(l_extendedprice*(1-l_discount))
        FROM lineitem JOIN part ON l_partkey=p_partkey
        WHERE l_shipdate>=DATE '1995-09-01' AND l_shipdate<DATE '1995-10-01'"""
    ).fetchone()[0]
    assert abs(100.0 * expected[1] / sum(expected) - ratio) < 1e-10
    write_case(
        f"q14_sf{sf}",
        part["k"],
        part["g"],
        fact["k"],
        fact["g"],
        fact["v"],
        2,
        1,
        1,
        expected,
        len(fact["k"]),
        dict(
            kind="tpch",
            query=14,
            sf=sf,
            reference_sql=sql14,
            q14_promo_percent=ratio,
            integer_scale=10000,
        ),
    )
    orders = con.execute(
        """SELECT o_orderkey::INT k,
        (o_orderpriority IN ('1-URGENT','2-HIGH'))::INT g FROM orders"""
    ).fetchnumpy()
    where = """l_shipmode IN ('MAIL','SHIP') AND l_commitdate<l_receiptdate
        AND l_shipdate<l_commitdate AND l_receiptdate>=DATE '1994-01-01'
        AND l_receiptdate<DATE '1995-01-01'"""
    fact = con.execute(
        f"SELECT l_orderkey::INT k,(l_shipmode='SHIP')::INT g,1::BIGINT v FROM lineitem WHERE {where}"
    ).fetchnumpy()
    sql12 = f"""SELECT (l_shipmode='SHIP')::INT*2+
        (o_orderpriority IN ('1-URGENT','2-HIGH'))::INT g, count(*)::BIGINT v
        FROM lineitem JOIN orders ON l_orderkey=o_orderkey WHERE {where} GROUP BY 1"""
    expected = [0] * 4
    for g, v in con.execute(sql12).fetchall():
        expected[g] = v
    write_case(
        f"q12_sf{sf}",
        orders["k"],
        orders["g"],
        fact["k"],
        fact["g"],
        fact["v"],
        2,
        2,
        1,
        expected,
        len(fact["k"]),
        dict(kind="tpch", query=12, sf=sf, reference_sql=sql12),
    )
    con.close()
    print(
        f"SF{sf}: generation and references took {time.monotonic()-start:.1f}s",
        flush=True,
    )


def synthetic(name, groups=2, hit=1.0, dup=1, skew=False, n=4_000_000, b=262144):
    rng = np.random.default_rng(42)
    bk = np.repeat(np.arange(b, dtype=np.int32), dup)
    bg = rng.integers(0, groups, len(bk), dtype=np.int32)
    if skew:
        bg[rng.random(len(bk)) < 0.99] = 0
    pk = rng.integers(0, b, n, dtype=np.int32)
    pk[rng.random(n) >= hit] += b
    pg = np.zeros(n, np.int32)
    val = rng.integers(-1000, 1001, n, dtype=np.int64)
    expected = np.zeros(groups, np.int64)
    valid = pk < b
    for d in range(dup):
        np.add.at(expected, bg[pk[valid] * dup + d], val[valid])
    write_case(
        name,
        bk,
        bg,
        pk,
        pg,
        val,
        groups,
        1,
        dup,
        expected,
        int(valid.sum()) * dup,
        dict(kind="synthetic", hit=hit, duplicate_factor=dup, skew=skew, seed=42),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="*", default=[1, 5, 10])
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    for sf in args.scales:
        tpch(sf)
    if args.synthetic:
        for g in [2, 32, 128, 256, 4096, 65536]:
            synthetic(f"groups_{g}", groups=g)
        for hit in [0.01, 0.1, 0.5]:
            synthetic(f"hit_{hit}", hit=hit)
        for d in [4, 16]:
            synthetic(f"dup_{d}", dup=d)
        synthetic("skew_4096", groups=4096, skew=True)
        # Boundary fixture: fewer rows than a block, duplicates, misses, signed values.
        synthetic("small_correctness", groups=32, hit=0.5, dup=4, n=97, b=13)


if __name__ == "__main__":
    main()
