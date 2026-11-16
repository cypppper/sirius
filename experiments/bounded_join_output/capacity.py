"""Exploratory capacity check beyond cuDF's single-table row limit."""

import hashlib
import json
import subprocess
import sys
from measure import evidence
from run import ROOT, LIBRARY

protocol = json.loads((ROOT / "results/capacity_protocol.json").read_text())
outputs = []
for mode in protocol["modes"]:
    name = "capacity_" + mode
    folder = ROOT / "results" / name
    command = [
        sys.executable,
        str(ROOT / "run.py"),
        "--case",
        protocol["case"],
        "--mode",
        mode,
        "--budget-mib",
        str(protocol["budget_mib"]),
        "--fraction",
        str(protocol["fraction"]),
        "--warmups",
        "0",
        "--repetitions",
        "1",
        "--name",
        name,
    ]
    print(mode, flush=True)
    timed_out = False
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=protocol["timeout_seconds"]
        )
        (folder / "stdout.txt").write_text(result.stdout)
        (folder / "stderr.txt").write_text(result.stderr)
    except subprocess.TimeoutExpired:
        timed_out = True
    measurement = json.loads((folder / "results.json").read_text())
    facts = evidence(folder)
    status = "timeout" if timed_out else measurement["status"]
    outputs.append(
        dict(
            mode=mode,
            status=status,
            measurement=measurement,
            environment=json.loads((folder / "environment.json").read_text()),
            **facts
        )
    )
    if status == "passed":
        assert measurement["matches_reference"]
        assert (
            sum(r["pairs"] for r in facts["consumer_records"])
            == protocol["expected_pairs"]
        )
        assert all(
            r["max_tile_rows"] * r["bytes_per_row"] <= r["tile_budget"]
            for r in facts["consumer_records"]
        )
    print(status, facts["tracked_process_peak_bytes"], flush=True)
    (ROOT / "results/capacity_results.json").write_text(
        json.dumps(outputs, indent=2) + "\n"
    )
