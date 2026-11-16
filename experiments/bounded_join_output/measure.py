"""Predeclared serial whole-SQL study; no concurrent GPU processes."""

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from run import LIBRARY, ROOT

MODES = ["native", "range_full", "bounded"]


def protocol():
    jobs = []
    main = [
        ("unique", 2),
        ("uniform_8", 2),
        ("uniform_128", 2),
        ("uniform_1024", 2),
        ("uniform_2048", 2),
        ("skew_25pct", 2),
        ("uniform_2048", 8),
        ("tpch_pairs", 2),
    ]
    for case, width in main:
        for trial in range(3):
            for mode in MODES[trial:] + MODES[:trial]:
                jobs.append(
                    dict(
                        section="main",
                        case=case,
                        width=width,
                        trial=trial,
                        mode=mode,
                        budget_mib=32,
                        fraction=0.75,
                        warmups=1,
                        repetitions=3,
                    )
                )
    for case in ["uniform_2048", "tpch_pairs"]:
        for trial in range(3):
            sizes = [8, 32, 128]
            for budget in sizes[trial:] + sizes[:trial]:
                jobs.append(
                    dict(
                        section="tile_sweep",
                        case=case,
                        width=2,
                        trial=trial,
                        mode="bounded",
                        budget_mib=budget,
                        fraction=0.75,
                        warmups=1,
                        repetitions=3,
                    )
                )
    for case in ["uniform_2048", "one_hot_256m", "tpch_pairs"]:
        for mode in MODES:
            jobs.append(
                dict(
                    section="pressure",
                    case=case,
                    width=2,
                    trial=0,
                    mode=mode,
                    budget_mib=32,
                    fraction=0.08,
                    warmups=0,
                    repetitions=1,
                )
            )
    return jobs


def evidence(folder):
    logs = "\n".join(
        p.read_text(errors="replace") for p in sorted((folder / "logs").glob("*.log"))
    )
    records = []
    for line in logs.splitlines():
        if "[BOUNDED_JOIN]" in line:
            records.append(
                {
                    k: int(v) if v.isdigit() else v
                    for k, v in re.findall(r"(\w+)=([\w]+)", line)
                }
            )
    peaks = [
        int(x) for x in re.findall(r"\[gpu_pool\].*?QueryEnd.*?peak=(\d+) bytes", logs)
    ]
    selected = [
        line
        for line in logs.splitlines()
        if "[BOUNDED_JOIN]" in line
        or "[gpu_pool]" in line
        or "OOM at operator" in line
        or "exceeded" in line
        or "out of memory" in line.lower()
    ]
    (folder / "execution_evidence.txt").write_text("\n".join(selected) + "\n")
    return dict(
        consumer_records=records,
        tracked_process_peak_bytes=max(peaks, default=0),
        oom_log_mentions=logs.count("OOM at operator"),
    )


def main():
    jobs = protocol()
    (ROOT / "results/protocol.json").write_text(json.dumps(jobs, indent=2) + "\n")
    library_sha = hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
    datasets = json.loads((ROOT / "results/datasets.json").read_text())
    tpch_pairs = json.loads((ROOT / "results/tpch_pair_reference.json").read_text())[
        "result"
    ][0][0]
    rows, outcomes = [], []
    for index, job in enumerate(jobs):
        name = f"final_{index:03d}_{job['section']}_{job['case']}_w{job['width']}_{job['mode']}_t{job['trial']}"
        folder = ROOT / "results" / name
        command = [sys.executable, str(ROOT / "run.py"), "--name", name]
        for key in [
            "case",
            "width",
            "mode",
            "budget_mib",
            "fraction",
            "warmups",
            "repetitions",
        ]:
            command += ["--" + key.replace("_", "-"), str(job[key])]
        print(f"[{index+1}/{len(jobs)}] {name}", flush=True)
        timed_out = False
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            (folder / "stdout.txt").write_text(result.stdout)
            (folder / "stderr.txt").write_text(result.stderr)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out, returncode = True, -1
        metadata = json.loads((folder / "environment.json").read_text())
        assert (
            metadata["library_sha256"] == library_sha
        ), "Binary changed during measurement"
        measurement = json.loads((folder / "results.json").read_text())
        facts = evidence(folder)
        status = "timeout" if timed_out else measurement["status"]
        outcome = (
            job
            | dict(
                name=name,
                status=status,
                returncode=returncode,
                library_sha256=library_sha,
            )
            | facts
        )
        outcome["environment"] = metadata
        outcome["measurement"] = measurement
        outcomes.append(outcome)
        if status == "passed":
            assert measurement["matches_reference"]
            expected_pairs = (
                tpch_pairs
                if job["case"] == "tpch_pairs"
                else datasets[job["case"]]["pairs"]
            )
            if job["mode"] != "native":
                records = facts["consumer_records"]
                assert records and sum(
                    r["pairs"] for r in records
                ) == expected_pairs * (job["warmups"] + job["repetitions"]), (
                    name,
                    records,
                )
                if job["mode"] == "bounded":
                    assert all(
                        r["max_tile_rows"] * r["bytes_per_row"] <= r["tile_budget"]
                        for r in records
                    )
                else:
                    assert all(r["tiles"] == 1 for r in records)
            else:
                assert not facts["consumer_records"]
            for repetition, ms in enumerate(measurement["milliseconds"]):
                rows.append(
                    job
                    | dict(
                        name=name,
                        repetition=repetition,
                        milliseconds=ms,
                        tracked_process_peak_bytes=facts["tracked_process_peak_bytes"],
                    )
                )
        elif job["section"] != "pressure":
            raise RuntimeError(f"Unexpected failed primary run: {name}; {measurement}")
        with (ROOT / "results/timings.csv").open("w", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        (ROOT / "results/outcomes.json").write_text(
            json.dumps(outcomes, indent=2) + "\n"
        )
        print(
            status,
            facts["tracked_process_peak_bytes"],
            "OOMs",
            facts["oom_log_mentions"],
            flush=True,
        )
    print("complete", len(rows), "timings", library_sha, flush=True)


if __name__ == "__main__":
    main()
