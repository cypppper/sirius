"""Bounded serial fresh-process study of two predeclared scenarios."""

import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def normalize(sql):
    return " ".join(sql.strip().rstrip(";").lower().split())


def collect(folder):
    labels = {normalize(p.read_text()): p.stem for p in folder.glob("*.sql")}
    counts = {name: 0 for name in labels.values()}
    logs = "\n".join(p.read_text() for p in (folder / "logs").glob("*.log"))
    current = None
    for line in logs.splitlines():
        if "QueryBegin:" in line and " SQL: " in line:
            current = labels.get(normalize(line.split(" SQL: ", 1)[1]))
        if "[JOIN_MAP_SUM] pairs=" in line and current:
            counts[current] += 1
    (folder / "operator_counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n"
    )
    (folder / "execution_evidence.txt").write_text(
        "\n".join(
            line.rstrip()
            for line in logs.splitlines()
            if "[JOIN_MAP_SUM]" in line
            or "Join-map SUM" in line
            or "[gpu_pool]" in line
        )
        + "\n"
    )
    return counts


def main():
    rows = []
    hashes = set()
    for scenario, sf, pinned in [
        ("parquet_sf1", 1, False),
        ("resident_sf10", 10, True),
    ]:
        for trial in range(3):
            modes = ["ordinary", "maps"] if trial % 2 == 0 else ["maps", "ordinary"]
            for mode in modes:
                name = f"finaltrial{trial}_{scenario}"
                command = [
                    sys.executable,
                    str(ROOT / "run.py"),
                    "--sf",
                    str(sf),
                    "--mode",
                    mode,
                    "--name",
                    name,
                ]
                if pinned:
                    command.append("--pin")
                folder = ROOT / "results" / f"{name}_{mode}_sf{sf}"
                print(" ".join(command), flush=True)
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=600
                )
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "stdout.txt").write_text(result.stdout)
                (folder / "stderr.txt").write_text(result.stderr)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                measurements = json.loads((folder / "results.json").read_text())
                metadata = json.loads((folder / "environment.json").read_text())
                hashes.add(metadata["library_sha256"])
                counts = collect(folder)
                if mode == "maps":
                    assert all(
                        counts[q] >= 7 for q in ["sum_price", "two_sums", "duplicates"]
                    ), counts
                    assert counts["sum_product"] == 0, counts
                else:
                    assert not any(counts.values()), counts
                for query, values in measurements.items():
                    assert values["matches_cpu"]
                    for repetition, ms in enumerate(values["milliseconds"]):
                        rows.append(
                            dict(
                                scenario=scenario,
                                sf=sf,
                                pinned=int(pinned),
                                trial=trial,
                                mode=mode,
                                query=query,
                                repetition=repetition,
                                milliseconds=ms,
                                consumer_executions=counts[query],
                            )
                        )
                with (ROOT / "results/timings.csv").open("w") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    assert len(hashes) == 1
    (ROOT / "results/campaign.json").write_text(
        json.dumps(
            dict(
                processes=12,
                measured_queries=len(rows),
                warmup_checks=96,
                library_sha256=next(iter(hashes)),
                same_binary=True,
                scenarios=["SF1 Parquet", "SF10 GPU-pinned"],
                trials=3,
                repetitions=5,
                warmups=2,
                note="sum_product is a deliberately unsupported negative control",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
