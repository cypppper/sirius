"""Retain compact plans and execution markers after an engine process exits."""

import argparse
import json
from pathlib import Path


def normalize(sql):
    return " ".join(sql.strip().rstrip(";").lower().split())


def collect(folder):
    folder = Path(folder)
    labels = {normalize(p.read_text()): p.stem for p in folder.glob("q*.sql")}
    lines = "\n".join(
        p.read_text() for p in sorted((folder / "logs").glob("*.log"))
    ).splitlines()
    markers = [
        line
        for line in lines
        if any(
            text in line
            for text in [
                "[FUSED_JOIN_AGGREGATE]",
                "[LAZY_EXPR_AUDIT]",
                "preserving original build child",
                "local I/O backend: kvikio",
                "[gpu_pool]",
                "partition strategy:",
                "using distinct_hash_join",
                "using runtime-distinct",
            ]
        )
    ]
    (folder / "operator_evidence.txt").write_text("\n".join(markers) + "\n")
    plans, counts = {}, {q: 0 for q in labels.values()}
    current, recording = None, False
    for line in lines:
        if "QueryBegin:" in line and " SQL: " in line:
            current = labels.get(normalize(line.split(" SQL: ", 1)[1]))
            recording = False
        if current and "[FUSED_JOIN_AGGREGATE] build_rows=" in line:
            counts[current] += 1
        if line == "=== Pipeline Overview ===" and current and current not in plans:
            plans[current] = []
            recording = True
        if line == "=== Query Plan DAG ===":
            recording = False
        if recording:
            plans[current].append(line.rstrip())
    (folder / "plan_evidence.txt").write_text(
        "\n\n".join(
            f"{q.upper()}: first recorded physical pipeline plan\n"
            + "\n".join(plan).strip()
            for q, plan in plans.items()
        )
        + "\n"
    )
    (folder / "operator_counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n"
    )
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", type=Path)
    for folder in parser.parse_args().folders:
        print(folder, collect(folder))
