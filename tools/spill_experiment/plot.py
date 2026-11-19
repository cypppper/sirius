"""Plot raw CUDA transfer timings or Sirius downgrade-executor timings."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

POLICIES = {
    "newest_first": ("Newest first", "#62666d", "s"),
    "best_fit": ("Sirius byte-targeted", "#006b71", "o"),
    "minimum_bytes": ("Minimum-byte subset", "#b44c12", "^"),
}


def plot_transfers(rows):
    samples = defaultdict(list)
    for row in rows:
        samples[row["policy"], int(row["deficit_mib"])].append(row)
    deficits = sorted({deficit for _, deficit in samples})
    fig, (volume, timing) = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True)
    x = list(range(len(deficits)))
    for policy, (label, color, marker) in POLICIES.items():
        groups = [samples[policy, deficit] for deficit in deficits]
        volume.plot(
            x,
            [int(group[0]["spilled_mib"]) for group in groups],
            color=color,
            marker=marker,
            label=label,
            markerfacecolor="white",
        )
        timing.plot(
            x,
            [
                median(
                    float(row["spill_ms"]) + float(row["restore_ms"]) for row in group
                )
                for group in groups
            ],
            color=color,
            marker=marker,
            markerfacecolor="white",
        )
    volume.set_ylabel("Evicted per request (MiB)")
    timing.set_ylabel("Median round trip (ms)")
    timing.set_xticks(x, deficits)
    timing.set_xlabel("Requested memory (MiB)")
    for axis in (volume, timing):
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        axis.axvspan(x[4] - 0.3, x[5] + 0.3, color="#ece6d9", zorder=0)
    volume.legend(loc="upper left")
    fig.suptitle("A small increase in the deficit can evict a much larger batch")
    fig.text(
        0.5,
        0.936,
        "RTX A4000 · pinned host memory · 1,536 / 1,024 / 384 / 256 MiB batches",
        ha="center",
        fontsize=9,
        color="#454545",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_executor(rows):
    samples = defaultdict(list)
    for row in rows:
        samples[int(row["deficit_mib"])].append(row)
    deficits = sorted(samples)
    x = list(range(len(deficits)))
    downgrade = [
        median(float(row["downgrade_ms"]) for row in samples[d]) for d in deficits
    ]
    restore = [median(float(row["restore_ms"]) for row in samples[d]) for d in deficits]
    fig, axis = plt.subplots(figsize=(8.6, 4.8))
    axis.bar(x, downgrade, label="Downgrade", color="#006b71")
    axis.bar(x, restore, bottom=downgrade, label="Restore", color="#b44c12")
    for i, deficit in enumerate(deficits):
        spilled = int(samples[deficit][0]["evicted_mib"])
        axis.annotate(
            f"{spilled:,} MiB evicted",
            (i, downgrade[i] + restore[i]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
        )
    axis.set_xticks(x, [f"{d:,}" for d in deficits])
    axis.set_xlabel("GPU reservation shortfall (MiB)")
    axis.set_ylabel("Median phase latency (ms)")
    axis.set_ylim(0, 1.15 * max(a + b for a, b in zip(downgrade, restore)))
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    axis.set_title("Sirius eviction under a 4 GiB GPU memory cap")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--executor", action="store_true", help="Plot executor timings")
    args = parser.parse_args()
    with args.csv.open(newline="") as source:
        rows = list(csv.DictReader(source))
    fig = plot_executor(rows) if args.executor else plot_transfers(rows)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
