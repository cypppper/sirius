"""Plot process-level summaries; repeated executions are not independent trials."""

import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
QUERIES = ["sum_price", "two_sums", "duplicates", "sum_product"]
LABELS = [
    "One SUM",
    "Two SUMs",
    "Duplicate join",
    "Decimal product\n(fallback control)",
]
SCENARIOS = [("parquet_sf1", "SF1 · Parquet"), ("resident_sf10", "SF10 · GPU-pinned")]
COLORS = ["#426c9d", "#d58138"]


def main():
    raw = defaultdict(list)
    with (ROOT / "results/timings.csv").open() as source:
        for row in csv.DictReader(source):
            raw[row["scenario"], row["query"], row["mode"], int(row["trial"])].append(
                float(row["milliseconds"])
            )
    values = {key: median(v) for key, v in raw.items()}
    summary = []
    output = ROOT / "plots"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (scenario, title) in zip(axes, SCENARIOS):
        for j, mode in enumerate(["ordinary", "maps"]):
            trials = np.array(
                [[values[scenario, q, mode, t] for t in range(3)] for q in QUERIES]
            )
            centers = np.median(trials, axis=1)
            positions = np.arange(4) + (j - 0.5) * 0.36
            bars = ax.bar(
                positions,
                centers,
                width=0.34,
                color=COLORS[j],
                label="Ordinary" if j == 0 else "Join-map consumer",
                yerr=[centers - trials.min(axis=1), trials.max(axis=1) - centers],
                capsize=3,
            )
            bars[-1].set_hatch("///")
            ax.scatter(
                np.repeat(positions, 3), trials.flatten(), color="black", s=10, zorder=3
            )
        ax.set(
            title=title,
            ylabel="Complete SQL latency (ms)",
            xticks=np.arange(4),
            xticklabels=LABELS,
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        "Median of three process medians; whiskers show min–max, dots show each process"
    )
    fig.savefig(output / "latency.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.4), constrained_layout=True, sharey=True
    )
    for ax, (scenario, title) in zip(axes, SCENARIOS):
        for i, query in enumerate(QUERIES):
            ordinary = [values[scenario, query, "ordinary", t] for t in range(3)]
            maps = [values[scenario, query, "maps", t] for t in range(3)]
            ratios = np.array(maps) / ordinary
            color = "#777777" if query == "sum_product" else COLORS[1]
            ax.scatter(i + np.array([-0.08, 0, 0.08]), ratios, color=color, s=30)
            ax.plot(
                [i - 0.18, i + 0.18], [median(ratios)] * 2, color=color, linewidth=3
            )
            summary.append(
                dict(
                    scenario=scenario,
                    query=query,
                    ordinary_ms=median(ordinary),
                    maps_ms=median(maps),
                    ratio_of_medians=median(maps) / median(ordinary),
                    paired_ratios=ratios.tolist(),
                    median_paired_ratio=median(ratios),
                )
            )
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set(title=title, xticks=np.arange(4), xticklabels=LABELS)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Consumer / ordinary latency\n(lower is faster)")
    fig.suptitle("Three fresh-process pairs; horizontal marks show median paired ratio")
    fig.savefig(output / "paired_ratios.png", dpi=180)
    plt.close(fig)
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
