"""Process-level plots of complete SQL execution and tracked GPU allocations."""

from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
MODES = ["native", "range_full", "bounded"]
LABELS = {
    "native": "Native cuDF join",
    "range_full": "Range join · full output",
    "bounded": "Range join · 32 MiB tiles",
}
COLORS = {"native": "#426c9d", "range_full": "#559a78", "bounded": "#d58138"}


def main():
    outcomes = json.loads((ROOT / "results/outcomes.json").read_text())
    datasets = json.loads((ROOT / "results/datasets.json").read_text())
    grouped = defaultdict(list)
    for run in outcomes:
        if run["status"] == "passed":
            grouped[
                run["section"],
                run["case"],
                run["width"],
                run["mode"],
                run["budget_mib"],
            ].append(run)
    summary = []

    def stats(section, case, width, mode, budget=32):
        runs = grouped[section, case, width, mode, budget]
        latencies = [median(r["measurement"]["milliseconds"]) for r in runs]
        memories = [r["tracked_process_peak_bytes"] / 1024**2 for r in runs]
        row = dict(
            section=section,
            case=case,
            width=width,
            mode=mode,
            budget_mib=budget,
            process_median_ms=latencies,
            process_peak_mib=memories,
            median_ms=median(latencies),
            median_peak_mib=median(memories),
        )
        return row

    for key in grouped:
        summary.append(stats(*key))
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    output = ROOT / "plots"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    cases = ["unique", "uniform_8", "uniform_128", "uniform_1024", "uniform_2048"]
    x = [datasets[c]["pairs"] / 1e6 for c in cases]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for mode in MODES:
        rows = [stats("main", c, 2, mode) for c in cases]
        for ax, field, center in zip(
            axes,
            ["process_median_ms", "process_peak_mib"],
            ["median_ms", "median_peak_mib"],
        ):
            y = np.array([r[center] for r in rows])
            low = np.array([min(r[field]) for r in rows])
            high = np.array([max(r[field]) for r in rows])
            ax.errorbar(
                x,
                y,
                yerr=[y - low, high - y],
                marker="o",
                capsize=3,
                color=COLORS[mode],
                label=LABELS[mode],
            )
            ax.set_xscale("log")
            ax.set_xlabel("Matching pairs (millions; logarithmic scale)")
            ax.grid(alpha=0.18)
    axes[0].set(
        ylabel="Complete SQL latency (ms)",
        title="Uniform duplicates · fixed 16,384 rows per input",
    )
    axes[1].set(
        ylabel="Tracked process GPU peak (MiB)",
        title="Includes pinning and retained inputs",
        yscale="log",
    )
    axes[0].legend(fontsize=8)
    fig.suptitle("Median of three fresh processes; whiskers show process min–max")
    fig.savefig(output / "expansion.png", dpi=180)
    plt.close(fig)

    comparisons = [
        ("uniform_1024", 2, "Uniform\n16.8M pairs"),
        ("skew_25pct", 2, "Skewed\n16.8M pairs"),
        ("uniform_2048", 8, "Eight expressions\n33.6M pairs"),
        ("tpch_pairs", 2, "TPC-H transaction pairs\n186.1M pairs"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for j, mode in enumerate(MODES):
        rows = [stats("main", case, width, mode) for case, width, _ in comparisons]
        for ax, field, center in zip(
            axes,
            ["process_median_ms", "process_peak_mib"],
            ["median_ms", "median_peak_mib"],
        ):
            y = np.array([r[center] for r in rows])
            low = np.array([min(r[field]) for r in rows])
            high = np.array([max(r[field]) for r in rows])
            ax.bar(
                np.arange(4) + (j - 1) * 0.25,
                y,
                width=0.24,
                yerr=[y - low, high - y],
                capsize=2,
                color=COLORS[mode],
                label=LABELS[mode],
            )
            ax.set_xticks(
                np.arange(4), [label for _, _, label in comparisons], fontsize=8
            )
            ax.grid(axis="y", alpha=0.18)
    axes[0].set(
        ylabel="Complete SQL latency (ms)",
        title="Skew, expression count and transaction-pair workload",
    )
    axes[1].set(
        ylabel="Tracked process GPU peak (MiB)",
        title="Includes pinning and retained inputs",
        yscale="log",
    )
    axes[0].legend(fontsize=8)
    fig.savefig(output / "workloads.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, case, title in zip(
        axes,
        ["uniform_2048", "tpch_pairs"],
        ["Uniform · 33.6M pairs", "TPC-H transaction pairs · 186.1M pairs"],
    ):
        rows = [
            stats("tile_sweep", case, 2, "bounded", budget) for budget in [8, 32, 128]
        ]
        ax.plot(
            [r["median_peak_mib"] for r in rows],
            [r["median_ms"] for r in rows],
            color=COLORS["bounded"],
            linewidth=1,
        )
        for r in rows:
            x, y = r["median_peak_mib"], r["median_ms"]
            ax.errorbar(
                x,
                y,
                xerr=[
                    [x - min(r["process_peak_mib"])],
                    [max(r["process_peak_mib"]) - x],
                ],
                yerr=[
                    [y - min(r["process_median_ms"])],
                    [max(r["process_median_ms"]) - y],
                ],
                fmt="o",
                capsize=3,
                color=COLORS["bounded"],
            )
            ax.annotate(
                f"{r['budget_mib']} MiB tiles",
                (x, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )
        for mode, marker in [("native", "s"), ("range_full", "D")]:
            r = stats("main", case, 2, mode)
            ax.scatter(
                r["median_peak_mib"],
                r["median_ms"],
                marker=marker,
                color=COLORS[mode],
                label=LABELS[mode],
            )
        ax.set(
            xlabel="Tracked process GPU peak (MiB; log scale)",
            ylabel="Complete SQL latency (ms)",
            title=title,
            xscale="log",
        )
        ax.grid(alpha=0.18)
        ax.margins(x=0.2, y=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Buffer budget trades memory for batch size; three fresh processes per point"
    )
    fig.savefig(output / "tile_budget.png", dpi=180)
    plt.close(fig)

    evidence = {}
    for run in outcomes:
        path = ROOT / "results" / run["name"] / "execution_evidence.txt"
        if path.exists():
            evidence[run["name"]] = path.read_text()
    archive = ROOT / "results/process_evidence.json.gz"
    if len(evidence) == len(outcomes):
        archive.write_bytes(gzip.compress(json.dumps(evidence).encode(), mtime=0))
    else:
        assert archive.exists(), "Missing retained execution evidence archive"

    profile = json.loads((ROOT / "results/profile_summary.json").read_text())["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, field, title in zip(
        axes,
        ["kernel_launches", "cuda_stream_synchronize_calls"],
        ["GPU kernel launches", "cudaStreamSynchronize calls"],
    ):
        values = [r[field] for r in profile]
        bars = ax.bar(np.arange(3), values, color=[COLORS[r["mode"]] for r in profile])
        ax.bar_label(bars, padding=3)
        ax.set_xticks(np.arange(3), ["Native", "Range-full", "Bounded · 32 MiB"])
        ax.set(title=title, ylabel="Calls / instances in the traced process")
        ax.set_ylim(top=max(values) * 1.2)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle(
        "Separate Nsight Systems runs: pinning + one warmup + one measured query"
    )
    fig.savefig(output / "profile_costs.png", dpi=180)
    plt.close(fig)
    print("Wrote four plots, process statistics and compressed execution evidence.")


if __name__ == "__main__":
    main()
