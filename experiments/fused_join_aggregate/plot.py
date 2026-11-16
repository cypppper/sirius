"""Rebuild plots and numerical summaries from raw, unprofiled measurements."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "plots"
METHODS = ["cudf", "materialized", "fused", "fused_global"]
LABELS = dict(
    cudf="cuDF ordinary",
    materialized="Same-hash materialized",
    fused="Fused adaptive",
    fused_global="Fused global atomics",
)
COLORS = dict(zip(METHODS, ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]))
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
)
RAW = {}
STATS = {}


def load():
    for file in sorted((ROOT / "results").glob("*.csv")):
        if file.name == "speedups.csv":
            continue
        rows = list(csv.DictReader(file.open()))
        RAW[file.stem] = rows
        STATS[file.stem] = {}
        for method in METHODS:
            selected = [r for r in rows if r["method"] == method]
            metrics = {}
            for key in [
                "build_gpu_ms",
                "probe_gpu_ms",
                "build_wall_ms",
                "probe_wall_ms",
            ]:
                values = np.array([float(r[key]) for r in selected])
                metrics[key] = dict(
                    median=float(np.median(values)),
                    p25=float(np.quantile(values, 0.25)),
                    p75=float(np.quantile(values, 0.75)),
                )
            totals = np.array(
                [float(r["build_gpu_ms"]) + float(r["probe_gpu_ms"]) for r in selected]
            )
            metrics["total_gpu_ms"] = dict(
                median=float(np.median(totals)),
                p25=float(np.quantile(totals, 0.25)),
                p75=float(np.quantile(totals, 0.75)),
            )
            metrics["logical_intermediate_bytes"] = int(
                selected[0]["logical_intermediate_bytes"]
            )
            STATS[file.stem][method] = metrics


def m(case, method, metric="probe_gpu_ms"):
    return STATS[case][method][metric]["median"]


def curve(ax, cases, xs, title, xlabel, logx=False):
    for method in METHODS:
        med = np.array([m(c, method) for c in cases])
        lo = np.array([STATS[c][method]["probe_gpu_ms"]["p25"] for c in cases])
        hi = np.array([STATS[c][method]["probe_gpu_ms"]["p75"] for c in cases])
        ax.plot(xs, med, "o-", label=LABELS[method], color=COLORS[method])
        ax.fill_between(xs, lo, hi, color=COLORS[method], alpha=0.13)
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel="Probe + aggregate elapsed time (ms)")
    ax.grid(alpha=0.2)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    load()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, q in zip(axes, [14, 12]):
        x = np.arange(3)
        for j, method in enumerate(METHODS[:3]):
            cases = [f"q{q}_sf{sf}" for sf in [1, 5, 10]]
            build = np.array([m(c, method, "build_gpu_ms") for c in cases])
            probe = np.array([m(c, method) for c in cases])
            pos = x + (j - 1) * 0.25
            ax.bar(pos, build, width=0.23, color=COLORS[method], alpha=0.4)
            ax.bar(
                pos,
                probe,
                bottom=build,
                width=0.23,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set(
            xticks=x,
            xticklabels=["SF1", "SF5", "SF10"],
            title=f"TPC-H Q{q} operator fragment",
            ylabel="Elapsed time (ms); pale = build, solid = probe + aggregate",
        )
        ax.legend(fontsize=8)
    fig.suptitle(
        "GPU-resident inputs; excludes scans, filters, transfers and SQL planning"
    )
    fig.tight_layout()
    save(fig, "tpch_total")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, q in zip(axes, [14, 12]):
        curve(
            ax,
            [f"q{q}_sf{s}" for s in [1, 5, 10]],
            [1, 5, 10],
            f"Q{q}: build excluded",
            "TPC-H scale factor",
        )
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "tpch_probe")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    gs = [2, 32, 128, 256, 4096, 65536]
    curve(
        ax,
        [f"groups_{g}" for g in gs],
        gs,
        "Group cardinality sweep (4M probes, all matched)",
        "Number of output groups",
        True,
    )
    ax.axvline(
        128,
        color="gray",
        linestyle=":",
        label="Adaptive kernel: shared → global above 128",
    )
    ax.legend(fontsize=8)
    save(fig, "groups")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    curve(
        ax,
        ["hit_0.01", "hit_0.1", "hit_0.5", "groups_2"],
        [1, 10, 50, 100],
        "Join selectivity sweep (4M probes, 2 output groups)",
        "Probe rows with a match (%)",
    )
    ax.legend(fontsize=8)
    save(fig, "selectivity")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    curve(
        ax,
        ["groups_2", "dup_4", "dup_16"],
        [1, 4, 16],
        "Build multiplicity sweep (4M probes, every pair visited)",
        "Build rows per key",
    )
    ax.legend(fontsize=8)
    save(fig, "multiplicity")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, method in enumerate(METHODS):
        cases = ["groups_4096", "skew_4096"]
        ax.bar(
            np.arange(2) + (j - 1.5) * 0.2,
            [m(c, method) for c in cases],
            width=0.18,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Uniform dimension groups", "99% of dimension rows in group 0"],
        ylabel="Probe + aggregate elapsed time (ms)",
        title="Skew at 4,096 possible output groups",
    )
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    save(fig, "skew")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cases = ["q14_sf10", "q12_sf10", "groups_2", "dup_4", "dup_16"]
    for j, method in enumerate(METHODS[:3]):
        ax.bar(
            np.arange(len(cases)) + (j - 1) * 0.25,
            [STATS[c][method]["logical_intermediate_bytes"] / 2**20 for c in cases],
            width=0.23,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set(
        xticks=np.arange(len(cases)),
        xticklabels=[
            "Q14 SF10",
            "Q12 SF10",
            "4M matches",
            "16M matches",
            "64M matches",
        ],
        ylabel="Explicit intermediate buffers (MiB)",
        title="Buffer-size accounting: excludes hash state and library scratch",
    )
    ax.legend(fontsize=8)
    save(fig, "intermediate")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cases = ["q14_sf10", "q12_sf10", "groups_2", "dup_4", "dup_16"]
    for j, method in enumerate(METHODS[:3]):
        peaks = []
        for case in cases:
            rows = list(
                csv.DictReader((ROOT / "results" / "memory" / f"{case}.csv").open())
            )
            peaks.append(
                np.median(
                    [
                        int(r["peak_operator_bytes"])
                        for r in rows
                        if r["method"] == method
                    ]
                )
                / 2**20
            )
        ax.bar(
            np.arange(len(cases)) + (j - 1) * 0.25,
            peaks,
            width=0.23,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set(
        xticks=np.arange(len(cases)),
        xticklabels=[
            "Q14 SF10",
            "Q12 SF10",
            "4M matches",
            "16M matches",
            "64M matches",
        ],
        ylabel="Peak live operator allocations (MiB)",
        title="Measured RMM allocation peaks: excludes resident inputs and pool reserve",
    )
    ax.legend(fontsize=8)
    save(fig, "peak_memory")
    (ROOT / "results" / "summary.json").write_text(json.dumps(STATS, indent=2) + "\n")
    rng = np.random.default_rng(7)
    with (ROOT / "results" / "speedups.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "baseline",
                "metric",
                "ratio_of_medians",
                "paired_bootstrap_ci_low",
                "paired_bootstrap_ci_high",
            ]
        )
        for case, rows in RAW.items():
            for baseline in ["cudf", "materialized"]:
                for metric in ["probe_gpu_ms", "total_gpu_ms"]:
                    values = []
                    for method in [baseline, "fused"]:
                        rr = sorted(
                            [r for r in rows if r["method"] == method],
                            key=lambda r: int(r["iteration"]),
                        )
                        values.append(
                            np.array(
                                [
                                    float(r["probe_gpu_ms"])
                                    + (
                                        float(r["build_gpu_ms"])
                                        if metric == "total_gpu_ms"
                                        else 0
                                    )
                                    for r in rr
                                ]
                            )
                        )
                    a, b = values
                    indices = rng.integers(0, len(a), (10000, len(a)))
                    ratios = np.median(a[indices], axis=1) / np.median(
                        b[indices], axis=1
                    )
                    writer.writerow(
                        [
                            case,
                            baseline,
                            metric,
                            np.median(a) / np.median(b),
                            *np.quantile(ratios, [0.025, 0.975]),
                        ]
                    )


if __name__ == "__main__":
    main()
