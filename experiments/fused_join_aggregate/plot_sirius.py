"""Plots for full Sirius SQL measurements; independent process runs form the units."""

import csv
import json
from pathlib import Path
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "plots"
MODES = ["ordinary", "eager", "fused"]
COLORS = ["#4c78a8", "#f58518", "#54a24b"]
LABELS = ["Ordinary Sirius", "Fused eager", "Fused lazy"]
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def main():
    rows = list(csv.DictReader((ROOT / "results/sirius/end_to_end.csv").open()))
    scales = sorted({int(r["sf"]) for r in rows})

    def medians(query, sf, pinned, mode):
        selected = [
            r
            for r in rows
            if r["query"] == query
            and int(r["sf"]) == sf
            and int(r["pinned"]) == pinned
            and r["mode"] == mode
        ]
        trials = sorted({int(r["trial"]) for r in selected})
        return np.array(
            [
                np.median(
                    [float(r["milliseconds"]) for r in selected if int(r["trial"]) == t]
                )
                for t in trials
            ]
        )

    summary = []
    rng = np.random.default_rng(23)
    for pinned in [0, 1]:
        for query in ["q12", "q14"]:
            for sf in scales:
                a, b = [medians(query, sf, pinned, m) for m in ["ordinary", "fused"]]
                samples = rng.integers(0, len(a), (10000, len(a)))
                ratios = np.median(a[samples], axis=1) / np.median(b[samples], axis=1)
                eager = medians(query, sf, pinned, "eager")
                lazy_ratios = np.median(eager[samples], axis=1) / np.median(
                    b[samples], axis=1
                )
                summary.append(
                    dict(
                        query=query,
                        sf=sf,
                        pinned=pinned,
                        ordinary_ms=float(np.median(a)),
                        fused_ms=float(np.median(b)),
                        eager_ms=float(np.median(eager)),
                        eager_speedup=float(np.median(eager) / np.median(b)),
                        eager_ci_low=float(np.quantile(lazy_ratios, 0.025)),
                        eager_ci_high=float(np.quantile(lazy_ratios, 0.975)),
                        speedup=float(np.median(a) / np.median(b)),
                        ci_low=float(np.quantile(ratios, 0.025)),
                        ci_high=float(np.quantile(ratios, 0.975)),
                    )
                )
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for row, pinned in enumerate([0, 1]):
        for col, query in enumerate(["q12", "q14"]):
            ax = axes[row, col]
            for j, mode in enumerate(MODES):
                data = [medians(query, sf, pinned, mode) for sf in scales]
                y = np.array([np.median(d) for d in data])
                error = np.array(
                    [y - [np.min(d) for d in data], [np.max(d) for d in data] - y]
                )
                ax.bar(
                    np.arange(len(scales)) + (j - 1) * 0.26,
                    y,
                    width=0.24,
                    yerr=error,
                    capsize=3,
                    color=COLORS[j],
                    label=LABELS[j],
                )
            ax.set(
                xticks=np.arange(len(scales)),
                xticklabels=[f"SF{x}" for x in scales],
                ylabel="Whole SQL execution (ms)",
                title=f"{query.upper()}: "
                + ("GPU-pinned inputs" if pinned else "Parquet via KvikIO"),
            )
            ax.legend(fontsize=8)
    fig.suptitle(
        "Same Sirius binary; CPU fallback disabled; bars = median of fresh-process medians"
    )
    fig.tight_layout()
    save(fig, "sirius_end_to_end")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for pinned in [0, 1]:
        for query, color in [("q12", "#4c78a8"), ("q14", "#f58518")]:
            data = [x for x in summary if x["pinned"] == pinned and x["query"] == query]
            y = np.array([x["speedup"] for x in data])
            ax.plot(
                scales,
                y,
                marker="o",
                linestyle="-" if pinned else "--",
                color=color,
                label=query.upper() + (" pinned" if pinned else " Parquet"),
            )
            ax.fill_between(
                scales,
                [x["ci_low"] for x in data],
                [x["ci_high"] for x in data],
                color=color,
                alpha=0.1,
            )
    ax.axhline(1, color="black", linewidth=0.9)
    ax.set(
        xlabel="TPC-H scale factor",
        ylabel="Ordinary / lazy-fused whole-query latency",
        title="Above 1 favors fusion; shading = paired process-bootstrap interval",
    )
    ax.legend()
    save(fig, "sirius_speedups")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for pinned in [0, 1]:
        for query, color in [("q12", "#4c78a8"), ("q14", "#f58518")]:
            data = [r for r in summary if r["pinned"] == pinned and r["query"] == query]
            ax.plot(
                scales,
                [r["eager_speedup"] for r in data],
                color=color,
                marker="o",
                linestyle="-" if pinned else "--",
                label=query.upper() + (" pinned" if pinned else " Parquet"),
            )
            ax.fill_between(
                scales,
                [r["eager_ci_low"] for r in data],
                [r["eager_ci_high"] for r in data],
                color=color,
                alpha=0.1,
            )
    ax.axhline(1, color="black", linewidth=0.9)
    ax.set(
        xlabel="TPC-H scale factor",
        ylabel="Eager-fused / lazy-fused whole-query latency",
        title="Same-binary expression-evaluation control; above 1 favors lazy",
    )
    ax.legend()
    save(fig, "sirius_lazy_gain")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, query, groups in zip(axes, ["q12", "q14"], [2, 1]):
        build, probe = [], []
        for sf in scales:
            evidence = (
                ROOT
                / "results/sirius"
                / f"e2e_t0_p1_fused_sf{sf}"
                / "operator_evidence.txt"
            )
            matches = re.findall(
                r"build_rows=(\d+) probe_rows=(\d+) groups=(\d+)", evidence.read_text()
            )
            b, p, _ = next(x for x in matches if int(x[2]) == groups)
            build.append(int(b) / 1e6)
            probe.append(int(p) / 1e6)
        x = np.arange(len(scales))
        build_bars = ax.bar(
            x - 0.16, build, width=0.3, label="Original build input", color="#4c78a8"
        )
        probe_bars = ax.bar(
            x + 0.16, probe, width=0.3, label="Original probe input", color="#f58518"
        )
        ax.bar_label(build_bars, fmt="%.2f", padding=3, fontsize=8)
        ax.bar_label(probe_bars, fmt="%.2f", padding=3, fontsize=8)
        ax.margins(y=0.15)
        ax.set(
            xticks=x,
            xticklabels=[f"SF{s}" for s in scales],
            ylabel="Rows entering fused operator (millions)",
            title=f"{query.upper()}: actual execution counters",
        )
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "sirius_input_rows")
    memory = list(csv.DictReader((ROOT / "results/sirius/process_memory.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, pinned in zip(axes, [0, 1]):
        for j, mode in enumerate(MODES):
            values = [
                np.median(
                    [
                        int(r["peak_bytes"]) / 2**30
                        for r in memory
                        if int(r["sf"]) == sf
                        and int(r["pinned"]) == pinned
                        and r["mode"] == mode
                    ]
                )
                for sf in scales
            ]
            ax.bar(
                np.arange(len(scales)) + (j - 1) * 0.26,
                values,
                width=0.24,
                color=COLORS[j],
                label=LABELS[j],
            )
        ax.set(
            xticks=np.arange(len(scales)),
            xticklabels=[f"SF{s}" for s in scales],
            ylabel="Tracked live-allocation high-water mark (GiB)",
            title="GPU-pinned inputs" if pinned else "Parquet via KvikIO",
        )
        ax.legend(fontsize=8)
    fig.suptitle(
        "Cumulative process high-water mark: includes setup, pinning and both queries"
    )
    fig.tight_layout()
    save(fig, "sirius_memory")
    profiles = json.loads((ROOT / "results/sirius/profile_summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, case in enumerate(["q12_sf10", "q14_sf1"]):
        selected = [
            next(r for r in profiles if r["case"] == case and r["mode"] == mode)
            for mode in MODES
        ]
        for row, field, ylabel in [
            (0, "kernel_ms", "Sum of CUDA kernel durations (ms)"),
            (1, "launches", "CUDA kernel launches"),
        ]:
            ax = axes[row, col]
            bars = ax.bar(
                ["Ordinary", "Eager", "Lazy"],
                [r[field] for r in selected],
                color=COLORS,
            )
            ax.bar_label(bars, fmt="%.3f" if row == 0 else "%d", padding=3)
            ax.margins(y=0.2)
            ax.set(ylabel=ylabel, title=case.upper().replace("_", " ") + " pinned")
    fig.suptitle("Nsight Systems: one warmed whole-query capture per case and mode")
    fig.tight_layout()
    save(fig, "sirius_profile")
    audit = (
        ROOT / "results/sirius/lazy_audit_fused_sf10/operator_evidence.txt"
    ).read_text()
    last, work = None, {}
    for line in audit.splitlines():
        match = re.search(r"predicate_evaluations=(\d+) value_evaluations=(\d+)", line)
        if match:
            last = tuple(map(int, match.groups()))
        match = re.search(r"build_rows=(\d+) probe_rows=(\d+) groups=(\d+)", line)
        if match and last:
            build, probe, groups = map(int, match.groups())
            query = "q12" if groups == 2 else "q14"
            work[query] = dict(
                eager_predicates=2 * probe if groups == 2 else build,
                eager_values=2 * build if groups == 2 else 2 * probe,
                lazy_predicates=last[0],
                lazy_values=last[1],
            )
    assert set(work) == {"q12", "q14"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, query in zip(axes, ["q12", "q14"]):
        data = work[query]
        for j, strategy in enumerate(["eager", "lazy"]):
            values = [
                data[strategy + "_" + field] / 1e6 for field in ["predicates", "values"]
            ]
            bars = ax.bar(
                np.arange(2) + (j - 0.5) * 0.32,
                values,
                width=0.3,
                color=[COLORS[1], COLORS[2]][j],
                label=(
                    "Eager input-row model"
                    if strategy == "eager"
                    else "Lazy device audit"
                ),
            )
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set(
            xticks=[0, 1],
            xticklabels=["CASE conditions", "SUM/THEN arguments"],
            ylabel="Top-level evaluations (millions)",
            title=query.upper() + " SF10",
        )
        ax.margins(y=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Expression work: eager model versus measured matched-pair evaluation counts"
    )
    fig.tight_layout()
    save(fig, "sirius_lazy_work")
    (ROOT / "results/sirius/lazy_work.json").write_text(
        json.dumps(work, indent=2) + "\n"
    )
    (ROOT / "results/sirius/summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (ROOT / "results/sirius/summary.md").open("w") as f:
        f.write(
            "| Query | SF | Input | Ordinary ms | Eager fused ms | Lazy fused ms | Ordinary/lazy | Eager/lazy | Process-bootstrap ordinary/lazy |\n"
        )
        f.write("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for r in summary:
            f.write(
                f"| {r['query'].upper()} | {r['sf']} | {'pinned' if r['pinned'] else 'Parquet'} | {r['ordinary_ms']:.2f} | {r['eager_ms']:.2f} | {r['fused_ms']:.2f} | {r['speedup']:.3f}× | {r['eager_speedup']:.3f}× | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] |\n"
            )


if __name__ == "__main__":
    main()
