"""Compare repeated SF1 all-query engine measurements without rerunning SQL."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/sirius"
PLOTS = ROOT / "plots"
COLORS = ["#4c78a8", "#54a24b"]


def save(fig, name):
    for suffix in ["png", "svg"]:
        fig.savefig(PLOTS / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main():
    PLOTS.mkdir(exist_ok=True)
    rows = list(csv.DictReader((RESULTS / "all22_end_to_end.csv").open()))
    metadata = json.loads((RESULTS / "all22_campaign.json").read_text())
    assert metadata["measured_queries"] == len(rows) == 660
    assert metadata["fallback_disabled"] and metadata["ordinary_fused_same_binary"]
    assert (
        metadata["library_sha256"]
        == json.loads((RESULTS / "campaign.json").read_text())["library_sha256"]
    )
    records = []
    samples_by_query = {}
    rng = np.random.default_rng(23)
    for i in range(1, 23):
        name = f"q{i}"
        modes = []
        for mode in ["ordinary", "fused"]:
            process_medians = []
            for trial in range(3):
                times = [
                    float(row["milliseconds"])
                    for row in rows
                    if row["query"] == name
                    and row["mode"] == mode
                    and int(row["trial"]) == trial
                ]
                assert len(times) == 5 and min(times) > 0
                process_medians.append(np.median(times))
            modes.append(np.array(process_medians))
        a, b = modes
        ordinary, enabled = np.median(a), np.median(b)
        indices = rng.integers(0, 3, (10000, 3))
        ratios = np.median(a[indices], axis=1) / np.median(b[indices], axis=1)
        samples_by_query[name.upper()] = modes
        records.append(
            dict(
                query=name.upper(),
                ordinary_ms=ordinary,
                fusion_enabled_ms=enabled,
                ordinary_over_fusion_enabled=ordinary / enabled,
                ci_low=np.quantile(ratios, 0.025),
                ci_high=np.quantile(ratios, 0.975),
            )
        )
    with (RESULTS / "all22_comparison.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    y = np.arange(len(records))
    labels = [r["query"] for r in records]
    note = (
        "SF1 · Parquet via KvikIO · CPU fallback disabled\n"
        "Three fresh processes per mode, five repetitions per query. "
        "Enabling the gate does not fuse every query."
    )
    fig, ax = plt.subplots(figsize=(11, 10))
    for mode_index, (offset, field, label, color) in enumerate(
        [
            (-0.19, "ordinary_ms", "Ordinary Sirius", COLORS[0]),
            (0.19, "fusion_enabled_ms", "Lazy fusion-enabled Sirius", COLORS[1]),
        ]
    ):
        data = [samples_by_query[r["query"]][mode_index] for r in records]
        values = np.array([r[field] for r in records])
        error = np.array(
            [values - [min(d) for d in data], [max(d) for d in data] - values]
        )
        ax.barh(
            y + offset,
            [r[field] for r in records],
            height=0.35,
            label=label,
            color=color,
            xerr=error,
            capsize=2,
            error_kw={"elinewidth": 0.8},
        )
        for row, (value, observed) in enumerate(zip(values, data)):
            ax.annotate(
                f"{value:.1f}",
                (max(observed), row + offset),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Whole SQL execution (ms; lower is faster)",
        title="All 22 TPC-H queries: ordinary versus fusion-enabled Sirius",
    )
    ax.invert_yaxis()
    ax.set_xlim(
        0, max(max(d) for modes in samples_by_query.values() for d in modes) * 1.18
    )
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.text(
        0.12,
        0.015,
        note
        + "\nBars: median of process medians; error bars: min/max process medians.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    save(fig, "sirius_all22_latency")

    fig, ax = plt.subplots(figsize=(10, 9))
    ratios = [r["ordinary_over_fusion_enabled"] for r in records]
    # Keep exploratory intervals visible; there are only three independent process pairs.
    ax.hlines(
        y,
        [min(1, ratio) for ratio in ratios],
        [max(1, ratio) for ratio in ratios],
        color=COLORS[0],
        linewidth=2,
    )
    ax.hlines(
        y,
        [r["ci_low"] for r in records],
        [r["ci_high"] for r in records],
        color=COLORS[0],
        alpha=0.3,
        linewidth=7,
    )
    ax.scatter(ratios, y, color=COLORS[0], s=36, zorder=3)
    for row, ratio in enumerate(ratios):
        ax.annotate(
            f"{ratio:.3f}×",
            (records[row]["ci_high"] if ratio >= 1 else records[row]["ci_low"], row),
            xytext=(7 if ratio >= 1 else -7, 0),
            textcoords="offset points",
            va="center",
            ha="left" if ratio >= 1 else "right",
            fontsize=9,
        )
    ax.axvline(1, color="#444444", linestyle="--", linewidth=1)
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Ordinary latency / fusion-enabled latency (>1 favors fusion-enabled)",
        title="All 22 queries: ratio of median whole-query latencies",
        xlim=(
            min(r["ci_low"] for r in records) - 0.10,
            max(r["ci_high"] for r in records) + 0.10,
        ),
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(
        0.12,
        0.015,
        note
        + "\nThick bands: exploratory 95% paired process-bootstrap intervals (three pairs).",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    save(fig, "sirius_all22_speedups")
    print("Generated all-22 latency and ratio plots (PNG/SVG), plus comparison CSV.")


if __name__ == "__main__":
    main()
