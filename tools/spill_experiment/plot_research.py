from itertools import combinations
import matplotlib.pyplot as plt

sizes = [1536, 1024, 384, 256]


def current(deficit):
    candidates = sizes[:]
    remaining = deficit
    while remaining > 0:
        viable = [i for i, size in enumerate(candidates) if size >= remaining]
        i = min(viable, key=candidates.__getitem__) if viable else 0
        remaining -= candidates.pop(i)
    return deficit - remaining


def totals(n):
    return sorted({sum(p) for k in range(1, n + 1) for p in combinations(sizes, k)})


def best(deficit, options, fallback):
    return next((size for size in options if size >= deficit), fallback)


x = range(1, sum(sizes) + 1)
baseline = [current(d) for d in x]
pair_totals = totals(2)
all_totals = totals(len(sizes))
pair = [min(b, best(d, pair_totals, b)) for d, b in zip(x, baseline)]
oracle = [best(d, all_totals, b) for d, b in zip(x, baseline)]
assert all(o <= p <= b for o, p, b in zip(oracle, pair, baseline))
assert sum(b > p for b, p in zip(baseline, pair)) == 640
fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=(8.5, 5.4), layout="constrained")
colors = ["#474e59", "#007579", "#ae481c"]
for ys, label, color, ls in zip(
    [baseline, pair, oracle],
    [
        "Current Sirius selector",
        "Pair check with current fallback",
        "Minimum-byte subset",
    ],
    colors,
    ["-", "--", ":"],
):
    a.step(x, ys, where="post", label=label, color=color, linestyle=ls, linewidth=2)
a.legend(loc="upper left", frameon=False)
a.set_ylabel("Evicted volume (MiB)")
b.fill_between(
    x, [v - p for v, p in zip(baseline, pair)], step="post", color=colors[1], alpha=0.78
)
b.set_ylabel("Saved by pair check (MiB)")
b.set_xlabel("Reservation shortfall (MiB)")
for ax in (a, b):
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.axvline(385, color="#919191", linewidth=1)
a.scatter([385], [1024], color=colors[0], zorder=4)
b.annotate(
    "385 MiB shortfall\n384 MiB saved",
    (385, 384),
    (620, 315),
    arrowprops={"arrowstyle": "->"},
)
fig.suptitle("Whole-batch eviction creates cliffs under GPU memory pressure")
fig.savefig("tools/spill_experiment/research-pair-sweep.png", dpi=165)
