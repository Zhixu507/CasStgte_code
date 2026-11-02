"""Paper figure: LungHist700 held-out test performance.

Each point is the mean of one seed over the five patient-level test folds.
The marker is the mean across seeds 42, 43 and 44. The line spans the three seeds.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "experiments"
OUT_DIR = ROOT / "figures"

METHODS = [
    ("cnn", "CNN", "single"),
    ("transformer", "Transformer", "single"),
    ("mean", "Mean fusion", "fusion"),
    ("mean_no_stain", "Mean fusion (no stain)", "fusion"),
    ("learned_gate", "Learned gate", "learned"),
    ("reliability_accuracy", "Reliability (accuracy)", "ablation"),
    ("reliability_stability", "Reliability (stability)", "ablation"),
    ("reliability_joint", "Reliability (joint)", "proposed"),
]

# Paul Tol bright, distinguishable under common color-vision deficiencies.
COLORS = {
    "single": "#555555",
    "fusion": "#0077BB",
    "learned": "#EE7733",
    "ablation": "#009988",
    "proposed": "#CC3311",
}

METRICS = [
    ("accuracy", "Accuracy", True),
    ("macro_f1", "Macro-F1", True),
    ("macro_auroc", "Macro-AUROC", True),
    ("nll", "NLL", False),
    ("ece_15_bins", "ECE", False),
]

# Extra space between method families so the grouping is visible without a box.
Y = {
    "cnn": 0.0,
    "transformer": 1.0,
    "mean": 2.45,
    "mean_no_stain": 3.45,
    "learned_gate": 4.45,
    "reliability_accuracy": 5.9,
    "reliability_stability": 6.9,
    "reliability_joint": 7.9,
}


def seed_values(method, metric):
    by_seed = {}
    for path in (RUNS / method).glob("fold*_seed*/test/metrics.json"):
        row = json.loads(path.read_text())
        by_seed.setdefault(row["seed"], []).append(row[metric])
    if sorted(by_seed) != [42, 43, 44] or any(len(v) != 5 for v in by_seed.values()):
        raise RuntimeError(f"Expected 5 folds x seeds 42/43/44 for {method}")
    return np.array([np.mean(by_seed[seed]) for seed in (42, 43, 44)])


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "xtick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
    })


def draw(ax, metric, title, higher, data, letter, show_y):
    color_of = {key: COLORS[group] for key, _, group in METHODS}
    anchor = float(np.mean(data["mean_no_stain"][metric]))
    ax.axvline(anchor, color="#0077BB", lw=0.6, ls=(0, (2.2, 1.6)), zorder=0, alpha=0.85)

    joint_y = Y["reliability_joint"]
    ax.axhspan(joint_y - 0.42, joint_y + 0.42, color="#CC3311", alpha=0.07, zorder=0)

    for key, _, _ in METHODS:
        values = data[key][metric]
        y = Y[key]
        color = color_of[key]
        ax.plot([values.min(), values.max()], [y, y], color=color, lw=1.15,
                solid_capstyle="butt", zorder=2, alpha=0.95)
        ax.scatter(values, np.full(3, y), s=9, c=color, alpha=0.55, zorder=3, linewidths=0)
        mean = float(values.mean())
        ax.scatter([mean], [y], s=36 if key == "reliability_joint" else 26,
                   c=color, edgecolors="#222222", linewidths=0.45, zorder=4)

    labels = [label for _, label, _ in METHODS]
    ax.set_yticks([Y[key] for key, _, _ in METHODS])
    ax.set_yticklabels(labels if show_y else [])
    ax.tick_params(axis="y", length=0, pad=2)
    if show_y:
        for tick, key in zip(ax.get_yticklabels(), [k for k, _, _ in METHODS]):
            if key == "reliability_joint":
                tick.set_fontweight("bold")
                tick.set_color(COLORS["proposed"])

    lo = min(data[k][metric].min() for k, _, _ in METHODS)
    hi = max(data[k][metric].max() for k, _, _ in METHODS)
    span = hi - lo
    left = lo - span * 0.12
    # Leave a right gutter so the printed means sit clear of the seed range.
    right = left + (hi - left) / 0.70
    ax.set_xlim(left, right)
    ax.set_ylim(-0.7, 8.6)
    ax.xaxis.grid(True, ls=":", lw=0.4, color="#D0D0D0")
    ax.set_axisbelow(True)
    direction = "higher is better" if higher else "lower is better"
    ax.set_xlabel(title, labelpad=2)
    ax.set_title(rf"$\mathbf{{{letter}}}$", loc="left", pad=7, fontsize=10)
    ax.text(1.0, 1.012, direction, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#666666", style="italic")

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for key, _, _ in METHODS:
        mean = float(data[key][metric].mean())
        weight = "bold" if key == "reliability_joint" else "normal"
        ax.text(0.985, Y[key], f"{mean:.3f}", transform=transform, ha="right", va="center",
                fontsize=6.5, color=color_of[key], fontweight=weight, zorder=5)


def main():
    style()
    data = {
        key: {metric: seed_values(key, metric) for metric, _, _ in METRICS}
        for key, _, _ in METHODS
    }

    fig = plt.figure(figsize=(7.16, 4.85), facecolor="white")
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.52, wspace=0.62,
                             left=0.205, right=0.988, top=0.93, bottom=0.18)
    panels = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:3]),
        fig.add_subplot(grid[1, 3:6]),
    ]
    letters = "abcde"
    for ax, (metric, title, higher), letter, show_y in zip(
            panels, METRICS, letters, [True, False, False, True, False]):
        draw(ax, metric, title, higher, data, letter, show_y)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["single"],
               markeredgecolor="#222222", markeredgewidth=0.4, markersize=5.5, label="Single model"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["fusion"],
               markeredgecolor="#222222", markeredgewidth=0.4, markersize=5.5, label="Equal-weight fusion"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["learned"],
               markeredgecolor="#222222", markeredgewidth=0.4, markersize=5.5, label="Learned gate"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["ablation"],
               markeredgecolor="#222222", markeredgewidth=0.4, markersize=5.5, label="Reliability ablation"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["proposed"],
               markeredgecolor="#222222", markeredgewidth=0.4, markersize=6.2,
               label="Proposed (joint)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.50, 0.012), fontsize=7, handletextpad=0.35,
               columnspacing=1.05, borderaxespad=0)

    fig.text(0.50, 0.072,
             "Small dots and the bar are the three seeds (each seed is the mean of five test folds). "
             "Large dot: mean over seeds. Dashed line: mean fusion without stain.",
             ha="center", va="center", fontsize=6.5, color="#444444")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "lunghist_test_performance.pdf"
    png = OUT_DIR / "lunghist_test_performance.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
