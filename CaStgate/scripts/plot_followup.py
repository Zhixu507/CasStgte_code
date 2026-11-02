"""Paper figures for the follow-up analyses. Skips panels whose inputs are not ready."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((ROOT / "runs" / "followup" / "existing_summary.json").read_text())
OUT = ROOT / "figures"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def confusion():
    matrix = np.asarray(SUMMARY["per_class"]["reliability_joint"]["confusion_row_normalized"])
    counts = np.asarray(SUMMARY["per_class"]["reliability_joint"]["confusion_mean_counts"])
    labels = ["ACA", "SCC", "NOR"]
    fig, ax = plt.subplots(figsize=(3.4, 3.15))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > 0.55 else "#222222"
            ax.text(j, i, f"{matrix[i, j] * 100:.1f}%\n({counts[i, j]:.0f})",
                    ha="center", va="center", color=color, fontsize=7.5)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_joint.pdf")
    fig.savefig(OUT / "confusion_joint.png", dpi=300)
    plt.close(fig)


def fusion_rules():
    rules = SUMMARY["fixed_branch"]["rules"]
    order = [
        ("cnn_branch", "CNN branch"),
        ("transformer_branch", "Transformer branch"),
        ("equal", "Equal weights"),
        ("shuffled_gate", "Shuffled gate"),
        ("learned_gate", "Learned gate"),
    ]
    if SUMMARY.get("validation") and SUMMARY["validation"].get("global_alpha"):
        order.insert(3, ("global_alpha", "Global α"))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ys = np.arange(len(order))
    for y, (key, label) in enumerate(order):
        if key == "global_alpha":
            item = SUMMARY["validation"]["global_alpha"]["test"]["macro_f1"]
        elif key == "shuffled_gate":
            item = {"mean": rules[key]["macro_f1_mean"], "sd": rules[key]["macro_f1_sd"]}
        else:
            item = rules[key]["macro_f1"]
        color = "#CC3311" if key == "learned_gate" else "#0077BB"
        ax.plot([item["mean"] - item["sd"], item["mean"] + item["sd"]], [y, y], color=color, lw=1.2)
        ax.scatter([item["mean"]], [y], s=28, c=color, zorder=3, edgecolors="#222222", linewidths=0.4)
        ax.text(item["mean"] + item["sd"] + 0.002, y, f"{item['mean']:.3f}", va="center", fontsize=7, color=color)
    ax.set_yticks(ys, [label for _, label in order])
    ax.set_xlabel("Macro-F1")
    ax.set_ylim(-0.6, len(order) - 0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fusion_rules.pdf")
    fig.savefig(OUT / "fusion_rules.png", dpi=300)
    plt.close(fig)


def reliability_diagram():
    """Seed 42 only: one patient partition, temperature fit on that fold's validation set."""
    import csv
    from scipy.optimize import minimize_scalar
    root = ROOT / "runs"
    methods = [("mean_no_stain", "#0077BB"), ("learned_gate", "#EE7733"), ("reliability_joint", "#CC3311")]

    def read(path):
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        y = np.array([int(row["label"]) for row in rows])
        p = np.array([[float(row[f"p_{name}"]) for name in ("aca", "scc", "nor")] for row in rows])
        return y, p

    def apply_t(p, temperature):
        logp = np.log(np.clip(p, 1e-8, 1)) / temperature
        logp -= logp.max(1, keepdims=True)
        q = np.exp(logp)
        return q / q.sum(1, keepdims=True)

    def fit_t(y, p):
        def nll(temperature):
            q = apply_t(p, temperature)
            return -np.log(q[np.arange(len(y)), y]).mean()
        return float(minimize_scalar(nll, bounds=(0.05, 5), method="bounded").x)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.15), sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, calibrated in zip(axes, (False, True)):
        ax.plot([0, 1], [0, 1], ls=(0, (2, 2)), color="#888888", lw=0.8)
        for method, color in methods:
            ys, ps = [], []
            for fold in range(5):
                yv, pv = read(root / "followup" / "val" / method / f"fold{fold}_seed42" / "predictions.csv")
                yt, pt = read(root / "experiments" / method / f"fold{fold}_seed42" / "test" / "predictions.csv")
                if calibrated:
                    pt = apply_t(pt, fit_t(yv, pv))
                ys.append(yt)
                ps.append(pt)
            y, p = np.concatenate(ys), np.concatenate(ps)
            conf, correct = p.max(1), p.argmax(1) == y
            centers, accs = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (conf > lo) & (conf <= hi) if lo else (conf >= lo) & (conf <= hi)
                if mask.sum() < 8:
                    continue
                centers.append(conf[mask].mean())
                accs.append(correct[mask].mean())
            ax.plot(centers, accs, marker="o", ms=3.5, lw=1.1, color=color, label=method.replace("_", " "))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_title("After temperature scaling" if calibrated else "Unscaled")
    axes[0].set_ylabel("Accuracy")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "reliability_diagram.pdf")
    fig.savefig(OUT / "reliability_diagram.png", dpi=300)
    plt.close(fig)


def stress_curves():
    colors = {
        "cnn": "#555555", "transformer": "#999999", "mean": "#0077BB", "mean_no_stain": "#33BBEE",
        "learned_gate": "#EE7733", "reliability_accuracy": "#009988", "reliability_stability": "#66CCEE",
        "reliability_joint": "#CC3311",
    }
    labels = {
        "cnn": "CNN", "transformer": "Transformer", "mean": "Mean", "mean_no_stain": "Mean, no stain",
        "learned_gate": "Learned gate", "reliability_accuracy": "Reliability, accuracy",
        "reliability_stability": "Reliability, stability", "reliability_joint": "Reliability, joint",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), sharey=True)
    for ax, family, title in zip(axes, ("multiplicative", "additive"),
                                 ("Multiplicative concentration", "Additive concentration")):
        for method, color in colors.items():
            means, sds = [], []
            for strength in (0.0, 0.1, 0.2, 0.3, 0.4):
                seed_means = []
                for seed in (42, 43, 44):
                    folds = []
                    for fold in range(5):
                        path = (ROOT / "runs" / "followup" / "stress" / family / method
                                / f"fold{fold}_seed{seed}.json")
                        folds.append(json.loads(path.read_text())["by_strength"][f"{strength:.1f}"]["macro_f1"])
                    seed_means.append(float(np.mean(folds)))
                means.append(float(np.mean(seed_means)))
                sds.append(float(np.std(seed_means, ddof=1)))
            lw = 1.6 if method == "reliability_joint" else 1.0
            ax.errorbar([0, 0.1, 0.2, 0.3, 0.4], means, yerr=sds, color=color, lw=lw,
                        marker="o", ms=3.2, label=labels[method], capsize=1.5, elinewidth=0.6)
        ax.set_xlabel("Perturbation amplitude")
        ax.set_title(title)
    axes[0].set_ylabel("Macro-F1")
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "stain_stress.pdf")
    fig.savefig(OUT / "stain_stress.png", dpi=300)
    plt.close(fig)


def main():
    style()
    OUT.mkdir(exist_ok=True)
    confusion()
    fusion_rules()
    reliability_diagram()
    stress_curves()
    print(OUT / "confusion_joint.pdf")
    print(OUT / "fusion_rules.pdf")
    print(OUT / "reliability_diagram.pdf")


if __name__ == "__main__":
    main()
