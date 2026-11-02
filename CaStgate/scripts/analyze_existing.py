"""Summaries that do not refit models.

Paired intervals use the five patient folds as the unit: each fold difference is
the mean over seeds 42, 43 and 44, and the 95% interval is a Student-t interval
on those five differences. Seeds are technical replicates, not extra patients.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from castgate.engine import DEFAULT_CONFIG
from castgate.metrics import classification_metrics
from castgate.model import build_model

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "runs" / "experiments"
VAL = ROOT / "runs" / "followup" / "val"
OUT = ROOT / "runs" / "followup"
SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)
LABELS = ("aca", "scc", "nor")
TCRIT = float(stats.t.ppf(0.975, df=len(FOLDS) - 1))
PERMUTATIONS = 200
PERM_SEED = 20261024
ALPHAS = np.round(np.linspace(0, 1, 21), 2)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def matrix(rows, prefix):
    return np.array([[float(row[f"{prefix}{label}"]) for label in LABELS] for row in rows], dtype=float)


def labels_of(rows):
    return np.array([int(row["label"]) for row in rows], dtype=int)


def run_dirs(method):
    found = {}
    for fold in FOLDS:
        for seed in SEEDS:
            directory = EXP / method / f"fold{fold}_seed{seed}" / "test"
            if not (directory / "metrics.json").exists():
                raise FileNotFoundError(directory)
            found[(fold, seed)] = directory
    return found


def metric_of(directory, key):
    return json.loads((directory / "metrics.json").read_text())[key]


def fold_seed_values(method, key):
    return {key_: metric_of(directory, key) for key_, directory in run_dirs(method).items()}


def mean_sd(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def seed_of_fold_means(keyed):
    """keyed: (fold, seed) -> scalar. Returns three seed means of the five folds."""
    return [float(np.mean([keyed[(fold, seed)] for fold in FOLDS])) for seed in SEEDS]


def paired_interval(left, right):
    fold_means = []
    for fold in FOLDS:
        fold_means.append(float(np.mean([left[(fold, seed)] - right[(fold, seed)] for seed in SEEDS])))
    estimate = float(np.mean(fold_means))
    sd = float(np.std(fold_means, ddof=1))
    half = TCRIT * sd / np.sqrt(len(FOLDS))
    return {"estimate": estimate, "low": estimate - half, "high": estimate + half,
            "fold_means": fold_means, "df": len(FOLDS) - 1, "unit": "patient_fold",
            "includes_zero": bool(estimate - half <= 0 <= estimate + half)}


def per_class(method):
    stored = []
    matrices = []
    for seed in SEEDS:
        rows = []
        for fold in FOLDS:
            rows.extend(read_rows(run_dirs(method)[(fold, seed)] / "predictions.csv"))
        y, p = labels_of(rows), matrix(rows, "p_")
        metrics = classification_metrics(y, p)
        cm = np.asarray(metrics["confusion_matrix"], dtype=float)
        predicted = cm.sum(0)
        precision = np.divide(cm.diagonal(), predicted, out=np.zeros(3), where=predicted > 0)
        recall = np.array([metrics["recall"][label] for label in LABELS], dtype=float)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3),
                       where=(precision + recall) > 0)
        stored.append({"precision": precision, "recall": recall, "f1": f1,
                       "macro_f1": metrics["macro_f1"], "support": cm.sum(1)})
        matrices.append(cm)
    summary = {}
    for name in ("precision", "recall", "f1"):
        stack = np.vstack([item[name] for item in stored])
        summary[name] = {label: {"mean": float(stack[:, i].mean()), "sd": float(stack[:, i].std(ddof=1))}
                         for i, label in enumerate(LABELS)}
    cm = np.mean(matrices, axis=0)
    summary["confusion_mean_counts"] = cm.round(3).tolist()
    row_sum = cm.sum(1, keepdims=True)
    normalized = np.zeros_like(cm)
    np.divide(cm, row_sum, out=normalized, where=row_sum > 0)
    summary["confusion_row_normalized"] = normalized.round(4).tolist()
    summary["support_per_seed"] = {label: int(stored[0]["support"][i]) for i, label in enumerate(LABELS)}
    return summary


def magnification(method):
    result = {}
    for mag in ("20x", "40x"):
        per_seed = []
        patients, images = None, None
        for seed in SEEDS:
            rows = []
            for fold in FOLDS:
                rows.extend(read_rows(run_dirs(method)[(fold, seed)] / "predictions.csv"))
            chosen = [row for row in rows if row["magnification"] == mag]
            metrics = classification_metrics(labels_of(chosen), matrix(chosen, "p_"))
            per_seed.append(metrics)
            patients = len({row["patient_id"] for row in chosen})
            images = len(chosen)
        result[mag] = {"n_images": images, "n_patients": patients}
        for key in ("accuracy", "macro_f1", "macro_auroc", "nll", "ece_15_bins"):
            result[mag][key] = dict(zip(("mean", "sd"), mean_sd([item[key] for item in per_seed])))
        result[mag]["recall"] = {label: dict(zip(("mean", "sd"), mean_sd([item["recall"][label] for item in per_seed])))
                                 for label in LABELS}
    return result


def fuse(cnn, transformer, cnn_weight):
    weight = np.asarray(cnn_weight, dtype=float).reshape(-1, 1)
    return weight * cnn + (1.0 - weight) * transformer


def fixed_branch():
    """Fusion rules applied to frozen joint-checkpoint predictions."""
    runs = run_dirs("reliability_joint")
    names = ("cnn_branch", "transformer_branch", "equal", "learned_gate", "shuffled_gate")
    keyed = {name: {} for name in names}
    group_rows = []
    for (fold, seed), directory in runs.items():
        rows = read_rows(directory / "predictions.csv")
        y = labels_of(rows)
        cnn, transformer = matrix(rows, "cnn_p_"), matrix(rows, "transformer_p_")
        weight = np.array([float(row["weight_cnn"]) for row in rows])
        fused = matrix(rows, "p_")
        equal = fuse(cnn, transformer, np.full(len(rows), 0.5))
        keyed["cnn_branch"][(fold, seed)] = classification_metrics(y, cnn)
        keyed["transformer_branch"][(fold, seed)] = classification_metrics(y, transformer)
        keyed["equal"][(fold, seed)] = classification_metrics(y, equal)
        keyed["learned_gate"][(fold, seed)] = classification_metrics(y, fused)
        rng = np.random.default_rng(PERM_SEED + fold * 100 + seed)
        shuffled = []
        true_f1 = keyed["learned_gate"][(fold, seed)]["macro_f1"]
        for _ in range(PERMUTATIONS):
            shuffled.append(classification_metrics(y, fuse(cnn, transformer, weight[rng.permutation(len(y))]))["macro_f1"])
        keyed["shuffled_gate"][(fold, seed)] = {"macro_f1": float(np.mean(shuffled)),
                                                 "null_low": float(np.quantile(shuffled, 0.025)),
                                                 "null_high": float(np.quantile(shuffled, 0.975)),
                                                 "fraction_null_ge_true": float(np.mean(np.asarray(shuffled) >= true_f1 - 1e-12))}
        cnn_ok, tr_ok = cnn.argmax(1) == y, transformer.argmax(1) == y
        fused_ok, equal_ok = fused.argmax(1) == y, equal.argmax(1) == y
        groups = {"both": cnn_ok & tr_ok, "cnn_only": cnn_ok & ~tr_ok,
                  "transformer_only": ~cnn_ok & tr_ok, "neither": ~cnn_ok & ~tr_ok}
        record = {"fold": fold, "seed": seed}
        for name, mask in groups.items():
            record[name] = {"n": int(mask.sum()),
                            "mean_cnn_weight": float(weight[mask].mean()) if mask.any() else 0.0,
                            "gate_accuracy": float(fused_ok[mask].mean()) if mask.any() else 0.0,
                            "equal_accuracy": float(equal_ok[mask].mean()) if mask.any() else 0.0}
        single = groups["cnn_only"] | groups["transformer_only"]
        correct_side = np.where(cnn_ok, weight > 0.5, weight < 0.5)
        record["single_correct"] = {
            "n": int(single.sum()),
            "gate_accuracy": float(fused_ok[single].mean()),
            "equal_accuracy": float(equal_ok[single].mean()),
            "always_cnn_accuracy": float(cnn_ok[single].mean()),
            "always_transformer_accuracy": float(tr_ok[single].mean()),
            "prefers_correct_branch": float(correct_side[single].mean()),
            "mean_weight_on_correct": float(np.where(cnn_ok[single], weight[single], 1 - weight[single]).mean()),
        }
        group_rows.append(record)
    table = {}
    for name in names:
        if name == "shuffled_gate":
            values = seed_of_fold_means({key: item["macro_f1"] for key, item in keyed[name].items()})
            table[name] = {"macro_f1_mean": float(np.mean(values)), "macro_f1_sd": float(np.std(values, ddof=1)),
                           "permutations": PERMUTATIONS, "permutation_seed": PERM_SEED}
            continue
        entry = {}
        for metric in ("accuracy", "macro_f1", "macro_auroc", "nll", "ece_15_bins"):
            values = seed_of_fold_means({key: item[metric] for key, item in keyed[name].items()})
            entry[metric] = dict(zip(("mean", "sd"), mean_sd(values)))
        table[name] = entry
    def seed_totals(field):
        totals = []
        for seed in SEEDS:
            chosen = [row for row in group_rows if row["seed"] == seed]
            counts = np.array([row[field]["n"] for row in chosen], dtype=float)
            weights = np.array([row[field]["mean_cnn_weight"] for row in chosen], dtype=float)
            gate = np.array([row[field]["gate_accuracy"] for row in chosen], dtype=float)
            equal = np.array([row[field]["equal_accuracy"] for row in chosen], dtype=float)
            total = counts.sum()
            totals.append({"n": float(total),
                           "mean_cnn_weight": float(np.average(weights, weights=counts)),
                           "gate_accuracy": float(np.average(gate, weights=counts)),
                           "equal_accuracy": float(np.average(equal, weights=counts))})
        return totals

    pooled = {}
    for name in ("both", "cnn_only", "transformer_only", "neither"):
        totals = seed_totals(name)
        pooled[name] = {"n": dict(zip(("mean", "sd"), mean_sd([item["n"] for item in totals]))),
                        "mean_cnn_weight": dict(zip(("mean", "sd"), mean_sd([item["mean_cnn_weight"] for item in totals]))),
                        "gate_accuracy": dict(zip(("mean", "sd"), mean_sd([item["gate_accuracy"] for item in totals]))),
                        "equal_accuracy": dict(zip(("mean", "sd"), mean_sd([item["equal_accuracy"] for item in totals])))}
    single_by_seed = []
    for seed in SEEDS:
        chosen = [row["single_correct"] for row in group_rows if row["seed"] == seed]
        counts = np.array([item["n"] for item in chosen], dtype=float)
        entry = {"n": float(counts.sum())}
        for key in ("gate_accuracy", "equal_accuracy", "always_cnn_accuracy", "always_transformer_accuracy",
                    "prefers_correct_branch", "mean_weight_on_correct"):
            entry[key] = float(np.average([item[key] for item in chosen], weights=counts))
        single_by_seed.append(entry)
    single = {"n": dict(zip(("mean", "sd"), mean_sd([item["n"] for item in single_by_seed])))}
    for key in ("gate_accuracy", "equal_accuracy", "always_cnn_accuracy", "always_transformer_accuracy",
                "prefers_correct_branch", "mean_weight_on_correct"):
        single[key] = dict(zip(("mean", "sd"), mean_sd([item[key] for item in single_by_seed])))
    null_fraction = seed_of_fold_means({key: item["fraction_null_ge_true"] for key, item in keyed["shuffled_gate"].items()})
    table["shuffled_gate"]["fraction_null_ge_true"] = dict(zip(("mean", "sd"), mean_sd(null_fraction)))
    return {"rules": table, "groups": pooled, "single_correct": single,
            "note": "All rules use the saved joint branch probabilities. Alpha is fitted later on validation."}


def fit_alpha(y, cnn, transformer):
    best_alpha, best_score = 0.5, -1.0
    for alpha in ALPHAS:
        score = classification_metrics(y, fuse(cnn, transformer, np.full(len(y), alpha)))["macro_f1"]
        closer = abs(alpha - 0.5) < abs(best_alpha - 0.5) - 1e-12
        smaller_tie = abs(abs(alpha - 0.5) - abs(best_alpha - 0.5)) <= 1e-12 and alpha < best_alpha
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and (closer or smaller_tie)):
            best_alpha, best_score = float(alpha), score
    return best_alpha, best_score


def nll_temperature(y, probabilities, temperature):
    logp = np.log(np.clip(probabilities, 1e-8, 1.0))
    scaled = logp / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    calibrated = np.exp(scaled)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return classification_metrics(y, calibrated), calibrated


def fit_temperature(y, probabilities):
    def objective(temperature):
        metrics, _ = nll_temperature(y, probabilities, temperature)
        return metrics["nll"]

    from scipy.optimize import minimize_scalar
    fitted = minimize_scalar(objective, bounds=(0.05, 5.0), method="bounded", options={"xatol": 1e-4})
    if not fitted.success:
        raise RuntimeError("Temperature fit did not converge")
    return float(fitted.x)


def validation_analyses():
    if not VAL.exists():
        return None
    methods = sorted(path.name for path in VAL.iterdir() if path.is_dir())
    temperature, alpha = {}, {}
    for method in methods:
        keyed_raw, keyed_cal, temperatures = {}, {}, {}
        for fold in FOLDS:
            for seed in SEEDS:
                test_dir = EXP / method / f"fold{fold}_seed{seed}" / "test"
                val_dir = VAL / method / f"fold{fold}_seed{seed}"
                if not (val_dir / "predictions.csv").exists():
                    return None
                val_rows = read_rows(val_dir / "predictions.csv")
                test_rows = read_rows(test_dir / "predictions.csv")
                y_val, p_val = labels_of(val_rows), matrix(val_rows, "p_")
                y_test, p_test = labels_of(test_rows), matrix(test_rows, "p_")
                chosen = fit_temperature(y_val, p_val)
                calibrated, _ = nll_temperature(y_test, p_test, chosen)
                raw = classification_metrics(y_test, p_test)
                temperatures[(fold, seed)] = chosen
                keyed_raw[(fold, seed)] = raw
                keyed_cal[(fold, seed)] = calibrated
        entry = {"temperature": dict(zip(("mean", "sd"), mean_sd(seed_of_fold_means(temperatures))))}
        for stage, keyed in (("raw", keyed_raw), ("calibrated", keyed_cal)):
            entry[stage] = {}
            for metric in ("nll", "ece_15_bins", "accuracy", "macro_f1"):
                entry[stage][metric] = dict(zip(("mean", "sd"), mean_sd(seed_of_fold_means(
                    {key: value[metric] for key, value in keyed.items()}))))
        temperature[method] = entry
        if method == "reliability_joint":
            alphas, scores = {}, {}
            for fold in FOLDS:
                for seed in SEEDS:
                    val_rows = read_rows(VAL / method / f"fold{fold}_seed{seed}" / "predictions.csv")
                    test_rows = read_rows(EXP / method / f"fold{fold}_seed{seed}" / "test" / "predictions.csv")
                    chosen, _ = fit_alpha(labels_of(val_rows), matrix(val_rows, "cnn_p_"), matrix(val_rows, "transformer_p_"))
                    test_metrics = classification_metrics(
                        labels_of(test_rows),
                        fuse(matrix(test_rows, "cnn_p_"), matrix(test_rows, "transformer_p_"),
                             np.full(len(test_rows), chosen)))
                    alphas[(fold, seed)] = chosen
                    scores[(fold, seed)] = test_metrics
            alpha = {"grid": ALPHAS.tolist(), "tie_break": "closest to 0.5, then smaller alpha",
                     "alpha": dict(zip(("mean", "sd"), mean_sd(seed_of_fold_means(alphas)))),
                     "test": {}}
            for metric in ("accuracy", "macro_f1", "macro_auroc", "nll", "ece_15_bins"):
                alpha["test"][metric] = dict(zip(("mean", "sd"), mean_sd(seed_of_fold_means(
                    {key: value[metric] for key, value in scores.items()}))))
    return {"temperature": temperature, "global_alpha": alpha}


def parameter_counts():
    counts = {}
    for mode in ("cnn", "transformer", "mean", "learned_gate", "reliability_gate"):
        model = build_model(dict(DEFAULT_CONFIG, mode=mode, pretrained=False))
        counts[mode] = int(sum(parameter.numel() for parameter in model.parameters()))
    return counts


def training_minutes():
    found = {}
    for method_dir in sorted(path for path in EXP.iterdir() if path.is_dir()):
        seconds = []
        for history_path in method_dir.glob("fold*_seed*/history.json"):
            history = json.loads(history_path.read_text())
            seconds.append(sum(row["seconds"] for row in history))
        if seconds:
            found[method_dir.name] = dict(zip(("mean_minutes", "sd_minutes"),
                                              mean_sd(np.asarray(seconds) / 60)))
    return found


def main():
    methods = ["cnn", "transformer", "mean", "mean_no_stain", "learned_gate",
               "reliability_accuracy", "reliability_stability", "reliability_joint"]
    f1 = {method: fold_seed_values(method, "macro_f1") for method in methods}
    joint = f1["reliability_joint"]
    paired = {baseline: paired_interval(joint, f1[baseline]) for baseline in
              ("learned_gate", "mean_no_stain", "reliability_accuracy", "reliability_stability", "mean", "cnn", "transformer")}
    payload = {"paired_macro_f1": paired,
               "per_class": {method: per_class(method) for method in
                             ("cnn", "transformer", "mean", "mean_no_stain", "learned_gate", "reliability_joint")},
               "magnification": {method: magnification(method) for method in
                                 ("cnn", "transformer", "mean_no_stain", "reliability_joint")},
               "fixed_branch": fixed_branch(),
               "parameters": parameter_counts(),
               "training_minutes": training_minutes(),
               "validation": validation_analyses()}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "existing_summary.json"
    path.write_text(json.dumps(payload, indent=2))
    print(path)
    for name, item in paired.items():
        if name in ("learned_gate", "mean_no_stain", "reliability_accuracy", "reliability_stability"):
            print(f"{name}: {item['estimate']:+.4f} [{item['low']:+.4f}, {item['high']:+.4f}] includes0={item['includes_zero']}")


if __name__ == "__main__":
    main()
