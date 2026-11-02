"""Aggregate follow-up outputs. Missing experiments are reported as absent."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLLOW = ROOT / "runs" / "followup"
SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)


def mean_sd(values):
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return None
    if len(array) == 1:
        return {"mean": float(array[0]), "sd": None, "n": 1}
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1)), "n": int(len(array))}


def seed_means(keyed):
    return [float(np.mean([keyed[(fold, seed)] for fold in FOLDS]))
            for seed in SEEDS if all((fold, seed) in keyed for fold in FOLDS)]


def stress(family):
    root = FOLLOW / "stress" / family
    if not root.exists():
        return None
    methods = {}
    for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        keyed = {}
        for path in method_dir.glob("fold*_seed*.json"):
            fold = int(path.stem.split("_")[0].replace("fold", ""))
            seed = int(path.stem.split("_")[1].replace("seed", ""))
            keyed[(fold, seed)] = json.loads(path.read_text())["by_strength"]
        if len(keyed) != 15:
            methods[method_dir.name] = {"complete": len(keyed)}
            continue
        summary = {}
        for strength in ("0.0", "0.1", "0.2", "0.3", "0.4"):
            summary[strength] = {}
            for metric in ("macro_f1", "nll", "ece_15_bins", "delta_f1"):
                values = seed_means({key: item[strength][metric] for key, item in keyed.items()})
                summary[strength][metric] = mean_sd(values)
        methods[method_dir.name] = summary
    return methods


def target():
    root = FOLLOW / "target"
    if not root.exists():
        return None
    rows = []
    for path in sorted(root.glob("fold*_seed*.json")):
        payload = json.loads(path.read_text())
        payload.pop("delta_error", None)
        payload.pop("beta_delta_stability", None)
        payload.pop("preference_changed", None)
        rows.append(payload)
    if len(rows) != 15:
        return {"complete": len(rows)}
    keys = ["preference_change_fraction", "median_abs_error_gap", "median_abs_stability_gap",
            "stability_to_error_median_ratio", "fraction_stability_exceeds_error", "pearson_error_stability"]
    # Pool within seed across folds by image-weighted... files are per fold, so weight by n_images.
    summary = {}
    for key in keys:
        per_seed = []
        for seed in SEEDS:
            chosen = [row for path, row in zip(sorted(root.glob("fold*_seed*.json")), rows)
                      if f"seed{seed}" in path.name]
            # zip above is unsafe if pop changed order. Re-read lightly.
        per_seed = []
        for seed in SEEDS:
            chosen = []
            for fold in FOLDS:
                payload = json.loads((root / f"fold{fold}_seed{seed}.json").read_text())
                chosen.append(payload)
            weights = np.array([item["n_images"] for item in chosen], dtype=float)
            per_seed.append(float(np.average([item[key] for item in chosen], weights=weights)))
        summary[key] = mean_sd(per_seed)
    return summary


def training():
    found = {}
    for method_dir in sorted(path for path in FOLLOW.iterdir() if (path / "fold0_seed42").exists()):
        keyed = {}
        for fold in FOLDS:
            for seed in SEEDS:
                metrics = method_dir / f"fold{fold}_seed{seed}" / "test" / "metrics.json"
                if metrics.exists():
                    keyed[(fold, seed)] = json.loads(metrics.read_text())
        if len(keyed) != 15:
            found[method_dir.name] = {"complete": len(keyed)}
            continue
        entry = {}
        for metric in ("accuracy", "macro_f1", "macro_auroc", "nll", "ece_15_bins"):
            values = seed_means({key: item[metric] for key, item in keyed.items() if item.get(metric) is not None})
            entry[metric] = mean_sd(values) if len(values) == 3 else {"incomplete_auroc_seeds": len(values)}
        found[method_dir.name] = entry
    return found


def main():
    payload = {"stress_multiplicative": stress("multiplicative"),
               "stress_additive": stress("additive"),
               "target": target(),
               "training": training(),
               "timing": json.loads((FOLLOW / "inference_timing.json").read_text())
               if (FOLLOW / "inference_timing.json").exists() else None}
    path = FOLLOW / "followup_summary.json"
    path.write_text(json.dumps(payload, indent=2))
    print(path)
    print("training", {name: (item.get("complete") if "complete" in item else "done")
                       for name, item in payload["training"].items()})


if __name__ == "__main__":
    main()
