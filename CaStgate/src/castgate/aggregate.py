import csv
import json
from pathlib import Path

import numpy as np

from .engine import save_json
from .metrics import classification_metrics, cluster_bootstrap


def aggregate(directories, output, bootstrap=1000):
    all_rows, metrics, image_ids, owners = [], [], set(), {}
    signature = None
    for fold_index, directory in enumerate(directories):
        directory = Path(directory)
        m = json.loads((directory / "metrics.json").read_text())
        if m["split"] != "test":
            raise ValueError("Aggregate only held-out fold test evaluations, not val/external results")
        current = {k: m[k] for k in ("mode", "seed", "synthetic", "stain_strength", "label_mapping",
                                      "development_manifest_sha256", "protocol")}
        if signature is not None and current != signature:
            raise ValueError("Cannot pool different protocols/seeds/datasets; summarize them separately")
        signature = current
        with (directory / "predictions.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["image_id"] in image_ids:
                raise ValueError("Duplicate test image: do not pool repeated seeds as independent observations")
            if row["patient_id"] in owners and owners[row["patient_id"]] != fold_index:
                raise ValueError("Patient appears in multiple fold test sets")
            image_ids.add(row["image_id"])
            owners[row["patient_id"]] = fold_index
        all_rows.extend(rows)
        metrics.append(m)
    if not all_rows:
        raise ValueError("No predictions")
    names = list(signature["label_mapping"])
    y = np.array([int(r["label"]) for r in all_rows])
    p = np.array([[float(r[f"p_{label}"]) for label in names] for r in all_rows])
    result = classification_metrics(y, p, names)
    result.update(signature, n_folds=len(metrics),
                  note="Pooled held-out images; confirm all planned folds are supplied. Fold models share training patients.",
                  bootstrap_ci95=cluster_bootstrap(y, p, [r["patient_id"] for r in all_rows], bootstrap,
                                                   label_names=names),
                  fold_macro_f1_mean=float(np.mean([m["macro_f1"] for m in metrics])),
                  fold_macro_f1_sd=float(np.std([m["macro_f1"] for m in metrics], ddof=1)) if len(metrics)>1 else None)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output)
    return result
