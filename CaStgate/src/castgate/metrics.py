import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from .manifests import LABELS


def classification_metrics(labels, probabilities, label_names=None):
    names = tuple(LABELS) if label_names is None else tuple(label_names)
    y, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if len(y) == 0 or p.shape != (len(y), len(names)):
        raise ValueError(f"Expected nonempty [N,{len(names)}] probabilities and N labels")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid probabilities")
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=list(range(len(names))))
    support = cm.sum(1)
    recalls = np.divide(cm.diagonal(), support, out=np.full(len(names), np.nan), where=support > 0)
    aucs = {}
    for index, label in enumerate(names):
        binary = y == index
        aucs[label] = float(roc_auc_score(binary, p[:, index])) if 0 < binary.sum() < len(y) else None
    complete = all(v is not None for v in aucs.values())
    confidence = p.max(1)
    correctness = pred == y
    ece = 0.
    for lower, upper in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean())
    return {"n_images": len(y), "accuracy": float(correctness.mean()),
            "macro_f1": float(f1_score(y, pred, labels=list(range(len(names))), average="macro", zero_division=0)),
            "balanced_accuracy": float(np.nanmean(recalls)),
            "macro_auroc": float(np.mean(list(aucs.values()))) if complete else None,
            "auroc_ovr": aucs, "recall": {label: (float(recalls[i]) if support[i] else None)
                                          for i, label in enumerate(names)},
            "support": {label: int(support[i]) for i, label in enumerate(names)},
            "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-8)).mean()),
            "ece_15_bins": float(ece), "confusion_matrix": cm.tolist()}


def cluster_bootstrap(labels, probabilities, patient_ids, samples=1000, seed=42, label_names=None):
    """Patient-cluster resampling of image-level metrics; never assumes one label per patient."""
    if samples <= 0:
        return {}
    y, p, patients = np.asarray(labels), np.asarray(probabilities), np.asarray(patient_ids)
    unique = np.unique(patients)
    if len(unique) < 2:
        return {"note": "At least two patients required"}
    lookup = {g: np.where(patients == g)[0] for g in unique}
    rng = np.random.default_rng(seed)
    results = {"accuracy": [], "macro_f1": [], "macro_auroc": []}
    for _ in range(samples):
        idx = np.concatenate([lookup[g] for g in rng.choice(unique, len(unique), replace=True)])
        # AUROC is undefined when any class is absent. Do not silently impute it.
        if len(np.unique(y[idx])) < p.shape[1]:
            continue
        m = classification_metrics(y[idx], p[idx], label_names)
        for key in results:
            results[key].append(m[key])
    return {key: {"low": float(np.quantile(values, .025)),
                  "high": float(np.quantile(values, .975)), "valid_resamples": len(values),
                  "requested_resamples": samples, "unit": "patient_cluster"}
            if values else {"low": None, "high": None, "valid_resamples": 0,
                            "requested_resamples": samples, "unit": "patient_cluster"}
            for key, values in results.items()}
