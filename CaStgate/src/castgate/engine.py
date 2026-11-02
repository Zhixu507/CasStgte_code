from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import HistologyDataset
from .losses import training_loss
from .manifests import LABELS, content_hashes, fingerprint, load_fold, mapping_for_labels, read_manifest
from .metrics import classification_metrics, cluster_bootstrap
from .model import build_model

DEFAULT_CONFIG = dict(manifest="data/lunghist700/manifest.csv", splits="data/lunghist700/splits.json",
                      output="runs/default", fold=0, seed=42, image_size=224, batch_size=8,
                      num_workers=0, epochs=40, warmup_epochs=5, patience=10, lr=1e-4,
                      weight_decay=.01, device="auto", cpu_threads=4, pretrained=True,
                      cnn="resnet18", transformer="vit_tiny_patch16_224", gate_hidden=128,
                      cnn_weights=None, transformer_weights=None,
                      mode="reliability_gate", num_stain_views=2, stain_strength=.15,
                      reliability="joint", stability_beta=1., temperature=.5, lambda_branch=1.,
                      lambda_gate=.5, bootstrap_samples=1000, synthetic=False,
                      shuffle_gate_target=False, train_magnification=None, test_magnification=None,
                      perturbation="stain", geometry_mode="histology", class_names=None)


def save_json(payload, path):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_config(path):
    c = DEFAULT_CONFIG.copy()
    supplied = json.loads(Path(path).read_text())
    unknown = set(supplied) - set(c)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    c.update(supplied)
    return c


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def label_dict(config):
    names = config.get("class_names")
    if not names:
        return dict(LABELS)
    if len(names) != len(set(names)):
        raise ValueError("class_names contains duplicates")
    return {name: index for index, name in enumerate(names)}


def loader(rows, config, training=False, eval_strength=0.):
    ds = HistologyDataset(rows, size=config["image_size"], training=training,
                         num_stain_views=config["num_stain_views"], stain_strength=config["stain_strength"],
                         eval_stain_strength=eval_strength, seed=config["seed"],
                         perturbation=config.get("perturbation", "stain"),
                         geometry_mode=config.get("geometry_mode", "histology"),
                         labels=label_dict(config))
    return DataLoader(ds, batch_size=config["batch_size"], shuffle=training,
                      num_workers=config["num_workers"], worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config["seed"]), drop_last=False)


@torch.inference_mode()
def predict(model, data_loader, device):
    model.eval()
    labels, probabilities, branch_probs, weights = [], [], [], []
    records = []
    for batch in data_loader:
        outputs = model(batch["views"][:, 0].to(device))
        labels.extend(batch["label"].tolist())
        probabilities.extend(outputs["probs"].cpu().tolist())
        branch_probs.extend(outputs["branch_logits"].float().softmax(-1).cpu().tolist())
        weights.extend(outputs["weights"].cpu().tolist())
        records.extend(dict(image_id=i, patient_id=p, magnification=m)
                       for i, p, m in zip(batch["image_id"], batch["patient_id"], batch["magnification"]))
    return np.asarray(labels), np.asarray(probabilities), np.asarray(branch_probs), np.asarray(weights), records


def validate_config(c):
    if c["epochs"] < 1 or c["batch_size"] < 1 or c["num_stain_views"] < 0:
        raise ValueError("epochs and batch_size must be positive; num_stain_views nonnegative")
    if c["mode"] in ("learned_gate", "reliability_gate") and not 0 <= c["warmup_epochs"] < c["epochs"]:
        raise ValueError("Gated modes require 0 <= warmup_epochs < epochs")
    if c["mode"] == "reliability_gate" and c["num_stain_views"] < 1:
        raise ValueError("Reliability supervision needs at least one additional stain view")
    if not 0 <= c["stain_strength"] <= 1 or c["temperature"] <= 0:
        raise ValueError("Invalid stain_strength or temperature")
    for key in ("stability_beta", "lambda_gate", "lambda_branch", "weight_decay"):
        if c[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if c["lr"] <= 0:
        raise ValueError("lr must be positive")
    if c.get("shuffle_gate_target") and c["mode"] != "reliability_gate":
        raise ValueError("shuffle_gate_target only applies to reliability_gate")
    if c.get("perturbation", "stain") not in ("stain", "illumination"):
        raise ValueError("perturbation must be stain or illumination")
    if c.get("geometry_mode", "histology") not in ("histology", "industrial"):
        raise ValueError("geometry_mode must be histology or industrial")
    if c.get("perturbation") == "illumination" and c.get("geometry_mode") != "industrial":
        raise ValueError("Illumination views require the shared full-image geometry")
    if c.get("class_names") is not None and (not c["class_names"] or len(c["class_names"]) != len(set(c["class_names"]))):
        raise ValueError("class_names must be a non-empty list of unique labels")


def filter_magnification(rows, magnification):
    """Keep one magnification. None keeps every row. Empty selections are rejected."""
    if not magnification:
        return rows
    kept = [row for row in rows if row["magnification"] == magnification]
    if not kept:
        raise ValueError(f"No images with magnification {magnification}")
    return kept


def train(config):
    validate_config(config)
    seed_all(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    subsets = load_fold(config["manifest"], config["splits"], config["fold"])
    rows = read_manifest(config["manifest"])
    mapping = mapping_for_labels(row["label"] for row in rows)
    if config.get("class_names") and list(config["class_names"]) != list(mapping):
        raise ValueError(f"class_names must use the published order {list(mapping)}")
    config["class_names"] = list(mapping)
    out = Path(config["output"])
    if (out / "config.json").exists():
        raise FileExistsError(f"Run already exists: {out}; choose a new --output")
    out.mkdir(parents=True, exist_ok=True)
    save_json(config, out / "config.json")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "timm", "numpy", "scikit-learn", "Pillow")}
    save_json(dict(python=platform.python_version(), packages=versions), out / "environment.json")
    if any(r["source"] == "SYNTHETIC" for r in rows) and not config["synthetic"]:
        raise ValueError("Synthetic data must be explicitly marked synthetic=true")
    current_hashes = content_hashes(rows)
    split_spec = json.loads(Path(config["splits"]).read_text())
    if split_spec["content_sha256"] != current_hashes:
        raise ValueError("Image files changed since split creation")
    if config.get("train_magnification"):
        subsets["train"] = filter_magnification(subsets["train"], config["train_magnification"])
        subsets["val"] = filter_magnification(subsets["val"], config["train_magnification"])
    if config.get("test_magnification"):
        subsets["test"] = filter_magnification(subsets["test"], config["test_magnification"])
    provenance = dict(manifest_sha256=fingerprint(rows),
                      development_content_sha256=sorted(set(current_hashes.values())),
                      development_patient_keys=sorted({r["source"] + ":" + r["patient_id"] for r in rows}),
                      train_ids=[r["image_id"] for r in subsets["train"]],
                      train_patients=sorted({r["patient_id"] for r in subsets["train"]}),
                      val_ids=[r["image_id"] for r in subsets["val"]],
                      test_ids=[r["image_id"] for r in subsets["test"]])
    save_json(provenance, out / "split_provenance.json")
    device = get_device(config["device"])
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    training_loader = loader(subsets["train"], config, training=True)
    validation_loader = loader(subsets["val"], config)
    history, best, stale = [], float("-inf"), 0
    for epoch in range(config["epochs"]):
        start = time.time()
        model.train()
        warmup = epoch < config["warmup_epochs"] and config["mode"] in ("learned_gate", "reliability_gate")
        sums, n = defaultdict(float), 0
        for batch in training_loader:
            images, labels = batch["views"].to(device), batch["label"].to(device)
            b, v = images.shape[:2]
            flat = images.flatten(0, 1)
            teacher_logits = None
            if config["mode"] == "reliability_gate" and not warmup:
                # Same current network in eval mode; no stochastic dropout or BN updates in targets.
                model.eval()
                with torch.no_grad():
                    teacher_logits = model(flat)["branch_logits"].reshape(b, v, 2, -1)
                model.train()
            outputs = model(flat, uniform_gate=warmup)
            loss, scalars = training_loss(outputs, labels, b, v, config, warmup, teacher_logits)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; no checkpoint was silently accepted")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            for key, value in scalars.items():
                sums[key] += value * b
            n += b
        scheduler.step()
        y, p, _, _, _ = predict(model, validation_loader, device)
        metrics = classification_metrics(y, p, config["class_names"])
        record = dict(epoch=epoch + 1, warmup=warmup, seconds=round(time.time() - start, 2),
                      train={k: value / n for k, value in sums.items()}, val=metrics)
        history.append(record)
        save_json(history, out / "history.json")
        print(f"epoch={epoch + 1} loss={sums['loss']/n:.4f} val_macro_f1={metrics['macro_f1']:.4f} warmup={warmup}", flush=True)
        # Do not select an untrained gate from warmup as the final model.
        if not warmup:
            if metrics["macro_f1"] > best:
                best, stale = metrics["macro_f1"], 0
                torch.save(dict(model=model.state_dict(), config=config, epoch=epoch + 1,
                                val_metrics=metrics, provenance=provenance,
                                labels=label_dict(config)), out / "best.pt")
            else:
                stale += 1
            if config["patience"] > 0 and stale >= config["patience"]:
                break
    # Test remains untouched until explicit evaluate; no hyperparameter search on it.
    return {"checkpoint": str(out / "best.pt"), "best_val_macro_f1": best,
            "synthetic": config["synthetic"], "test_evaluated": False}


def evaluate(checkpoint, output, split="test", manifest=None, stain_strength=0., bootstrap=None,
             device_name="auto"):
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    c = payload["config"].copy()
    seed_all(c["seed"])
    torch.set_num_threads(c["cpu_threads"])
    saved_labels = payload["labels"]
    if list(saved_labels) != list(label_dict(c)):
        raise ValueError("Checkpoint label mapping does not match its config")
    if manifest:
        rows = read_manifest(manifest)
        if fingerprint(rows) == payload["provenance"]["manifest_sha256"]:
            raise ValueError("External manifest is the development dataset; use --split instead")
        if set(content_hashes(rows).values()).intersection(payload["provenance"]["development_content_sha256"]):
            raise ValueError("External set contains byte-identical development images")
        keys = {r["source"] + ":" + r["patient_id"] for r in rows}
        if keys.intersection(payload["provenance"]["development_patient_keys"]):
            raise ValueError("External set overlaps development patients from the same source")
        split = "external"
    else:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train/val/test")
        all_rows = read_manifest(c["manifest"])
        if fingerprint(all_rows) != payload["provenance"]["manifest_sha256"]:
            raise ValueError("Development manifest does not match checkpoint")
        if sorted(set(content_hashes(all_rows).values())) != payload["provenance"]["development_content_sha256"]:
            raise ValueError("Development image content changed since training")
        ids = set(payload["provenance"][f"{split}_ids"])
        rows = [r for r in all_rows if r["image_id"] in ids]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"Evaluation already exists: {out}")
    device = get_device(device_name)
    model = build_model(c, pretrained=False).to(device)
    model.load_state_dict(payload["model"])
    y, p, bp, g, records = predict(model, loader(rows, c, eval_strength=stain_strength), device)
    names = list(saved_labels)
    result = classification_metrics(y, p, names)
    result.update(split=split, checkpoint=str(checkpoint.resolve()), epoch=payload["epoch"],
                  synthetic=c["synthetic"] or any(r["source"] == "SYNTHETIC" for r in rows),
                  stain_strength=stain_strength, label_mapping=saved_labels, mode=c["mode"],
                  seed=c["seed"], fold=c["fold"],
                  development_manifest_sha256=payload["provenance"]["manifest_sha256"],
                  protocol={k: c.get(k) for k in ("cnn", "transformer", "cnn_weights", "transformer_weights",
                            "image_size", "pretrained", "mode",
                            "num_stain_views", "stain_strength", "reliability", "stability_beta",
                            "temperature", "lambda_branch", "lambda_gate", "warmup_epochs",
                            "epochs", "lr", "weight_decay", "batch_size", "patience",
                            "shuffle_gate_target", "train_magnification", "test_magnification",
                            "perturbation", "geometry_mode", "class_names")},
                  evaluation_unit="image", n_patients=len({r["patient_id"] for r in records}),
                  evaluation_manifest_sha256=fingerprint(rows),
                  bootstrap_ci95=cluster_bootstrap(y, p, [r["patient_id"] for r in records],
                                                   c["bootstrap_samples"] if bootstrap is None else bootstrap,
                                                   c["seed"], names))
    result["by_magnification"] = {}
    for mag in sorted({r["magnification"] for r in records}):
        mask = np.array([r["magnification"] == mag for r in records])
        result["by_magnification"][mag] = classification_metrics(y[mask], p[mask], names)
    if c["mode"] not in ("cnn", "transformer"):
        result["branches"] = {"cnn": classification_metrics(y, bp[:, 0], names),
                              "transformer": classification_metrics(y, bp[:, 1], names)}
        correct = bp.argmax(-1) == y[:, None]
        informative = correct[:, 0] != correct[:, 1]
        chosen = g.argmax(-1)
        result["gate_diagnostics"] = {
            "mean_cnn_weight": float(g[:, 0].mean()),
            "one_branch_correct_images": int(informative.sum()),
            "fraction_weight_ties": float(np.isclose(g[:, 0], g[:, 1]).mean()),
            "correct_branch_preference": float(correct[np.arange(len(y)), chosen][informative].mean())
            if informative.any() and c["mode"] != "mean" else None,
            "note": "Descriptive single-view correctness diagnostic; not a causal explanation."}
    save_json(result, out / "metrics.json")
    fields = ["image_id", "patient_id", "magnification", "label", "prediction",
              *[f"p_{k}" for k in saved_labels], "weight_cnn", "weight_transformer",
              *[f"{branch}_p_{k}" for branch in ("cnn", "transformer") for k in saved_labels]]
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, record in enumerate(records):
            row = dict(record, label=int(y[i]), prediction=int(p[i].argmax()),
                       weight_cnn=float(g[i, 0]), weight_transformer=float(g[i, 1]))
            row.update({f"p_{k}": float(p[i, j]) for k, j in saved_labels.items()})
            row.update({f"{branch}_p_{k}": float(bp[i, b, j])
                        for b, branch in enumerate(("cnn", "transformer")) for k, j in saved_labels.items()})
            writer.writerow(row)
    return result
