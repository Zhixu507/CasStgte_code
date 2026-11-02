"""Validation predictions, stain-stress tests, and target diagnostics.

Stress strengths, replicate count, and the additive stain family were fixed
before any stress-test metric was computed. Replicate seeds depend only on the
image, family, strength, and replicate index, so every method sees the same pixels.
"""

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from castgate.augment import MEAN, STD, STAIN_INVERSE, STAIN_MATRIX, geometry, stain_perturb, stain_shift, tensorize
from castgate.engine import get_device, load_config, save_json, seed_all
from castgate.losses import reliability_target
from castgate.manifests import LABELS, read_manifest
from castgate.metrics import classification_metrics
from castgate.model import build_model

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "runs" / "experiments"
FOLLOW = ROOT / "runs" / "followup"
STRENGTHS = (0.0, 0.1, 0.2, 0.3, 0.4)
REPLICATES = 5


def version_seed(family, strength, replicate, image_id):
    raw = f"{family}:{strength:.2f}:{replicate}:{image_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def checkpoints():
    return sorted(EXP.glob("*/fold*_seed*/best.pt"))


def perturb(image, family, strength, replicate, image_id):
    if strength == 0:
        return image
    rng = np.random.default_rng(version_seed(family, strength, replicate, image_id))
    if family == "multiplicative":
        return stain_perturb(image, strength, rng)
    if family == "additive":
        return stain_shift(image, strength, rng)
    raise ValueError(family)


def load_rows(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]
    by_id = {row["image_id"]: row for row in read_manifest(config["manifest"])}
    provenance = json.loads((checkpoint.parent / "split_provenance.json").read_text())
    return payload, config, by_id, provenance


def load_base_images(rows):
    images = []
    for row in rows:
        with Image.open(row["path"]) as handle:
            image = geometry(handle.convert("RGB"), 224, False)
        images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images)


def apply_family(images, family, strength, replicate, image_ids):
    if strength == 0:
        return images
    factors = np.zeros((len(images), 3), dtype=np.float32)
    for index, image_id in enumerate(image_ids):
        rng = np.random.default_rng(version_seed(family, strength, replicate, image_id))
        if family == "multiplicative":
            factors[index, :2] = rng.uniform(1 - strength, 1 + strength, size=2)
            factors[index, 2] = 1
        else:
            factors[index, :2] = rng.uniform(-strength, strength, size=2)
    concentrations = (-np.log((images.astype(np.float32) + 1.0) / 256.0)) @ STAIN_INVERSE
    if family == "multiplicative":
        concentrations = concentrations * factors[:, None, None, :]
    else:
        concentrations = concentrations + factors[:, None, None, :]
    restored = np.exp(-np.maximum(concentrations @ STAIN_MATRIX, 0)) * 256.0 - 1
    return np.clip(restored, 0, 255).round().astype(np.uint8)


def predict_arrays(model, images, labels, device):
    array = images.astype(np.float32) / 255.0
    array = (array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2)))
    probabilities = []
    for start in range(0, len(tensor), 32):
        batch = tensor[start:start + 32].to(device)
        with torch.inference_mode():
            probabilities.append(model(batch)["probs"].cpu())
    return labels, torch.cat(probabilities).numpy()


def stress_one(checkpoint, family, device_name):
    method = checkpoint.parents[1].name
    leaf = checkpoint.parent.name
    output = FOLLOW / "stress" / family / method / f"{leaf}.json"
    if output.exists():
        return
    payload, config, by_id, provenance = load_rows(checkpoint)
    rows = [by_id[image_id] for image_id in provenance["test_ids"]]
    seed_all(config["seed"])
    device = get_device(device_name)
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    images = load_base_images(rows)
    labels = np.array([LABELS[row["label"]] for row in rows])
    image_ids = [row["image_id"] for row in rows]
    _, clean_p = predict_arrays(model, images, labels, device)
    clean = classification_metrics(labels, clean_p)
    result = {"family": family, "strengths": STRENGTHS, "replicates": REPLICATES,
              "checkpoint": str(checkpoint), "n_images": len(rows), "by_strength": {}}
    for strength in STRENGTHS:
        reps = [0] if strength == 0 else list(range(REPLICATES))
        metrics = []
        for replicate in reps:
            if strength == 0:
                metrics.append(clean)
            else:
                perturbed = apply_family(images, family, strength, replicate, image_ids)
                _, probabilities = predict_arrays(model, perturbed, labels, device)
                metrics.append(classification_metrics(labels, probabilities))
        entry = {}
        for key in ("macro_f1", "nll", "ece_15_bins", "accuracy"):
            values = [item[key] for item in metrics]
            entry[key] = float(np.mean(values))
            entry[f"{key}_sd_replicates"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        entry["delta_f1"] = float(clean["macro_f1"] - entry["macro_f1"])
        result["by_strength"][f"{strength:.1f}"] = entry
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output)


def target_one(checkpoint, device_name):
    leaf = checkpoint.parent.name
    output = FOLLOW / "target" / f"{leaf}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return
    payload, config, by_id, provenance = load_rows(checkpoint)
    rows = [by_id[image_id] for image_id in provenance["val_ids"]]
    device = get_device(device_name)
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    delta_e, delta_s, changed = [], [], []
    for start in range(0, len(rows), 8):
        batch_rows = rows[start:start + 8]
        views, labels = [], []
        for row in batch_rows:
            with Image.open(row["path"]) as handle:
                image = geometry(handle.convert("RGB"), config["image_size"], False)
            rng = np.random.default_rng(version_seed("target", config["stain_strength"], 0, row["image_id"]))
            stained = [image]
            for _ in range(config["num_stain_views"]):
                stained.append(stain_perturb(image, config["stain_strength"], rng))
            views.append(torch.stack([tensorize(item) for item in stained]))
            labels.append(LABELS[row["label"]])
        flat = torch.stack(views).to(device)
        b, v = flat.shape[:2]
        label = torch.tensor(labels, device=device)
        with torch.inference_mode():
            logits = model(flat.flatten(0, 1))["branch_logits"].reshape(b, v, 2, -1)
        joint, parts = reliability_target(logits, label, beta=config["stability_beta"],
                                          temperature=config["temperature"], component="joint")
        accuracy, _ = reliability_target(logits, label, beta=config["stability_beta"],
                                         temperature=config["temperature"], component="accuracy")
        nll, js = parts["nll"], parts["js"]
        error = nll[:, 0] - nll[:, 1]
        stability = config["stability_beta"] * (js[:, 0] - js[:, 1])
        delta_e.extend(error.cpu().tolist())
        delta_s.extend(stability.cpu().tolist())
        changed.extend((joint.argmax(-1) != accuracy.argmax(-1)).cpu().tolist())
    error = np.asarray(delta_e)
    stability = np.asarray(delta_s)
    changed = np.asarray(changed)
    scale = np.median(np.abs(stability)) / max(np.median(np.abs(error)), 1e-8)
    save_json({"n_images": len(rows), "preference_change_fraction": float(changed.mean()),
               "median_abs_error_gap": float(np.median(np.abs(error))),
               "median_abs_stability_gap": float(np.median(np.abs(stability))),
               "stability_to_error_median_ratio": float(scale),
               "fraction_stability_exceeds_error": float(np.mean(np.abs(stability) > np.abs(error))),
               "pearson_error_stability": float(np.corrcoef(error, stability)[0, 1]),
               "delta_error": error.tolist(), "beta_delta_stability": stability.tolist(),
               "preference_changed": changed.astype(int).tolist()}, output)


def run_pool(jobs, worker, gpus):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    failed = []

    def loop(gpu):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                worker(job, gpu)
                print(f"gpu{gpu} ok {job}", flush=True)
            except Exception as exc:
                failed.append((str(job), str(exc)))
                print(f"gpu{gpu} FAILED {job}: {exc}", flush=True)

    import threading
    threads = [threading.Thread(target=loop, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failed:
        raise RuntimeError(failed)


def command_val(gpus):
    jobs = []
    for checkpoint in checkpoints():
        output = FOLLOW / "val" / checkpoint.parents[1].name / checkpoint.parent.name
        if (output / "metrics.json").exists():
            continue
        jobs.append((checkpoint, output))
    python = sys.executable

    def worker(job, gpu):
        checkpoint, output = job
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        subprocess.run([python, "-m", "castgate", "evaluate", "--checkpoint", str(checkpoint),
                        "--output", str(output), "--split", "val", "--bootstrap", "0", "--device", "cuda"],
                       check=True, env=env)

    run_pool(jobs, worker, gpus)


def command_stress(gpus, family):
    jobs = [checkpoint for checkpoint in checkpoints()
            if not (FOLLOW / "stress" / family / checkpoint.parents[1].name / f"{checkpoint.parent.name}.json").exists()]

    def worker(checkpoint, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        subprocess.run([sys.executable, __file__, "stress-one", "--checkpoint", str(checkpoint),
                        "--family", family], check=True, env=env)

    run_pool(jobs, worker, gpus)


def command_target(gpus):
    jobs = [checkpoint for checkpoint in (EXP / "reliability_joint").glob("fold*_seed*/best.pt")
            if not (FOLLOW / "target" / f"{checkpoint.parent.name}.json").exists()]
    (FOLLOW / "target").mkdir(parents=True, exist_ok=True)

    def worker(checkpoint, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        subprocess.run([sys.executable, __file__, "target-one", "--checkpoint", str(checkpoint)],
                       check=True, env=env)

    run_pool(jobs, worker, gpus)


def command_timing(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = get_device("cuda")
    config = load_config(ROOT / "configs" / "lunghist700.json")
    batch = torch.randn(8, 3, config["image_size"], config["image_size"], device=device)
    report = {}
    for mode in ("cnn", "transformer", "mean", "reliability_gate"):
        model = build_model(dict(config, mode=mode), pretrained=False).to(device).eval()
        with torch.inference_mode():
            for _ in range(10):
                model(batch)
            torch.cuda.synchronize()
            start = time.perf_counter()
            repeats = 30
            for _ in range(repeats):
                model(batch)
            torch.cuda.synchronize()
        milliseconds = (time.perf_counter() - start) / repeats / batch.shape[0] * 1000
        parameters = sum(item.numel() for item in model.parameters())
        report[mode] = {"parameters": int(parameters), "forward_ms_per_image": milliseconds,
                        "batch_size": int(batch.shape[0]), "warmup_batches": 10, "timed_batches": repeats,
                        "includes_data_loading": False, "device": torch.cuda.get_device_name(device)}
    save_json(report, FOLLOW / "inference_timing.json")
    print(json.dumps(report, indent=2))


def command_preview():
    provenance = json.loads((EXP / "reliability_joint" / "fold0_seed42" / "split_provenance.json").read_text())
    by_id = {row["image_id"]: row for row in read_manifest(ROOT / "data" / "lunghist700" / "manifest.csv")}
    row = by_id[provenance["train_ids"][0]]
    with Image.open(row["path"]) as handle:
        image = geometry(handle.convert("RGB"), 224, False)
    tiles = []
    for family in ("multiplicative", "additive"):
        for strength in STRENGTHS:
            tiles.append(perturb(image, family, strength, 0, row["image_id"]))
    sheet = Image.new("RGB", (224 * len(STRENGTHS), 224 * 2))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % len(STRENGTHS)) * 224, (index // len(STRENGTHS)) * 224))
    output = FOLLOW / "stain_preview_train.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(output, row["image_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "val", "stress", "target", "timing", "stress-one", "target-one"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--family", choices=("multiplicative", "additive"), default="multiplicative")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "preview":
        command_preview()
    elif args.command == "val":
        command_val(args.gpus)
    elif args.command == "stress":
        command_stress(args.gpus, args.family)
    elif args.command == "stress-one":
        stress_one(args.checkpoint, args.family, "cuda")
    elif args.command == "target":
        command_target(args.gpus)
    elif args.command == "target-one":
        target_one(args.checkpoint, "cuda")
    else:
        command_timing(args.gpus[0])


if __name__ == "__main__":
    main()
