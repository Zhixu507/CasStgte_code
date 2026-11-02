from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .engine import DEFAULT_CONFIG, evaluate, load_config, save_json, train
from .manifests import create_balanced_splits, create_splits, load_fold, prepare_lunghist, prepare_neu, read_manifest
from .model import MODES


def print_json(result):
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description="CaStGate illumination-reliability fusion research")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="Download the official LungHist700 RAR (~639 MB)")
    p.add_argument("--output", default="data/raw")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("inspect-data", help="Show CSV headers; no assumed official column names")
    p.add_argument("--root", required=True)
    p = sub.add_parser("prepare", help="Convert official image-to-patient CSV to a canonical manifest")
    p.add_argument("--root", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", default="data/lunghist700/manifest.csv")
    p.add_argument("--image-col")
    p.add_argument("--patient-col")
    p.add_argument("--label-col")
    p.add_argument("--magnification-col")
    p.add_argument("--label-from-path", action="store_true")
    p.add_argument("--label-map", help="JSON object mapping raw labels to aca/scc/nor")
    p.add_argument("--delimiter", default=",")
    p = sub.add_parser("prepare-neu", help="Index a NEU-CLS directory into a six-class manifest")
    p.add_argument("--root", required=True)
    p.add_argument("--output", default="data/neu_cls/manifest.csv")
    p = sub.add_parser("split", help="Create fixed patient-grouped rotating train/val/test folds")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="data/lunghist700/splits.json")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("split-balanced", help="Class-balanced rotating groups; NEU-CLS uses seed 2026")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="data/neu_cls/splits.json")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p = sub.add_parser("audit")
    p.add_argument("--manifest", required=True)
    p.add_argument("--splits", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/lunghist700.json")
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--fold", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--output")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    p.add_argument("--epochs", type=int)
    p.add_argument("--no-pretrained", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--manifest", help="Independent external manifest; never used for fitting")
    p.add_argument("--stain-strength", type=float, default=0.)
    p.add_argument("--bootstrap", type=int)
    p.add_argument("--device", default="auto")
    p = sub.add_parser("infer", help="Predict one image without a label or perturbation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--device", default="auto")
    p = sub.add_parser("smoke", help="Offline synthetic end-to-end execution check")
    p.add_argument("--output", default="runs/smoke")
    p = sub.add_parser("aggregate", help="Aggregate existing fold test predictions; no fitting")
    p.add_argument("--evaluations", nargs="+", required=True, help="Directories containing predictions.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "download":
        from .download import download_lunghist
        print_json(download_lunghist(args.output, args.dry_run))
    elif args.command == "inspect-data":
        result = {}
        for path in sorted(Path(args.root).rglob("*.csv")):
            with path.open(encoding="utf-8-sig", newline="") as f:
                result[str(path)] = [next(csv.reader(f), [])]
        print_json(result)
    elif args.command == "prepare":
        print_json(prepare_lunghist(args.root, args.metadata, args.output, args.image_col,
                                   args.patient_col, args.label_col, args.magnification_col,
                                   args.label_from_path, json.loads(args.label_map) if args.label_map else None,
                                   args.delimiter))
    elif args.command == "prepare-neu":
        print_json(prepare_neu(args.root, args.output))
    elif args.command == "split":
        spec = create_splits(args.manifest, args.output, args.folds, args.seed)
        print_json(dict(output=args.output, folds=spec["n_splits"], manifest_sha256=spec["manifest_sha256"]))
    elif args.command == "split-balanced":
        spec = create_balanced_splits(args.manifest, args.output, args.folds, args.seed)
        print_json(dict(output=args.output, folds=spec["n_splits"], seed=spec["seed"],
                        manifest_sha256=spec["manifest_sha256"]))
    elif args.command == "audit":
        from collections import Counter
        from .manifests import content_hashes
        spec = json.loads(Path(args.splits).read_text())
        if spec["content_sha256"] != content_hashes(read_manifest(args.manifest)):
            raise ValueError("Image files changed since split creation")
        result = []
        for fold in range(spec["n_splits"]):
            data = load_fold(args.manifest, args.splits, fold)
            result.append({name: dict(images=len(rows), patients=len({r["patient_id"] for r in rows}),
                                     classes=dict(Counter(r["label"] for r in rows)))
                           for name, rows in data.items()})
        print_json(result)
    elif args.command == "train":
        c = load_config(args.config)
        for key in ("mode", "fold", "seed", "output", "device", "epochs"):
            value = getattr(args, key)
            if value is not None:
                c[key] = value
        if args.no_pretrained:
            c["pretrained"] = False
        print_json(train(c))
    elif args.command == "evaluate":
        print_json(evaluate(args.checkpoint, args.output, args.split, args.manifest,
                            args.stain_strength, args.bootstrap, args.device))
    elif args.command == "infer":
        import torch
        from PIL import Image
        from .augment import geometry, industrial_geometry, tensorize
        from .engine import get_device, label_dict
        from .model import build_model
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        c = payload["config"]
        labels = payload.get("labels") or label_dict(c)
        torch.set_num_threads(c["cpu_threads"])
        device = get_device(args.device)
        model = build_model(c, pretrained=False).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        with Image.open(args.image) as image:
            if c.get("geometry_mode") == "industrial":
                prepared = industrial_geometry(image.convert("L"), c["image_size"], False).convert("RGB")
            else:
                prepared = geometry(image.convert("RGB"), c["image_size"], False)
            x = tensorize(prepared).unsqueeze(0).to(device)
        with torch.inference_mode():
            result = model(x)
        p = result["probs"][0].cpu().tolist()
        print_json(dict(probabilities={k: p[i] for k, i in labels.items()},
                        prediction=max(labels, key=lambda k: p[labels[k]]),
                        branch_weights=result["weights"][0].cpu().tolist(), synthetic=c["synthetic"]))
    elif args.command == "smoke":
        from .synthetic import create_synthetic
        out = Path(args.output).resolve()
        manifest = create_synthetic(out / "SYNTHETIC_DATA")
        splits = out / "splits.json"
        create_splits(manifest, splits, n_splits=3)
        c = dict(DEFAULT_CONFIG, manifest=str(manifest), splits=str(splits), output=str(out / "train"),
                 cnn="tiny_cnn", transformer="tiny_transformer", image_size=32, pretrained=False,
                 device="cpu", epochs=2, warmup_epochs=1, batch_size=6, cpu_threads=2,
                 bootstrap_samples=20, synthetic=True)
        result = train(c)
        metrics = evaluate(result["checkpoint"], out / "eval", bootstrap=20, device_name="cpu")
        print_json(dict(status="SYNTHETIC_PIPELINE_CHECK_ONLY", checkpoint=result["checkpoint"],
                        evaluation=str(out / "eval" / "metrics.json"), n_images=metrics["n_images"]))
    elif args.command == "aggregate":
        from .aggregate import aggregate
        print_json(aggregate(args.evaluations, args.output, args.bootstrap))


if __name__ == "__main__":
    main()
