"""Explicit image metadata, fail-closed patient splits, and split provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

LABELS = {"aca": 0, "scc": 1, "nor": 2}
# NEU-CLS order is fixed: crazing, inclusion, patches, pitted surface, rolled-in scale, scratches.
NEU_LABELS = {"crazing": 0, "inclusion": 1, "patches": 2, "pitted_surface": 3,
              "rolled_in_scale": 4, "scratches": 5}
NEU_ALIASES = {"cr": "crazing", "crazing": "crazing", "in": "inclusion", "inclusion": "inclusion",
               "pa": "patches", "patches": "patches", "ps": "pitted_surface",
               "pitted_surface": "pitted_surface", "pitted surface": "pitted_surface",
               "rs": "rolled_in_scale", "rolled_in_scale": "rolled_in_scale",
               "rolled-in_scale": "rolled_in_scale", "rolled-in scale": "rolled_in_scale",
               "sc": "scratches", "scratches": "scratches"}
FIELDS = ["image_id", "path", "patient_id", "label", "magnification", "source"]


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain {FIELDS}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    ids, paths = set(), set()
    for row in rows:
        if not row["image_id"].strip() or not row["patient_id"].strip() or not row["source"].strip():
            raise ValueError("Missing image_id, patient_id, or source: grouping must come from metadata")
        if row["label"] not in LABELS and row["label"] not in NEU_LABELS:
            raise ValueError(f"Unknown label {row['label']}; expected {list(LABELS)} or {list(NEU_LABELS)}")
        p = Path(row["path"])
        if not p.is_absolute():
            p = path.parent / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        if row["image_id"] in ids or str(p) in paths:
            raise ValueError("Duplicate image ID or path in manifest")
        ids.add(row["image_id"])
        paths.add(str(p))
        row["path"] = str(p)
    found = {row["label"] for row in rows}
    if not (found <= set(LABELS) or found <= set(NEU_LABELS)):
        raise ValueError(f"Manifest mixes label sets: {sorted(found)}")
    return rows


def mapping_for_labels(names):
    """Return the canonical index map. A subset keeps the published indices."""
    found = set(names)
    if found <= set(LABELS):
        return dict(LABELS)
    if found <= set(NEU_LABELS):
        return dict(NEU_LABELS)
    raise ValueError(f"Unrecognized or mixed labels: {sorted(found)}")


def fingerprint(rows):
    # Absolute data locations can change without changing a split.
    identity = sorted((r["image_id"], r["patient_id"], r["label"], r["source"],
                       r["magnification"]) for r in rows)
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def content_hashes(rows):
    hashes = {}
    for row in rows:
        digest = hashlib.sha256()
        with Path(row["path"]).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        hashes[row["image_id"]] = digest.hexdigest()
    return hashes


def _column(headers, requested, aliases, required=True):
    if requested:
        if requested not in headers:
            raise ValueError(f"Column {requested!r} not found. Available: {headers}")
        return requested
    candidates = [h for h in headers if h.strip().lower() in aliases]
    if len(candidates) == 1:
        return candidates[0]
    if required:
        raise ValueError(f"Specify the column explicitly; available headers: {headers}")
    return None


def canonical_label(value):
    # Only published LungHist700 superclasses and their differentiation suffixes.
    value = value.strip().lower()
    for label in LABELS:
        if value == label or value.startswith(label + "_"):
            return label
    raise ValueError(f"Unrecognized label {value!r}; use aca/scc/nor or an explicit label map")


def prepare_lunghist(root, metadata, output, image_col=None, patient_col=None,
                     label_col=None, magnification_col=None, label_from_path=False,
                     label_map=None, delimiter=","):
    root = Path(root).resolve()
    images = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            images.setdefault(p.name, []).append(p)
    with Path(metadata).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = reader.fieldnames or []
        ic = _column(headers, image_col, {"image", "image_id", "image_name", "filename", "file_name", "file", "path"})
        pc = _column(headers, patient_col, {"patient", "patient_id", "patientid", "patient id"})
        lc = _column(headers, label_col, {"label", "class", "category", "diagnosis"},
                     required=not label_from_path)
        mc = _column(headers, magnification_col, {"magnification", "magnification_level", "resolution"}, False)
        rows = []
        for record in reader:
            raw = record[ic].strip().replace("\\", "/")
            candidate = (root / raw).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError(f"Image path escapes data root: {raw}")
            if not candidate.is_file():
                matches = images.get(Path(raw).name, [])
                if len(matches) != 1:
                    raise ValueError(f"Cannot uniquely locate {raw!r}: {len(matches)} matches")
                candidate = matches[0]
            patient = record[pc].strip()
            if not patient:
                raise ValueError(f"Missing patient ID for {raw}")
            if lc:
                value = record[lc].strip()
                label = canonical_label(label_map.get(value, value) if label_map else value)
            else:
                # Opt-in; never infer patient identity from file names.
                codes = set(re.findall(r"(?:^|[/_\-])(aca|scc|nor)(?=[/_\-.]|$)",
                                       candidate.relative_to(root).as_posix().lower()))
                if len(codes) != 1:
                    raise ValueError(f"Cannot uniquely find a class code in {candidate}")
                label = codes.pop()
            rows.append(dict(image_id=candidate.relative_to(root).as_posix(), path=str(candidate),
                             patient_id=patient, label=label,
                             magnification=record[mc].strip() if mc else "unknown",
                             source="LungHist700"))
    write_manifest(rows, output)
    validated = read_manifest(output)
    return {"images": len(validated), "patients": len({r["patient_id"] for r in rows}),
            "classes": dict(Counter(r["label"] for r in rows))}


def validate_fold(rows, fold):
    by_id = {r["image_id"]: r for r in rows}
    seen_ids, seen_patients = set(), set()
    for name in ("train", "val", "test"):
        ids = fold.get(name, [])
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate IDs in {name}")
        if not set(ids).issubset(by_id):
            raise ValueError(f"Unknown image IDs in {name}")
        patients = {by_id[i]["patient_id"] for i in ids}
        if seen_ids.intersection(ids) or seen_patients.intersection(patients):
            raise ValueError("Patient or image leakage across train/val/test")
        seen_ids.update(ids)
        seen_patients.update(patients)
        if {by_id[i]["label"] for i in ids} != {row["label"] for row in rows}:
            raise ValueError(f"{name} does not contain every class")
    if seen_ids != set(by_id):
        raise ValueError("Split does not cover the entire manifest")


def create_splits(manifest, output, n_splits=5, seed=42):
    rows = read_manifest(manifest)
    owners = {}
    hashes = content_hashes(rows)
    for row in rows:
        digest = hashes[row["image_id"]]
        if digest in owners and owners[digest] != row["patient_id"]:
            raise ValueError("Byte-identical images assigned to different patients; audit metadata first")
        owners[digest] = row["patient_id"]
    if n_splits < 3:
        raise ValueError("At least 3 folds are needed for train/val/test")
    y = np.array([LABELS[r["label"]] for r in rows])
    groups = np.array([r["patient_id"] for r in rows])
    # A patient can have both normal and tumor images: do not assign one class per patient.
    for label in LABELS:
        count = len({r["patient_id"] for r in rows if r["label"] == label})
        if count < n_splits:
            raise ValueError(f"Only {count} patients contain {label}; reduce n_splits")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    buckets = [idx for _, idx in splitter.split(np.zeros(len(rows)), y, groups)]
    folds = []
    for k in range(n_splits):
        test, val = buckets[k], buckets[(k + 1) % n_splits]
        train = np.concatenate([buckets[j] for j in range(n_splits)
                                if j not in (k, (k + 1) % n_splits)])
        fold = {name: [rows[i]["image_id"] for i in indices]
                for name, indices in (("train", train), ("val", val), ("test", test))}
        validate_fold(rows, fold)
        folds.append(fold)
    payload = dict(seed=seed, n_splits=n_splits, manifest_sha256=fingerprint(rows),
                   content_sha256=hashes, folds=folds)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload


def prepare_neu(root, output):
    """Index a NEU-CLS directory. Each image is its own split unit.

    Class names come from the parent directory. There is no patient identifier
    in this dataset, so patient_id is the image id and cannot group images.
    """
    root = Path(root).resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        key = path.parent.name.strip().lower().replace(" ", "_")
        if key not in NEU_ALIASES:
            raise ValueError(f"Unrecognized NEU-CLS class directory: {path.parent.name}")
        label = NEU_ALIASES[key]
        image_id = f"{label}_{path.stem}"
        rows.append(dict(image_id=image_id, path=str(path), patient_id=image_id, label=label,
                         magnification="native", source="NEU-CLS"))
    if not rows:
        raise ValueError(f"No images found under {root}")
    counts = Counter(row["label"] for row in rows)
    if set(counts) != set(NEU_LABELS):
        raise ValueError(f"NEU-CLS requires all six classes; found {dict(counts)}")
    write_manifest(rows, output)
    validated = read_manifest(output)
    return {"images": len(validated), "classes": dict(counts)}


def create_balanced_splits(manifest, output, n_splits=5, seed=2026):
    """Five class-balanced groups. Fold k tests group k and validates group k+1."""
    rows = read_manifest(manifest)
    mapping = mapping_for_labels(row["label"] for row in rows)
    if n_splits < 3:
        raise ValueError("At least 3 folds are needed for train/val/test")
    rng = np.random.default_rng(seed)
    groups = [[] for _ in range(n_splits)]
    for label in mapping:
        indices = [i for i, row in enumerate(rows) if row["label"] == label]
        if not indices:
            continue
        if len(indices) % n_splits:
            raise ValueError(f"{label} has {len(indices)} images, not divisible by {n_splits}")
        order = np.array(indices)
        rng.shuffle(order)
        for group, chunk in zip(groups, np.array_split(order, n_splits)):
            group.extend(int(i) for i in chunk)
    folds = []
    for k in range(n_splits):
        held_out = {k, (k + 1) % n_splits}
        parts = {"test": groups[k], "val": groups[(k + 1) % n_splits],
                 "train": [i for j, group in enumerate(groups) if j not in held_out for i in group]}
        fold = {name: [rows[i]["image_id"] for i in indices] for name, indices in parts.items()}
        validate_fold(rows, fold)
        folds.append(fold)
    payload = dict(seed=seed, n_splits=n_splits, scheme="class_balanced_groups",
                   manifest_sha256=fingerprint(rows), content_sha256=content_hashes(rows), folds=folds)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload


def load_fold(manifest, splits, fold):
    rows = read_manifest(manifest)
    spec = json.loads(Path(splits).read_text())
    if spec["manifest_sha256"] != fingerprint(rows):
        raise ValueError("Manifest changed since split creation; regenerate splits")
    selected = spec["folds"][fold]
    validate_fold(rows, selected)
    by_id = {r["image_id"]: r for r in rows}
    return {name: [by_id[i] for i in selected[name]] for name in ("train", "val", "test")}
