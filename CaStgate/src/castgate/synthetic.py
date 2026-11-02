"""Artificial colored shapes for execution checks only, not histological evidence."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .manifests import LABELS, write_manifest


def create_synthetic(root, seed=42):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.csv").exists():
        raise FileExistsError(f"Synthetic manifest already exists in {root}")
    rng = np.random.default_rng(seed)
    rows = []
    for label, index in LABELS.items():
        for patient in range(6):
            for view in range(2):
                name = f"SYNTHETIC_{label}_{patient}_{view}.png"
                a = rng.integers(180, 245, (64, 64, 3), dtype=np.uint8)
                image = Image.fromarray(a)
                draw = ImageDraw.Draw(image)
                color = [(80, 20, 120), (180, 90, 145), (110, 140, 190)][index]
                for _ in range(12):
                    x, y = rng.integers(0, 48, 2).tolist()
                    draw.ellipse([x, y, x + 12, y + 12], fill=color)
                image.save(root / name)
                rows.append(dict(image_id=name, path=str(root / name), patient_id=f"SYNTH_{label}_{patient}",
                                 label=label, magnification="synthetic", source="SYNTHETIC"))
    manifest = root / "manifest.csv"
    write_manifest(rows, manifest)
    return manifest
