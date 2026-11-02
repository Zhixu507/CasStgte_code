import hashlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import geometry, illumination_view, industrial_geometry, stain_perturb, tensorize
from .manifests import LABELS


class HistologyDataset(Dataset):
    def __init__(self, rows, size=224, training=False, num_stain_views=0,
                 stain_strength=0.15, eval_stain_strength=0.0, seed=42,
                 perturbation="stain", geometry_mode="histology", labels=None):
        if perturbation not in ("stain", "illumination"):
            raise ValueError(f"Unknown perturbation {perturbation}")
        if geometry_mode not in ("histology", "industrial"):
            raise ValueError(f"Unknown geometry {geometry_mode}")
        self.rows = rows
        self.size = size
        self.training = training
        self.num_stain_views = num_stain_views if training else 0
        self.stain_strength = stain_strength
        self.eval_stain_strength = eval_stain_strength
        self.seed = seed
        self.perturbation = perturbation
        self.geometry_mode = geometry_mode
        self.labels = labels or LABELS

    def __len__(self):
        return len(self.rows)

    def _aligned_views(self, image, rng):
        # Geometry is applied once. Perturbations must not move pixels.
        views = [image.convert("RGB") if image.mode != "RGB" else image]
        for _ in range(self.num_stain_views):
            if self.perturbation == "illumination":
                views.append(illumination_view(image, rng))
            else:
                views.append(stain_perturb(views[0], self.stain_strength, rng))
        return views

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as handle:
            if self.geometry_mode == "industrial":
                rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
                image = industrial_geometry(handle.convert("L"), self.size, self.training, rng)
            else:
                # Keep the histology generator order: geometry consumes torch RNG first.
                image = geometry(handle.convert("RGB"), self.size, self.training)
                rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
        views = self._aligned_views(image, rng)
        if not self.training and self.eval_stain_strength:
            key = f"{self.seed}:{row['image_id']}".encode()
            stable_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
            stable = np.random.default_rng(stable_seed)
            if self.perturbation == "illumination":
                views = [illumination_view(image, stable)]
            else:
                views = [stain_perturb(image if image.mode == "RGB" else image.convert("RGB"),
                                       self.eval_stain_strength, stable)]
        return {"views": torch.stack([tensorize(v) for v in views]),
                "label": self.labels[row["label"]], "image_id": row["image_id"],
                "patient_id": row["patient_id"], "magnification": row["magnification"]}
