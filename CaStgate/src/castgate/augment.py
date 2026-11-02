"""Approximate H&E optical-density perturbations, not a causal intervention model."""

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms import RandomResizedCrop

# Rows: fixed H/E/residual optical density directions. No target-domain fitting.
STAIN_MATRIX = np.array([[0.650, 0.704, 0.286], [0.072, 0.990, 0.105],
                         [0.268, 0.570, 0.776]], dtype=np.float32)
STAIN_INVERSE = np.linalg.inv(STAIN_MATRIX)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def stain_perturb(image, strength=0.15, rng=None):
    if not 0 <= strength <= 1:
        raise ValueError("stain strength must be between 0 and 1")
    if strength == 0:
        return image.copy()
    rng = np.random.default_rng() if rng is None else rng
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    od = -np.log((rgb + 1.0) / 256.0)
    concentrations = od @ STAIN_INVERSE
    scales = np.array([*rng.uniform(1 - strength, 1 + strength, size=2), 1], dtype=np.float32)
    shifted = np.maximum((concentrations * scales) @ STAIN_MATRIX, 0)
    restored = np.clip(np.exp(-shifted) * 256.0 - 1, 0, 255).round().astype(np.uint8)
    return Image.fromarray(restored)


def stain_shift(image, amplitude=0.2, rng=None):
    """Additive H/E concentration offset. Residual is unchanged.

    This is a different family from multiplicative ``stain_perturb`` and is used
    only as a pre-specified robustness check, not as a training augmentation.
    """
    if not 0 <= amplitude <= 1:
        raise ValueError("stain shift amplitude must be between 0 and 1")
    if amplitude == 0:
        return image.copy()
    rng = np.random.default_rng() if rng is None else rng
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    od = -np.log((rgb + 1.0) / 256.0)
    concentrations = od @ STAIN_INVERSE
    shifts = np.array([*rng.uniform(-amplitude, amplitude, size=2), 0], dtype=np.float32)
    shifted = np.maximum((concentrations + shifts) @ STAIN_MATRIX, 0)
    restored = np.clip(np.exp(-shifted) * 256.0 - 1, 0, 255).round().astype(np.uint8)
    return Image.fromarray(restored)


def apply_illumination(intensity, mu, alpha, contrast, gamma, offset, horizontal, vertical):
    """CaStGate illumination proxy on intensities in [0, 1].

    Contrast is centered on the reference mean. The low-frequency field is
    normalized to unit mean. Gain is applied before gamma, and the offset is
    applied after the inner clip.
    """
    if intensity.ndim != 2:
        raise ValueError("Illumination is applied to a single intensity plane")
    height, width = intensity.shape
    rows = np.zeros((height, 1), dtype=np.float64) if height == 1 else np.linspace(-1, 1, height, dtype=np.float64)[:, None]
    cols = np.zeros((1, width), dtype=np.float64) if width == 1 else np.linspace(-1, 1, width, dtype=np.float64)[None, :]
    field = np.exp(horizontal * cols + vertical * rows)
    field /= field.mean()
    contrasted = mu + contrast * (np.asarray(intensity, dtype=np.float64) - mu)
    base = np.clip(alpha * field * contrasted, 0, 1)
    return np.clip(np.power(base, gamma) + offset, 0, 1)


def _sample_illumination(rng):
    # Initial training ranges fixed before testing. They are not tuned optima.
    return dict(alpha=float(rng.uniform(0.85, 1.15)), contrast=float(rng.uniform(0.85, 1.15)),
                gamma=float(rng.uniform(0.90, 1.10)), offset=float(rng.uniform(-0.03, 0.03)),
                horizontal=float(rng.uniform(-0.15, 0.15)), vertical=float(rng.uniform(-0.15, 0.15)))


def illumination_view(image, rng=None, parameters=None):
    """One illumination view of a spatially aligned reference image.

    The reference mean is measured on this image. Parameters are sampled
    independently when not supplied. The result is replicated to three channels.
    """
    intensity = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    if parameters is None:
        parameters = _sample_illumination(np.random.default_rng() if rng is None else rng)
    transformed = apply_illumination(intensity, float(intensity.mean()), **parameters)
    rgb = np.repeat(np.clip(np.round(transformed * 255.0), 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def industrial_geometry(image, size, training, rng=None):
    """Full-image resize with shared flips. No crop and no rotation.

    Orientation is left unchanged because it can be class-relevant for scratches
    and rolled-in scale. Flips are shared by every illumination view.
    """
    image = TF.resize(image, [size, size], InterpolationMode.BICUBIC)
    if training:
        if rng is None:
            raise ValueError("Training geometry requires the shared random generator")
        if rng.random() < 0.5:
            image = TF.hflip(image)
        if rng.random() < 0.5:
            image = TF.vflip(image)
    return image


def geometry(image, size, training):
    if training:
        i, j, h, w = RandomResizedCrop.get_params(image, scale=(0.75, 1.0), ratio=(0.9, 1.1))
        image = TF.resized_crop(image, i, j, h, w, [size, size], InterpolationMode.BICUBIC)
        if torch.rand(()) < 0.5:
            image = TF.hflip(image)
        if torch.rand(()) < 0.5:
            image = TF.vflip(image)
        image = TF.rotate(image, 90 * int(torch.randint(0, 4, ()).item()))
        return image
    # Deterministic whole-image resize; distortion is documented as a first-version limitation.
    return TF.resize(image, [size, size], InterpolationMode.BICUBIC)


def tensorize(image):
    return TF.normalize(TF.to_tensor(image), MEAN, STD)
