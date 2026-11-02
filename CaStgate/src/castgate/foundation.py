"""Frozen CTransPath and UNI encoders with a linear three-class head.

Training uses the same patient folds, seeds, augmentation, optimizer, epoch
budget, and validation selection as the single-branch models. The encoder is
not a training hyperparameter and is not updated.


CTransPath is a Swin-T whose patch stem is convolutional. Its checkpoint matches
the timm 0.6 block layout, which current timm no longer uses, so the blocks are
reimplemented here. UNI is a ViT-L/16 loaded through timm. Neither encoder is
updated; BatchNorm in the CTransPath stem stays in evaluation mode.
"""

import csv
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import Mlp, _assert, to_2tuple

from .engine import get_device, loader, predict, save_json, seed_all
from .manifests import LABELS, content_hashes, fingerprint, load_fold, read_manifest
from .metrics import classification_metrics, cluster_bootstrap


def _window_partition(x, window_size):
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window_size, window_size, width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)


def _window_reverse(windows, window_size, height, width):
    batch = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(batch, height // window_size, width // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


def _relative_position_index(window_h, window_w):
    coords = torch.stack(torch.meshgrid(torch.arange(window_h), torch.arange(window_w), indexing="ij"))
    flat = torch.flatten(coords, 1)
    relative = (flat[:, :, None] - flat[:, None, :]).permute(1, 2, 0).contiguous()
    relative[:, :, 0] += window_h - 1
    relative[:, :, 1] += window_w - 1
    relative[:, :, 0] *= 2 * window_w - 1
    return relative.sum(-1)


class _ConvStem(nn.Module):
    """CTransPath stem: two stride-2 convolutions, then a 1x1 projection to 96 channels."""

    def __init__(self, img_size=224, embed_dim=96):
        super().__init__()
        self.grid_size = (img_size // 4, img_size // 4)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        channels = embed_dim // 8
        layers = []
        in_channels = 3
        for _ in range(2):
            layers += [nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
            in_channels = channels
            channels *= 2
        layers.append(nn.Conv2d(in_channels, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class _WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        window_h, window_w = to_2tuple(window_size)
        self.window_area = window_h * window_w
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_h - 1) * (2 * window_w - 1), num_heads))
        self.register_buffer("relative_position_index", _relative_position_index(window_h, window_w), persistent=True)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = query * self.scale @ key.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_area, self.window_area, -1).permute(2, 0, 1).contiguous()
        attention = attention + bias.unsqueeze(0)
        if mask is not None:
            windows = mask.shape[0]
            attention = attention.view(-1, windows, self.num_heads, length, length) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, length, length)
        attention = self.softmax(attention)
        return self.proj((attention @ value).transpose(1, 2).reshape(batch, length, -1))


class _SwinBlock(nn.Module):
    def __init__(self, dim, resolution, num_heads, shift_size):
        super().__init__()
        self.resolution = resolution
        self.window_size = 7
        if min(resolution) <= self.window_size:
            shift_size = 0
            self.window_size = min(resolution)
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, num_heads, self.window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * 4))
        if shift_size:
            height, width = resolution
            image_mask = torch.zeros((1, height, width, 1))
            count = 0
            for h_slice in (slice(0, -7), slice(-7, -shift_size), slice(-shift_size, None)):
                for w_slice in (slice(0, -7), slice(-7, -shift_size), slice(-shift_size, None)):
                    image_mask[:, h_slice, w_slice, :] = count
                    count += 1
            windows = _window_partition(image_mask, 7).view(-1, 49)
            mask = windows.unsqueeze(1) - windows.unsqueeze(2)
            mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        else:
            mask = None
        self.register_buffer("attn_mask", mask, persistent=mask is not None)

    def forward(self, x):
        height, width = self.resolution
        batch, length, channels = x.shape
        _assert(length == height * width, "input feature has wrong size")
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = _window_partition(x, 7).view(-1, 49, channels)
        windows = self.attn(windows, mask=self.attn_mask)
        x = _window_reverse(windows.view(-1, 7, 7, channels), 7, height, width)
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x.view(batch, height * width, channels)
        return x + self.mlp(self.norm2(x))


class _PatchMerging(nn.Module):
    def __init__(self, resolution, dim):
        super().__init__()
        self.resolution = resolution
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        height, width = self.resolution
        batch, length, channels = x.shape
        _assert(length == height * width, "input feature has wrong size")
        x = x.view(batch, height, width, channels)
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], -1)
        return self.reduction(self.norm(x.view(batch, -1, 4 * channels)))


class _SwinStage(nn.Module):
    def __init__(self, dim, resolution, depth, num_heads, downsample):
        super().__init__()
        self.blocks = nn.Sequential(*[
            _SwinBlock(dim, resolution, num_heads, 0 if i % 2 == 0 else 3) for i in range(depth)
        ])
        self.downsample = _PatchMerging(resolution, dim) if downsample else None

    def forward(self, x):
        x = self.blocks(x)
        return self.downsample(x) if self.downsample is not None else x


class CTransPathEncoder(nn.Module):
    """Swin-T feature extractor. Output is the mean of the final 768-d tokens."""

    num_features = 768

    def __init__(self, img_size=224):
        super().__init__()
        self.patch_embed = _ConvStem(img_size, 96)
        grid = self.patch_embed.grid_size
        depths, heads = (2, 2, 6, 2), (3, 6, 12, 24)
        stages = []
        for index, (depth, num_heads) in enumerate(zip(depths, heads)):
            stages.append(_SwinStage(96 * 2 ** index, (grid[0] // 2 ** index, grid[1] // 2 ** index),
                                     depth, num_heads, downsample=index < 3))
        self.layers = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(self.num_features)
        self.pos_drop = nn.Dropout(0.)

    def forward(self, x):
        x = self.pos_drop(self.patch_embed(x))
        return self.norm(self.layers(x)).mean(1)


def load_ctranspath(path, img_size=224):
    model = CTransPathEncoder(img_size)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    ignored = [key for key in unexpected if key.endswith("num_batches_tracked")]
    if missing or set(unexpected) - set(ignored):
        raise RuntimeError(f"CTransPath checkpoint mismatch missing={missing} unexpected={unexpected}")
    return model


def load_uni(path):
    model = timm.create_model("vit_large_patch16_224", pretrained=False, img_size=224, patch_size=16,
                              init_values=1e-5, num_classes=0, dynamic_img_size=True)
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"UNI checkpoint mismatch missing={missing} unexpected={unexpected}")
    model.num_features = 1024
    return model


class FrozenLinearProbe(nn.Module):
    def __init__(self, encoder, num_classes=3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.num_features, num_classes)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        logits = self.head(features)
        probabilities = logits.float().softmax(-1)
        weights = logits.new_tensor([1., 0.]).expand(x.shape[0], -1)
        return {"branch_logits": torch.stack([logits, logits], 1), "weights": weights, "probs": probabilities}


def build_probe(name, weights, img_size=224):
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(path)
    if name == "ctranspath":
        encoder = load_ctranspath(path, img_size)
    elif name == "uni":
        encoder = load_uni(path)
    else:
        raise ValueError(f"Unknown foundation model {name}")
    return FrozenLinearProbe(encoder)


def train_probe(config):
    """Fit only the linear head. The test split is not read for selection."""
    if (Path(config["output"]) / "config.json").exists():
        raise FileExistsError(f"Run already exists: {config['output']}")
    seed_all(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    subsets = load_fold(config["manifest"], config["splits"], config["fold"])
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    save_json(config, out / "config.json")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "timm", "numpy", "scikit-learn", "Pillow")}
    save_json(dict(python=platform.python_version(), packages=versions), out / "environment.json")
    rows = read_manifest(config["manifest"])
    current_hashes = content_hashes(rows)
    split_spec = json.loads(Path(config["splits"]).read_text())
    if split_spec["content_sha256"] != current_hashes:
        raise ValueError("Image files changed since split creation")
    provenance = dict(manifest_sha256=fingerprint(rows),
                      development_content_sha256=sorted(set(current_hashes.values())),
                      development_patient_keys=sorted({r["source"] + ":" + r["patient_id"] for r in rows}),
                      train_ids=[r["image_id"] for r in subsets["train"]],
                      train_patients=sorted({r["patient_id"] for r in subsets["train"]}),
                      val_ids=[r["image_id"] for r in subsets["val"]],
                      test_ids=[r["image_id"] for r in subsets["test"]])
    save_json(provenance, out / "split_provenance.json")
    device = get_device(config["device"])
    model = build_probe(config["foundation"], config["weights"], config["image_size"]).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    training_loader = loader(subsets["train"], config, training=True)
    validation_loader = loader(subsets["val"], config)
    history, best, stale = [], float("-inf"), 0
    for epoch in range(config["epochs"]):
        start = time.time()
        model.train()
        total, seen = 0.0, 0
        for batch in training_loader:
            images, labels = batch["views"].to(device), batch["label"].to(device)
            batch_size, views = images.shape[:2]
            outputs = model(images.flatten(0, 1))
            repeated = labels[:, None].expand(-1, views).reshape(-1)
            loss = F.cross_entropy(outputs["branch_logits"][:, 0], repeated)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; no checkpoint was silently accepted")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * batch_size
            seen += batch_size
        scheduler.step()
        y, probabilities, _, _, _ = predict(model, validation_loader, device)
        metrics = classification_metrics(y, probabilities)
        history.append(dict(epoch=epoch + 1, seconds=round(time.time() - start, 2),
                            train_loss=total / seen, val=metrics))
        save_json(history, out / "history.json")
        print(f"epoch={epoch + 1} loss={total / seen:.4f} val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
        if metrics["macro_f1"] > best:
            best, stale = metrics["macro_f1"], 0
            torch.save(dict(head=model.head.state_dict(), config=config, epoch=epoch + 1,
                            val_metrics=metrics, provenance=provenance, labels=LABELS), out / "best.pt")
        else:
            stale += 1
        if config["patience"] > 0 and stale >= config["patience"]:
            break
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evaluate_probe(out / "best.pt", out / "test", device_name=config["device"])
    return {"checkpoint": str(out / "best.pt"), "best_val_macro_f1": best}


def evaluate_probe(checkpoint, output, device_name="auto"):
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]
    if payload["labels"] != LABELS:
        raise ValueError("Checkpoint label mapping is incompatible")
    seed_all(config["seed"])
    rows = read_manifest(config["manifest"])
    if fingerprint(rows) != payload["provenance"]["manifest_sha256"]:
        raise ValueError("Development manifest does not match checkpoint")
    ids = set(payload["provenance"]["test_ids"])
    test_rows = [row for row in rows if row["image_id"] in ids]
    out = Path(output)
    if (out / "metrics.json").exists():
        raise FileExistsError(f"Evaluation already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(device_name)
    model = build_probe(config["foundation"], config["weights"], config["image_size"]).to(device)
    model.head.load_state_dict(payload["head"])
    y, probabilities, branch, weights, records = predict(model, loader(test_rows, config), device)
    result = classification_metrics(y, probabilities)
    result.update(split="test", checkpoint=str(checkpoint.resolve()), epoch=payload["epoch"],
                  synthetic=False, stain_strength=0.0, label_mapping=LABELS, mode="linear_probe",
                  foundation=config["foundation"], encoder_frozen=True, seed=config["seed"], fold=config["fold"],
                  development_manifest_sha256=payload["provenance"]["manifest_sha256"],
                  protocol={key: config.get(key) for key in (
                      "foundation", "weights", "encoder_frozen", "image_size", "num_stain_views",
                      "stain_strength", "epochs", "lr", "weight_decay", "batch_size", "patience")},
                  evaluation_unit="image", n_patients=len({record["patient_id"] for record in records}),
                  evaluation_manifest_sha256=fingerprint(test_rows),
                  bootstrap_ci95=cluster_bootstrap(y, probabilities, [record["patient_id"] for record in records],
                                                   config["bootstrap_samples"], config["seed"]))
    result["by_magnification"] = {}
    for magnification in sorted({record["magnification"] for record in records}):
        mask = np.array([record["magnification"] == magnification for record in records])
        result["by_magnification"][magnification] = classification_metrics(y[mask], probabilities[mask])
    save_json(result, out / "metrics.json")
    fields = ["image_id", "patient_id", "magnification", "label", "prediction",
              *[f"p_{name}" for name in LABELS], "weight_cnn", "weight_transformer",
              *[f"{branch_name}_p_{name}" for branch_name in ("cnn", "transformer") for name in LABELS]]
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            row = dict(record, label=int(y[index]), prediction=int(probabilities[index].argmax()),
                       weight_cnn=float(weights[index, 0]), weight_transformer=float(weights[index, 1]))
            row.update({f"p_{name}": float(probabilities[index, class_index]) for name, class_index in LABELS.items()})
            row.update({f"{branch_name}_p_{name}": float(branch[index, branch_index, class_index])
                        for branch_index, branch_name in enumerate(("cnn", "transformer"))
                        for name, class_index in LABELS.items()})
            writer.writerow(row)
    print(f"test macro_f1={result['macro_f1']:.4f} accuracy={result['accuracy']:.4f}", flush=True)
    return result
