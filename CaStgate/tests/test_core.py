import json

import numpy as np
import pytest
import torch
from collections import Counter

from PIL import Image

from castgate.augment import apply_illumination, illumination_view, industrial_geometry, stain_perturb, stain_shift
from castgate.data import HistologyDataset
from castgate.engine import DEFAULT_CONFIG, evaluate, filter_magnification, train, validate_config
from castgate.losses import reliability_target, training_loss
from castgate.manifests import (NEU_LABELS, canonical_label, create_balanced_splits, create_splits,
                                        load_fold, prepare_lunghist, read_manifest, validate_fold, write_manifest)
from castgate.metrics import classification_metrics, cluster_bootstrap
from castgate.model import FusionModel
from castgate.synthetic import create_synthetic
from castgate.aggregate import aggregate

torch.set_num_threads(2)


def tiny_config(tmp_path):
    manifest = create_synthetic(tmp_path / "data")
    splits = tmp_path / "splits.json"
    create_splits(manifest, splits, n_splits=3)
    return dict(DEFAULT_CONFIG, manifest=str(manifest), splits=str(splits),
                output=str(tmp_path / "run"), cnn="tiny_cnn", transformer="tiny_transformer",
                image_size=32, pretrained=False, epochs=2, warmup_epochs=1, batch_size=6,
                bootstrap_samples=5, cpu_threads=2, synthetic=True, device="cpu")


def test_stable_wrong_branch_not_rewarded():
    # Both branches are perfectly stable; only first is correct.
    logits = torch.tensor([[[[6., 0., 0.], [0., 6., 0.]],
                            [[6., 0., 0.], [0., 6., 0.]]]], requires_grad=True)
    q, stats = reliability_target(logits, torch.tensor([0]))
    assert q[0, 0] > .99
    assert torch.allclose(stats["js"], torch.zeros_like(stats["js"]), atol=1e-6)
    assert not q.requires_grad
    stable, _ = reliability_target(logits, torch.tensor([0]), component="stability")
    assert torch.allclose(stable, torch.full_like(stable, .5))


def test_stability_detects_prediction_changes():
    logits = torch.tensor([[[[6., 0., 0.], [6., 0., 0.]],
                            [[6., 0., 0.], [0., 6., 0.]]]])
    q, stats = reliability_target(logits, torch.tensor([0]), component="stability")
    assert stats["js"][0, 1] > stats["js"][0, 0]
    assert q[0, 0] > q[0, 1]


@pytest.mark.parametrize("mode", ["cnn", "transformer", "mean", "learned_gate", "reliability_gate"])
def test_forward_and_gradient(mode):
    torch.manual_seed(4)
    model = FusionModel("tiny_cnn", "tiny_transformer", mode=mode, image_size=32)
    outputs = model(torch.randn(6, 3, 32, 32))
    assert outputs["probs"].shape == (6, 3)
    assert torch.allclose(outputs["probs"].sum(-1), torch.ones(6), atol=1e-6)
    cfg = dict(DEFAULT_CONFIG, mode=mode)
    loss, scalars = training_loss(outputs, torch.tensor([0, 1]), 2, 3, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    if mode in ("learned_gate", "reliability_gate"):
        assert model.gate[-1].weight.grad.abs().sum() > 0
    if mode == "reliability_gate":
        assert scalars["gate_kl"] >= -1e-6


def test_production_backbones_forward_backward():
    # Real ResNet18 + ViT-Tiny, no weight download; verifies actual timm API and gradients.
    model = FusionModel(pretrained=False, image_size=224)
    outputs = model(torch.randn(2, 3, 224, 224))
    loss = -outputs["probs"][:, 0].clamp_min(1e-8).log().mean()
    loss.backward()
    assert model.cnn_head.weight.grad is not None
    assert model.transformer_head.weight.grad is not None
    assert model.gate[-1].weight.grad is not None


def test_shuffled_reliability_target_breaks_sample_alignment():
    torch.manual_seed(0)
    batch, views = 4, 2
    logits = torch.zeros(batch, views, 2, 3)
    labels = torch.tensor([0, 0, 1, 1])
    for index, label in enumerate(labels.tolist()):
        preferred = 0 if index < 2 else 1
        logits[index, :, preferred, label] = 8.
    target, _ = reliability_target(logits, labels)
    weights = target[:, None, :].expand(batch, views, 2).contiguous()
    outputs = {"branch_logits": logits.reshape(batch * views, 2, 3),
               "probs": logits[:, :, 0].softmax(-1).reshape(batch * views, 3),
               "weights": weights.reshape(batch * views, 2)}
    config = dict(DEFAULT_CONFIG, mode="reliability_gate")
    _, aligned = training_loss(outputs, labels, batch, views, config, teacher_logits=logits)
    assert aligned["gate_kl"] < 1e-5
    changed = False
    for seed in range(30):
        torch.manual_seed(seed)
        _, shuffled = training_loss(outputs, labels, batch, views,
                                   dict(config, shuffle_gate_target=True), teacher_logits=logits)
        if shuffled["gate_kl"] > 1e-3:
            changed = True
            break
    assert changed


def test_magnification_filter_rejects_empty_selection():
    rows = [{"magnification": "synthetic"}]
    assert filter_magnification(rows, None) == rows
    with pytest.raises(ValueError, match="20x"):
        filter_magnification(rows, "20x")


def test_stain_shift_is_deterministic_and_zero_is_exact():
    image = Image.fromarray(np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3))
    assert np.array_equal(np.asarray(image), np.asarray(stain_shift(image, 0)))
    one = stain_shift(image, 0.2, np.random.default_rng(3))
    two = stain_shift(image, 0.2, np.random.default_rng(3))
    assert np.array_equal(np.asarray(one), np.asarray(two))
    assert not np.array_equal(np.asarray(image), np.asarray(one))


def test_stain_zero_is_exact_and_seeded():
    image = Image.fromarray(np.arange(12*12*3, dtype=np.uint8).reshape(12, 12, 3))
    assert np.array_equal(np.asarray(image), np.asarray(stain_perturb(image, 0)))
    a = stain_perturb(image, .2, np.random.default_rng(2))
    b = stain_perturb(image, .2, np.random.default_rng(2))
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_aligned_geometry_and_deterministic_evaluation(tmp_path):
    manifest = create_synthetic(tmp_path)
    rows = read_manifest(manifest)
    ds = HistologyDataset(rows, size=32, training=True, num_stain_views=2, stain_strength=0)
    views = ds[0]["views"]
    assert torch.equal(views[0], views[1]) and torch.equal(views[1], views[2])
    ev = HistologyDataset(rows, size=32, eval_stain_strength=.3)
    assert torch.equal(ev[0]["views"], ev[0]["views"])


def test_patient_splits_allow_mixed_patient_labels(tmp_path):
    cfg = tiny_config(tmp_path)
    rows = read_manifest(cfg["manifest"])
    # Include normal image in tumor patient's group without conflating normal with patient diagnosis.
    rows[0]["patient_id"] = rows[12]["patient_id"]
    write_manifest(rows, cfg["manifest"])
    spec = create_splits(cfg["manifest"], cfg["splits"], n_splits=3)
    for index in range(len(spec["folds"])):
        subsets = load_fold(cfg["manifest"], cfg["splits"], index)
        groups = {k: {r["patient_id"] for r in v} for k, v in subsets.items()}
        assert groups["train"].isdisjoint(groups["val"])
        assert groups["train"].isdisjoint(groups["test"])
        assert groups["val"].isdisjoint(groups["test"])


def test_invalid_overlap_and_modified_manifest_rejected(tmp_path):
    cfg = tiny_config(tmp_path)
    rows = read_manifest(cfg["manifest"])
    spec = json.loads(open(cfg["splits"]).read())
    fold = spec["folds"][0]
    fold["val"][0] = fold["train"][0]
    with pytest.raises(ValueError, match="leakage"):
        validate_fold(rows, fold)
    rows[0]["patient_id"] = "changed"
    write_manifest(rows, cfg["manifest"])
    with pytest.raises(ValueError, match="changed"):
        load_fold(cfg["manifest"], cfg["splits"], 0)


def test_missing_class_auc_is_not_fabricated():
    m = classification_metrics([0, 0], [[.9, .05, .05], [.8, .1, .1]])
    assert m["macro_auroc"] is None
    assert m["recall"]["scc"] is None
    assert m["accuracy"] == 1


def test_patient_cluster_bootstrap_with_mixed_labels():
    y = [0, 1, 2, 0, 1, 2]
    p = np.eye(3)[y] * .8 + .2 / 3
    ci = cluster_bootstrap(y, p, ["a", "a", "a", "b", "b", "b"], samples=10)
    assert ci["accuracy"]["low"] == 1
    assert ci["accuracy"]["valid_resamples"] == 10


def test_prepare_requires_patient_metadata_and_collapses_grade(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (16, 16), "pink").save(image_root / "a.jpg")
    metadata = tmp_path / "raw.csv"
    metadata.write_text("filename,patient_id,class\na.jpg,7,aca_pd\n")
    output = tmp_path / "manifest.csv"
    summary = prepare_lunghist(image_root, metadata, output)
    assert summary["classes"] == {"aca": 1}
    assert read_manifest(output)[0]["patient_id"] == "7"
    metadata.write_text("filename,class\na.jpg,aca_pd\n")
    with pytest.raises(ValueError, match="column"):
        prepare_lunghist(image_root, metadata, output)


def test_reject_unrecognized_label_and_bad_config():
    with pytest.raises(ValueError):
        canonical_label("healthy")
    with pytest.raises(ValueError):
        validate_config(dict(DEFAULT_CONFIG, num_stain_views=0))


def test_end_to_end_checkpoint_eval_and_external_leakage(tmp_path):
    cfg = tiny_config(tmp_path)
    result = train(cfg)
    assert result["test_evaluated"] is False
    metrics = evaluate(result["checkpoint"], tmp_path / "eval", bootstrap=5, device_name="cpu")
    assert metrics["synthetic"] is True
    assert (tmp_path / "eval" / "predictions.csv").is_file()
    assert metrics["split"] == "test"
    assert metrics["n_images"] > 0
    # Distinct metadata cannot conceal byte-identical development images in external testing.
    rows = read_manifest(cfg["manifest"])
    for row in rows:
        row["image_id"] = "external_" + row["image_id"]
        row["source"] = "OTHER"
        row["patient_id"] = "OTHER_" + row["patient_id"]
    ext = tmp_path / "external.csv"
    write_manifest(rows, ext)
    with pytest.raises(ValueError, match="byte-identical"):
        evaluate(result["checkpoint"], tmp_path / "ext_eval", manifest=ext, device_name="cpu")


def test_byte_duplicate_across_patients_is_rejected(tmp_path):
    cfg = tiny_config(tmp_path)
    rows = read_manifest(cfg["manifest"])
    from pathlib import Path
    Path(rows[2]["path"]).write_bytes(Path(rows[0]["path"]).read_bytes())
    with pytest.raises(ValueError, match="Byte-identical"):
        create_splits(cfg["manifest"], cfg["splits"], n_splits=3)


def test_image_mutation_after_training_is_rejected(tmp_path):
    cfg = tiny_config(tmp_path)
    result = train(cfg)
    row = read_manifest(cfg["manifest"])[0]
    Image.new("RGB", (64, 64), "black").save(row["path"])
    with pytest.raises(ValueError, match="content changed"):
        evaluate(result["checkpoint"], tmp_path / "eval", bootstrap=0, device_name="cpu")


def test_gate_teacher_is_detached():
    cfg = dict(DEFAULT_CONFIG)
    model = FusionModel("tiny_cnn", "tiny_transformer", image_size=32)
    outputs = model(torch.randn(6, 3, 32, 32))
    teacher = torch.randn(2, 3, 2, 3, requires_grad=True)
    loss, _ = training_loss(outputs, torch.tensor([0, 1]), 2, 3, cfg, teacher_logits=teacher)
    loss.backward()
    assert teacher.grad is None
    assert model.gate[-1].weight.grad is not None


def test_no_warmup_checkpoint_can_be_selected(tmp_path):
    cfg = tiny_config(tmp_path)
    result = train(cfg)
    checkpoint = torch.load(result["checkpoint"], weights_only=True)
    assert checkpoint["epoch"] > cfg["warmup_epochs"]


def test_aggregate_duplicate_tests_and_non_test_rejected(tmp_path):
    cfg = tiny_config(tmp_path)
    result = train(cfg)
    ev = tmp_path / "eval"
    evaluate(result["checkpoint"], ev, bootstrap=0, device_name="cpu")
    one = aggregate([ev], tmp_path / "pooled.json", bootstrap=5)
    assert one["synthetic"] is True
    with pytest.raises(ValueError, match="Duplicate test image"):
        aggregate([ev, ev], tmp_path / "bad.json")
    m = json.loads((ev / "metrics.json").read_text())
    m["split"] = "val"
    (ev / "metrics.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="held-out"):
        aggregate([ev], tmp_path / "bad_val.json")


def test_illumination_identity_is_replicated_gray_and_seeded():
    image = Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16), mode="L")
    identity = dict(alpha=1, contrast=1, gamma=1, offset=0, horizontal=0, vertical=0)
    out = illumination_view(image, parameters=identity)
    assert np.array_equal(np.asarray(out)[..., 0], np.asarray(image))
    assert np.array_equal(np.asarray(out)[..., 0], np.asarray(out)[..., 2])
    intensity = np.asarray(image, dtype=np.float64) / 255.0
    transformed = apply_illumination(intensity, float(intensity.mean()), 1.1, 0.9, 1.05, -0.02, 0.1, -0.1)
    assert transformed.min() >= 0 and transformed.max() <= 1
    one = illumination_view(image, np.random.default_rng(4))
    two = illumination_view(image, np.random.default_rng(4))
    assert np.array_equal(np.asarray(one), np.asarray(two))
    assert not np.array_equal(np.asarray(one)[..., 0], np.asarray(image))


def test_industrial_geometry_resizes_without_crop_or_rotation():
    image = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8), mode="L")

    class NoFlip:
        def random(self):
            return 0.9

    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize
    expected = resize(image, [16, 16], InterpolationMode.BICUBIC)
    out = industrial_geometry(image, 16, True, NoFlip())
    assert np.array_equal(np.asarray(out), np.asarray(expected))
    with pytest.raises(ValueError, match="full-image"):
        validate_config(dict(DEFAULT_CONFIG, perturbation="illumination", geometry_mode="histology"))


def test_neu_groups_are_balanced_and_rotate(tmp_path):
    rows = []
    for label, index in NEU_LABELS.items():
        for number in range(5):
            path = tmp_path / f"{label}_{number}.png"
            Image.fromarray(np.full((8, 8), index * 20, dtype=np.uint8), mode="L").save(path)
            image_id = f"{label}_{number}"
            rows.append(dict(image_id=image_id, path=str(path.name), patient_id=image_id, label=label,
                             magnification="native", source="NEU-CLS"))
    manifest = tmp_path / "manifest.csv"
    write_manifest(rows, manifest)
    spec = create_balanced_splits(manifest, tmp_path / "splits.json", n_splits=5, seed=2026)
    assert spec["seed"] == 2026 and spec["scheme"] == "class_balanced_groups"
    tests, vals = [], []
    for fold in range(5):
        parts = load_fold(manifest, tmp_path / "splits.json", fold)
        covered = {row["image_id"] for name in parts for row in parts[name]}
        assert covered == {row["image_id"] for row in rows}
        assert len({row["patient_id"] for name in parts for row in parts[name]}) == len(covered)
        for name, expected in (("train", 3), ("val", 1), ("test", 1)):
            assert Counter(row["label"] for row in parts[name]) == {label: expected for label in NEU_LABELS}
        tests.append({row["image_id"] for row in parts["test"]})
        vals.append({row["image_id"] for row in parts["val"]})
    for fold in range(5):
        assert tests[(fold + 1) % 5] == vals[fold]
        assert tests[fold].isdisjoint(tests[(fold + 1) % 5])
