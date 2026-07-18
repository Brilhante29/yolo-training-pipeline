import json
from pathlib import Path

import pytest

from yolo_training_pipeline import dataset as dataset_module
from yolo_training_pipeline.dataset import CLASS_NAMES, generate_dataset
from yolo_training_pipeline.domain import YoloBox


def test_generated_dataset_is_valid_deterministic_and_disjoint(tmp_path: Path) -> None:
    first = generate_dataset(
        tmp_path / "first",
        seed=42,
        train_samples=6,
        validation_samples=4,
        image_size=96,
    )
    second = generate_dataset(
        tmp_path / "second",
        seed=42,
        train_samples=6,
        validation_samples=4,
        image_size=96,
    )

    assert first.manifest["dataset_sha256"] == second.manifest["dataset_sha256"]
    assert first.manifest["counts"] == {"train": 6, "val": 4}
    train_hashes = {
        item["image_sha256"] for item in first.manifest["splits"]["train"]
    }
    validation_hashes = {
        item["image_sha256"] for item in first.manifest["splits"]["val"]
    }
    assert train_hashes.isdisjoint(validation_hashes)

    label_path = next((first.root / "labels" / "train").glob("*.txt"))
    box = YoloBox.from_line(label_path.read_text(encoding="utf-8").strip())
    box.validate(len(CLASS_NAMES))

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "deterministic synthetic generator"
    assert first.data_yaml.read_text(encoding="utf-8").count("images/") == 2


def test_generated_dataset_replaces_existing_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    generate_dataset(root, seed=42, train_samples=2, validation_samples=1, image_size=96)
    artifact = generate_dataset(
        root, seed=43, train_samples=1, validation_samples=1, image_size=96
    )

    assert artifact.manifest["seed"] == 43
    assert len(list((root / "images" / "train").glob("*.png"))) == 1


def test_split_validation_rejects_wrong_layout(tmp_path: Path) -> None:
    artifact = generate_dataset(
        tmp_path / "dataset", seed=42, train_samples=1, validation_samples=1, image_size=96
    )
    label_path = next((artifact.root / "labels" / "train").glob("*.txt"))

    with pytest.raises(ValueError, match="expected 2 images"):
        dataset_module._validate_split(artifact.root, "train", 2)

    label_path.unlink()
    with pytest.raises(ValueError, match="Missing label"):
        dataset_module._validate_split(artifact.root, "train", 1)

    generate_dataset(
        artifact.root, seed=42, train_samples=1, validation_samples=1, image_size=96
    )
    label_path = next((artifact.root / "labels" / "train").glob("*.txt"))
    label_path.write_text(
        label_path.read_text(encoding="utf-8") * 2,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one object"):
        dataset_module._validate_split(artifact.root, "train", 1)


def test_dataset_rejects_cross_split_content_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dataset_module, "_sha256", lambda path: "a" * 64)

    with pytest.raises(ValueError, match="disjoint"):
        generate_dataset(
            tmp_path / "dataset", seed=42, train_samples=1, validation_samples=1, image_size=96
        )
