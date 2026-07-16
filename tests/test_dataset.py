import json
from pathlib import Path

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
