import hashlib
from pathlib import Path

import pytest

from yolo_training_pipeline.artifact import (
    build_model_manifest,
    validate_model_manifest,
)


def test_model_manifest_binds_checkpoint_input_classes_and_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"deterministic-checkpoint")

    manifest = build_model_manifest(
        checkpoint_path=checkpoint,
        framework_version="8.4.96",
        architecture="yolo26n.yaml",
        parameters=2_400_000,
        image_size=160,
        classes=("warm_rectangle", "cool_ellipse"),
        dataset_sha256="a" * 64,
        seed=42,
        created_at="2026-07-15T00:00:00+00:00",
        metrics={"map50_95": 0.5, "map50": 0.8, "training_seconds": 12.0},
    )

    assert manifest["schema_version"] == 1
    assert manifest["checkpoint"]["sha256"] == hashlib.sha256(
        b"deterministic-checkpoint"
    ).hexdigest()
    assert manifest["checkpoint"]["bytes"] == checkpoint.stat().st_size
    assert manifest["input"]["batch"] == 1
    assert manifest["classes"][1] == {"id": 1, "name": "cool_ellipse"}
    assert manifest["provenance"]["dataset_sha256"] == "a" * 64
    assert "training_seconds" not in manifest["metrics"]


def test_model_manifest_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"first-checkpoint")
    manifest = build_model_manifest(
        checkpoint_path=checkpoint,
        framework_version="8.4.96",
        architecture="yolo26n.yaml",
        parameters=2_400_000,
        image_size=160,
        classes=("warm_rectangle", "cool_ellipse"),
        dataset_sha256="a" * 64,
        seed=42,
        created_at="2026-07-15T00:00:00+00:00",
        metrics={"map50_95": 0.5},
    )
    checkpoint.write_bytes(b"tampered-checkpoint")

    with pytest.raises(ValueError, match="Checkpoint size"):
        validate_model_manifest(manifest, checkpoint)