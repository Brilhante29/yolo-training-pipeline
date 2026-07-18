import copy
import hashlib
from pathlib import Path

import pytest

from yolo_training_pipeline.artifact import (
    build_model_manifest,
    validate_model_manifest,
)


def _valid_manifest(tmp_path: Path) -> tuple[Path, dict]:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"deterministic-checkpoint")
    return checkpoint, build_model_manifest(
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

def test_build_manifest_rejects_invalid_checkpoint_and_model(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="non-empty checkpoint"):
        build_model_manifest(
            checkpoint_path=empty,
            framework_version="8.4.96",
            architecture="yolo26n.yaml",
            parameters=1,
            image_size=160,
            classes=("warm_rectangle",),
            dataset_sha256="a" * 64,
            seed=42,
            created_at="2026-07-15T00:00:00+00:00",
            metrics={},
        )

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    for classes, parameters, message in [
        ((), 1, "ordered class"),
        (("warm_rectangle",), 0, "parameter count"),
    ]:
        with pytest.raises(ValueError, match=message):
            build_model_manifest(
                checkpoint_path=checkpoint,
                framework_version="8.4.96",
                architecture="yolo26n.yaml",
                parameters=parameters,
                image_size=160,
                classes=classes,
                dataset_sha256="a" * 64,
                seed=42,
                created_at="2026-07-15T00:00:00+00:00",
                metrics={},
            )


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("remove", "fields"),
        ("version", "Unsupported"),
        ("classes", "ordered classes"),
        ("ids", "contiguous"),
        ("name", "non-empty"),
        ("filename", "filename"),
        ("hash", "hash"),
        ("dataset", "SHA-256"),
        ("metric", "numeric"),
    ],
)
def test_model_manifest_rejects_invalid_contract_fields(
    tmp_path: Path, mutation: str, message: str
) -> None:
    _checkpoint, original = _valid_manifest(tmp_path)
    manifest = copy.deepcopy(original)
    if mutation == "remove":
        manifest.pop("metrics")
    elif mutation == "version":
        manifest["schema_version"] = 2
    elif mutation == "classes":
        manifest["classes"] = []
    elif mutation == "ids":
        manifest["classes"][1]["id"] = 3
