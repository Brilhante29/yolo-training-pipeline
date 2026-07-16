from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model_manifest(
    *,
    checkpoint_path: Path,
    framework_version: str,
    architecture: str,
    parameters: int,
    image_size: int,
    classes: tuple[str, ...],
    dataset_sha256: str,
    seed: int,
    created_at: str,
    metrics: dict[str, float],
) -> dict[str, Any]:
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise ValueError("A non-empty checkpoint is required")
    if not classes:
        raise ValueError("At least one ordered class is required")
    if parameters <= 0:
        raise ValueError("Model parameter count must be greater than zero")

    manifest = {
        "schema_version": 1,
        "task": "detection",
        "format": "ultralytics-pt",
        "framework": {
            "name": "ultralytics",
            "version": framework_version,
        },
        "model": {
            "architecture": architecture,
            "parameters": parameters,
        },
        "input": {
            "height": image_size,
            "width": image_size,
            "channels": 3,
            "color_space": "RGB",
            "dtype": "uint8",
            "batch": 1,
        },
        "classes": [
            {"id": class_id, "name": name}
            for class_id, name in enumerate(classes)
        ],
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "provenance": {
            "dataset_sha256": dataset_sha256,
            "seed": seed,
            "initialization": "architecture-only-random-seeded",
            "source_repository": "Brilhante29/yolo-training-pipeline",
            "created_at": created_at,
        },
        "metrics": {
            name: float(value)
            for name, value in metrics.items()
            if name in {"map50_95", "map50", "precision", "recall"}
        },
    }
    validate_model_manifest(manifest, checkpoint_path)
    return manifest


def validate_model_manifest(
    manifest: Mapping[str, Any], checkpoint_path: Path
) -> None:
    required = {
        "schema_version",
        "task",
        "format",
        "framework",
        "model",
        "input",
        "classes",
        "checkpoint",
        "provenance",
        "metrics",
    }
    if set(manifest) != required:
        raise ValueError("Model manifest fields do not match schema version 1")
    if manifest["schema_version"] != 1 or manifest["task"] != "detection":
        raise ValueError("Unsupported model artifact schema or task")

    classes = manifest["classes"]
    if not isinstance(classes, list) or not classes:
        raise ValueError("Model manifest must contain ordered classes")
    expected_ids = list(range(len(classes)))
    if [item.get("id") for item in classes] != expected_ids:
        raise ValueError("Model class identifiers must be contiguous and ordered")
    if any(not item.get("name") for item in classes):
        raise ValueError("Model class names must be non-empty")

    checkpoint = manifest["checkpoint"]
    if checkpoint["file"] != checkpoint_path.name:
        raise ValueError("Checkpoint filename does not match its manifest")
    if checkpoint["bytes"] != checkpoint_path.stat().st_size:
        raise ValueError("Checkpoint size does not match its manifest")
    if checkpoint["sha256"] != sha256_file(checkpoint_path):
        raise ValueError("Checkpoint hash does not match its manifest")

    provenance = manifest["provenance"]
    dataset_sha256 = provenance.get("dataset_sha256", "")
    if len(dataset_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in dataset_sha256
    ):
        raise ValueError("Dataset SHA-256 must be a lowercase hexadecimal digest")
    if any(
        not isinstance(value, (int, float))
        for value in manifest["metrics"].values()
    ):
        raise ValueError("Model artifact metrics must be numeric")