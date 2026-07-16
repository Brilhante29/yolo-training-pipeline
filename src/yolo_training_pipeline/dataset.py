from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from yolo_training_pipeline.domain import YoloBox

CLASS_NAMES = ("warm_rectangle", "cool_ellipse")


@dataclass(frozen=True)
class DatasetArtifact:
    root: Path
    data_yaml: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_sample(
    image_path: Path,
    label_path: Path,
    image_size: int,
    seed: int,
    index: int,
) -> YoloBox:
    rng = random.Random(seed + index * 104729)
    background = (
        rng.randint(18, 46),
        rng.randint(18, 46),
        rng.randint(18, 46),
    )
    image = Image.new("RGB", (image_size, image_size), background)
    draw = ImageDraw.Draw(image)

    for _ in range(8):
        x = rng.randrange(0, image_size)
        y = rng.randrange(0, image_size)
        length = rng.randrange(6, 24)
        shade = rng.randint(35, 70)
        draw.line((x, y, min(image_size - 1, x + length), y), fill=(shade,) * 3)

    class_id = index % len(CLASS_NAMES)
    width = rng.randrange(image_size // 4, image_size // 2)
    height = rng.randrange(image_size // 4, image_size // 2)
    left = rng.randrange(4, image_size - width - 4)
    top = rng.randrange(4, image_size - height - 4)
    right = left + width - 1
    bottom = top + height - 1

    if class_id == 0:
        draw.rectangle((left, top, right, bottom), fill=(225, 55, 45))
    else:
        draw.ellipse((left, top, right, bottom), fill=(40, 205, 225))

    box = YoloBox(
        class_id=class_id,
        center_x=(left + width / 2) / image_size,
        center_y=(top + height / 2) / image_size,
        width=width / image_size,
        height=height / image_size,
    )
    box.validate(len(CLASS_NAMES))
    image.save(image_path, format="PNG", optimize=False)
    label_path.write_text(box.to_line() + chr(10), encoding="utf-8")
    return box


def _validate_split(
    root: Path, split: str, expected_count: int
) -> list[dict[str, Any]]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = sorted(image_dir.glob("*.png"))
    if len(images) != expected_count:
        raise ValueError(f"{split} expected {expected_count} images, found {len(images)}")

    records: list[dict[str, Any]] = []
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"Missing label for {image_path.name}")
        lines = [
            line for line in label_path.read_text(encoding="utf-8").splitlines() if line
        ]
        if len(lines) != 1:
            raise ValueError(f"{label_path.name} must contain exactly one object")
        YoloBox.from_line(lines[0]).validate(len(CLASS_NAMES))
        with Image.open(image_path) as image:
            dimensions = list(image.size)
        records.append(
            {
                "image": image_path.relative_to(root).as_posix(),
                "label": label_path.relative_to(root).as_posix(),
                "image_sha256": _sha256(image_path),
                "label_sha256": _sha256(label_path),
                "dimensions": dimensions,
            }
        )
    return records


def generate_dataset(
    root: Path,
    *,
    seed: int,
    train_samples: int,
    validation_samples: int,
    image_size: int,
) -> DatasetArtifact:
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    offsets = {"train": 0, "val": 1_000_000}
    counts = {"train": train_samples, "val": validation_samples}
    for split, count in counts.items():
        for index in range(count):
            name = f"{split}-{index:04d}"
            _render_sample(
                root / "images" / split / f"{name}.png",
                root / "labels" / split / f"{name}.txt",
                image_size,
                seed + offsets[split],
                index,
            )

    splits = {
        split: _validate_split(root, split, count) for split, count in counts.items()
    }
    train_hashes = {record["image_sha256"] for record in splits["train"]}
    validation_hashes = {record["image_sha256"] for record in splits["val"]}
    if train_hashes & validation_hashes:
        raise ValueError("Train and validation image content must be disjoint")

    manifest_core = {
        "format": "yolo-detection",
        "source": "deterministic synthetic generator",
        "license": "generated project fixture under AGPL-3.0-only",
        "seed": seed,
        "image_size": [image_size, image_size],
        "classes": list(CLASS_NAMES),
        "counts": counts,
        "splits": splits,
    }
    canonical = json.dumps(manifest_core, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        **manifest_core,
        "dataset_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    data_yaml = root / "dataset.yaml"
    data_yaml.write_text(
        chr(10).join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                *[
                    f"  {class_id}: {name}"
                    for class_id, name in enumerate(CLASS_NAMES)
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return DatasetArtifact(root, data_yaml, manifest_path, manifest)
