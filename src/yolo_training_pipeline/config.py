from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class Settings:
    runtime_dir: Path
    output_path: Path
    artifact_dir: Path
    seed: int
    train_samples: int
    validation_samples: int
    image_size: int
    epochs: int
    batch_size: int
    torch_threads: int
    warmup_images: int
    measured_images: int
    model_architecture: str
    device: str

    @classmethod
    def from_env(cls) -> Settings:
        runtime_dir = Path(
            os.getenv("YOLO_RUNTIME_DIR", "/tmp/yolo-training-pipeline")
        ).resolve()
        settings = cls(
            runtime_dir=runtime_dir,
            output_path=Path(
                os.getenv(
                    "BENCHMARK_OUTPUT",
                    "/tmp/yolo-training-pipeline/benchmark.json",
                )
            ).resolve(),
            artifact_dir=Path(
                os.getenv(
                    "MODEL_ARTIFACT_DIR",
                    str(runtime_dir / "artifacts"),
                )
            ).resolve(),
            seed=int(os.getenv("YOLO_SEED", "42")),
            train_samples=_positive_int("YOLO_TRAIN_SAMPLES", 80),
            validation_samples=_positive_int("YOLO_VALIDATION_SAMPLES", 20),
            image_size=_positive_int("YOLO_IMAGE_SIZE", 160),
            epochs=_positive_int("YOLO_EPOCHS", 25),
            batch_size=_positive_int("YOLO_BATCH_SIZE", 8),
            torch_threads=_positive_int("YOLO_TORCH_THREADS", 4),
            warmup_images=_positive_int("YOLO_WARMUP_IMAGES", 10),
            measured_images=_positive_int("YOLO_MEASURED_IMAGES", 60),
            model_architecture=os.getenv("YOLO_MODEL_ARCHITECTURE", "yolo26n.yaml"),
            device=os.getenv("YOLO_DEVICE", "cpu"),
        )
        if settings.image_size % 32 != 0:
            raise ValueError("YOLO_IMAGE_SIZE must be divisible by 32")
        if settings.device != "cpu":
            raise ValueError("The default reproducible contract accepts YOLO_DEVICE=cpu only")
        return settings
