from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def validate(self, class_count: int) -> None:
        if not 0 <= self.class_id < class_count:
            raise ValueError(f"class_id {self.class_id} is outside [0, {class_count})")
        values = (self.center_x, self.center_y, self.width, self.height)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("YOLO coordinates must be normalized to [0, 1]")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("YOLO width and height must be greater than zero")
        if self.center_x - self.width / 2 < 0 or self.center_x + self.width / 2 > 1:
            raise ValueError("YOLO box crosses the horizontal image boundary")
        if self.center_y - self.height / 2 < 0 or self.center_y + self.height / 2 > 1:
            raise ValueError("YOLO box crosses the vertical image boundary")

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.center_x:.6f} {self.center_y:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )

    @classmethod
    def from_line(cls, line: str) -> YoloBox:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError("A YOLO detection row must contain five values")
        return cls(int(parts[0]), *(float(value) for value in parts[1:]))


def nearest_rank_percentile(values: Iterable[float], quantile: float) -> float:
    samples = sorted(float(value) for value in values)
    if not samples:
        raise ValueError("At least one sample is required")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    index = max(0, int(len(samples) * quantile + 0.999999) - 1)
    return samples[index]


def aggregate_results(result_paths: list[Path]) -> dict[str, Any]:
    if len(result_paths) < 3:
        raise ValueError("At least three successful run files are required")

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    failed = [run for run in runs if int(run.get("failures", 0)) != 0]
    if failed:
        raise ValueError("Failed runs cannot be included in a successful summary")

    comparable_environment_fields = (
        "batch_size",
        "device",
        "epochs",
        "image_size",
        "measured_images",
        "seed",
        "torch",
        "torch_threads",
        "torchvision",
        "train_samples",
        "ultralytics",
        "validation_samples",
        "warmup_images",
    )
    signatures = {
        tuple(run["environment"].get(field) for field in comparable_environment_fields)
        for run in runs
    }
    dataset_hashes = {run["proof"]["dataset_sha256"] for run in runs}
    architectures = {run["proof"]["model_architecture"] for run in runs}
    if len(signatures) != 1 or len(dataset_hashes) != 1 or len(architectures) != 1:
        raise ValueError("Runs are not comparable")

    map_values = [float(run["metrics"]["map50_95"]) for run in runs]
    map50_values = [float(run["metrics"]["map50"]) for run in runs]
    latency_values = [
        float(run["metrics"]["inference_latency_ms_p95"]) for run in runs
    ]
    latency_p50_values = [
        float(run["metrics"]["inference_latency_ms_p50"]) for run in runs
    ]
    training_values = [float(run["metrics"]["training_seconds"]) for run in runs]
    ordered_map = sorted(map_values)
    value = float(statistics.median(ordered_map))
    metrics = {
        "repetitions": len(runs),
        "map50_95_median": round(value, 6),
        "map50_95_min": round(min(map_values), 6),
        "map50_95_max": round(max(map_values), 6),
        "map50_median": round(float(statistics.median(map50_values)), 6),
        "inference_latency_ms_p50_median": round(
            float(statistics.median(latency_p50_values)), 3
        ),
        "inference_latency_ms_p95_median": round(
            float(statistics.median(latency_values)), 3
        ),
        "training_seconds_median": round(
            float(statistics.median(training_values)), 3
        ),
    }
    return {
        "project": "yolo-training-pipeline",
        "metric": "map50_95_median",
        "value": round(value, 6),
        "unit": "ratio",
        "timestamp": max(str(run["timestamp"]) for run in runs),
        "command": (
            "docker run --rm yolo-training-pipeline aggregate "
            "--input-dir /results --output /results/summary.json"
        ),
        "repeat": len(runs),
        "measured_iterations": len(runs),
        "samples": [round(item, 6) for item in map_values],
        "environment": runs[0]["environment"],
        "failures": 0,
        "metrics": metrics,
        "summary": metrics,
        "proof": {
            "dataset_sha256": next(iter(dataset_hashes)),
            "dataset_sha256_identical": True,
            "model_architecture": next(iter(architectures)),
            "model_architecture_identical": True,
            "all_checkpoints_reloaded": all(
                run["proof"]["checkpoint_reloaded"] for run in runs
            ),
        },
        "results": [
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in result_paths
        ],
    }
