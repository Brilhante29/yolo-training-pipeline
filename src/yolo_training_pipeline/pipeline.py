from __future__ import annotations

import json
import os
import platform
import random
import shutil
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO, settings as ultralytics_settings

from yolo_training_pipeline.artifact import build_model_manifest
from yolo_training_pipeline.config import Settings
from yolo_training_pipeline.dataset import (
    CLASS_NAMES,
    DatasetArtifact,
    generate_dataset,
)
from yolo_training_pipeline.domain import nearest_rank_percentile


def _configure_runtime(settings: Settings) -> None:
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.set_num_threads(settings.torch_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    ultralytics_root = settings.runtime_dir / "ultralytics"
    ultralytics_settings.update(
        {
            "datasets_dir": str(settings.runtime_dir / "datasets"),
            "weights_dir": str(ultralytics_root / "weights"),
            "runs_dir": str(ultralytics_root / "runs"),
            "sync": False,
        }
    )


def _train_and_validate(
    settings: Settings, dataset: DatasetArtifact
) -> tuple[YOLO, Path, dict[str, float]]:
    runs_dir = settings.runtime_dir / "runs"
    model = YOLO(settings.model_architecture, task="detect")
    training_started = time.perf_counter()
    model.train(
        data=str(dataset.data_yaml),
        epochs=settings.epochs,
        imgsz=settings.image_size,
        batch=settings.batch_size,
        device=settings.device,
        workers=0,
        seed=settings.seed,
        deterministic=True,
        pretrained=False,
        amp=False,
        cache=False,
        val=True,
        plots=False,
        save=True,
        verbose=False,
        optimizer="AdamW",
        cos_lr=True,
        project=str(runs_dir),
        name="train",
        exist_ok=True,
    )
    training_seconds = time.perf_counter() - training_started
    if model.trainer is None:
        raise RuntimeError("Ultralytics trainer did not expose the completed run")
    checkpoint_path = Path(model.trainer.best).resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Best checkpoint was not produced: {checkpoint_path}")

    trained_model = YOLO(str(checkpoint_path), task="detect")
    validation = trained_model.val(
        data=str(dataset.data_yaml),
        split="val",
        imgsz=settings.image_size,
        batch=1,
        device=settings.device,
        workers=0,
        plots=False,
        save_json=False,
        verbose=False,
        project=str(runs_dir),
        name="validation",
    )
    metrics = {
        "training_seconds": round(training_seconds, 3),
        "map50_95": round(float(validation.box.map), 6),
        "map50": round(float(validation.box.map50), 6),
        "precision": round(float(validation.box.mp), 6),
        "recall": round(float(validation.box.mr), 6),
    }
    return trained_model, checkpoint_path, metrics


def _benchmark_inference(
    model: YOLO, settings: Settings, validation_dir: Path
) -> dict[str, float]:
    images = sorted(validation_dir.glob("*.png"))
    if not images:
        raise ValueError("Validation images are required for inference benchmarking")

    for index in range(settings.warmup_images):
        model.predict(
            source=str(images[index % len(images)]),
            imgsz=settings.image_size,
            batch=1,
            device=settings.device,
            verbose=False,
            save=False,
        )

    wall_samples: list[float] = []
    preprocess_samples: list[float] = []
    inference_samples: list[float] = []
    postprocess_samples: list[float] = []
    for index in range(settings.measured_images):
        started = time.perf_counter()
        predictions = model.predict(
            source=str(images[index % len(images)]),
            imgsz=settings.image_size,
            batch=1,
            device=settings.device,
            verbose=False,
            save=False,
        )
        wall_samples.append((time.perf_counter() - started) * 1_000)
        speed = predictions[0].speed
        preprocess_samples.append(float(speed["preprocess"]))
        inference_samples.append(float(speed["inference"]))
        postprocess_samples.append(float(speed["postprocess"]))

    return {
        "inference_latency_ms_p50": round(
            nearest_rank_percentile(wall_samples, 0.50), 3
        ),
        "inference_latency_ms_p95": round(
            nearest_rank_percentile(wall_samples, 0.95), 3
        ),
        "framework_preprocess_ms_mean": round(
            sum(preprocess_samples) / len(preprocess_samples), 3
        ),
        "framework_inference_ms_mean": round(
            sum(inference_samples) / len(inference_samples), 3
        ),
        "framework_postprocess_ms_mean": round(
            sum(postprocess_samples) / len(postprocess_samples), 3
        ),
    }


def run_pipeline() -> dict[str, Any]:
    settings = Settings.from_env()
    pipeline_started = time.perf_counter()
    current_stage = "runtime_initialization"
    if settings.runtime_dir.exists():
        shutil.rmtree(settings.runtime_dir)
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)

    try:
        _configure_runtime(settings)

        current_stage = "dataset_generation_and_validation"
        dataset_started = time.perf_counter()
        dataset = generate_dataset(
            settings.runtime_dir / "dataset",
            seed=settings.seed,
            train_samples=settings.train_samples,
            validation_samples=settings.validation_samples,
            image_size=settings.image_size,
        )
        dataset_seconds = time.perf_counter() - dataset_started

        current_stage = "training_and_held_out_validation"
        model, checkpoint_path, model_metrics = _train_and_validate(settings, dataset)

        current_stage = "checkpoint_reload_and_inference"
        inference_metrics = _benchmark_inference(
            model,
            settings,
            dataset.root / "images" / "val",
        )
        parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
        total_seconds = time.perf_counter() - pipeline_started
        created_at = datetime.now(UTC).isoformat()

        settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        bundle_checkpoint = settings.artifact_dir / "best.pt"
        shutil.copy2(checkpoint_path, bundle_checkpoint)
        model_manifest = build_model_manifest(
            checkpoint_path=bundle_checkpoint,
            framework_version=version("ultralytics"),
            architecture=settings.model_architecture,
            parameters=parameter_count,
            image_size=settings.image_size,
            classes=CLASS_NAMES,
            dataset_sha256=dataset.manifest["dataset_sha256"],
            seed=settings.seed,
            created_at=created_at,
            metrics=model_metrics,
        )
        manifest_path = settings.artifact_dir / "model-manifest.json"
        manifest_path.write_text(
            json.dumps(model_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        result: dict[str, Any] = {
            "project": "yolo-training-pipeline",
            "metric": "map50_95",
            "value": model_metrics["map50_95"],
            "unit": "ratio",
            "timestamp": created_at,
            "command": "docker run --rm yolo-training-pipeline",
            "environment": {
                "os": platform.platform(),
                "architecture": platform.machine(),
                "cpu_count": os.cpu_count(),
                "torch_threads": torch.get_num_threads(),
                "python": platform.python_version(),
                "torch": version("torch"),
                "torchvision": version("torchvision"),
                "ultralytics": version("ultralytics"),
                "device": settings.device,
                "seed": settings.seed,
                "train_samples": settings.train_samples,
                "validation_samples": settings.validation_samples,
                "image_size": settings.image_size,
                "batch_size": settings.batch_size,
                "epochs": settings.epochs,
                "warmup_images": settings.warmup_images,
                "measured_images": settings.measured_images,
            },
            "failures": 0,
            "metrics": {
                **model_metrics,
                **inference_metrics,
                "dataset_seconds": round(dataset_seconds, 3),
                "pipeline_seconds": round(total_seconds, 3),
            },
            "proof": {
                "dataset_sha256": dataset.manifest["dataset_sha256"],
                "dataset_manifest": dataset.manifest_path.as_posix(),
                "model_architecture": settings.model_architecture,
                "initialization": "architecture-only-random-seeded",
                "checkpoint": bundle_checkpoint.as_posix(),
                "checkpoint_sha256": model_manifest["checkpoint"]["sha256"],
                "checkpoint_bytes": model_manifest["checkpoint"]["bytes"],
                "model_manifest": manifest_path.as_posix(),
                "checkpoint_reloaded": True,
                "model_parameters": parameter_count,
                "runtime_network_required": False,
                "ultralytics_sync": False,
                "validation_split": "held-out-val",
                "fixture_scope": "synthetic-pipeline-proof-not-domain-accuracy",
            },
        }
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        settings.output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    except Exception as exc:
        failure = {
            "project": "yolo-training-pipeline",
            "metric": "map50_95",
            "value": 0.0,
            "unit": "ratio",
            "timestamp": datetime.now(UTC).isoformat(),
            "command": "docker run --rm yolo-training-pipeline",
            "environment": {
                "os": platform.platform(),
                "architecture": platform.machine(),
                "python": platform.python_version(),
                "device": settings.device,
                "seed": settings.seed,
            },
            "failures": 1,
            "proof": {"pipeline_completed": False},
            "failure": {
                "stage": current_stage,
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        settings.output_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise


def write_summary(input_dir: Path, output_path: Path) -> dict[str, Any]:
    from yolo_training_pipeline.domain import aggregate_results

    paths = sorted(input_dir.glob("run-*.json"))
    summary = aggregate_results(paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary
