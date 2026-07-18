import json
from pathlib import Path

import pytest

from yolo_training_pipeline.domain import (
    YoloBox,
    aggregate_results,
    nearest_rank_percentile,
)


def test_box_round_trip_preserves_valid_normalized_values() -> None:
    box = YoloBox(1, 0.5, 0.4, 0.25, 0.30)

    parsed = YoloBox.from_line(box.to_line())
    parsed.validate(class_count=2)

    assert parsed == box


@pytest.mark.parametrize(
    "box",
    [
        YoloBox(2, 0.5, 0.5, 0.2, 0.2),
        YoloBox(0, -0.1, 0.5, 0.2, 0.2),
        YoloBox(0, 0.05, 0.5, 0.2, 0.2),
        YoloBox(0, 0.5, 0.95, 0.2, 0.2),
    ],
)
def test_box_validation_rejects_invalid_annotations(box: YoloBox) -> None:
    with pytest.raises(ValueError):
        box.validate(class_count=2)


@pytest.mark.parametrize(
    "box",
    [YoloBox(0, 0.5, 0.5, 0.0, 0.2), YoloBox(0, 0.5, 0.5, 0.2, 0.0)],
)
def test_box_validation_rejects_empty_dimensions(box: YoloBox) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        box.validate(class_count=2)


def test_box_parser_rejects_wrong_column_count() -> None:
    with pytest.raises(ValueError, match="five values"):
        YoloBox.from_line("0 0.5 0.5 0.2")


def test_nearest_rank_percentile_is_explicit() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]

    assert nearest_rank_percentile(values, 0.50) == 3.0
    assert nearest_rank_percentile(values, 0.95) == 5.0


@pytest.mark.parametrize("quantile", [0.0, 1.1])
def test_nearest_rank_percentile_rejects_invalid_input(quantile: float) -> None:
    with pytest.raises(ValueError):
        nearest_rank_percentile([1.0], quantile)


def _write_run(path: Path, map_value: float, latency: float, training: float) -> None:
    path.write_text(
        json.dumps(
            {
                "timestamp": f"2026-07-15T00:00:0{path.stem[-1]}+00:00",
                "environment": {"device": "cpu"},
                "failures": 0,
                "metrics": {
                    "map50_95": map_value,
                    "inference_latency_ms_p95": latency,
                    "training_seconds": training,
                },
                "proof": {
                    "dataset_sha256": "same",
                    "model_architecture": "yolo26n.yaml",
                    "checkpoint_reloaded": True,
                },
            }
        ),
        encoding="utf-8",
    )


def test_aggregation_requires_three_successful_runs(tmp_path: Path) -> None:
    paths = [tmp_path / f"run-{index}.json" for index in range(1, 4)]
    _write_run(paths[0], 0.4, 12.0, 30.0)
    _write_run(paths[1], 0.6, 10.0, 32.0)
    _write_run(paths[2], 0.5, 11.0, 31.0)

    summary = aggregate_results(paths)

    assert summary["metric"] == "map50_95_median"
    assert summary["value"] == 0.5
    assert summary["metrics"]["inference_latency_ms_p95_median"] == 11.0
    assert summary["proof"]["dataset_sha256_identical"] is True


def test_aggregation_rejects_too_few_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="three successful"):
        aggregate_results([tmp_path / "run-1.json"])


def test_aggregation_rejects_failed_runs(tmp_path: Path) -> None:
    paths = [tmp_path / f"run-{index}.json" for index in range(1, 4)]
    for path in paths:
        _write_run(path, 0.5, 11.0, 31.0)
    failed = json.loads(paths[0].read_text(encoding="utf-8"))
    failed["failures"] = 1
    paths[0].write_text(json.dumps(failed), encoding="utf-8")

    with pytest.raises(ValueError, match="Failed runs"):
        aggregate_results(paths)
