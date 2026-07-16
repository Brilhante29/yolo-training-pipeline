import pytest

from yolo_training_pipeline.config import Settings


def test_settings_expose_reproducible_cpu_defaults(monkeypatch) -> None:
    for name in (
        "YOLO_IMAGE_SIZE",
        "YOLO_DEVICE",
        "YOLO_EPOCHS",
        "YOLO_TRAIN_SAMPLES",
        "YOLO_VALIDATION_SAMPLES",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.seed == 42
    assert settings.image_size == 160
    assert settings.device == "cpu"
    assert settings.model_architecture == "yolo26n.yaml"


def test_settings_reject_non_cpu_default(monkeypatch) -> None:
    monkeypatch.setenv("YOLO_DEVICE", "cuda:0")

    with pytest.raises(ValueError, match="cpu"):
        Settings.from_env()


def test_settings_reject_non_stride_aligned_image(monkeypatch) -> None:
    monkeypatch.setenv("YOLO_IMAGE_SIZE", "150")

    with pytest.raises(ValueError, match="divisible by 32"):
        Settings.from_env()
