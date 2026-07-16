# Design: yolo-training-pipeline

## Primary Architecture

Pipeline architecture with pure annotation/benchmark contracts and a direct Ultralytics adapter.

```text
generate -> validate annotations -> prove split isolation -> train
train -> best checkpoint -> held-out validate
best checkpoint -> reload -> warm -> measure -> JSON
three JSON runs -> aggregate -> README evidence
```

## Dependency Direction

`domain.py` and benchmark aggregation know no Ultralytics, Torch, API, cloud, broker, database, or UI. `dataset.py` depends on pure `YoloBox` validation. `pipeline.py` composes framework behavior and maps it into stable evidence.

## Self-Challenge

| Question | Answer | Decision |
|---|---|---|
| Is a real dataset needed to prove this repository? | No; this repository proves training mechanics, not domain efficacy. | Generate and label a deterministic fixture; state its limitation beside metrics. |
| Does pretrained initialization solve the present force? | It improves mAP but adds downloaded weights and inherited knowledge. | Start from packaged architecture YAML with seeded random weights. |
| Is Ultralytics license compatible with MIT? | Open-source Ultralytics is AGPL-3.0, not an invisible permissive dependency. | License this repository AGPL-3.0-only and document Enterprise separately. |
| Is FastAPI needed? | No request/throughput claim exists. | Keep CLI; #7 serves the artifact. |
| Are MLflow and Airflow needed? | No registry/promotion/orchestration metric exists here. | Reuse #21 as the lifecycle system instead of duplicating it. |
| Is GPU needed? | Universal reproducibility is the current force. | CPU default; GPU requires a separate benchmark profile. |
| Is ONNX/OpenVINO needed? | Export speed is a different question. | Measure native persisted checkpoint only. |
| Is cloud/Kumo needed? | No AWS behavior exists. | No cloud adapter or emulator. |
| Are framework ports useful? | A port that mirrors train/val/predict adds no substitution value. | Keep pure contracts and direct framework composition. |

## Revisit Triggers

- A legally redistributable domain dataset becomes the public benchmark.
- #7 requires a stable cross-repository checkpoint metadata contract.
- GPU or export runtime becomes the explicit metric.
- A second trainer framework demonstrates real substitution requirements.
- Concrete AWS artifact storage enters acceptance criteria; use Kumo before real AWS.
