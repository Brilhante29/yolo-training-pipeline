# Architecture Decision: Artifact Pipeline With Pure Contracts

## Status

Accepted; runtime benchmark confirmed.

## Context

The dominant problem is an ordered and auditable transition from image fixture to labels, split manifest, model checkpoint, held-out metrics, warmed predictions, and JSON evidence. Data reproducibility and auditability are high. UI state, asynchronous throughput, independent deployment, and external integration pressure are low.

## Decision

Use pipeline architecture. Keep annotation rules, dataset manifest logic, percentile calculation, and result aggregation pure. Use Ultralytics directly as the training/validation/prediction adapter.

```text
src/yolo_training_pipeline/
  config.py       explicit reproducibility settings
  domain.py       annotation and benchmark policy
  dataset.py      deterministic fixture and split manifest
  pipeline.py     Ultralytics composition and evidence
  cli.py          run/aggregate command
tests/             pure contract tests
benchmarks/        retained run and summary JSON
```

The dependency direction is toward pure contracts: framework code consumes validated artifacts and maps framework results into project evidence. Pure modules do not import Ultralytics, Torch, cloud SDKs, APIs, brokers, databases, or UI code.

## Why It Fits

- Training is an artifact pipeline, not a request-driven application.
- Direct framework calls keep the concept visible and avoid wrappers that only rename `train`, `val`, and `predict`.
- The split manifest and result JSON make the proof inspectable.
- One process and one image keep training and latency attributable.

## Rejected Alternatives

| Alternative | Why rejected |
|---|---|
| MVC/layered | Controller and service layers do not express artifact transitions or data leakage gates. |
| Hexagonal as primary | There are few external actors; a framework port would not make training behavior substitutable. |
| Notebook-first | Hidden state and manual cells violate one-command evidence. |
| Microservices | No independent scaling or deployment force exists. |
| Event-driven | No event stream, fan-out, or asynchronous throughput target exists. |
| MLOps platform | Registry and orchestration are already proved by #21 and are not required for mAP/latency here. |

## OpenSpec Self-Challenge

| Question | Answer |
|---|---|
| What force dominates? | Reproducible ordered artifacts and held-out evidence. |
| What easiest complexity could be added? | GPU profiles, serving, ONNX, MLflow, Airflow, cloud storage, and brokers. |
| Why not use pretrained weights? | They improve quality but introduce implicit artifact download and attribution into the runtime path. |
| Why synthetic data? | It makes offline mechanics deterministic; the README explicitly rejects domain claims. |
| Why Ultralytics despite AGPL? | The project is explicitly YOLO-focused and adopts AGPL-3.0-only rather than hiding the license. |
| What invalidates this design? | A later claim requiring provider substitution, independent deployment, a real domain dataset, or export-runtime comparison. |

## Revisit Triggers

- Introduce a dataset adapter when a legally redistributable real-domain fixture becomes the benchmark.
- Introduce an artifact-storage port only when #7 or cloud storage consumes checkpoints across process boundaries.
- Add GPU only with separate hardware, image, and benchmark evidence.
- Add ONNX/OpenVINO only when export speed or portability is the primary metric.
- Split serving into #7 rather than changing this repository's training boundary.
