# Proposal: Ship yolo-training-pipeline

## Intent

Create the Applied Computer Vision training foundation: one offline CPU Docker command that generates and validates a YOLO-format fixture, trains YOLO26n from architecture, evaluates held-out mAP, reloads the best checkpoint, measures warmed inference, and emits reproducible JSON.

## Portfolio Impact

The repository demonstrates Python, PyTorch, Ultralytics YOLO, dataset contracts, reproducible training, detection metrics, checkpoint governance, latency methodology, Docker, tests, and CI. Its checkpoint and evidence contract are consumed by #7; its rules guide #4, #5, and #6 without forcing one framework onto those domain problems.

## In Scope

- Original deterministic two-class detection fixture.
- YOLO annotation and split leakage validation.
- Architecture-only YOLO26n CPU training.
- Held-out mAP50-95 and secondary quality.
- Best-checkpoint reload and warmed p50/p95 latency.
- Three-run summary and failure evidence.
- AGPL-compatible publication.

## Out Of Scope

Real-domain claims, pretrained weights, external datasets, GPU, export formats, API, tracking, registry, orchestration, cloud, broker, database, and frontend.
