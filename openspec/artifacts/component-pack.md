# Component Pack: yolo-training-pipeline

## Selected Pack

- Pack id: `applied-computer-vision`
- Pack name: Applied Computer Vision and Medical AI
- Problem: Train, evaluate, and serve visual models with reproducible metrics and domain-specific proof.

## Benchmark Focus

- map
- auc
- sensitivity
- accuracy
- confusion_matrix
- latency_per_image_ms
- throughput_rps

## Preferred Artifacts

- dataset card
- preprocessing manifest
- model card
- confusion matrix
- inference latency script

## Rejection Rules

- Reject health or medical claims without clear demo limitation.
- Reject training notebooks without Docker path.
- Reject serving APIs without latency benchmark.

## Reuse Priority

1. Use repo-local `.codex/skills/` and `.claude/skills/`.
2. Use `.portfolio/` and upstream `portfolio-reuse-kit`.
3. Use external repositories as references for organization, workflow, schemas, tests, benchmarks, and docs.
4. Use external code only with license compatibility, attribution, and a decision record.
