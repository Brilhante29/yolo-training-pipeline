# Portfolio Control: #1 yolo-training-pipeline

- **Program:** applied-computer-vision
- **Status:** benchmarked; provenance-rich V2 pending
- **Proves:** offline YOLO training, held-out evaluation, reloadable checkpoint and warmed inference evidence
- **Primary benchmark:** median held-out mAP50-95 over three complete CPU runs

| Evidence | Location | State |
|---|---|---|
| Specification and decisions | `sdd/` | complete |
| Raw runs and aggregate | `benchmarks/results/` | measured and hash-linked |
| Publication evidence | `benchmarks/publication/yolo-training-v2.json` | pending source-image run |
| Model producer contract | `project.yaml`, `.portfolio/contracts/vision-model-artifact.schema.json` | complete |
| Reuse review | `sdd/reuse-improvement-review.md` | complete |
