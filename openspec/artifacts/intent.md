# Intent: yolo-training-pipeline

## Measurable Claim

One local CPU Docker command generates a deterministic detection fixture, validates YOLO annotations, trains YOLO26 from architecture without downloading weights, evaluates held-out mAP, emits a verifiable model bundle, reloads the best checkpoint, and measures warmed inference latency.

## Problem

Establishes the reproducible training, held-out evaluation, checkpoint, and inference evidence contract consumed by later vision serving and domain-specific computer-vision repositories.

## In Scope

- Use the selected component pack: `applied-computer-vision`.
- Keep the project under the Applied Computer Vision and Medical AI program.
- Preserve the benchmark contract: `map50_95_median` in `benchmarks/results/summary.json`.
- Keep the default path local-first and reproducible.

## Out Of Scope

- Paid credentials for the default demo.
- External infrastructure that is not required by the benchmark.
- Replacing local portfolio skills with external components silently.

## Default Demo Path

- Status: specified
- Runtime: Python 3.12.13 slim image pinned by OCI index digest with CPU-only PyTorch; no runtime network or credential.
- Benchmark command: `docker run --rm yolo-training-pipeline`

## Public Proof

- Benchmark: map50_95_median = 0.002420
- Result path: `benchmarks/results/summary.json`
