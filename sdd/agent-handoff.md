# Agent Handoff: yolo-training-pipeline

This file stores observable state, not private reasoning.

- Project: `1 - yolo-training-pipeline`.
- Program: `applied-computer-vision`.
- Status: published.
- Raw evidence: three complete comparable CPU runs under `benchmarks/results/`.
- Result: median held-out mAP50-95 `0.002420`; warmed p95 `68.303 ms/image`.
- V2: `benchmarks/publication/yolo-training-v2.json` from source `a7ca83e52145f726346a762294fc8d3ad1c7d590` and image `sha256:926babbaf404a8d5268449e129674846c3cc2d2677fc440f7439a9d42e812aeb`.
- Source CI: run `31341449701` passed.
- Consumer: #7 verifies and loads checkpoint SHA-256 `41598b005401dab39969719bf0678b55415121911fa72c110b21b75bab9c2e96`.
- Limitation: synthetic-fixture mAP is engineering evidence, not domain accuracy.

Preserve architecture-only initialization, held-out evaluation, AGPL licensing, offline Docker and raw-run comparability. The central portfolio registry records final exact-head publication CI.
