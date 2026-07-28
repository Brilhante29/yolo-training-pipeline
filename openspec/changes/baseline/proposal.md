# Change Proposal: baseline

Project: `yolo-training-pipeline` (#1)

## Intent

One local CPU Docker command generates a deterministic detection fixture, validates YOLO annotations, trains YOLO26 from architecture without downloading weights, evaluates held-out mAP, emits a verifiable model bundle, reloads the best checkpoint, and measures warmed inference latency.

## Why This Change Exists

Describe the smallest change that improves the measurable claim or removes a
known portfolio risk.

## Scope

- In scope: Deterministic detection fixture generation, YOLO annotation validation, local training pipeline, and mAP evaluation.
- Out of scope: paid credentials, unrelated infrastructure, and unmeasured features.

## Portfolio Impact

Program: `applied-computer-vision`

This change should produce evidence, fixtures, decisions, or components that
can be reused by sibling repositories without moving project-specific behavior
into the kit.

## Acceptance Signal

The benchmark in `project.yaml` remains reproducible and its result is recorded
in `benchmarks/results/`.
