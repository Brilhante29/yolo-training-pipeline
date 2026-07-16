# Agent Handoff: yolo-training-pipeline

## Mission

Ship #1 as the reusable training/evidence foundation of the Applied Computer Vision program. Preserve the offline runtime, CPU default, held-out split, AGPL decision, and explicit synthetic-fixture limitation.

## Source Order

1. `openspec/changes/ship-yolo-training-pipeline/` for approved scope and self-challenge.
2. `project.yaml` for the portfolio contract.
3. `sdd/` for architecture, stack, benchmark, license, and release evidence.
4. `.portfolio/` and local `python-computer-vision` skill for reusable standards.
5. Product code and tests for implementation truth.

## Invariants

- The default run downloads no model, weight, image, annotation, or dataset.
- Train and validation image hashes are disjoint.
- Held-out mAP never comes from the training split.
- Latency starts only after best-checkpoint reload and warmup.
- CPU, image size, batch, threads, warmup, and sample count are explicit.
- Synthetic-fixture metrics are never described as real-domain performance.
- Open-source Ultralytics use remains visibly AGPL-3.0-only.
- Ultralytics imports do not enter pure annotation or aggregation policy.
- Serving, registry, cloud, broker, tracking, export, and UI stay out without an OpenSpec change and benchmark force.

## Verification Order

1. Parse project YAML, JSON, PowerShell, and Python source.
2. Build the pinned CPU image and inspect resolved dependency versions.
3. Run Ruff and pure tests with at least 90% focused coverage.
4. Run complete dataset/train/validate/reload/predict integration.
5. Retain three successful runs and every failure.
6. Aggregate median/range, update README, and reconcile SDD.
7. Complete reuse review and synchronize the published kit commit.
8. Publish and inspect GitHub Actions plus benchmark artifact.

## Failure Triage

- Dependency failure: preserve CPU PyTorch and exact Ultralytics before changing versions.
- Model architecture lookup: confirm `yolo26n.yaml` is packaged in the pinned release; do not fall back to remote weights.
- Low mAP: inspect annotations, class balance, validation isolation, and learning curves before changing the fixture.
- Slow training: report stage/device/thread evidence before reducing the proof.
- Variable latency: confirm reload, warmup, batch 1, threads, image size, and competing host load.
- License failure: stop publication until AGPL packaging and attribution are correct.
