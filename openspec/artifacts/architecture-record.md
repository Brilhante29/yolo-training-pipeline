# Architecture Record: yolo-training-pipeline

## Decision

- Architecture: `pipeline`
- Stack profile: `python-ml`
- API style: `cli`
- Messaging: `none`
- Database/runtime: `none` / `Python 3.12.13 slim image pinned by OCI index digest with CPU-only PyTorch; no runtime network or credential.`

## Reason

The dominant force is an ordered transition from fixture and annotations to validated split, trained checkpoint, held-out metrics, reloaded inference, and benchmark evidence.

## Dependency Direction

The Ultralytics adapter consumes validated local artifacts and emits checkpoint/metric records; pure dataset and benchmark policy never import the training framework.

## Boundaries

- deterministic image and YOLO annotation generation
- annotation and split contract
- Ultralytics training and validation adapter
- best-checkpoint artifact contract
- versioned model manifest and checkpoint bundle
- warmed inference measurement
- benchmark aggregation and JSON evidence

## Library Policy

Use current pinned Ultralytics for the explicit YOLO claim, CPU-only PyTorch for the universal path, standard-library/Pillow fixture generation, pytest for pure contracts, and AGPL-3.0 for license compatibility.

## Principle Check

- SRP: keep benchmark, API, use cases, and adapters separate.
- OCP: new providers must be adapters, not domain rewrites.
- LSP: replacement providers must preserve observable behavior.
- ISP: ports stay narrow.
- DIP: application depends on behavior, not infrastructure.
- KISS/YAGNI: leave out anything that does not improve the benchmark.
