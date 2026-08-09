# Quality Gates: #1 yolo-training-pipeline

- [x] README opens with project number, measured mAP and warmed p95 latency.
- [x] `project.yaml` records problem, pipeline architecture, exact stack, metric and V2 result path.
- [x] SDD and OpenSpec agree with implementation and synthetic-fixture limitations.
- [x] Pure dataset, annotation, aggregation and artifact policy stay outside Ultralytics composition.
- [x] SOLID, LSP, KISS, YAGNI, DRY and dependency-direction decisions are explicit.
- [x] Unit and failure-path tests pass; focused domain logic exceeds the required coverage gate.
- [x] Pinned Docker runs offline from clean checkout and CI executes the full training path.
- [x] Three raw runs are comparable, hash-linked and aggregated without handwritten metrics.
- [x] Provenance-rich V2 binds source `a7ca83e`, image, fixture, config, lock and raw aggregate.
- [x] AGPL-3.0-only packaging and Ultralytics licensing scope are explicit.
- [x] Reuse review separates reusable evidence contracts from project-specific training code.

Exact-head GitHub status is an external release fact recorded by the central portfolio registry; the workflow validates every pushed head and avoids self-referential commit metadata.
