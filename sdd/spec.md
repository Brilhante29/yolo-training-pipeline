# Spec: yolo-training-pipeline

## Number And Claim

#1 proves that one CPU Docker command can create a deterministic YOLO-format fixture, validate annotations and split isolation, train YOLO26n without pretrained weights, evaluate held-out detection quality, reload the best checkpoint, and measure warmed batch-1 latency.

## Portfolio Program

`applied-computer-vision`. The checkpoint, dataset manifest, metric mapping, and benchmark conventions become inputs to #7 vision serving and references for #4, #5, and #6 domain projects.

## In Scope

- Deterministic two-class synthetic detection fixture.
- Normalized YOLO annotation and split validation.
- Architecture-only YOLO26n initialization.
- CPU-only deterministic training and best-checkpoint retention.
- Held-out mAP50-95, mAP50, precision, and recall.
- Best-checkpoint reload and warmed batch-1 latency samples.
- Versioned model bundle manifest conforming to the shared vision artifact schema.
- Failure evidence and three-run summary.
- AGPL-3.0 license decision and source attribution.

## Out Of Scope

- Natural-image or domain accuracy claims.
- Pretrained weights, external datasets, GPU/multi-GPU, export formats, serving API, tracking, registry, drift, cloud, broker, database, or frontend.

## Runtime Contract

```bash
docker build -t yolo-training-pipeline .
docker run --rm yolo-training-pipeline
```

The run SHALL require no secret, runtime network, host Python, GPU, cloud account, or manual dataset step. It SHALL print and write JSON containing the held-out metrics, training duration, warmed latency, environment, dataset hash, checkpoint identity/bytes, model parameters, and failure count.

## Dataset Contract

- Source: original deterministic generator.
- License: project fixture under AGPL-3.0-only.
- Classes: `warm_rectangle`, `cool_ellipse`.
- Default split: 80 training and 20 held-out validation images.
- Image size: 160 x 160 RGB.
- Annotation: one normalized YOLO box per image.
- Seed: 42.
- Leakage gate: no identical image hash may cross splits.
- Limitation: engineering fixture only; no domain generalization claim.

## Definition Of Done

- [x] Architecture, stack, license, and rejected alternatives answer the OpenSpec self-challenge.
- [x] Dataset generator, annotation validation, pipeline, failure output, and aggregation are implemented.
- [x] Pure tests cover annotation, split, deterministic manifest, percentile, settings, and summary behavior.
- [ ] Pinned Docker image builds and imports CPU PyTorch/Ultralytics.
- [ ] Lint and at least 90% focused pure-module coverage pass in the image.
- [ ] Three complete train/validate/reload/predict runs succeed.
- [ ] Summary JSON and README open with the confirmed median mAP50-95 and p95 latency.
- [ ] Full AGPL license text or canonical license packaging is confirmed.
- [ ] Reuse kit improvements are published and synchronized by commit.
- [ ] Public CI is green and its benchmark artifact is downloadable.
