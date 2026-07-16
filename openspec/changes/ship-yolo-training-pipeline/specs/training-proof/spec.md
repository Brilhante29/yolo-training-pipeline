# Training Proof Requirements

## Requirement: Offline Deterministic Fixture

The system SHALL generate all default images and YOLO labels locally from seed 42, validate normalized boxes, retain a manifest, and reject identical content across train and validation splits.

### Scenario: Clean Container

Given a clean built image with no network credential, when the default command runs, then dataset generation completes without downloading images, labels, or weights.

## Requirement: Architecture-Only CPU Training

The system SHALL instantiate the packaged YOLO26n architecture without pretrained weights and train with explicit CPU, seed, image size, batch, epochs, workers, precision, cache, optimizer, and thread settings.

### Scenario: Training Completion

Given a valid fixture, when training completes, then a non-empty best checkpoint exists and its size and parameter count are recorded.

## Requirement: Held-Out Quality

The system SHALL evaluate the best checkpoint on the validation split and expose mAP50-95, mAP50, precision, and recall.

### Scenario: Metric Provenance

Given a trained checkpoint, when validation runs, then the evidence names `held-out-val` and never substitutes training metrics.

## Requirement: Warmed Inference

The system SHALL reload the persisted checkpoint, execute 10 warmup predictions, measure 60 batch-1 predictions at fixed image size, and expose nearest-rank p50/p95 wall latency plus framework timing.

### Scenario: Checkpoint Reload

Given the best checkpoint, when inference measurement starts, then the proof records `checkpoint_reloaded=true`.

## Requirement: Verifiable Model Bundle

The system SHALL copy the best checkpoint into a portable artifact directory and emit a schema-versioned `model-manifest.json` containing framework, architecture, input, ordered classes, checkpoint SHA-256 and bytes, dataset provenance, seed, initialization, and held-out metrics.

### Scenario: Consumer Integrity Check

Given `best.pt` and `model-manifest.json`, when a serving repository validates the bundle, then the recorded filename, byte count, and SHA-256 match the checkpoint before model loading.

## Requirement: Honest Scope

The README and result SHALL label the fixture as synthetic engineering proof and SHALL NOT claim real-domain accuracy.

## Requirement: Reproducible Publication

Publication SHALL use at least three successful run JSON files, preserve failures, aggregate median/min/max, and match the opening README number.

## Requirement: License Visibility

The repository SHALL declare AGPL-3.0-only while importing open-source Ultralytics and SHALL cite the canonical framework and license sources.
