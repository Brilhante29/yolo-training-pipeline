# Benchmark Plan: Held-Out mAP And Warmed Latency

## Hypothesis

A seeded YOLO26n model trained from architecture on a fixed synthetic detection fixture can produce stable held-out mAP evidence and a reloadable checkpoint while keeping warmed batch-1 CPU inference measurable from one Docker path.

## Primary Metric

`map50_95_median`: median held-out mAP averaged across IoU thresholds 0.50 through 0.95 over three complete successful runs.

## Secondary Metrics

| Metric | Unit | Window |
|---|---:|---|
| `map50` | ratio | held-out validation split |
| `precision` | ratio | held-out validation split |
| `recall` | ratio | held-out validation split |
| `training_seconds` | seconds | model construction through best-checkpoint completion |
| `inference_latency_ms_p50` | ms/image | after checkpoint reload and 10 warmups |
| `inference_latency_ms_p95` | ms/image | same 60 batch-1 samples |
| framework preprocess/inference/postprocess | ms/image | Ultralytics timing for measured predictions |
| checkpoint bytes | bytes | persisted best checkpoint |
| model parameters | count | reloaded model |

## Fixed Inputs

- Seed 42.
- 80 train and 20 held-out validation images.
- Two classes, one object per 160 x 160 image.
- YOLO26n architecture-only initialization.
- 25 epochs, batch 8, AdamW, cosine schedule.
- CPU, four Torch threads, AMP off, cache off, workers 0.
- 10 warmups and 60 measured images, batch 1.

## Commands

```bash
docker build -t yolo-training-pipeline .
docker run --rm yolo-training-pipeline
```

Repeat three times with the same image content and mounted outputs named `run-1.json` through `run-3.json`. Then run aggregate mode over that directory.

## Evidence Rules

- The default run must perform no model or dataset download.
- Validate annotations and split hashes before training.
- Evaluate only the held-out split.
- Reload the persisted best checkpoint before latency measurement.
- Preserve every failure JSON with stage and error type.
- Report median, min, max, all raw files, environment, image ID/size, dataset hash, checkpoint bytes, and model parameters.
- Do not compare across CPU/GPU, thread counts, image sizes, batches, or machines without labeling the difference.
- Do not present synthetic-fixture mAP as domain accuracy.

## Publication Gate

The README number must come from `benchmarks/results/summary.json`. Three successful runs, focused coverage, complete Docker integration, license validation, and green CI are mandatory.

## Recorded Result

- Image: `sha256:2ca7c9aa87e939aa66f09f41ae7d098ea864ce7d8dd19db6491e982ff74c412f` (`1,842,427,744` bytes).
- Runs: `run-1.json`, `run-2.json`, and `run-3.json`; all completed with `failures=0`, identical dataset hash, and checkpoint reload proof.
- `map50_95_median`: `0.002420` ratio.
- `inference_latency_ms_p95_median`: `68.303` ms/image.
- `training_seconds_median`: `199.350` seconds.
