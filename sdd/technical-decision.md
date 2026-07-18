# Technical Decision: Offline CPU YOLO26 Training

## Status

Accepted; dependency resolution, Docker, and runtime benchmark confirmed.

## Selected Stack

| Concern | Selection | Problem fit |
|---|---|---|
| Runtime | Python 3.12.13 slim, OCI index digest pinned | Multi-platform Python image with explicit supply-chain identity. |
| Model framework | Ultralytics 8.4.96 | Current YOLO26 train, validation, and prediction surfaces. |
| Tensor runtime | PyTorch 2.13.0+cpu / TorchVision 0.28.0+cpu | CPU-only universal path without CUDA runtime weight. |
| Model | YOLO26n architecture YAML, seeded random initialization | No mutable pretrained-weight download in `docker run`. |
| Fixture | Pillow deterministic generator, YOLO labels | Small original dataset with transparent boxes, classes, and hashes. |
| Tests | pytest 9.1.1, pytest-cov 7.0.0 | Pure contract and focused coverage evidence. |
| Lint | Ruff 0.15.21 | Fast deterministic Python style and static checks. |
| Packaging | Docker | Same run contract on Linux, macOS, and Windows hosts. |
| License | AGPL-3.0-only | Compatible posture for open-source Ultralytics use. |

## Reproducibility Controls

- Seed 42 for Python, Torch, fixture, and Ultralytics.
- CPU device, four Torch threads, one interop thread, zero data-loader workers.
- Deterministic algorithms with warnings, AMP disabled, cache disabled.
- 160-pixel images, batch 8, 25 epochs, AdamW, cosine learning rate.
- 80/20 generated split with content-hash leakage gate.
- 10 warmup images, 60 measured batch-1 predictions.
- Framework analytics synchronization disabled.
- No runtime download of model weights or data.

## Recorded Runtime Evidence

- Image: `sha256:2ca7c9aa87e939aa66f09f41ae7d098ea864ce7d8dd19db6491e982ff74c412f` (`1,842,427,744` bytes).
- Three successful CPU runs on the same image and fixture produced median held-out mAP50-95 `0.002420` and warmed p95 `68.303 ms/image`.
- The container embeds a local Matplotlib font under the Ultralytics filename expected by the ASCII fixture, avoiding the auxiliary font download path.

## Metric Mapping

- Primary run metric: `validation.box.map` as held-out mAP50-95.
- Secondary quality: `map50`, mean precision, and mean recall.
- Public primary: median mAP50-95 over three complete runs.
- Latency: nearest-rank p50/p95 wall time after reload and warmup.
- Framework timing: mean preprocess, inference, and postprocess milliseconds.
- Cost evidence: training seconds, pipeline seconds, model parameter count, and checkpoint bytes.
- Cross-repository bundle: `best.pt` plus a schema-versioned manifest with SHA-256, input, classes, provenance, and held-out metrics.

## License Decision

Ultralytics publishes its open-source package under AGPL-3.0 and offers a separate Enterprise license. This repository therefore declares AGPL-3.0-only, cites the canonical license, includes source and build instructions, and makes no claim that the open-source option is suitable for proprietary commercial integration.

## Rejected Options

| Option | Why rejected |
|---|---|
| TorchVision detector | Strong permissive default for generic detection, but misses the named YOLO skill signal. |
| COCO-pretrained YOLO26n | Better quality but adds remote weights and makes the proof partly inherited. |
| COCO8 | Official smoke dataset, but automatically downloads and has four validation images. |
| CUDA image | Hardware-specific and substantially larger; not universal CI. |
| ONNX/OpenVINO | Useful CPU deployment targets, but export comparison is not the current claim. |
| FastAPI | Serving is #7 and would mix training with a different benchmark. |
| MLflow/Airflow | Lifecycle governance is #21; adding it here duplicates rather than reuses. |

## Known Risks

- CPU training may be slow on GitHub-hosted runners.
- Deterministic deep-learning operations can still vary across hardware/runtime implementations.
- A small synthetic split can overstate apparent quality and cannot support domain claims.
- Ultralytics APIs and architecture files are version-sensitive; exact package pinning and Docker integration contain that risk.
- The concise license notice must be reviewed against the publication gate for complete license packaging.
