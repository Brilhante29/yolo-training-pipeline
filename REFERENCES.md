# References

| Area | Source | Use in this repository |
|---|---|---|
| Framework release | [Ultralytics 8.4.96 on PyPI](https://pypi.org/project/ultralytics/8.4.96/) | Exact framework package and release provenance. |
| Model family | [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) | Architecture-only YOLO26n training and CPU-oriented model choice. |
| Training | [Ultralytics train mode](https://docs.ultralytics.com/modes/train/) | Explicit epochs, image size, batch, device, seed, deterministic mode, and validation. |
| Detection metrics | [Ultralytics validation mode](https://docs.ultralytics.com/modes/val/) | Held-out mAP50-95, mAP50, precision, and recall mapping. |
| Latency | [Ultralytics benchmark mode](https://docs.ultralytics.com/modes/benchmark/) | Warmed inference and per-image latency terminology. |
| Smoke dataset alternative | [Ultralytics COCO8](https://docs.ultralytics.com/datasets/detect/coco8/) | Considered and rejected for the offline default because first use downloads data and validation has four images. |
| Framework license | [Ultralytics licensing](https://github.com/ultralytics/ultralytics#license) | AGPL-3.0-only repository decision; Enterprise is a separate option. |
| License text | [GNU AGPL v3](https://www.gnu.org/licenses/agpl-3.0.txt) | Canonical terms referenced by `LICENSE`. |
| ML runtime | [PyTorch local installation](https://pytorch.org/get-started/locally/) | CPU-only package index and universal default. |
| Base image | [Python official image](https://hub.docker.com/_/python) | Python 3.12.13 slim base and reviewed OCI index digest. |
| Organization reference | [Paulescu](https://github.com/Paulescu) | Problem-first ML repository organization and reproducible workflow inspiration; no code copied. |

No external image, annotation, checkpoint, or source-code fixture is copied into the repository. Generated images and labels are original deterministic test data.
