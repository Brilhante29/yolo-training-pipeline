# #1 yolo-training-pipeline: map50_95_median = 0.002420

One local CPU Docker command generates a deterministic detection fixture, validates YOLO annotations, trains YOLO26 from architecture without downloading weights, evaluates held-out mAP, emits a verifiable model bundle, reloads the best checkpoint, and measures warmed inference latency.

This repository belongs to the Applied Computer Vision and Medical AI program. Its job is narrow: prove the measurable claim through the selected component pack before adding unrelated infrastructure or features.

The benchmark is the proof. `map50_95_median = 0.002420` and warmed p95 latency is `68.303 ms/image`. The result is stored in `benchmarks/results/summary.json` and can be reproduced from the Docker path.

The important architecture decision is pipeline. The dominant force is an ordered transition from fixture and annotations to validated split, trained checkpoint, held-out metrics, reloaded inference, and benchmark evidence.

The default path stays local-first. The project uses python-ml, exposes cli, uses messaging mode `none`, and stores data with `none`. The dependency rule is explicit: The Ultralytics adapter consumes validated local artifacts and emits checkpoint/metric records; pure dataset and benchmark policy never import the training framework.

The rejected work matters as much as the implemented work. Anything that does not improve the benchmark stays out of the first version.

Post angle: start with the number, show the architecture boundary, then explain which future adapter can be added without changing the core use cases.
