# Reuse Map: #1 yolo-training-pipeline

| Kit input or delta | Use | Resolution |
|---|---|---|
| agent graph and architecture matrix | select pipeline and execute gates | reused |
| Python computer-vision profile | deterministic Docker, tests and benchmark conventions | reused and improved |
| vision model artifact contract | publish checkpoint bytes, SHA-256, classes and input metadata | reusable producer contract |
| multi-run aggregate contract | bind three comparable runs and their hashes | reusable publication pattern |
| synthetic fixture and training choices | prove this repository's pipeline only | keep project-local |

The reusable surface is a validated artifact/evidence contract. Model selection, fixture geometry and training hyperparameters are not shared abstractions.
