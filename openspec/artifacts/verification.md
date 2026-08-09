# Verification: yolo-training-pipeline

| Gate | Evidence |
|---|---|
| Dataset and annotation contracts | deterministic 80/20 split, disjoint content hashes, validated YOLO labels |
| Training integration | pinned CPU Docker trains YOLO26n from architecture without runtime downloads |
| Tests and lint | Ruff and 35 tests pass in the project image |
| Benchmark | three complete runs, median mAP50-95 `0.002420`, warmed p95 `68.303 ms/image` |
| Artifact | reloaded `best.pt`, 5,333,317 bytes, manifest plus SHA-256 contract |
| Publication | V2 binds source `a7ca83e`, image `sha256:926babbaf404...`, config, lock and raw-run fixture |
| Source CI | exact source head passed GitHub Actions run `31341449701` |

The benchmark proves pipeline mechanics on a synthetic fixture. It does not claim traffic, medical, industrial or natural-image accuracy.
