# #1 yolo-training-pipeline

**Status:** scaffold

**Proves:** treino e inferencia de deteccao.

**Benchmark target:** mAP e latency_ms_per_image.

**Stack:** python, pytorch, ultralytics, opencv, docker.

## Next milestone

Implement the smallest Docker-runnable version and produce the first JSON benchmark under enchmarks/results/.

## Run

`ash
docker build -t yolo-training-pipeline .
docker run --rm yolo-training-pipeline
`

## Benchmark

`ash
docker run --rm yolo-training-pipeline benchmark
`

| Metric | Value | Unit |
|---|---:|---|
| mAP e latency_ms_per_image | pending | pending |

## Architecture

Defined in sdd/spec.md before implementation.

## References

See REFERENCES.md.