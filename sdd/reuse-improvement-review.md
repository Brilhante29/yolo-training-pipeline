# Reuse Improvement Review

Project: `1 - yolo-training-pipeline`

## Review Points

- [x] After scaffold: selected `applied-computer-vision` and pipeline architecture.
- [x] After research: recorded framework, runtime, dataset, benchmark, and license forces.
- [x] Before implementation: patched a reusable computer-vision skill and standard into the kit.
- [x] During specification: rejected serving, registry, export, GPU, cloud, broker, and notebook scope.
- [ ] After first image: confirm dependency pins, packaged architecture, and offline behavior.
- [ ] After benchmark: patch only repeated CV evidence gaps into the kit.
- [ ] Before publication: synchronize the final kit commit and validate license packaging.

## Findings

| Finding | Classification | Kit area | Action | Status |
|---|---|---|---|---|
| The Python profile lacked dataset, annotation, held-out metric, and warmed vision latency rules. | `patch_now` | language profile / skill | Added `python-computer-vision` for Codex and Claude. | patched and kit-validated |
| Applied Computer Vision selected only the FastAPI skill even for training repositories. | `patch_now` | component pack | Added the new vision skill without forcing FastAPI into CLI training proofs. | patched and kit-validated |
| Framework, weight, image, and annotation licenses must be architecture inputs. | `patch_now` | decision brain / docs | Added AGPL/Enterprise decision and dataset/license gates. | patched and kit-validated |
| Generic benchmark rules did not distinguish training time from warmed per-image inference. | `patch_now` | CV standard / skill | Added reload, warmup, batch/image size, median/p95, and framework stage timing rules. | patched and kit-validated |
| Synthetic fixture mAP can be misread as domain quality. | `patch_now` | CV standard / README | Require explicit engineering-only limitation next to public metrics. | patched and kit-validated |
| The kit README had a broken Node TypeScript skill row from a newline conversion. | `patch_now` | kit README / validator | Repaired the row and added a validation pattern. | patched and kit-validated |
| Vision training and serving lacked a stable cross-repository artifact boundary. | `patch_now` | contracts / installer / CV skill | Added schema version 1 for checkpoint identity, input, classes, provenance, and metrics; the producer validates the emitted bundle. | patched and kit-validated |
| Ultralytics-specific training code should move into the kit now. | `reject` | templates | Keep framework composition local until another vision repository proves stable duplication. | rejected |
| Every vision project should use Ultralytics. | `reject` | library selection | TorchVision remains the permissive generic default; Ultralytics is selected when YOLO is the claim. | rejected |
| Every vision project should expose FastAPI. | `reject` | component pack | Serving belongs only where throughput or API behavior is measured. | rejected |

## Final Gate

- [x] Reusable improvements were patched or recorded.
- [x] Project-specific implementation was not moved into the kit.
- [x] Validation reflects runtime dependency, license, offline, held-out metric, model-bundle integrity, and warmed latency evidence.
