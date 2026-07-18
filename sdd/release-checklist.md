# Release Checklist: yolo-training-pipeline

- [x] OpenSpec proposal, design, requirements, and tasks exist.
- [x] Architecture, stack, license, and rejected alternatives are explicit.
- [x] SOLID, LSP, DIP, KISS, YAGNI, DRY, coupling, and testability are concrete.
- [x] Default path is Docker-based, CPU-only, cross-platform, local-first, and secret-free.
- [x] Dataset source, license, class map, size, split, seed, hashes, and limitation are documented.
- [x] Framework, Python base, PyTorch CPU, and top-level tooling versions are pinned.
- [ ] Full AGPL license packaging is confirmed.
- [x] Docker build resolves and imports the pinned dependencies.
- [x] Lint and at least 90% focused pure coverage pass.
- [x] Three full training and inference runs pass on one image.
- [x] Summary JSON validates and README opens with measured mAP and latency.
- [ ] Reuse review is resolved and published kit commit is synchronized.
- [x] Git status contains only intentional source/evidence and no runtime data.
- [ ] Public repository description, topics, default branch, and license are correct.
- [ ] GitHub Actions is green and benchmark artifact is downloadable.
