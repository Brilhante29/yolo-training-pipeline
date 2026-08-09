from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["project"] == "yolo-training-pipeline"
    assert summary["metric"] == "map50_95_median"
    assert summary["repeat"] == 3
    assert summary["measured_iterations"] == 3
    assert len(summary["samples"]) == 3
    assert summary["value"] == summary["metrics"]["map50_95_median"]
    assert summary["failures"] == 0
    assert summary["proof"]["dataset_sha256_identical"] is True
    assert summary["proof"]["model_architecture_identical"] is True
    assert summary["proof"]["all_checkpoints_reloaded"] is True
    assert len(summary["results"]) == 3
    for item in summary["results"]:
        path = RESULTS / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{summary['value']:.6f}" in readme
    assert f"{summary['metrics']['inference_latency_ms_p95_median']:.3f}" in readme
    print("benchmark_contract=passed")


if __name__ == "__main__":
    main()
