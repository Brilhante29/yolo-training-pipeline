from __future__ import annotations

import argparse
from pathlib import Path

from yolo_training_pipeline.pipeline import run_pipeline, write_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, validate, reload, and benchmark a local YOLO model."
    )
    parser.add_argument("mode", nargs="?", choices=("run", "aggregate"), default="run")
    parser.add_argument("--input-dir", type=Path, default=Path("/results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/results/summary.json"),
    )
    args = parser.parse_args()

    if args.mode == "aggregate":
        write_summary(args.input_dir, args.output)
        return
    run_pipeline()


if __name__ == "__main__":
    main()
