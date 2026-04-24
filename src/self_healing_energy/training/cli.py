from __future__ import annotations

import argparse
from pathlib import Path

from self_healing_energy.training.trainer import ForecastTrainingService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate forecasting models on parquet data.")
    parser.add_argument("dataset", help="Parquet dataset path relative to DATA_ROOT.")
    parser.add_argument(
        "--test-horizon-hours",
        type=int,
        default=24,
        help="Number of final hourly observations reserved for holdout scoring.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Optional output directory for the training summary artifact.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = ForecastTrainingService.build_default()
    summary = service.train_and_evaluate(
        dataset=args.dataset,
        test_horizon_hours=args.test_horizon_hours,
        artifact_dir=args.artifact_dir,
    )
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

