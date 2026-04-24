from __future__ import annotations

import argparse
from pathlib import Path

from self_healing_energy.training.testing import ForecastTestingService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest forecasting models on a parquet holdout window.")
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
        help="Optional output directory for the backtest summary artifact.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = ForecastTestingService.build_default()
    summary = service.backtest(
        dataset=args.dataset,
        test_horizon_hours=args.test_horizon_hours,
        artifact_dir=args.artifact_dir,
    )
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
