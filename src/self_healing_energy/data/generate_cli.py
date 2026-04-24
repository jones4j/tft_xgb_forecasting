from __future__ import annotations

import argparse
from pathlib import Path

from self_healing_energy.data.synthetic import SyntheticDatasetConfig, write_synthetic_parquet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic parquet inputs for the energy platform.")
    parser.add_argument("output", type=Path, help="Output parquet path.")
    parser.add_argument("--num-sites", type=int, default=12)
    parser.add_argument("--horizon-days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--outage-site-count", type=int, default=2)
    parser.add_argument("--drift-site-count", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SyntheticDatasetConfig(
        num_sites=args.num_sites,
        horizon_days=args.horizon_days,
        seed=args.seed,
        start=args.start,
        outage_site_count=args.outage_site_count,
        drift_site_count=args.drift_site_count,
    )
    path = write_synthetic_parquet(args.output, config)
    print(path)


if __name__ == "__main__":
    main()
