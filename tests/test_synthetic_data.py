from pathlib import Path

import pandas as pd

from self_healing_energy.data.synthetic import SyntheticDatasetConfig, generate_synthetic_energy_data, write_synthetic_parquet


def test_generate_synthetic_energy_data_shape() -> None:
    config = SyntheticDatasetConfig(num_sites=3, horizon_days=2, seed=11)
    frame = generate_synthetic_energy_data(config)

    assert len(frame) == 3 * 2 * 24
    assert set(["timestamp", "site_id", "demand_kwh"]).issubset(frame.columns)
    assert frame["demand_kwh"].ge(0.0).all()


def test_write_synthetic_parquet_round_trip(tmp_path: Path) -> None:
    config = SyntheticDatasetConfig(num_sites=2, horizon_days=1, seed=13)
    path = write_synthetic_parquet(tmp_path / "demo.parquet", config)
    frame = pd.read_parquet(path)

    assert path.exists()
    assert len(frame) == 48
