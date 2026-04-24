from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    num_sites: int = 12
    horizon_days: int = 60
    seed: int = 7
    start: str = "2025-01-01"
    outage_site_count: int = 2
    drift_site_count: int = 2


def generate_synthetic_energy_data(config: SyntheticDatasetConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    timestamps = pd.date_range(config.start, periods=config.horizon_days * 24, freq="h", tz="UTC")
    outage_sites = {f"site_{idx:03d}" for idx in range(config.outage_site_count)}
    drift_sites = {
        f"site_{idx:03d}"
        for idx in range(config.outage_site_count, config.outage_site_count + config.drift_site_count)
    }
    outage_start_idx = min(len(timestamps) // 3, max(len(timestamps) - 1, 0))
    outage_end_idx = min(outage_start_idx + 72, len(timestamps))
    drift_start_idx = min((2 * len(timestamps)) // 3, max(len(timestamps) - 1, 0))
    outage_start = timestamps[outage_start_idx] if len(timestamps) else None
    outage_end = timestamps[outage_end_idx - 1] if outage_end_idx > outage_start_idx else None
    drift_start = timestamps[drift_start_idx] if len(timestamps) else None

    rows: list[dict] = []
    for site_idx in range(config.num_sites):
        site_id = f"site_{site_idx:03d}"
        base_load = 80 + site_idx * 18
        solar_capacity = 20 + site_idx * 2
        tariff_code = "PEAK" if site_idx % 3 == 0 else "STANDARD"

        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.dayofweek
            seasonal = 8 * np.sin(2 * np.pi * hour / 24)
            weekly = -6 if day_of_week >= 5 else 4
            temperature = 15 + 10 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 1.5)
            solar_kw = max(0.0, solar_capacity * np.sin(np.pi * hour / 24) + rng.normal(0, 1.0))
            battery_soc = float(np.clip(0.55 + 0.25 * np.sin(2 * np.pi * hour / 24), 0.1, 0.95))
            demand = base_load + seasonal + weekly + 0.65 * max(temperature - 18, 0) - 0.18 * solar_kw
            demand += rng.normal(0, 2.0)

            if outage_start is not None and outage_end is not None and site_id in outage_sites and outage_start <= ts <= outage_end:
                demand *= 0.08
            if drift_start is not None and site_id in drift_sites and ts >= drift_start:
                demand *= 1.18

            rows.append(
                {
                    "timestamp": ts,
                    "site_id": site_id,
                    "demand_kwh": round(max(demand, 0.0), 3),
                    "temperature_c": round(float(temperature), 3),
                    "solar_kw": round(float(solar_kw), 3),
                    "battery_soc": round(battery_soc, 3),
                    "tariff_code": tariff_code,
                    "is_holiday": bool(day_of_week >= 5),
                }
            )

    return pd.DataFrame(rows)


def write_synthetic_parquet(output_path: Path, config: SyntheticDatasetConfig) -> Path:
    frame = generate_synthetic_energy_data(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path
