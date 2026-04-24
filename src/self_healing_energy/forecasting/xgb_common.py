from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


MODEL_FEATURES = [
    "site_code",
    "tariff_code_num",
    "hour",
    "day_of_week",
    "month",
    "is_holiday",
    "is_weekend",
    "temperature_c",
    "solar_kw",
    "battery_soc",
    "lag_1",
    "lag_24",
    "rolling_mean_24",
    "rolling_std_24",
]


def build_category_mapping(values: Iterable[str]) -> dict[str, int]:
    unique = sorted({str(value) for value in values})
    return {value: idx for idx, value in enumerate(unique)}


def add_tabular_forecast_features(
    frame: pd.DataFrame,
    site_mapping: dict[str, int],
    tariff_mapping: dict[str, int],
) -> pd.DataFrame:
    enriched = frame.sort_values(["site_id", "timestamp"]).copy()
    enriched["hour"] = enriched["timestamp"].dt.hour if "hour" not in enriched.columns else enriched["hour"]
    enriched["day_of_week"] = (
        enriched["timestamp"].dt.dayofweek
        if "day_of_week" not in enriched.columns
        else enriched["day_of_week"]
    )
    enriched["month"] = enriched["timestamp"].dt.month if "month" not in enriched.columns else enriched["month"]
    enriched["is_weekend"] = (
        enriched["day_of_week"].isin([5, 6])
        if "is_weekend" not in enriched.columns
        else enriched["is_weekend"]
    )
    enriched["is_holiday"] = enriched["is_holiday"].fillna(False).astype(int)
    enriched["is_weekend"] = enriched["is_weekend"].fillna(False).astype(int)
    enriched["temperature_c"] = enriched["temperature_c"].fillna(enriched["temperature_c"].median())
    enriched["solar_kw"] = enriched["solar_kw"].fillna(0.0)
    enriched["battery_soc"] = enriched["battery_soc"].fillna(0.5)
    enriched["site_code"] = enriched["site_id"].map(site_mapping).fillna(-1).astype(int)
    enriched["tariff_code_num"] = (
        enriched["tariff_code"].fillna("unknown").astype(str).map(tariff_mapping).fillna(-1).astype(int)
    )

    grouped = enriched.groupby("site_id", sort=False)
    enriched["lag_1"] = grouped["demand_kwh"].shift(1)
    enriched["lag_24"] = grouped["demand_kwh"].shift(24)
    enriched["rolling_mean_24"] = grouped["demand_kwh"].transform(
        lambda series: series.shift(1).rolling(window=24, min_periods=6).mean()
    )
    enriched["rolling_std_24"] = grouped["demand_kwh"].transform(
        lambda series: series.shift(1).rolling(window=24, min_periods=6).std()
    )

    enriched["lag_1"] = enriched["lag_1"].fillna(grouped["demand_kwh"].transform("median"))
    enriched["lag_24"] = enriched["lag_24"].fillna(enriched["lag_1"])
    enriched["rolling_mean_24"] = enriched["rolling_mean_24"].fillna(enriched["lag_24"])
    enriched["rolling_std_24"] = enriched["rolling_std_24"].fillna(0.0)
    return enriched


def build_training_matrix(
    frame: pd.DataFrame,
    site_mapping: dict[str, int],
    tariff_mapping: dict[str, int],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    featured = add_tabular_forecast_features(frame, site_mapping, tariff_mapping)
    design = featured.loc[:, MODEL_FEATURES].astype(float)
    target = featured["demand_kwh"].astype(float)
    return design, target, featured
