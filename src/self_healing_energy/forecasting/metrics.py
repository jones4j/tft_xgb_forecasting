from __future__ import annotations

import math

import pandas as pd

from self_healing_energy.forecasting.contracts import ForecastPoint


def forecasts_to_frame(forecasts: list[ForecastPoint]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site_id": forecast.site_id,
                "timestamp": pd.Timestamp(forecast.timestamp),
                "prediction_kwh": forecast.prediction_kwh,
                "lower_kwh": forecast.lower_kwh,
                "upper_kwh": forecast.upper_kwh,
            }
            for forecast in forecasts
        ]
    )


def score_forecasts(actuals: pd.DataFrame, forecasts: list[ForecastPoint]) -> dict[str, float]:
    forecast_frame = forecasts_to_frame(forecasts)
    if forecast_frame.empty:
        raise ValueError("No forecasts were generated for scoring.")

    joined = actuals.merge(
        forecast_frame,
        on=["site_id", "timestamp"],
        how="inner",
    )
    if joined.empty:
        raise ValueError("Forecasts did not overlap with the actual holdout timestamps.")

    error = joined["prediction_kwh"] - joined["demand_kwh"]
    abs_error = error.abs()
    denominator = joined["demand_kwh"].abs().replace(0.0, pd.NA)

    return {
        "rows_scored": float(len(joined)),
        "mae": float(abs_error.mean()),
        "rmse": float(math.sqrt((error.pow(2)).mean())),
        "mape": float((abs_error / denominator).dropna().mean()),
        "coverage_90": float(
            (
                (joined["demand_kwh"] >= joined["lower_kwh"])
                & (joined["demand_kwh"] <= joined["upper_kwh"])
            ).mean()
        ),
    }

