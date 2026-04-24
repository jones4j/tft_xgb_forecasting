from datetime import timedelta

import pandas as pd

from self_healing_energy.forecasting.base import ForecastModel
from self_healing_energy.forecasting.contracts import ForecastPoint


class LightweightBaselineForecaster(ForecastModel):
    """Cheap baseline used for lower-impact sites."""

    def fit(self, frame: pd.DataFrame) -> None:
        self._hourly_profile = (
            frame.assign(hour=frame["timestamp"].dt.hour)
            .groupby(["site_id", "hour"])["demand_kwh"]
            .mean()
            .to_dict()
        )

    def predict(self, frame: pd.DataFrame, horizon_hours: int) -> list[ForecastPoint]:
        forecasts: list[ForecastPoint] = []
        latest = frame.groupby("site_id")["timestamp"].max().to_dict()
        for site_id, start_ts in latest.items():
            for step in range(1, horizon_hours + 1):
                ts = start_ts + timedelta(hours=step)
                hour = ts.hour
                baseline = float(self._hourly_profile.get((site_id, hour), 0.0))
                forecasts.append(
                    ForecastPoint(
                        site_id=site_id,
                        timestamp=str(ts),
                        prediction_kwh=baseline,
                        lower_kwh=baseline * 0.85,
                        upper_kwh=baseline * 1.15,
                    )
                )
        return forecasts

    def backtest(self, frame: pd.DataFrame) -> list[ForecastPoint]:
        forecasts: list[ForecastPoint] = []
        for row in frame.itertuples(index=False):
            baseline = float(self._hourly_profile.get((row.site_id, row.timestamp.hour), 0.0))
            forecasts.append(
                ForecastPoint(
                    site_id=row.site_id,
                    timestamp=str(row.timestamp),
                    prediction_kwh=baseline,
                    lower_kwh=baseline * 0.85,
                    upper_kwh=baseline * 1.15,
                )
            )
        return forecasts
