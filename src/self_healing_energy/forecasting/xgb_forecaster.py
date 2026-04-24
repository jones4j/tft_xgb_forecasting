from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from self_healing_energy.forecasting.base import ForecastModel
from self_healing_energy.forecasting.contracts import ForecastPoint
from self_healing_energy.forecasting.xgb_common import (
    MODEL_FEATURES,
    add_tabular_forecast_features,
    build_category_mapping,
    build_training_matrix,
)


class XGBoostDemandForecaster(ForecastModel):
    """Gradient-boosted hourly demand forecaster with recursive rollout."""

    def __init__(self) -> None:
        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=7,
        )

    def fit(self, frame: pd.DataFrame) -> None:
        self.site_mapping = build_category_mapping(frame["site_id"].astype(str))
        self.tariff_mapping = build_category_mapping(frame["tariff_code"].fillna("unknown").astype(str))
        design, target, featured = build_training_matrix(frame, self.site_mapping, self.tariff_mapping)
        self.model.fit(design, target)

        train_predictions = self.model.predict(design)
        absolute_error = np.abs(train_predictions - target.to_numpy())
        self.interval_margin = float(np.quantile(absolute_error, 0.9)) if len(absolute_error) else 0.0
        self.training_history = featured.copy()
        self._site_profiles = self._build_site_profiles(featured)

    def predict(self, frame: pd.DataFrame, horizon_hours: int) -> list[ForecastPoint]:
        forecasts: list[ForecastPoint] = []
        for site_id, site_frame in frame.groupby("site_id", sort=False):
            history = site_frame.sort_values("timestamp").copy()
            if history.empty:
                continue

            last_timestamp = history["timestamp"].max()
            for step in range(1, horizon_hours + 1):
                future_timestamp = last_timestamp + timedelta(hours=step)
                candidate = self._build_future_row(site_id, history, future_timestamp)
                projected = pd.concat([history, pd.DataFrame([candidate])], ignore_index=True)
                featured = add_tabular_forecast_features(projected, self.site_mapping, self.tariff_mapping)
                prediction = float(self.model.predict(featured.loc[[featured.index[-1]], MODEL_FEATURES].astype(float))[0])
                candidate["demand_kwh"] = max(prediction, 0.0)
                history = pd.concat([history, pd.DataFrame([candidate])], ignore_index=True)
                forecasts.append(
                    ForecastPoint(
                        site_id=site_id,
                        timestamp=str(future_timestamp),
                        prediction_kwh=candidate["demand_kwh"],
                        lower_kwh=max(candidate["demand_kwh"] - self.interval_margin, 0.0),
                        upper_kwh=candidate["demand_kwh"] + self.interval_margin,
                    )
                )
        return forecasts

    def backtest(self, frame: pd.DataFrame) -> list[ForecastPoint]:
        combined = (
            pd.concat([self.training_history, frame], ignore_index=True)
            .drop_duplicates(subset=["site_id", "timestamp"], keep="last")
        )
        featured = add_tabular_forecast_features(combined, self.site_mapping, self.tariff_mapping)
        test_keys = frame.loc[:, ["site_id", "timestamp"]].copy()
        scored = featured.merge(test_keys, on=["site_id", "timestamp"], how="inner")
        predictions = self.model.predict(scored.loc[:, MODEL_FEATURES].astype(float))

        forecasts: list[ForecastPoint] = []
        for row, prediction in zip(scored.itertuples(index=False), predictions, strict=False):
            value = max(float(prediction), 0.0)
            forecasts.append(
                ForecastPoint(
                    site_id=row.site_id,
                    timestamp=str(row.timestamp),
                    prediction_kwh=value,
                    lower_kwh=max(value - self.interval_margin, 0.0),
                    upper_kwh=value + self.interval_margin,
                )
            )
        return forecasts

    def _build_site_profiles(self, frame: pd.DataFrame) -> dict[str, dict[str, object]]:
        profiles: dict[str, dict[str, object]] = {}
        for site_id, site_frame in frame.groupby("site_id", sort=False):
            site_profile = (
                site_frame.assign(hour=site_frame["timestamp"].dt.hour)
                .groupby("hour", as_index=True)[["temperature_c", "solar_kw", "battery_soc"]]
                .median()
            )
            profiles[site_id] = {
                "hourly_medians": site_profile,
                "tariff_code": str(site_frame["tariff_code"].fillna("unknown").mode().iloc[0]),
            }
        return profiles

    def _build_future_row(
        self,
        site_id: str,
        history: pd.DataFrame,
        future_timestamp: pd.Timestamp,
    ) -> dict[str, object]:
        profile = self._site_profiles.get(site_id, {})
        hourly_medians = profile.get("hourly_medians")
        default_temperature = float(history["temperature_c"].median()) if "temperature_c" in history else 20.0
        default_solar = float(history["solar_kw"].median()) if "solar_kw" in history else 0.0
        default_battery = float(history["battery_soc"].median()) if "battery_soc" in history else 0.5

        if hourly_medians is not None and future_timestamp.hour in hourly_medians.index:
            hourly_row = hourly_medians.loc[future_timestamp.hour]
            temperature_c = float(hourly_row["temperature_c"])
            solar_kw = float(hourly_row["solar_kw"])
            battery_soc = float(hourly_row["battery_soc"])
        else:
            temperature_c = default_temperature
            solar_kw = default_solar
            battery_soc = default_battery

        return {
            "timestamp": future_timestamp,
            "site_id": site_id,
            "demand_kwh": float(history["demand_kwh"].iloc[-1]),
            "temperature_c": temperature_c,
            "solar_kw": max(solar_kw, 0.0),
            "battery_soc": float(np.clip(battery_soc, 0.0, 1.0)),
            "tariff_code": str(profile.get("tariff_code", "unknown")),
            "is_holiday": bool(future_timestamp.dayofweek >= 5),
            "is_weekend": bool(future_timestamp.dayofweek >= 5),
            "hour": future_timestamp.hour,
            "day_of_week": future_timestamp.dayofweek,
            "month": future_timestamp.month,
        }
