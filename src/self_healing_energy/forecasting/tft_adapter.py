from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from self_healing_energy.forecasting.base import ForecastModel
from self_healing_energy.forecasting.contracts import ForecastPoint

try:
    from lightning.pytorch import Trainer
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
except ImportError:  # pragma: no cover - exercised only when optional deps are missing
    Trainer = None
    TemporalFusionTransformer = None
    TimeSeriesDataSet = None
    GroupNormalizer = None
    QuantileLoss = None


@dataclass(frozen=True)
class TFTTrainingConfig:
    max_encoder_length: int = 48
    min_encoder_length: int = 24
    max_epochs: int = 3
    learning_rate: float = 0.03
    hidden_size: int = 8
    attention_head_size: int = 1
    dropout: float = 0.1
    hidden_continuous_size: int = 4
    batch_size: int = 64


class TFTForecastAdapter(ForecastModel):
    """PyTorch Forecasting Temporal Fusion Transformer adapter."""

    quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

    def __init__(self, config: TFTTrainingConfig | None = None) -> None:
        self.config = config or TFTTrainingConfig()

    def fit(self, frame: pd.DataFrame) -> None:
        self._require_tft_dependencies()
        prepared = self._prepare_frame(frame)
        min_history = prepared.groupby("site_id").size().min()
        if min_history is None or min_history < 30:
            raise ValueError("TFT requires at least 30 hourly observations per site for this configuration.")

        max_encoder_length = min(self.config.max_encoder_length, max(int(min_history) - 2, 12))
        min_encoder_length = min(self.config.min_encoder_length, max_encoder_length)

        self.training_frame = prepared.copy()
        self.training_start_timestamp = prepared["timestamp"].min()
        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.site_profiles = self._build_site_profiles(prepared)
        self.training_dataset = TimeSeriesDataSet(
            prepared,
            time_idx="time_idx",
            target="demand_kwh",
            group_ids=["site_id"],
            min_encoder_length=min_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=1,
            static_categoricals=["site_id", "tariff_code"],
            time_varying_known_reals=[
                "time_idx",
                "hour",
                "day_of_week",
                "month",
                "is_holiday",
                "is_weekend",
                "temperature_c",
                "solar_kw",
                "battery_soc",
            ],
            time_varying_unknown_reals=["demand_kwh"],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            target_normalizer=GroupNormalizer(groups=["site_id"]),
        )
        train_loader = self.training_dataset.to_dataloader(
            train=True,
            batch_size=self.config.batch_size,
            num_workers=0,
        )
        val_loader = self.training_dataset.to_dataloader(
            train=False,
            batch_size=self.config.batch_size,
            num_workers=0,
        )
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_continuous_size,
            loss=QuantileLoss(),
            output_size=len(self.quantiles),
            log_interval=-1,
            reduce_on_plateau_patience=2,
        )
        trainer = Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=0.1,
            log_every_n_steps=1,
        )
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def predict(self, frame: pd.DataFrame, horizon_hours: int) -> list[ForecastPoint]:
        self._check_is_fit()
        prepared = self._prepare_frame(frame)
        forecasts: list[ForecastPoint] = []

        for site_id, site_frame in prepared.groupby("site_id", sort=False):
            history = site_frame.sort_values("timestamp").copy()
            if history.empty:
                continue

            for step in range(1, horizon_hours + 1):
                next_timestamp = history["timestamp"].max() + pd.Timedelta(hours=1)
                candidate = self._build_future_row(site_id, history, next_timestamp)
                quantiles = self._predict_single_row(pd.concat([history, pd.DataFrame([candidate])], ignore_index=True))
                median = quantiles[3]
                history = pd.concat(
                    [
                        history,
                        pd.DataFrame(
                            [
                                {
                                    **candidate,
                                    "demand_kwh": median,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
                forecasts.append(
                    ForecastPoint(
                        site_id=site_id,
                        timestamp=str(next_timestamp),
                        prediction_kwh=median,
                        lower_kwh=max(quantiles[1], 0.0),
                        upper_kwh=max(quantiles[5], quantiles[1]),
                    )
                )

        return forecasts

    def backtest(self, frame: pd.DataFrame) -> list[ForecastPoint]:
        self._check_is_fit()
        prepared = self._prepare_frame(frame)
        forecasts: list[ForecastPoint] = []

        for site_id, site_frame in prepared.groupby("site_id", sort=False):
            history = self.training_frame[self.training_frame["site_id"] == site_id].sort_values("timestamp").copy()
            if history.empty:
                continue

            for row in site_frame.sort_values("timestamp").itertuples(index=False):
                candidate = {
                    "timestamp": row.timestamp,
                    "site_id": row.site_id,
                    "demand_kwh": float(history["demand_kwh"].iloc[-1]),
                    "temperature_c": float(row.temperature_c),
                    "solar_kw": float(row.solar_kw),
                    "battery_soc": float(row.battery_soc),
                    "tariff_code": str(row.tariff_code),
                    "is_holiday": bool(row.is_holiday),
                    "is_weekend": bool(row.is_weekend),
                    "hour": int(row.hour),
                    "day_of_week": int(row.day_of_week),
                    "month": int(row.month),
                    "time_idx": int(row.time_idx),
                }
                quantiles = self._predict_single_row(pd.concat([history, pd.DataFrame([candidate])], ignore_index=True))
                forecasts.append(
                    ForecastPoint(
                        site_id=row.site_id,
                        timestamp=str(row.timestamp),
                        prediction_kwh=quantiles[3],
                        lower_kwh=max(quantiles[1], 0.0),
                        upper_kwh=max(quantiles[5], quantiles[1]),
                    )
                )
                history = pd.concat(
                    [
                        history,
                        pd.DataFrame(
                            [
                                {
                                    **candidate,
                                    "demand_kwh": float(row.demand_kwh),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

        return forecasts

    def _predict_single_row(self, frame: pd.DataFrame) -> list[float]:
        prediction_frame = frame.sort_values(["site_id", "timestamp"]).copy()
        dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            prediction_frame,
            predict=True,
            stop_randomization=True,
        )
        prediction = self.model.predict(
            dataset,
            mode="quantiles",
            return_index=False,
            trainer_kwargs={
                "accelerator": "cpu",
                "devices": 1,
                "logger": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        )
        values = prediction.output if hasattr(prediction, "output") else prediction
        return [float(item) for item in values[0, 0, :].tolist()]

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.sort_values(["site_id", "timestamp"]).copy()
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True).dt.tz_localize(None)
        prepared["hour"] = prepared["timestamp"].dt.hour if "hour" not in prepared.columns else prepared["hour"]
        prepared["day_of_week"] = (
            prepared["timestamp"].dt.dayofweek
            if "day_of_week" not in prepared.columns
            else prepared["day_of_week"]
        )
        prepared["month"] = prepared["timestamp"].dt.month if "month" not in prepared.columns else prepared["month"]
        prepared["is_holiday"] = prepared["is_holiday"].fillna(False).astype(int)
        prepared["is_weekend"] = (
            prepared["day_of_week"].isin([5, 6])
            if "is_weekend" not in prepared.columns
            else prepared["is_weekend"]
        )
        prepared["is_weekend"] = prepared["is_weekend"].fillna(False).astype(int)
        prepared["temperature_c"] = prepared["temperature_c"].fillna(prepared["temperature_c"].median())
        prepared["solar_kw"] = prepared["solar_kw"].fillna(0.0)
        prepared["battery_soc"] = prepared["battery_soc"].fillna(0.5)
        prepared["tariff_code"] = prepared["tariff_code"].fillna("unknown").astype(str)
        if "time_idx" in prepared.columns:
            prepared["time_idx"] = prepared["time_idx"].astype(int)
        else:
            start_timestamp = getattr(self, "training_start_timestamp", prepared["timestamp"].min())
            prepared["time_idx"] = (
                (prepared["timestamp"] - start_timestamp).dt.total_seconds() // 3600
            ).astype(int)
        return prepared

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
                "tariff_code": str(site_frame["tariff_code"].mode().iloc[0]),
            }
        return profiles

    def _build_future_row(
        self,
        site_id: str,
        history: pd.DataFrame,
        future_timestamp: pd.Timestamp,
    ) -> dict[str, object]:
        profile = self.site_profiles.get(site_id, {})
        hourly_medians = profile.get("hourly_medians")
        default_temperature = float(history["temperature_c"].median())
        default_solar = float(history["solar_kw"].median())
        default_battery = float(history["battery_soc"].median())

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
            "is_holiday": int(future_timestamp.dayofweek >= 5),
            "is_weekend": int(future_timestamp.dayofweek >= 5),
            "hour": int(future_timestamp.hour),
            "day_of_week": int(future_timestamp.dayofweek),
            "month": int(future_timestamp.month),
            "time_idx": int(history["time_idx"].max()) + 1,
        }

    def _check_is_fit(self) -> None:
        if not hasattr(self, "model"):
            raise ValueError("TFT model has not been fit yet.")

    def _require_tft_dependencies(self) -> None:
        if any(item is None for item in (Trainer, TemporalFusionTransformer, TimeSeriesDataSet, GroupNormalizer, QuantileLoss)):
            raise ImportError(
                "TFT dependencies are not installed. Install the project with the 'tft' extra."
            )
