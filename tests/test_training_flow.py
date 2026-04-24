from pathlib import Path

import pandas as pd

from self_healing_energy.anomalies.detector import ThresholdAnomalyDetector
from self_healing_energy.data.splits import split_by_time
from self_healing_energy.forecasting.lightweight_model import LightweightBaselineForecaster
from self_healing_energy.forecasting.metrics import score_forecasts
from self_healing_energy.forecasting.router import ForecastRouter
from self_healing_energy.forecasting.tft_adapter import TFTForecastAdapter
from self_healing_energy.forecasting.xgb_explainer import XGBoostShadowExplainer
from self_healing_energy.forecasting.xgb_forecaster import XGBoostDemandForecaster
from self_healing_energy.training.trainer import ForecastTrainingService


class InMemoryLoader:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def load(self, dataset: str) -> pd.DataFrame:
        return self.frame.copy()


def make_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=96, freq="h")
    rows = []
    for site_id, base in (("alpha", 100.0), ("beta", 220.0)):
        for idx, ts in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": ts,
                    "site_id": site_id,
                    "demand_kwh": base + (idx % 24) * 2.0,
                    "temperature_c": 20.0 + (idx % 10),
                    "solar_kw": max(0.0, 15.0 - abs(12 - ts.hour)),
                    "battery_soc": 0.5,
                    "tariff_code": "A",
                    "is_holiday": False,
                }
            )
    return pd.DataFrame(rows)


def test_split_by_time_produces_train_and_test() -> None:
    frame = make_frame()
    split = split_by_time(frame, test_horizon_hours=24)
    assert len(split.train) > 0
    assert len(split.test) > 0
    assert split.train["timestamp"].max() <= split.cutoff_timestamp
    assert split.test["timestamp"].min() > split.cutoff_timestamp


def test_training_service_returns_metrics_and_assignments(tmp_path: Path) -> None:
    frame = make_frame()
    service = ForecastTrainingService(
        loader=InMemoryLoader(frame),
        detector=ThresholdAnomalyDetector(min_duration_hours=48, min_drop_fraction=0.20),
        router=ForecastRouter(impact_threshold=0.85),
        heavyweight_model=TFTForecastAdapter(),
        lightweight_model=LightweightBaselineForecaster(),
        explainer=XGBoostShadowExplainer(),
    )

    summary = service.train_and_evaluate(
        dataset="ignored.parquet",
        test_horizon_hours=24,
        artifact_dir=tmp_path,
    )

    assert summary.train_rows > 0
    assert summary.test_rows > 0
    assert "mae" in summary.metrics
    assert len(summary.site_assignments) == 2
    assert (tmp_path / "training_summary.json").exists()


def test_score_forecasts_returns_expected_metrics() -> None:
    actuals = pd.DataFrame(
        [
            {"site_id": "alpha", "timestamp": pd.Timestamp("2025-01-01 01:00:00"), "demand_kwh": 10.0},
            {"site_id": "alpha", "timestamp": pd.Timestamp("2025-01-01 02:00:00"), "demand_kwh": 20.0},
        ]
    )
    forecasts = [
        type("ForecastPointLike", (), {
            "site_id": "alpha",
            "timestamp": "2025-01-01 01:00:00",
            "prediction_kwh": 12.0,
            "lower_kwh": 9.0,
            "upper_kwh": 13.0,
        })(),
        type("ForecastPointLike", (), {
            "site_id": "alpha",
            "timestamp": "2025-01-01 02:00:00",
            "prediction_kwh": 18.0,
            "lower_kwh": 17.0,
            "upper_kwh": 22.0,
        })(),
    ]

    metrics = score_forecasts(actuals, forecasts)
    assert metrics["rows_scored"] == 2.0
    assert metrics["mae"] == 2.0
    assert metrics["coverage_90"] == 1.0


def test_xgboost_forecaster_and_explainer_fit_real_models() -> None:
    frame = make_frame()
    model = XGBoostDemandForecaster()
    model.fit(frame)
    backtest = model.backtest(frame.tail(24))
    explainer = XGBoostShadowExplainer()
    explainer.fit(frame)
    attributions = explainer.explain(frame.tail(24))

    assert len(backtest) == 24
    assert attributions
    assert any(item.feature == "lag_24" for item in attributions)
