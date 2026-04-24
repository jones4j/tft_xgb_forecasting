from self_healing_energy.config.settings import settings
from self_healing_energy.forecasting.tft_adapter import TFTForecastAdapter
from self_healing_energy.serving.contracts import BatchForecastRequest


def test_settings_defaults_are_sane() -> None:
    assert settings.forecast_horizon_hours == 24
    assert settings.repair_min_duration_hours == 48


def test_request_contract_bounds() -> None:
    request = BatchForecastRequest(dataset="hourly/site_batch.parquet", horizon_hours=48)
    assert request.horizon_hours == 48


def test_forecaster_has_fit_predict_contract() -> None:
    model = TFTForecastAdapter()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
