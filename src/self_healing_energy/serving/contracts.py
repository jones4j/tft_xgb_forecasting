from pydantic import BaseModel, Field

from self_healing_energy.anomalies.contracts import AnomalyEvent, RepairRecord
from self_healing_energy.config.models import RetrainDecision
from self_healing_energy.forecasting.contracts import FeatureAttribution, ForecastPoint
from self_healing_energy.optimization.contracts import AssetDecision


class BatchForecastRequest(BaseModel):
    dataset: str = Field(description="Parquet file or dataset path relative to DATA_ROOT.")
    horizon_hours: int = Field(default=24, ge=1, le=168)


class BatchForecastResponse(BaseModel):
    dataset: str
    anomalies: list[AnomalyEvent]
    repairs: list[RepairRecord]
    forecasts: list[ForecastPoint]
    attributions: list[FeatureAttribution]
    decisions: list[AssetDecision]
    retrain_decisions: list[RetrainDecision]
