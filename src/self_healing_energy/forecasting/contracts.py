from pydantic import BaseModel


class ForecastPoint(BaseModel):
    site_id: str
    timestamp: str
    prediction_kwh: float
    lower_kwh: float
    upper_kwh: float


class FeatureAttribution(BaseModel):
    site_id: str
    feature: str
    contribution: float

