from pydantic import BaseModel


class AnomalyEvent(BaseModel):
    site_id: str
    anomaly_type: str
    start_ts: str
    end_ts: str
    magnitude_fraction: float
    confidence: float
    should_repair: bool


class RepairRecord(BaseModel):
    site_id: str
    timestamp: str
    original_demand_kwh: float
    repaired_demand_kwh: float
    repair_method: str
    anomaly_type: str
