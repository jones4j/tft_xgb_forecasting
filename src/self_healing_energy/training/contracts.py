from __future__ import annotations

from pydantic import BaseModel

from self_healing_energy.anomalies.contracts import RepairRecord


class SiteModelAssignment(BaseModel):
    site_id: str
    assigned_tier: str
    business_impact_score: float
    volatility_score: float


class TrainingRunSummary(BaseModel):
    dataset: str
    cutoff_timestamp: str
    train_rows: int
    test_rows: int
    anomaly_count: int
    repair_count: int
    metrics: dict[str, float]
    site_assignments: list[SiteModelAssignment]
    repair_records: list[RepairRecord]
