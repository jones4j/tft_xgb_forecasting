import pandas as pd

from self_healing_energy.anomalies.contracts import AnomalyEvent
from self_healing_energy.anomalies.repair import repair_with_audit


def test_repair_with_audit_returns_records() -> None:
    frame = pd.DataFrame(
        [
            {"site_id": "alpha", "timestamp": pd.Timestamp("2025-01-01 00:00:00"), "demand_kwh": 100.0},
            {"site_id": "alpha", "timestamp": pd.Timestamp("2025-01-02 00:00:00"), "demand_kwh": 10.0},
        ]
    )
    anomalies = [
        AnomalyEvent(
            site_id="alpha",
            anomaly_type="sustained_drop",
            start_ts="2025-01-02 00:00:00",
            end_ts="2025-01-02 00:00:00",
            magnitude_fraction=0.2,
            confidence=0.9,
            should_repair=True,
        )
    ]

    repaired, audit = repair_with_audit(frame, anomalies)

    assert repaired.loc[1, "demand_kwh"] == 55.0
    assert len(audit) == 1
    assert audit[0].original_demand_kwh == 10.0
    assert audit[0].repaired_demand_kwh == 55.0
