from __future__ import annotations

import pandas as pd

from self_healing_energy.anomalies.detector import ThresholdAnomalyDetector
from self_healing_energy.anomalies.repair import repair_with_audit
from self_healing_energy.data.validation import validate_hourly_grain
from self_healing_energy.features.engineering import build_time_features


def prepare_training_frame(
    frame: pd.DataFrame,
    detector: ThresholdAnomalyDetector,
) -> tuple[pd.DataFrame, list, list]:
    issues = validate_hourly_grain(frame)
    if issues:
        raise ValueError(f"Input validation failed: {issues}")

    anomalies = detector.detect(frame)
    repaired, repair_records = repair_with_audit(frame, anomalies)
    featured = build_time_features(repaired)
    return featured, anomalies, repair_records
