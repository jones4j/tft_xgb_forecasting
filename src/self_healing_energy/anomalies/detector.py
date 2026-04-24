import pandas as pd

from self_healing_energy.anomalies.contracts import AnomalyEvent


class ThresholdAnomalyDetector:
    """Flags major sustained drops as outage-like anomalies."""

    def __init__(self, min_duration_hours: int, min_drop_fraction: float) -> None:
        self.min_duration_hours = min_duration_hours
        self.min_drop_fraction = min_drop_fraction

    def detect(self, frame: pd.DataFrame) -> list[AnomalyEvent]:
        if frame.empty:
            return []

        events: list[AnomalyEvent] = []
        grouped = frame.groupby("site_id", sort=False)
        for site_id, group in grouped:
            baseline = group["demand_kwh"].median()
            threshold = baseline * (1.0 - self.min_drop_fraction)
            impacted = group[group["demand_kwh"] <= threshold]
            if len(impacted) < self.min_duration_hours:
                continue

            events.append(
                AnomalyEvent(
                    site_id=site_id,
                    anomaly_type="sustained_drop",
                    start_ts=str(impacted["timestamp"].min()),
                    end_ts=str(impacted["timestamp"].max()),
                    magnitude_fraction=self.min_drop_fraction,
                    confidence=0.8,
                    should_repair=True,
                )
            )
        return events

