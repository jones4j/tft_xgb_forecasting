import pandas as pd

from self_healing_energy.anomalies.contracts import AnomalyEvent, RepairRecord


def repair_from_hourly_profile(
    frame: pd.DataFrame,
    anomalies: list[AnomalyEvent],
) -> pd.DataFrame:
    repaired, _ = repair_with_audit(frame, anomalies)
    return repaired


def repair_with_audit(
    frame: pd.DataFrame,
    anomalies: list[AnomalyEvent],
) -> tuple[pd.DataFrame, list[RepairRecord]]:
    if not anomalies:
        return frame, []

    repaired = frame.copy()
    repair_records: list[RepairRecord] = []
    profile = (
        repaired.assign(hour=repaired["timestamp"].dt.hour)
        .groupby(["site_id", "hour"], as_index=False)["demand_kwh"]
        .median()
        .rename(columns={"demand_kwh": "profile_demand_kwh"})
    )

    for event in anomalies:
        if not event.should_repair:
            continue

        mask = (
            (repaired["site_id"] == event.site_id)
            & (repaired["timestamp"] >= pd.Timestamp(event.start_ts))
            & (repaired["timestamp"] <= pd.Timestamp(event.end_ts))
        )
        event_frame = repaired.loc[mask].copy()
        if event_frame.empty:
            continue

        event_frame["hour"] = event_frame["timestamp"].dt.hour
        event_frame = event_frame.merge(profile, on=["site_id", "hour"], how="left")
        for row in event_frame.itertuples(index=False):
            repair_records.append(
                RepairRecord(
                    site_id=row.site_id,
                    timestamp=str(row.timestamp),
                    original_demand_kwh=float(row.demand_kwh),
                    repaired_demand_kwh=float(row.profile_demand_kwh),
                    repair_method="hourly_median_profile",
                    anomaly_type=event.anomaly_type,
                )
            )
        repaired.loc[mask, "demand_kwh"] = event_frame["profile_demand_kwh"].to_numpy()

    return repaired, repair_records
