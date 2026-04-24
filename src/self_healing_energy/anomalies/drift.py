import pandas as pd


def mean_shift_score(reference: pd.Series, current: pd.Series) -> float:
    ref_mean = float(reference.mean()) if not reference.empty else 0.0
    cur_mean = float(current.mean()) if not current.empty else 0.0
    denominator = abs(ref_mean) if ref_mean else 1.0
    return abs(cur_mean - ref_mean) / denominator


def detect_site_drift(frame: pd.DataFrame, lookback_hours: int = 168) -> dict[str, float]:
    if frame.empty:
        return {}

    drift_scores: dict[str, float] = {}
    for site_id, group in frame.groupby("site_id", sort=False):
        ordered = group.sort_values("timestamp")
        reference = ordered["demand_kwh"].iloc[:-lookback_hours]
        current = ordered["demand_kwh"].iloc[-lookback_hours:]
        if reference.empty or current.empty:
            drift_scores[site_id] = 0.0
            continue
        drift_scores[site_id] = mean_shift_score(reference, current)
    return drift_scores

