import pandas as pd


def build_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["hour"] = enriched["timestamp"].dt.hour
    enriched["day_of_week"] = enriched["timestamp"].dt.dayofweek
    enriched["month"] = enriched["timestamp"].dt.month
    enriched["is_weekend"] = enriched["day_of_week"] >= 5
    return enriched

