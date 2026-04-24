import pandas as pd


def validate_hourly_grain(frame: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if frame.empty:
        issues.append("input frame is empty")
        return issues

    if frame["timestamp"].isna().any():
        issues.append("timestamp contains null values")

    if frame["site_id"].isna().any():
        issues.append("site_id contains null values")

    duplicated = frame.duplicated(subset=["site_id", "timestamp"]).sum()
    if duplicated:
        issues.append(f"found {duplicated} duplicated site_id/timestamp records")

    return issues

