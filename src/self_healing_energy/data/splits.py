from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff_timestamp: pd.Timestamp


def split_by_time(frame: pd.DataFrame, test_horizon_hours: int) -> TimeSeriesSplit:
    if frame.empty:
        raise ValueError("Cannot split an empty frame.")

    ordered = frame.sort_values(["site_id", "timestamp"]).copy()
    max_timestamp = ordered["timestamp"].max()
    cutoff = max_timestamp - pd.Timedelta(hours=test_horizon_hours)

    train = ordered[ordered["timestamp"] <= cutoff].copy()
    test = ordered[ordered["timestamp"] > cutoff].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Split produced an empty partition. Increase dataset history or reduce the test horizon."
        )

    return TimeSeriesSplit(train=train, test=test, cutoff_timestamp=cutoff)

