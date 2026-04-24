from pathlib import Path

import pandas as pd

from self_healing_energy.data.contracts import CANONICAL_COLUMNS


class ParquetDatasetLoader:
    """Loads parquet files and enforces a minimal canonical schema."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, dataset: str) -> pd.DataFrame:
        dataset_path = self.root / dataset
        frame = pd.read_parquet(dataset_path)
        missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Dataset {dataset_path} is missing required columns: {missing}")
        return frame.loc[:, CANONICAL_COLUMNS].sort_values(["site_id", "timestamp"])

