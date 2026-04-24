from __future__ import annotations

from pathlib import Path

from self_healing_energy.training.contracts import TrainingRunSummary
from self_healing_energy.training.trainer import ForecastTrainingService


class ForecastTestingService:
    """
    Dedicated backtesting surface.

    This currently reuses the same routed training stack and scores the final
    holdout window, which keeps the workflow explicit for experimentation and CI.
    """

    def __init__(self, trainer: ForecastTrainingService) -> None:
        self.trainer = trainer

    @classmethod
    def build_default(cls) -> "ForecastTestingService":
        return cls(ForecastTrainingService.build_default())

    def backtest(
        self,
        dataset: str,
        test_horizon_hours: int,
        artifact_dir: Path | None = None,
    ) -> TrainingRunSummary:
        return self.trainer.train_and_evaluate(
            dataset=dataset,
            test_horizon_hours=test_horizon_hours,
            artifact_dir=artifact_dir,
        )

