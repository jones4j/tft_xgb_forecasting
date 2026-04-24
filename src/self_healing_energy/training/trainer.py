from __future__ import annotations

from pathlib import Path

import pandas as pd

from self_healing_energy.anomalies.drift import detect_site_drift
from self_healing_energy.anomalies.detector import ThresholdAnomalyDetector
from self_healing_energy.config.models import SiteRoutingProfile
from self_healing_energy.config.settings import settings
from self_healing_energy.data.parquet_loader import ParquetDatasetLoader
from self_healing_energy.data.splits import split_by_time
from self_healing_energy.forecasting.metrics import score_forecasts
from self_healing_energy.forecasting.router import ForecastRouter
from self_healing_energy.forecasting.tft_adapter import TFTForecastAdapter
from self_healing_energy.forecasting.xgb_forecaster import XGBoostDemandForecaster
from self_healing_energy.forecasting.xgb_explainer import XGBoostShadowExplainer
from self_healing_energy.training.contracts import SiteModelAssignment, TrainingRunSummary
from self_healing_energy.training.dataset_preparation import prepare_training_frame
from self_healing_energy.utils.serialization import write_json


class ForecastTrainingService:
    def __init__(
        self,
        loader: ParquetDatasetLoader,
        detector: ThresholdAnomalyDetector,
        router: ForecastRouter,
        heavyweight_model: TFTForecastAdapter,
        lightweight_model: XGBoostDemandForecaster,
        explainer: XGBoostShadowExplainer,
    ) -> None:
        self.loader = loader
        self.detector = detector
        self.router = router
        self.heavyweight_model = heavyweight_model
        self.lightweight_model = lightweight_model
        self.explainer = explainer

    @classmethod
    def build_default(cls) -> "ForecastTrainingService":
        return cls(
            loader=ParquetDatasetLoader(settings.data_root),
            detector=ThresholdAnomalyDetector(
                min_duration_hours=settings.repair_min_duration_hours,
                min_drop_fraction=settings.repair_min_drop_fraction,
            ),
            router=ForecastRouter(settings.high_impact_site_threshold),
            heavyweight_model=TFTForecastAdapter(),
            lightweight_model=XGBoostDemandForecaster(),
            explainer=XGBoostShadowExplainer(),
        )

    def train_and_evaluate(
        self,
        dataset: str,
        test_horizon_hours: int,
        artifact_dir: Path | None = None,
    ) -> TrainingRunSummary:
        raw = self.loader.load(dataset)
        prepared, anomalies, repair_records = prepare_training_frame(raw, self.detector)
        split = split_by_time(prepared, test_horizon_hours=test_horizon_hours)

        self.heavyweight_model.fit(split.train)
        self.lightweight_model.fit(split.train)
        self.explainer.fit(split.train)

        drift_scores = detect_site_drift(split.train)
        site_assignments = self._assign_sites(split.train, drift_scores)
        forecasts = self._routed_predict(split.test, site_assignments)
        metrics = score_forecasts(split.test[["site_id", "timestamp", "demand_kwh"]], forecasts)

        summary = TrainingRunSummary(
            dataset=dataset,
            cutoff_timestamp=str(split.cutoff_timestamp),
            train_rows=len(split.train),
            test_rows=len(split.test),
            anomaly_count=len(anomalies),
            repair_count=len(repair_records),
            metrics=metrics,
            site_assignments=site_assignments,
            repair_records=repair_records,
        )

        if artifact_dir is not None:
            self._persist_summary(summary, artifact_dir)

        return summary

    def _assign_sites(
        self,
        train_frame: pd.DataFrame,
        drift_scores: dict[str, float],
    ) -> list[SiteModelAssignment]:
        site_medians = train_frame.groupby("site_id")["demand_kwh"].median().to_dict()
        max_median = max(site_medians.values()) if site_medians else 1.0

        assignments: list[SiteModelAssignment] = []
        for site_id, site_median in site_medians.items():
            impact = float(site_median / max_median) if max_median else 0.0
            profile = SiteRoutingProfile(
                site_id=site_id,
                business_impact_score=impact,
                volatility_score=min(drift_scores.get(site_id, 0.0), 1.0),
            )
            assignments.append(
                SiteModelAssignment(
                    site_id=site_id,
                    assigned_tier=self.router.choose_tier(profile),
                    business_impact_score=profile.business_impact_score,
                    volatility_score=profile.volatility_score,
                )
            )
        return assignments

    def _routed_predict(
        self,
        test_frame: pd.DataFrame,
        site_assignments: list[SiteModelAssignment],
    ) -> list:
        forecasts = []
        for assignment in site_assignments:
            site_frame = test_frame[test_frame["site_id"] == assignment.site_id]
            if site_frame.empty:
                continue
            model = (
                self.heavyweight_model
                if assignment.assigned_tier == "heavyweight"
                else self.lightweight_model
            )
            forecasts.extend(model.backtest(site_frame))
        return forecasts

    def _persist_summary(self, summary: TrainingRunSummary, artifact_dir: Path) -> None:
        target = artifact_dir / "training_summary.json"
        write_json(target, summary.model_dump())
