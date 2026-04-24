from self_healing_energy.anomalies.drift import detect_site_drift
from self_healing_energy.anomalies.detector import ThresholdAnomalyDetector
from self_healing_energy.anomalies.repair import repair_with_audit
from self_healing_energy.config.models import SiteRoutingProfile
from self_healing_energy.config.settings import settings
from self_healing_energy.data.parquet_loader import ParquetDatasetLoader
from self_healing_energy.data.validation import validate_hourly_grain
from self_healing_energy.features.engineering import build_time_features
from self_healing_energy.forecasting.router import ForecastRouter
from self_healing_energy.forecasting.tft_adapter import TFTForecastAdapter
from self_healing_energy.forecasting.xgb_forecaster import XGBoostDemandForecaster
from self_healing_energy.forecasting.xgb_explainer import XGBoostShadowExplainer
from self_healing_energy.optimization.heuristics import recommend_assets
from self_healing_energy.orchestration.corrective_loop import decide_retraining
from self_healing_energy.serving.contracts import BatchForecastRequest, BatchForecastResponse


class BatchForecastPipeline:
    def __init__(
        self,
        loader: ParquetDatasetLoader,
        detector: ThresholdAnomalyDetector,
        forecaster: TFTForecastAdapter,
        lightweight_forecaster: XGBoostDemandForecaster,
        explainer: XGBoostShadowExplainer,
        router: ForecastRouter,
    ) -> None:
        self.loader = loader
        self.detector = detector
        self.forecaster = forecaster
        self.lightweight_forecaster = lightweight_forecaster
        self.explainer = explainer
        self.router = router

    @classmethod
    def build_default(cls) -> "BatchForecastPipeline":
        return cls(
            loader=ParquetDatasetLoader(settings.data_root),
            detector=ThresholdAnomalyDetector(
                min_duration_hours=settings.repair_min_duration_hours,
                min_drop_fraction=settings.repair_min_drop_fraction,
            ),
            forecaster=TFTForecastAdapter(),
            lightweight_forecaster=XGBoostDemandForecaster(),
            explainer=XGBoostShadowExplainer(),
            router=ForecastRouter(settings.high_impact_site_threshold),
        )

    def run(self, request: BatchForecastRequest) -> BatchForecastResponse:
        frame = self.loader.load(request.dataset)
        issues = validate_hourly_grain(frame)
        if issues:
            raise ValueError(f"Input validation failed: {issues}")

        anomalies = self.detector.detect(frame)
        repaired, repair_records = repair_with_audit(frame, anomalies)
        featured = build_time_features(repaired)
        drift_scores = detect_site_drift(featured)

        self.forecaster.fit(featured)
        self.lightweight_forecaster.fit(featured)
        self.explainer.fit(featured)

        forecasts = []
        impact_scores: dict[str, float] = {}
        site_profiles: list[SiteRoutingProfile] = []
        site_medians = featured.groupby("site_id")["demand_kwh"].median().to_dict()
        max_median = max(site_medians.values()) if site_medians else 1.0
        for site_id, site_median in site_medians.items():
            impact = float(site_median / max_median) if max_median else 0.0
            impact_scores[site_id] = impact
            site_profiles.append(
                SiteRoutingProfile(
                    site_id=site_id,
                    business_impact_score=impact,
                    volatility_score=min(drift_scores.get(site_id, 0.0), 1.0),
                )
            )

        for profile in site_profiles:
            site_frame = featured[featured["site_id"] == profile.site_id]
            tier = self.router.choose_tier(profile)
            model = self.forecaster if tier == "heavyweight" else self.lightweight_forecaster
            forecasts.extend(model.predict(site_frame, request.horizon_hours))

        attributions = self.explainer.explain(featured)
        decisions = recommend_assets(forecasts)
        anomaly_counts: dict[str, int] = {}
        for anomaly in anomalies:
            anomaly_counts[anomaly.site_id] = anomaly_counts.get(anomaly.site_id, 0) + 1
        retrain_decisions = decide_retraining(drift_scores, anomaly_counts, impact_scores)

        return BatchForecastResponse(
            dataset=request.dataset,
            anomalies=anomalies,
            repairs=repair_records,
            forecasts=forecasts,
            attributions=attributions,
            decisions=decisions,
            retrain_decisions=retrain_decisions,
        )
