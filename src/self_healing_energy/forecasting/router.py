from self_healing_energy.config.models import SiteRoutingProfile


class ForecastRouter:
    """Routes sites to heavyweight or lightweight model tiers."""

    def __init__(self, impact_threshold: float) -> None:
        self.impact_threshold = impact_threshold

    def choose_tier(self, profile: SiteRoutingProfile) -> str:
        if profile.preferred_model_tier != "auto":
            return profile.preferred_model_tier
        if profile.business_impact_score >= self.impact_threshold or profile.volatility_score >= 0.8:
            return "heavyweight"
        return "lightweight"

