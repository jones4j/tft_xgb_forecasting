from pydantic import BaseModel, Field


class SiteRoutingProfile(BaseModel):
    site_id: str
    business_impact_score: float = Field(ge=0.0, le=1.0)
    volatility_score: float = Field(ge=0.0, le=1.0, default=0.5)
    preferred_model_tier: str = "auto"


class RetrainDecision(BaseModel):
    site_id: str
    should_retrain: bool
    reason: str
    priority: str

