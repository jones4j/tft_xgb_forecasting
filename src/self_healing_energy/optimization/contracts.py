from pydantic import BaseModel


class AssetDecision(BaseModel):
    site_id: str
    action: str
    estimated_value_usd: float
    rationale: str

