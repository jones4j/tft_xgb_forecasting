from datetime import datetime

from pydantic import BaseModel, Field


class EnergyRecord(BaseModel):
    timestamp: datetime
    site_id: str
    demand_kwh: float = Field(ge=0.0)
    temperature_c: float | None = None
    solar_kw: float | None = None
    battery_soc: float | None = None
    tariff_code: str | None = None
    is_holiday: bool = False


CANONICAL_COLUMNS = tuple(EnergyRecord.model_fields.keys())

