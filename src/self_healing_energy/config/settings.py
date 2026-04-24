from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    app_env: str = "local"
    log_level: str = "INFO"
    data_root: Path = Path("./data")
    artifact_root: Path = Path("./artifacts")
    default_timezone: str = "UTC"
    forecast_horizon_hours: int = 24
    repair_min_duration_hours: int = 48
    repair_min_drop_fraction: float = 0.20
    high_impact_site_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = AppSettings()

