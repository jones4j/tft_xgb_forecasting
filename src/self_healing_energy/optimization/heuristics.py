from self_healing_energy.forecasting.contracts import ForecastPoint
from self_healing_energy.optimization.contracts import AssetDecision


def recommend_assets(forecasts: list[ForecastPoint]) -> list[AssetDecision]:
    decisions: list[AssetDecision] = []
    totals: dict[str, float] = {}
    for forecast in forecasts:
        totals[forecast.site_id] = totals.get(forecast.site_id, 0.0) + forecast.prediction_kwh

    for site_id, demand_total in totals.items():
        if demand_total > 500:
            decisions.append(
                AssetDecision(
                    site_id=site_id,
                    action="evaluate_battery_expansion",
                    estimated_value_usd=12000.0,
                    rationale="High forecasted load makes storage arbitrage and peak shaving attractive.",
                )
            )
        else:
            decisions.append(
                AssetDecision(
                    site_id=site_id,
                    action="maintain_current_assets",
                    estimated_value_usd=1500.0,
                    rationale="Load profile does not justify larger capital allocation in the current horizon.",
                )
            )
    return decisions

