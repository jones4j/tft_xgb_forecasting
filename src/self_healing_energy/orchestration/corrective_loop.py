from self_healing_energy.config.models import RetrainDecision


def decide_retraining(
    drift_scores: dict[str, float],
    anomaly_counts: dict[str, int],
    impact_scores: dict[str, float],
) -> list[RetrainDecision]:
    decisions: list[RetrainDecision] = []
    site_ids = set(drift_scores) | set(anomaly_counts) | set(impact_scores)

    for site_id in sorted(site_ids):
        drift = drift_scores.get(site_id, 0.0)
        anomalies = anomaly_counts.get(site_id, 0)
        impact = impact_scores.get(site_id, 0.0)
        should_retrain = drift >= 0.15 and anomalies >= 1 and impact >= 0.5
        reason = (
            "drift and anomaly thresholds exceeded for a material site"
            if should_retrain
            else "monitor only"
        )
        priority = "high" if impact >= 0.85 else "normal"
        decisions.append(
            RetrainDecision(
                site_id=site_id,
                should_retrain=should_retrain,
                reason=reason,
                priority=priority,
            )
        )
    return decisions

