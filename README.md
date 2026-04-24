# Self-Healing Energy Forecasting & Optimization System

This repository is a parquet-native framework for forecasting hourly energy demand across many sites, detecting and repairing data quality failures, retraining models when conditions materially change, and producing decision-ready outputs that remain explainable to non-ML stakeholders.

The project is intentionally not organized as "pick one model and maximize accuracy." Its core idea is that enterprise energy systems are shaped by competing constraints:

- forecasts must be accurate enough to drive asset and operational decisions
- explanations must be concrete enough for business users to trust
- anomaly repair must be strong enough to stabilize models without erasing real events
- automation must be selective enough to be safe under drift, outages, and bad telemetry

The architecture is built around those tensions rather than around a single algorithm.

## System Flow

```text
parquet inputs
  -> schema validation
  -> feature engineering
  -> anomaly detection
  -> drift monitoring
  -> threshold-gated repair
  -> forecasting
       -> site-tier router
       -> heavyweight path: Temporal Fusion Transformer
       -> lower-cost path: XGBoost demand forecaster
       -> shadow explainability path: XGBoost feature contributions
  -> optimization
  -> explainability + reporting
  -> corrective loop for retraining and repair escalation
```

## Why This Is Complex

The complexity in this system is not just "time-series forecasting is hard." The deeper challenge is that the system has to make several different kinds of decisions well at the same time:

- statistical decisions: what signal is predictive, what is noise, what is drift
- operational decisions: when to repair data, when to leave it untouched, when to retrain
- economic decisions: where higher model cost is justified and where it is not
- communication decisions: how to explain model outputs in ways that preserve trust

That is why the modeling layer is split rather than unified.

## Forecasting Strategy

The primary forecasting path is a real Temporal Fusion Transformer implemented through PyTorch Forecasting. That choice is purposeful because the forecasting problem is structurally complex:

- load is strongly temporal, with multiple interacting seasonalities
- weather, tariff regime, holiday effects, solar generation, and battery state all influence demand jointly
- relationships are site-dependent rather than globally uniform
- the same site can behave differently under normal operation, outage recovery, and post-drift conditions

The TFT is used because it can absorb that complexity better than a shallow tabular model. It handles:

- long encoder windows over historical demand
- known future covariates such as calendar structure
- mixed static and time-varying signals
- nonlinear feature interactions that would otherwise need extensive manual engineering
- quantile outputs for uncertainty-aware forecasts instead of only point estimates

In this repository, the TFT is the heavyweight path for higher-impact sites. That is not just a performance optimization. It reflects an architectural decision that the most expensive model should be reserved for the sites where forecast error matters most.

## Why XGBoost Still Exists

If the TFT is better at absorbing sequence complexity, it is also harder to explain in a way that is persuasive to operators, analysts, and business stakeholders. Even when attention weights or internal variable importance are available, that does not necessarily produce explanations people can act on with confidence.

That is why XGBoost is present in two different roles.

First, it serves as the lower-cost forecasting path for less critical sites. In that role it is a practical engineering choice:

- fast to train
- cheap to score
- effective on lagged, engineered time-series features
- easier to scale broadly across long-tail site populations

Second, and more importantly, it acts as the shadow explainability model. This is a deliberate separation of concerns:

- the TFT is optimized for forecast quality
- the XGBoost shadow model is optimized for explanation quality

That means the system does not force one model to satisfy two conflicting goals equally well. Instead:

- the deep model can remain specialized for temporal prediction
- the tabular model can expose concrete, feature-level contributions over familiar variables like lagged demand, hour, temperature, solar output, and battery state

This is the key modeling choice in the repository. The system is intentionally saying:

`use the richer model to predict, but use the more interpretable model to explain`

That is a purposeful enterprise design choice, not a compromise born of incomplete implementation.

## Purposeful Tradeoffs

### Accuracy vs Interpretability

The repository does not pretend that one modeling approach can cleanly maximize both. The TFT and XGBoost split exists because the forecasting and explanation problems are adjacent, but not identical.

### Robustness vs Truthfulness

Anomaly repair is threshold-gated. Sustained drops are repaired only when duration and magnitude criteria are met. This is intentional: aggressive repair improves model stability, but it can also erase real operational events. The system therefore treats repaired values as derived artifacts and keeps an audit trail.

### Cost vs Coverage

Not every site deserves the same model budget. The router exists because enterprise systems usually have a small set of high-impact assets and a large tail of lower-impact ones. Heavyweight forecasting is focused where the marginal accuracy gain is worth the compute and maintenance cost.

### Automation vs Control

The self-healing loop is not fully autonomous. Validation, repair, drift scoring, and retraining recommendations are structured and rule-governed. The system automates repeatable judgment, not arbitrary judgment.

## Self-Healing Behavior

The "self-healing" aspect of the platform is not a single model behavior. It emerges from several coordinated mechanisms:

- anomaly detection identifies outage-like or suspicious demand drops
- repair logic reconstructs only sufficiently severe corrupted segments
- repair records preserve original versus corrected values
- drift scoring tracks changes in demand behavior across time
- retraining decisions are gated by drift, anomaly presence, and business impact

This matters because the repository is designed for systems that continue operating under degraded data conditions, not just for ideal historical datasets.

## Repository Layout

```text
src/self_healing_energy/
  api/                    FastAPI service surface
  config/                 typed settings and model registry
  data/                   parquet ingestion, schema checks, synthetic data generation
  features/               temporal and site-level feature engineering
  forecasting/            TFT, XGBoost, metrics, routing, explainability
  anomalies/              detection, repair, drift monitors, audit records
  optimization/           asset heuristics and constraint models
  orchestration/          batch forecasting and corrective workflows
  serving/                response assembly and business-facing outputs
  training/               train, backtest, and run-summary workflows
  utils/                  shared helpers
tests/
```

## Input Contract

Expected parquet inputs are partitionable by `site_id` and `date`, with a canonical schema such as:

- `timestamp`: UTC hourly timestamp
- `site_id`: unique site identifier
- `demand_kwh`: observed load
- `temperature_c`: weather feature
- `solar_kw`: on-site solar generation
- `battery_soc`: battery state of charge
- `tariff_code`: price regime or business segment
- `is_holiday`: calendar indicator

## Running The Project

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev,tft]
pytest
```

Synthetic parquet generation:

```bash
python -m self_healing_energy.data.generate_cli data/hourly/demo_sites.parquet --num-sites 24 --horizon-days 90
```

Training and holdout evaluation:

```bash
python -m self_healing_energy.training.cli hourly/site_batch.parquet --test-horizon-hours 24 --artifact-dir artifacts/run_001
python -m self_healing_energy.training.test_cli hourly/site_batch.parquet --test-horizon-hours 72 --artifact-dir artifacts/backtest_001
```

Those workflows:

- load parquet from `DATA_ROOT`
- validate schema and hourly grain
- detect anomalies and repair only threshold-qualified events
- record repair audit trails
- engineer temporal features
- train routed forecasting paths
- score holdout forecasts with `MAE`, `RMSE`, `MAPE`, and interval coverage
- persist a structured training summary

## Possible Improvements

- Persist TFT and XGBoost artifacts so training and inference are cleanly decoupled.
- Batch TFT prediction more aggressively for large multi-site inference workloads.
- Enrich the optimization layer so it consumes forecast uncertainty directly rather than point forecasts plus heuristics.
- Add richer business-facing reporting that summarizes top forecast drivers and repair rationale per site.
