import pandas as pd
from xgboost import DMatrix, XGBRegressor

from self_healing_energy.forecasting.base import ExplainerModel
from self_healing_energy.forecasting.contracts import FeatureAttribution
from self_healing_energy.forecasting.xgb_common import (
    MODEL_FEATURES,
    add_tabular_forecast_features,
    build_category_mapping,
    build_training_matrix,
)


class XGBoostShadowExplainer(ExplainerModel):
    """Shadow-model explainer backed by a real XGBoost regressor."""

    def fit(self, frame: pd.DataFrame) -> None:
        self.site_mapping = build_category_mapping(frame["site_id"].astype(str))
        self.tariff_mapping = build_category_mapping(frame["tariff_code"].fillna("unknown").astype(str))
        design, target, _ = build_training_matrix(frame, self.site_mapping, self.tariff_mapping)
        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=11,
        )
        self.model.fit(design, target)

    def explain(self, frame: pd.DataFrame) -> list[FeatureAttribution]:
        attributions: list[FeatureAttribution] = []
        featured = add_tabular_forecast_features(frame, self.site_mapping, self.tariff_mapping)

        for site_id, group in featured.groupby("site_id", sort=False):
            latest = group.sort_values("timestamp").tail(1)
            matrix = DMatrix(latest.loc[:, MODEL_FEATURES].astype(float), feature_names=MODEL_FEATURES)
            contributions = self.model.get_booster().predict(matrix, pred_contribs=True)[0]
            feature_contributions = contributions[:-1]

            for feature_name, contribution in zip(MODEL_FEATURES, feature_contributions, strict=False):
                attributions.append(
                    FeatureAttribution(
                        site_id=site_id,
                        feature=feature_name,
                        contribution=float(contribution),
                    )
                )
        return attributions
