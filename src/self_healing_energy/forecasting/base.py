from abc import ABC, abstractmethod

import pandas as pd

from self_healing_energy.forecasting.contracts import FeatureAttribution, ForecastPoint


class ForecastModel(ABC):
    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, frame: pd.DataFrame, horizon_hours: int) -> list[ForecastPoint]:
        raise NotImplementedError

    @abstractmethod
    def backtest(self, frame: pd.DataFrame) -> list[ForecastPoint]:
        raise NotImplementedError


class ExplainerModel(ABC):
    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> None:
        raise NotImplementedError

    @abstractmethod
    def explain(self, frame: pd.DataFrame) -> list[FeatureAttribution]:
        raise NotImplementedError
