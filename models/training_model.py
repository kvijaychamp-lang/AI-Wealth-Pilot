"""Model definition and training helpers."""
from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class StockModel:
    """Build a simple regression pipeline for stock price prediction."""

    @staticmethod
    def create() -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression()),
        ])
