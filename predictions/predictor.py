"""Predictor module for generating stock forecasts."""
from __future__ import annotations

from pathlib import Path

from models.model_factory import ModelFactory
from utils.constants import MODEL_FILENAME


class StockPredictor:
    """Load a trained model and generate simple price predictions."""

    def __init__(self, config) -> None:
        self.model_path = config.models.random_forest_path
        self.model = ModelFactory.load(self.model_path)

    def predict(self, symbol: str) -> float:
        from data.loader import HistoricalDataLoader
        from utils.constants import DEFAULT_DATA_WINDOW

        start_date, end_date = HistoricalDataLoader.default_date_range(days=DEFAULT_DATA_WINDOW)
        loader = HistoricalDataLoader(symbols=[symbol], start_date=start_date, end_date=end_date)
        data = loader.fetch()
        df = data.get(symbol)
        last_close = df['Close'].iloc[-1]
        prediction = self.model.predict([[last_close]])
        return float(prediction[0])
