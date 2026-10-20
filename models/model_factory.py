"""Model persistence and factory helpers."""
from pathlib import Path

import joblib

from models.training_model import StockModel


class ModelFactory:
    """Create, save, and load models for stock prediction."""

    @staticmethod
    def build() -> object:
        return StockModel.create()

    @staticmethod
    def save(model: object, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    @staticmethod
    def load(model_path: Path) -> object:
        return joblib.load(model_path)
