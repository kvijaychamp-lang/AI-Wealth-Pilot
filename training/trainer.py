"""Training logic for machine learning models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from data.repository import StockDataRepository
from models.model_factory import ModelFactory
from utils.constants import MODEL_FILENAME
from utils.logger import get_training_logger, log_model_training

logger = get_training_logger()


class TrainingPipeline:
    """Orchestrate historical data preparation and model training."""

    def __init__(self, config):
        self.config = config
        self.model_path = self.config.models.random_forest_path

    def prepare_features(self, data_repo: StockDataRepository, symbol: str):
        df = data_repo.get_symbol_data(symbol).copy()
        df = df.dropna(subset=['Close'])
        df['target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        features = df[['Close']].to_numpy()
        target = df['target'].to_numpy()
        return train_test_split(features, target, test_size=0.2, random_state=42)

    def run(self) -> None:
        from data.loader import HistoricalDataLoader
        from utils.constants import DEFAULT_STOCK_SYMBOLS

        logger.info("Starting model training pipeline")
        log_model_training("RandomForest", "STARTED", symbols=DEFAULT_STOCK_SYMBOLS)

        try:
            start_date, end_date = HistoricalDataLoader.default_date_range(days=90)
            logger.info(f"Loading data from {start_date} to {end_date}")

            loader = HistoricalDataLoader(symbols=DEFAULT_STOCK_SYMBOLS, start_date=start_date, end_date=end_date)
            data = loader.fetch()
            repo = StockDataRepository(data)

            logger.info("Building and training model")
            model = ModelFactory.build()
            X_train, X_test, y_train, y_test = self.prepare_features(repo, DEFAULT_STOCK_SYMBOLS[0])

            logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
            model.fit(X_train, y_train)

            score = model.score(X_test, y_test)
            logger.info(f"Model training completed with score: {score:.4f}")

            ModelFactory.save(model, self.model_path)
            log_model_training("RandomForest", "COMPLETED", score=f"{score:.4f}", model_path=str(self.model_path))

            logger.info('Trained model saved to %s', self.model_path)

        except Exception as e:
            log_model_training("RandomForest", "FAILED", error=str(e))
            logger.error(f"Model training failed: {e}")
            raise
        logger.info('Model R^2 score: %.4f', score)
