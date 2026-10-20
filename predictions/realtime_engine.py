"""Real-time AI stock prediction engine using live NSE data."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from data.loader import HistoricalDataLoader

from utils.logger import get_prediction_logger, get_api_logger, log_prediction, log_api_call

logger = get_prediction_logger()

DEFAULT_REFRESH_MINUTES = 5
DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_INTERVAL = '1d'


@dataclass
class PredictionResult:
    """Structured prediction result for a stock symbol."""
    symbol: str
    timestamp: str
    direction: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, Any]


def load_model(model_path: Path) -> BaseEstimator:
    """Load a trained scikit-learn model from disk."""
    if not model_path.exists():
        logger.error('Model file does not exist: %s', model_path)
        raise FileNotFoundError(f'Model file does not exist: {model_path}')

    model = joblib.load(model_path)
    if not hasattr(model, 'predict'):
        logger.error('Loaded object is not a scikit-learn estimator: %s', type(model))
        raise ValueError('Model must support predict()')

    logger.info('Loaded trained model from %s', model_path)
    return model


def load_feature_columns(feature_csv_path: Path) -> list[str]:
    """Read saved feature column names from a training feature CSV."""
    if not feature_csv_path.exists():
        logger.error('Feature file not found: %s', feature_csv_path)
        raise FileNotFoundError(f'Feature file not found: {feature_csv_path}')

    columns = pd.read_csv(feature_csv_path, nrows=0).columns.tolist()
    logger.info('Loaded feature column names from %s', feature_csv_path)
    return columns


def parse_live_price(payload: dict[str, Any]) -> tuple[float, str]:
    """Extract a usable live price and timestamp from the NSE API payload."""
    price_info = payload.get('priceInfo') or {}
    if not price_info:
        logger.error('Live payload does not contain price information')
        raise ValueError('Unexpected live payload format')

    for key in ('lastPrice', 'closePrice', 'lastTradedPrice'):
        price_value = price_info.get(key)
        if price_value is not None:
            try:
                price = float(price_value)
                timestamp = price_info.get('lastUpdatedTime') or price_info.get('lastPriceTime') or str(pd.Timestamp.now())
                return price, timestamp
            except (TypeError, ValueError):
                continue

    logger.error('Unable to extract numeric live price from payload')
    raise ValueError('Live payload does not contain a valid price')


def load_symbol_history(symbol: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """Load recent historical OHLCV data for a symbol, preferring processed CSV files."""
    # Always load raw historical data to preserve unscaled OHLCV values for display
    start_date, end_date = HistoricalDataLoader.default_date_range(days=lookback_days)
    loader = HistoricalDataLoader(symbols=[symbol], start_date=start_date, end_date=end_date)
    df = loader.fetch()[symbol]
    df['Date'] = pd.to_datetime(df['Date'])
    logger.info('Loaded raw historical data for %s using HistoricalDataLoader', symbol)
    return df


def append_live_quote(df: pd.DataFrame, price: float, timestamp: str) -> pd.DataFrame:
    """Append the latest live quote to the historical dataset for indicator generation."""
    if df.empty:
        logger.error('Historical dataframe is empty, cannot append live quote')
        raise ValueError('No historical data available')

    latest_row = df.iloc[-1].copy()
    latest_row['Date'] = pd.to_datetime(timestamp, errors='coerce')
    latest_row['Open'] = price
    latest_row['High'] = price
    latest_row['Low'] = price
    latest_row['Close'] = price
    if 'Volume' in latest_row.index:
        latest_row['Volume'] = latest_row.get('Volume', 0) or 0

    combined = pd.concat([df, pd.DataFrame([latest_row])], ignore_index=True)
    combined = combined.sort_values(by='Date', ascending=True).reset_index(drop=True)
    logger.debug('Appended live quote for %s at %s', latest_row.get('symbol', ''), timestamp)
    return combined


def build_realtime_features(symbol: str, feature_columns: list[str]) -> pd.Series:
    """Build a single row of real-time features for prediction."""
    df = load_symbol_history(symbol)

    # Respect market active window — do not append a new live quote when market is closed
    from utils.market_schedule import is_market_active, NSE_TIMEZONE
    now = pd.Timestamp.now(tz=NSE_TIMEZONE).to_pydatetime()
    if is_market_active(now):
        payload = fetch_live_quote(symbol)
        price, timestamp = parse_live_price(payload)
        df = append_live_quote(df, price, timestamp)
    else:
        # Market closed: use last available historical bar without creating a new current-timestamp row
        logger.info('Market closed; using last historical bar for realtime features for %s', symbol)

    feature_df = build_feature_set(df)
    if feature_df.empty:
        logger.error('No valid features generated for %s', symbol)
        raise ValueError('Unable to build features for symbol')

    if not set(feature_columns).issubset(set(feature_df.columns)):
        missing = set(feature_columns) - set(feature_df.columns)
        logger.error('Missing feature columns in generated data: %s', missing)
        raise ValueError(f'Missing feature columns: {missing}')

    latest_features = feature_df.loc[feature_df.index[-1], feature_columns]
    logger.info('Built realtime features for %s', symbol)
    return latest_features


def predict_stock_direction(model: BaseEstimator, features: pd.Series) -> tuple[str, float, dict[str, float]]:
    """Predict the stock direction and confidence for one feature row."""
    if features.empty:
        logger.error('No features provided for prediction')
        raise ValueError('Features cannot be empty')

    x = features.to_frame().T

    try:
        # If model is a Pipeline with a scaler step, apply scaler only to a copy
        from sklearn.pipeline import Pipeline

        scaler = None
        model_to_use = model
        if isinstance(model, Pipeline) and 'scaler' in model.named_steps:
            scaler = model.named_steps['scaler']
            model_to_use = Pipeline([(n, s) for n, s in model.steps if n != 'scaler'])

        if scaler is not None:
            X_copy = x.copy()
            X_scaled = scaler.transform(X_copy)
            prediction = model_to_use.predict(X_scaled)[0]
        else:
            prediction = model.predict(x)[0]

    except NotFittedError as exc:
        logger.error('Model is not fitted for prediction')
        raise RuntimeError('Loaded model is not fitted') from exc

    label = 'UP' if int(prediction) == 1 else 'DOWN'
    probabilities: dict[str, float] = {}
    confidence = 1.0

    # Predict probabilities using the same scaler-aware approach
    if hasattr(model, 'predict_proba'):
        if scaler is not None:
            proba = model_to_use.predict_proba(X_scaled)[0]
        else:
            proba = model.predict_proba(x)[0]

        if len(proba) == 2:
            confidence = float(max(proba))
            probabilities = {'DOWN': float(proba[0]), 'UP': float(proba[1])}
        else:
            confidence = float(np.max(proba))
            probabilities = {f'Class_{i}': float(p) for i, p in enumerate(proba)}

    logger.info('Predicted %s with confidence %.4f', label, confidence)
    return label, confidence, probabilities


class RealTimePredictionEngine:
    """Engine for continuous real-time stock direction prediction."""

    def __init__(
        self,
        model_path: Path,
        feature_csv_path: Path,
        symbols: Iterable[str],
        refresh_minutes: int = DEFAULT_REFRESH_MINUTES,
    ) -> None:
        self.model = load_model(model_path)
        self.feature_columns = load_feature_columns(feature_csv_path)
        self.symbols = list(symbols)
        self.refresh_interval = max(1, refresh_minutes) * 60
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._subscribers: list[Callable[[PredictionResult], None]] = []

    def start(self) -> None:
        """Start continuous predictions in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning('Prediction engine is already running')
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info('Started real-time prediction engine')

    def stop(self) -> None:
        """Stop the prediction engine gracefully."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.refresh_interval + 5)
            logger.info('Stopped real-time prediction engine')

    def subscribe(self, callback: Callable[[PredictionResult], None]) -> None:
        """Register a callback to receive live prediction updates."""
        self._subscribers.append(callback)

    def _run_loop(self) -> None:
        """Refresh predictions at the configured interval."""
        while not self._stop_event.is_set():
            for symbol in self.symbols:
                if self._stop_event.is_set():
                    break
                try:
                    result = self.predict_symbol(symbol)
                    self._notify_subscribers(result)
                except Exception as exc:
                    logger.exception('Prediction failed for %s', symbol)
            time.sleep(self.refresh_interval)

    def _notify_subscribers(self, result: PredictionResult) -> None:
        """Notify subscribers of a new prediction result."""
        for callback in self._subscribers:
            try:
                callback(result)
            except Exception:
                logger.exception('Subscriber callback failed for %s', result.symbol)

    def predict_symbol(self, symbol: str) -> PredictionResult:
        """Generate a live prediction for a single symbol."""
        features = build_realtime_features(symbol, self.feature_columns)
        direction, confidence, probabilities = predict_stock_direction(self.model, features)
        timestamp = pd.Timestamp.now(tz='Asia/Kolkata').isoformat()

        result = PredictionResult(
            symbol=symbol,
            timestamp=timestamp,
            direction=direction,
            confidence=confidence,
            probabilities=probabilities,
            features=features.to_dict(),
        )

        logger.info('Generated prediction for %s: %s (%.4f)', symbol, direction, confidence)
        return result

    def predict_all(self) -> list[PredictionResult]:
        """Predict all configured symbols once and return their results."""
        results: list[PredictionResult] = []
        for symbol in self.symbols:
            try:
                results.append(self.predict_symbol(symbol))
            except Exception as exc:
                logger.warning('Skipping symbol %s due to failure: %s', symbol, exc)
        return results


def run_example() -> None:
    """Example entrypoint for real-time prediction."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    engine = RealTimePredictionEngine(
        model_path=Path('models') / 'saved' / 'random_forest_stock_direction.joblib',
        feature_csv_path=Path('data') / 'ml' / 'X_train.csv',
        symbols=['RELIANCE.NS', 'TCS.NS', 'INFY.NS'],
        refresh_minutes=5,
    )

    def print_prediction(result: PredictionResult) -> None:
        print(
            f"{result.symbol} at {result.timestamp}: {result.direction} "
            f"({result.confidence:.2f}) {result.probabilities}"
        )

    engine.subscribe(print_prediction)
    engine.start()

    try:
        time.sleep(60 * 5)
    finally:
        engine.stop()


if __name__ == '__main__':
    run_example()
