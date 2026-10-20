"""Real-time AI/ML prediction engine for NSE stocks.

This module orchestrates live data downloading, rolling window calculation,
technical feature engineering, model inference, and volatility regime control.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.base import BaseEstimator

from utils.logger import get_prediction_logger, log_prediction
from live_data.nse_client import fetch_live_quote
from data.loader import HistoricalDataLoader

logger = get_prediction_logger()

# Constants
DEFAULT_MODEL_PATH = Path("models/saved/ensemble_stock_direction.joblib")
DEFAULT_FEATURES_PATH = Path("data/ml/processed/feature_names.csv")
DEFAULT_FEATURE_COLUMNS = ["SMA_10", "SMA_20", "EMA_12", "EMA_26", "MACD", "Volume_MA", "Price_Change"]
NSE_TIMEZONE = ZoneInfo("Asia/Kolkata")
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0
YFINANCE_HISTORY_DAYS = 90
INTRADAY_INTERVAL = "1m"
INTRADAY_ROLLING_PERIODS = 100
INTRADAY_FETCH_PERIOD = "5d"
INTRADAY_CACHE_TTL_SECONDS = 10


def _retry_action(
    action: Callable[..., Any],
    *args: Any,
    retries: int = RETRY_ATTEMPTS,
    backoff: float = RETRY_BACKOFF_SECONDS,
    **kwargs: Any,
) -> Any:
    """Execute a callable with retry/backoff for transient failures."""
    last_exception: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return action(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            logger.warning(
                "Retry %d/%d failed for %s: %s",
                attempt,
                retries,
                getattr(action, '__name__', str(action)),
                exc,
            )
            time.sleep(backoff * attempt)
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Retry failed without exception")


def create_mock_history(symbol: str, days: int = 90) -> pd.DataFrame:
    """Generate fallback historical OHLCV data anchored to realistic INR range."""
    row_count = max(days, 60)
    symbol_upper = symbol.upper()
    reference_prices = {
        'RELIANCE.NS': 2450.0,
        'RELIANCE': 2450.0,
        'TCS.NS': 3850.0,
        'TCS': 3850.0,
        'INFY.NS': 1850.0,
        'INFY': 1850.0,
        'HDFC.NS': 1650.0,
        'HDFC': 1650.0,
        'HDFCBANK.NS': 1650.0,
        'ICICIBANK.NS': 1250.0,
        'ICICIBANK': 1250.0,
    }
    base_price = reference_prices.get(symbol_upper, 1500.0)
    dates = pd.date_range(end=datetime.now(tz=NSE_TIMEZONE).normalize(), periods=row_count, freq='D')
    np.random.seed(abs(hash(symbol_upper)) % (2**31 - 1))
    returns = np.random.randn(row_count) * 0.012
    prices = base_price * np.exp(np.cumsum(returns))
    noise = np.random.randn(row_count) * 0.002
    return pd.DataFrame(
        {
            'Date': dates,
            'Open': prices * (1.0 + noise),
            'High': prices * (1.0 + abs(np.random.randn(row_count) * 0.007)),
            'Low': prices * (1.0 - abs(np.random.randn(row_count) * 0.007)),
            'Close': prices,
            'Volume': np.random.randint(1_000_000, 8_000_000, size=row_count).astype(float),
        }
    )


def is_nse_market_hours() -> bool:
    """Check if the current time falls within NSE trading hours.

    NSE Trading Hours: Monday - Friday, 09:15 AM to 03:30 PM IST.
    Does a fast time-based check and falls back to calendar schedules if available.
    """
    tz = NSE_TIMEZONE
    now = datetime.now(tz)

    # 1. Quick day check: Saturday/Sunday
    if now.weekday() >= 5:
        logger.debug("Market closed: Weekend (%s)", now.strftime("%A"))
        return False

    # 2. Check time bounds (9:15 AM to 3:30 PM)
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if not (market_start <= now <= market_end):
        logger.debug("Market closed: Outside trading hours (%s)", now.strftime("%I:%M %p"))
        return False

    # 3. Holiday check (requires pandas_market_calendars via utils)
    try:
        from utils.market_schedule import is_market_holiday
        if is_market_holiday(now.date()):
            logger.debug("Market closed: Scheduled NSE Holiday (%s)", now.date())
            return False
    except Exception as exc:
        logger.warning("Market calendar holiday check failed (defaulting to time check): %s", exc)

    logger.debug("Market is open and active")
    return True


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all required technical features dynamically on unscaled scale.

    Calculates: SMA(10), SMA(20), EMA(12), EMA(26), MACD (Line and Signal),
    RSI(14), Bollinger Bands (Upper/Lower), Volume Moving Average (10),
    and Average True Range (ATR 14).
    """
    # 1. Ensure the dataframe is fully stitched before calculation (works on ENTIRE dataframe)
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # 2. Recalculate technical metrics over the ENTIRE stitched dataframe
    # Simple Moving Averages
    df["SMA_10"] = close.rolling(window=10, min_periods=1).mean()
    df["SMA_20"] = close.rolling(window=20, min_periods=1).mean()

    # Exponential Moving Averages
    df["EMA_12"] = close.ewm(span=12, adjust=False, min_periods=1).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False, min_periods=1).mean()

    # MACD Line and Signal
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=1).mean()

    # RSI(14) - Wilder's Smoothing EWM technique
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - (100 / (1 + rs))).fillna(50.0)
    df["RSI_14"] = df["RSI"]

    # Bollinger Bands (Upper/Lower)
    bb_middle = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std()
    df["BB_Upper"] = bb_middle + (2.0 * bb_std)
    df["BB_Lower"] = bb_middle - (2.0 * bb_std)
    df["BB_Upper"] = df["BB_Upper"].fillna(close)
    df["BB_Lower"] = df["BB_Lower"].fillna(close)

    # Volume Moving Average (10)
    df["Volume_MA"] = volume.rolling(window=10, min_periods=1).mean()

    # Average True Range (ATR 14)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14, min_periods=1).mean()
    # Handle first row NaN
    if not df.empty:
        df.loc[df.index[0], "ATR_14"] = float(high.iloc[0] - low.iloc[0])

    # Price Change (unscaled percent change)
    df["Price_Change"] = close.pct_change().fillna(0.0)

    # 3. Final safety check: Bulletproof Last-Row Fallback Mechanism
    feature_cols = [
        "SMA_10", "SMA_20", "EMA_12", "EMA_26", "MACD", "MACD_Signal",
        "RSI", "RSI_14", "BB_Upper", "BB_Lower", "Volume_MA", "ATR_14", "Price_Change"
    ]
    if len(df) >= 2:
        for col in feature_cols:
            if pd.isna(df.iloc[-1][col]):
                df.iloc[-1, df.columns.get_loc(col)] = df.iloc[-2][col]
    
    # Log successful computation for debugging indicator tracking
    logger.debug("Computed technical indicators and applied explicit last-row safety checks across %d rows.", len(df))

    return df


class LivePredictionEngine:
    """Thread-safe prediction engine for live market status and model inference."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        feature_names_path: Path = DEFAULT_FEATURES_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.feature_names_path = Path(feature_names_path)
        self.model: Optional[BaseEstimator] = None
        self.feature_columns: List[str] = DEFAULT_FEATURE_COLUMNS.copy()
        self._intraday_cache: dict[str, dict[str, Any]] = {}
        self._intraday_cache_lock = threading.Lock()
        
        # Load resources eagerly
        self._load_model()
        self._load_feature_columns()

    def _load_model(self) -> None:
        """Load the ensemble model from disk, falling back gracefully if not found."""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info("Ensemble model loaded successfully from %s", self.model_path)
            else:
                logger.warning(
                    "Model file not found at %s. Attempting fallback paths...", self.model_path
                )
                # Fallback to general random forest model
                rf_fallback = Path("models/random_forest_classifier.pkl")
                if rf_fallback.exists():
                    self.model = joblib.load(rf_fallback)
                    logger.info("Fallback Random Forest model loaded from %s", rf_fallback)
                else:
                    logger.error("No valid model file located in fallback directories.")
        except Exception as exc:
            logger.exception("Failed to load machine learning model: %s", exc)

    def _load_feature_columns(self) -> None:
        """Load exact feature columns from configuration, falling back to defaults."""
        try:
            if self.feature_names_path.exists():
                features_df = pd.read_csv(self.feature_names_path)
                if "feature" in features_df.columns:
                    self.feature_columns = features_df["feature"].dropna().tolist()
                else:
                    self.feature_columns = features_df.columns.tolist()
                logger.info("Loaded features from config: %s", self.feature_columns)
            else:
                logger.info("Using default ML features: %s", self.feature_columns)
        except Exception as exc:
            logger.warning("Feature config loading failed, using defaults: %s", exc)

    def get_rolling_window(self, symbol: str, is_live: bool) -> pd.DataFrame:
        """Get the base historical data and stitch the latest live tick if live.

        Prevents data leakage by ensuring strictly chronological ordering and no lookahead.
        """
        # 1. Download last 60 days of historical daily data
        df_history = self._load_historical_base(symbol)

        if not is_live:
            logger.debug("[%s] Running in Simulation Mode (Market Closed)", symbol)
            return df_history

        logger.debug(
            "[%s] Running live intraday rolling window fetch (%s, last %d candles)",
            symbol,
            INTRADAY_INTERVAL,
            INTRADAY_ROLLING_PERIODS,
        )

        try:
            df_intraday = self._load_intraday_rolling_window(symbol)
            if not df_intraday.empty:
                return df_intraday
        except Exception as exc:
            logger.warning("[%s] Intraday live fetch failed: %s", symbol, exc)

        live_tick = _retry_action(self._fetch_live_quote_data, symbol, retries=2, backoff=1.0)
        if not live_tick:
            logger.warning("[%s] Live quote unavailable, returning history only", symbol)
            return df_history

        df_history = self._stitch_live_quote(df_history, live_tick)
        return df_history

    def _normalize_ohlcv_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize historical and intraday data into clean Date/Open/High/Low/Close/Volume schema."""
        clean = df.copy()

        if isinstance(clean.columns, pd.MultiIndex):
            clean.columns = [col[0] if isinstance(col, tuple) else col for col in clean.columns]

        if 'Date' not in clean.columns:
            if isinstance(clean.index, pd.DatetimeIndex):
                clean = clean.reset_index()
            elif 'datetime' in clean.columns:
                clean = clean.rename(columns={'datetime': 'Date'})

        rename_map = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
        }
        clean = clean.rename(columns={k: v for k, v in rename_map.items() if k in clean.columns})

        for col in ('Open', 'High', 'Low', 'Close', 'Volume'):
            if col not in clean.columns:
                raise ValueError(f'Missing OHLCV column: {col}')

        clean['Date'] = pd.to_datetime(clean['Date'], errors='coerce')
        clean = clean.dropna(subset=['Date'])
        clean['Date'] = clean['Date'].dt.tz_localize(None)

        # Preserve numeric scale and avoid auto-adjustment.
        clean['Open'] = clean['Open'].astype(float)
        clean['High'] = clean['High'].astype(float)
        clean['Low'] = clean['Low'].astype(float)
        clean['Close'] = clean['Close'].astype(float)
        clean['Volume'] = clean['Volume'].astype(float)

        return clean[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def _load_historical_base(self, symbol: str) -> pd.DataFrame:
        """Download or read 60 days of historical daily stock data.

        The returned OHLCV data remains on the native price scale and is
        normalized for safe feature engineering.
        """
        try:
            start_date, end_date = HistoricalDataLoader.default_date_range(days=60)
            loader = HistoricalDataLoader(symbols=[symbol], start_date=start_date, end_date=end_date)
            data_dict = loader.fetch()
            if symbol in data_dict and not data_dict[symbol].empty:
                return self._normalize_ohlcv_dataframe(data_dict[symbol]).sort_values(by="Date").reset_index(drop=True)
        except Exception as exc:
            logger.warning("Loader failed for %s historical base, falling back to yfinance: %s", symbol, exc)

        try:
            ticker = yf.Ticker(symbol)
            # Hard override: fetch recent intraday 5-minute candles instead of daily bars
            try:
                df = _retry_action(
                    yf.download,
                    tickers=symbol,
                    period='5d',
                    interval='5m',
                    progress=False,
                    auto_adjust=False,
                )
            except Exception as exc:
                logger.error("yf.download intraday fetch failed for %s: %s", symbol, exc)
                df = pd.DataFrame()

            if not df.empty:
                df = df.dropna()
                df = df.tail(100).reset_index()
                if 'Datetime' not in df.columns:
                    if 'Date' in df.columns:
                        df.rename(columns={'Date': 'Datetime'}, inplace=True)
                    else:
                        df['Datetime'] = df.iloc[:, 0]
                df['Date'] = pd.to_datetime(df['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                return self._normalize_ohlcv_dataframe(df).sort_values('Date').reset_index(drop=True)
        except Exception as exc:
            logger.error("Failed downloading base history from yfinance for %s: %s", symbol, exc)

        logger.warning("Generating mock daily history fallback for %s", symbol)
        return create_mock_history(symbol, days=YFINANCE_HISTORY_DAYS)

    def _load_intraday_rolling_window(self, symbol: str) -> pd.DataFrame:
        """Fetch a cached rolling intraday OHLCV window to avoid yfinance throttling."""
        now = pd.Timestamp.now(tz=NSE_TIMEZONE)
        with self._intraday_cache_lock:
            cached = self._intraday_cache.get(symbol)
            if cached and (now - cached["fetched_at"]).total_seconds() < INTRADAY_CACHE_TTL_SECONDS:
                return cached["data"].copy()

        # Hard override: always fetch intraday 5-minute data via yf.download
        try:
            df_intraday = _retry_action(
                yf.download,
                tickers=symbol,
                period=INTRADAY_FETCH_PERIOD,
                interval="5m",
                progress=False,
                auto_adjust=False,
            )
        except Exception as exc:
            logger.warning("yf.download intraday fetch failed for %s: %s", symbol, exc)
            df_intraday = pd.DataFrame()

        if df_intraday.empty:
            raise ValueError(f"Intraday yfinance returned no data for {symbol}")

        # Safety cleaning and rolling window trimming
        df_intraday = df_intraday.dropna()
        df_intraday = df_intraday.tail(INTRADAY_ROLLING_PERIODS).reset_index()

        # Ensure there's a Datetime column and a formatted Date string for display
        if "Datetime" not in df_intraday.columns:
            if "Date" in df_intraday.columns:
                df_intraday.rename(columns={"Date": "Datetime"}, inplace=True)
            else:
                df_intraday["Datetime"] = df_intraday.iloc[:, 0]
        df_intraday["Date"] = pd.to_datetime(df_intraday["Datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        clean_intraday = self._normalize_ohlcv_dataframe(df_intraday)
        clean_intraday = clean_intraday.sort_values("Date").reset_index(drop=True)

        with self._intraday_cache_lock:
            self._intraday_cache[symbol] = {
                "fetched_at": now,
                "data": clean_intraday.copy(),
            }

        return clean_intraday

    def _get_atr_from_last_closed_candle(self, df_features: pd.DataFrame) -> float:
        """Return ATR corresponding to the last fully closed intraday candle."""
        if df_features.empty:
            return 0.0

        last_idx = len(df_features) - 1
        last_ts = pd.to_datetime(df_features.loc[last_idx, "Date"])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize(NSE_TIMEZONE)
        else:
            last_ts = last_ts.tz_convert(NSE_TIMEZONE)

        cutoff = pd.Timestamp.now(tz=NSE_TIMEZONE) - pd.Timedelta(INTRADAY_INTERVAL)
        if last_ts >= cutoff and len(df_features) > 1:
            closed_row = df_features.iloc[-2]
        else:
            closed_row = df_features.iloc[-1]

        return float(closed_row.get("ATR_14", 0.0) or 0.0)

    def _is_intraday_active_candle(self, df_features: pd.DataFrame) -> bool:
        if len(df_features) < 2:
            return False
        last_ts = pd.to_datetime(df_features.loc[len(df_features) - 1, "Date"])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize(NSE_TIMEZONE)
        else:
            last_ts = last_ts.tz_convert(NSE_TIMEZONE)

        cutoff = pd.Timestamp.now(tz=NSE_TIMEZONE) - pd.Timedelta(INTRADAY_INTERVAL)
        return last_ts >= cutoff

    def _fetch_live_quote_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest real-time stock price and volume data point."""
        # Try live NSE api client with retry support
        try:
            payload = _retry_action(fetch_live_quote, symbol, retries=2, backoff=0.8)
            price_info = payload.get("priceInfo", {})
            price = float(
                price_info.get("lastPrice")
                or price_info.get("closePrice")
                or price_info.get("lastTradedPrice")
                or 0.0
            )
            if price > 0:
                ts_str = (
                    price_info.get("lastUpdatedTime")
                    or price_info.get("lastPriceTime")
                    or datetime.now(tz=NSE_TIMEZONE).isoformat()
                )
                timestamp = pd.to_datetime(ts_str, utc=True, errors="coerce")
                if pd.isna(timestamp):
                    timestamp = datetime.now(tz=NSE_TIMEZONE)
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.tz_convert(NSE_TIMEZONE).tz_localize(None)

                volume = float(
                    price_info.get("volume", 0.0)
                    or payload.get("volume", {}).get("totSellQty", 0.0)
                    or 0.0
                )
                return {
                    "Date": timestamp,
                    "Open": price,
                    "High": price,
                    "Low": price,
                    "Close": price,
                    "Volume": volume,
                }
        except Exception as exc:
            logger.debug("NSE quote client fetch failed for %s: %s", symbol, exc)

        # Fallback: yfinance intraday
        try:
            # Hard override: use yf.download to fetch recent 5-minute candles
            df_intraday = _retry_action(
                yf.download,
                tickers=symbol,
                period=INTRADAY_FETCH_PERIOD,
                interval="5m",
                progress=False,
                auto_adjust=False,
            )
            if df_intraday.empty:
                raise ValueError("yfinance returned empty intraday frame")

            df_intraday = df_intraday.dropna()
            df_intraday = df_intraday.tail(INTRADAY_ROLLING_PERIODS).reset_index()
            # ensure Datetime column exists
            if "Datetime" not in df_intraday.columns:
                if "Date" in df_intraday.columns:
                    df_intraday.rename(columns={"Date": "Datetime"}, inplace=True)
                else:
                    df_intraday["Datetime"] = df_intraday.iloc[:, 0]

            last_row = df_intraday.iloc[-1]
            timestamp = pd.to_datetime(last_row["Datetime"]) 
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(NSE_TIMEZONE).tz_localize(None)
            else:
                timestamp = timestamp.tz_convert(NSE_TIMEZONE).tz_localize(None)

            return {
                "Date": timestamp,
                "Open": float(last_row["Open"]),
                "High": float(last_row["High"]),
                "Low": float(last_row["Low"]),
                "Close": float(last_row["Close"]),
                "Volume": float(last_row["Volume"]),
            }
        except Exception as exc:
            logger.warning("Intraday yfinance fallback failed for %s: %s", symbol, exc)

        return None

    def _stitch_live_quote(self, df_history: pd.DataFrame, live_tick: Dict[str, Any]) -> pd.DataFrame:
        """Safely stitch real-time tick onto the tail of daily history.

        If timestamps match by date, overwrite the final row (today's update).
        Otherwise, append it as a new chronological row.
        """
        df = df_history.copy()
        if df.empty:
            return pd.DataFrame([live_tick])

        history_last_date = pd.to_datetime(df["Date"].iloc[-1]).date()
        live_date = pd.to_datetime(live_tick["Date"]).date()

        row_data = {
            "Date": pd.to_datetime(live_tick["Date"]).tz_localize(None),
            "Open": float(live_tick["Open"]),
            "High": float(live_tick["High"]),
            "Low": float(live_tick["Low"]),
            "Close": float(live_tick["Close"]),
            "Volume": float(live_tick["Volume"]),
        }

        if history_last_date == live_date:
            # Overwrite the final row (accumulating intraday high/low)
            idx = df.index[-1]
            df.loc[idx, "Close"] = row_data["Close"]
            df.loc[idx, "High"] = max(float(df.loc[idx, "High"]), row_data["High"])
            df.loc[idx, "Low"] = min(float(df.loc[idx, "Low"]), row_data["Low"])
            df.loc[idx, "Volume"] = row_data["Volume"]
            df.loc[idx, "Date"] = row_data["Date"]
        else:
            # Append new day
            row_data["Date"] = row_data["Date"].replace(hour=0, minute=0, second=0, microsecond=0)
            new_row = pd.DataFrame([row_data])
            df = pd.concat([df, new_row], ignore_index=True)

        return df.sort_values(by="Date").reset_index(drop=True)

    def determine_signal(self, up_prob: float, atr: float) -> Tuple[str, float]:
        """Apply dynamic volatility target control (ATR Engine) to choose signal.

        - ATR < 25 (Low Volatility Regime): Threshold = 52.0%
        - ATR >= 25 (High Volatility Regime): Threshold = 57.0%
        """
        if atr < 25.0:
            threshold = 0.52
        else:
            threshold = 0.57

        down_prob = 1.0 - up_prob

        if up_prob >= threshold:
            return "UP", threshold
        elif down_prob >= threshold:
            return "DOWN", threshold
        else:
            return "HOLD", threshold

    def _get_label_probability(self, proba: np.ndarray, classes: np.ndarray, label: str = 'UP') -> float:
        """Resolve the probability of the requested label from model output."""
        try:
            index = int(np.where(classes == label)[0][0])
            return float(proba[index])
        except Exception:
            return float(proba[1]) if len(proba) > 1 else float(proba[0])

    def _extract_individual_probas(self, X_live: pd.DataFrame, ensemble_prob: float) -> Tuple[float, float]:
        """Extract XGBoost and LightGBM individual probabilities from VotingClassifier.

        Falls back to the ensemble probability if individual models are inaccessible.
        """
        xgb_prob = ensemble_prob
        lgbm_prob = ensemble_prob

        if self.model is None:
            return xgb_prob, lgbm_prob

        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import VotingClassifier

        try:
            classifier = self.model
            scaler = None

            if isinstance(self.model, Pipeline):
                if "scaler" in self.model.named_steps:
                    scaler = self.model.named_steps["scaler"]
                if "classifier" in self.model.named_steps:
                    classifier = self.model.named_steps["classifier"]

            if isinstance(classifier, VotingClassifier):
                X_scaled = scaler.transform(X_live) if scaler is not None else X_live

                if "xgb" in classifier.named_estimators_:
                    xgb_model = classifier.named_estimators_["xgb"]
                    xgb_proba = xgb_model.predict_proba(X_scaled)[0]
                    xgb_prob = self._get_label_probability(xgb_proba, np.asarray(xgb_model.classes_), label='UP')
                if "lgbm" in classifier.named_estimators_:
                    lgbm_model = classifier.named_estimators_["lgbm"]
                    lgbm_proba = lgbm_model.predict_proba(X_scaled)[0]
                    lgbm_prob = self._get_label_probability(lgbm_proba, np.asarray(lgbm_model.classes_), label='UP')
        except Exception as exc:
            logger.debug("Could not extract XGBoost/LightGBM subclass probabilities: %s", exc)

        return xgb_prob, lgbm_prob

    def predict_live(self, symbol: str, df_rolling: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate a complete real-time prediction payload.

        Automatically branch based on market hours, form the rolling window if not provided,
        calculate indicators, execute ML soft voting probabilities, and output signals.
        """
        is_active = is_nse_market_hours()
        status_message = "Live Mode (Market Open)" if is_active else "Simulation Mode (Market Closed)"
        
        # 1. Stitch and get the complete rolling window if not provided
        if df_rolling is None:
            df_rolling = self.get_rolling_window(symbol, is_live=is_active)
        
        # 2. Calculate technical features
        df_features = compute_technical_indicators(df_rolling)
        
        # Extract latest values for logging and return fields
        latest_row = df_features.iloc[-1]
        live_price = float(latest_row["Close"])
        rsi = float(latest_row.get("RSI", 50.0))
        timestamp_str = pd.to_datetime(latest_row["Date"]).isoformat()

        atr = self._get_atr_from_last_closed_candle(df_features)
        active_candle = self._is_intraday_active_candle(df_features)
        if active_candle:
            logger.debug(
                "[%s] Active intraday candle detected; threshold anchored to last closed candle.",
                symbol,
            )

        # 3. Model Inference
        up_prob = 0.50
        xgb_prob = 0.50
        lgbm_prob = 0.50
        
        if self.model is not None:
            try:
                # Align features exactly to columns expected by ML model
                aligned_dict = {col: float(latest_row.get(col, 0.0)) for col in self.feature_columns}
                X_live = pd.DataFrame([aligned_dict], columns=self.feature_columns)
                
                # Fetch prediction probabilities
                ensemble_proba = self.model.predict_proba(X_live)[0]
                model_classes = np.asarray(getattr(self.model, 'classes_', np.array([0, 1])))
                up_prob = self._get_label_probability(ensemble_proba, model_classes, label='UP')
                
                # Fetch XGB and LightGBM sub-probabilities
                xgb_prob, lgbm_prob = self._extract_individual_probas(X_live, up_prob)
            except Exception as exc:
                logger.error("[%s] Inference failed. Falling back to default baseline: %s", symbol, exc)
        else:
            logger.warning("[%s] Model artifact not loaded. Falling back to heuristic baseline.", symbol)
            # Heuristic baseline probability based on RSI
            up_prob = 0.5 + 0.1 * ((rsi - 50.0) / 50.0)
            up_prob = max(0.1, min(0.9, up_prob))
            xgb_prob, lgbm_prob = up_prob, up_prob

        # 4. Volatility Target Selector
        signal, threshold = self.determine_signal(up_prob, atr)
        
        # Confidence is the score of the selected class
        confidence = up_prob if signal == "UP" else (1.0 - up_prob if signal == "DOWN" else max(up_prob, 1.0 - up_prob))

        result = {
            "symbol": symbol,
            "timestamp": timestamp_str,
            "live_price": live_price,
            "signal": signal,
            "confidence": confidence,
            "atr": atr,
            "rsi": rsi,
            "xgb_prob": xgb_prob,
            "lgbm_prob": lgbm_prob,
            "ensemble_prob": up_prob,
            "threshold": threshold,
            "mode": "Live" if is_active else "Simulation",
            "mode_message": status_message,
            "df_features": df_features,
        }

        # Log prediction details centrally
        log_kwargs = {k: v for k, v in result.items() if k not in ("symbol", "signal", "confidence", "df_features")}
        log_prediction(symbol, signal, confidence, **log_kwargs)
        logger.info(
            "[%s] Signal: %s (Conf: %.2f%%) | ATR: %.2f | Price: INR %.2f | Mode: %s",
            symbol,
            signal,
            confidence * 100,
            atr,
            live_price,
            result["mode"],
        )

        return result


def execute_virtual_trade_callback(
    payload: Dict[str, Any],
    wallet: Dict[str, Any],
    execute_buy_fn: Callable[[str, float, int], bool],
    execute_sell_fn: Callable[[str, float], bool],
    shares: int = 10,
) -> Optional[Dict[str, Any]]:
    """A clean callback callback / virtual wallet trigger.

    Checks payload signal changes and executes trades automatically.
    
    Args:
        payload (dict): The dictionary returned by predict_live().
        wallet (dict): Session state portfolio dict.
        execute_buy_fn (callable): Callback to purchase shares.
        execute_sell_fn (callable): Callback to liquidate position.
        shares (int): Size of trades (default: 10).
    """
    symbol = payload["symbol"]
    signal = payload["signal"]
    price = payload["live_price"]
    confidence = payload["confidence"]
    threshold = payload["threshold"]

    # Only execute on high-confidence triggers
    is_high_confidence = confidence >= threshold
    if not is_high_confidence or signal not in ("UP", "DOWN"):
        return None

    # Handle buy
    if signal == "UP":
        # Check if we already have a position
        positions = wallet.get("positions", {})
        if symbol not in positions or positions[symbol].get("quantity", 0) == 0:
            success = execute_buy_fn(symbol, price, shares)
            if success:
                logger.info("[%s] Virtual Trade: AUTO BUY %d shares at %.2f", symbol, shares, price)
                return {"action": "BUY", "quantity": shares, "price": price}

    # Handle sell
    elif signal == "DOWN":
        positions = wallet.get("positions", {})
        if symbol in positions and positions[symbol].get("quantity", 0) > 0:
            qty = positions[symbol].get("quantity", 0)
            success = execute_sell_fn(symbol, price)
            if success:
                logger.info("[%s] Virtual Trade: AUTO SELL %d shares at %.2f", symbol, qty, price)
                return {"action": "SELL", "quantity": qty, "price": price}

    return None
