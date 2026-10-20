"""Build technical indicator features for stock market machine learning datasets."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure the input dataframe contains the required OHLCV columns."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.error('Dataframe is missing required columns: %s', missing)
        raise ValueError(f'Missing required columns: {missing}')
    logger.debug('OHLCV validation passed')


def add_rsi(df: pd.DataFrame, window: int = 14, close_col: str = 'Close') -> pd.Series:
    """Add Relative Strength Index (RSI) to the dataset."""
    indicator = RSIIndicator(close=df[close_col], window=window)
    return indicator.rsi()


def add_sma(df: pd.DataFrame, window: int = 20, close_col: str = 'Close') -> pd.Series:
    """Add Simple Moving Average (SMA) for the close price."""
    indicator = SMAIndicator(close=df[close_col], window=window)
    return indicator.sma_indicator()


def add_ema(df: pd.DataFrame, window: int = 20, close_col: str = 'Close') -> pd.Series:
    """Add Exponential Moving Average (EMA) for the close price."""
    indicator = EMAIndicator(close=df[close_col], window=window)
    return indicator.ema_indicator()


def add_macd(df: pd.DataFrame, close_col: str = 'Close') -> pd.DataFrame:
    """Add MACD line, MACD signal line, and MACD histogram."""
    indicator = MACD(close=df[close_col])
    return pd.DataFrame(
        {
            'MACD': indicator.macd(),
            'MACD_Signal': indicator.macd_signal(),
            'MACD_Hist': indicator.macd_diff(),
        }
    )


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, close_col: str = 'Close') -> pd.DataFrame:
    """Add Bollinger Bands components for price volatility."""
    indicator = BollingerBands(close=df[close_col], window=window)
    return pd.DataFrame(
        {
            'BB_Middle': indicator.bollinger_mavg(),
            'BB_Upper': indicator.bollinger_hband(),
            'BB_Lower': indicator.bollinger_lband(),
            'BB_Width': indicator.bollinger_wband(),
        }
    )


def add_volume_indicators(df: pd.DataFrame, close_col: str = 'Close', volume_col: str = 'Volume', window: int = 20) -> pd.DataFrame:
    """Add volume-based indicators to capture momentum and buying pressure."""
    obv = OnBalanceVolumeIndicator(close=df[close_col], volume=df[volume_col])
    vwap = VolumeWeightedAveragePrice(
        high=df['High'],
        low=df['Low'],
        close=df[close_col],
        volume=df[volume_col],
        window=window,
    )
    return pd.DataFrame(
        {
            'OBV': obv.on_balance_volume(),
            'VWAP': vwap.volume_weighted_average_price(),
        }
    )


def add_returns(df: pd.DataFrame, close_col: str = 'Close') -> pd.Series:
    """Add daily return percentages for the close price."""
    return df[close_col].pct_change()


def add_volatility(df: pd.DataFrame, window: int = 20, close_col: str = 'Close') -> pd.Series:
    """Add rolling volatility measured as the standard deviation of daily returns."""
    returns = add_returns(df, close_col=close_col)
    return returns.rolling(window=window).std()


def build_feature_set(
    df: pd.DataFrame,
    rsi_window: int = 14,
    sma_window: int = 20,
    ema_window: int = 20,
    bollinger_window: int = 20,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Create a final feature dataset that is ready for machine learning."""
    validate_required_columns(df)

    data = df.copy()
    logger.info('Building technical features for dataset with %d rows', len(data))

    data['RSI'] = add_rsi(data, window=rsi_window)
    data['SMA'] = add_sma(data, window=sma_window)
    data['EMA'] = add_ema(data, window=ema_window)

    macd_features = add_macd(data)
    data = pd.concat([data, macd_features], axis=1)

    bollinger_features = add_bollinger_bands(data, window=bollinger_window)
    data = pd.concat([data, bollinger_features], axis=1)

    volume_features = add_volume_indicators(data)
    data = pd.concat([data, volume_features], axis=1)

    data['Daily_Return'] = add_returns(data)
    data['Volatility'] = add_volatility(data, window=volatility_window)

    logger.debug('Dropping rows with any NaN values after feature creation')
    data = data.dropna().reset_index(drop=True)

    if data.empty:
        logger.error('Feature dataset is empty after dropping NaNs')
        raise ValueError('Feature dataset contains no valid rows')

    logger.info('Final feature dataset has %d rows and %d columns', len(data), len(data.columns))
    return data


def save_feature_dataset(df: pd.DataFrame, output_path: Path) -> Path:
    """Save the final technical feature dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info('Saved feature dataset to %s', output_path)
    return output_path


def prepare_and_save_features(
    raw_csv_path: Path,
    feature_csv_path: Path,
    **kwargs,
) -> Path:
    """Load raw CSV, generate features, and save the processed dataset."""
    try:
        logger.info('Preparing features from raw file %s', raw_csv_path)
        raw_df = pd.read_csv(raw_csv_path)
        feature_df = build_feature_set(raw_df, **kwargs)
        return save_feature_dataset(feature_df, feature_csv_path)
    except Exception as exc:
        logger.exception('Unable to prepare features for %s: %s', raw_csv_path, exc)
        raise
