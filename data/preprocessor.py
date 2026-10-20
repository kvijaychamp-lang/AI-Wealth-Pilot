"""Data preprocessing utilities for stock market CSV files."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
DEFAULT_PROCESSED_DIR = Path('data') / 'processed'
REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


def ensure_directory(path: Path) -> None:
    """Ensure the destination directory exists before saving files."""
    path.mkdir(parents=True, exist_ok=True)
    logger.debug('Ensured directory exists: %s', path)


def validate_ohlcv_columns(df: pd.DataFrame) -> None:
    """Check that the required OHLCV columns are present in the dataframe."""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        logger.error('Missing expected columns: %s', missing_columns)
        raise ValueError(f'Missing required columns: {missing_columns}')
    logger.debug('All required OHLCV columns are present')


def load_stock_csv(file_path: Path) -> pd.DataFrame:
    """Load raw stock CSV data into a pandas DataFrame."""
    logger.info('Loading CSV file: %s', file_path)
    df = pd.read_csv(file_path)
    validate_ohlcv_columns(df)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw stock data through validation, deduplication, and sorting."""
    validate_ohlcv_columns(df)

    logger.debug('Converting Date column to datetime type')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    logger.debug('Removing rows with invalid dates or missing OHLCV values')
    df = df.dropna(subset=REQUIRED_COLUMNS)

    logger.debug('Dropping duplicate rows based on date and OHLCV fields')
    df = df.drop_duplicates(subset=REQUIRED_COLUMNS, keep='first')

    logger.debug('Converting OHLCV values to numeric dtype when possible')
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    logger.debug('Removing rows with non-numeric OHLCV values')
    df = df.dropna(subset=numeric_columns)

    logger.debug('Sorting data chronologically by Date')
    df = df.sort_values(by='Date', ascending=True)
    df = df.reset_index(drop=True)

    if df.empty:
        logger.error('Cleaned dataframe is empty after preprocessing')
        raise ValueError('No valid stock data remaining after cleanup')

    logger.info('Cleaned dataframe contains %d rows after preprocessing', len(df))
    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV data and engineer rolling technical indicators."""
    validate_ohlcv_columns(df)

    result = df.copy()
    result['Date'] = pd.to_datetime(result['Date'], errors='coerce')

    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors='coerce')

    logger.debug('Dropping rows with invalid date or OHLCV values')
    result = result.dropna(subset=['Date'] + numeric_columns)

    logger.debug('Sorting and deduplicating cleaned data')
    result = result.sort_values(by='Date', ascending=True)
    result = result.drop_duplicates(subset=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], keep='first')

    logger.debug('Dropping market holiday rows where Volume is zero')
    result = result[result['Volume'].ne(0)]
    result = result.reset_index(drop=True)

    if result.empty:
        logger.error('No trading days remain after dropping zero-volume rows')
        raise ValueError('No valid trading rows remain after cleaning')

    logger.debug('Forward-filling minor missing price gaps')
    result[['Open', 'High', 'Low', 'Close']] = result[['Open', 'High', 'Low', 'Close']].ffill()
    result = result.dropna(subset=numeric_columns + ['Date']).reset_index(drop=True)

    close = result['Close']
    result['SMA_10'] = close.rolling(window=10, min_periods=10).mean()
    result['SMA_20'] = close.rolling(window=20, min_periods=20).mean()
    result['EMA_12'] = close.ewm(span=12, adjust=False).mean()

    ema_26 = close.ewm(span=26, adjust=False).mean()
    result['MACD'] = result['EMA_12'] - ema_26

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain.div(avg_loss).replace([np.inf, -np.inf], np.nan)
    result['RSI_14'] = 100 - (100 / (1 + rs))
    result.loc[avg_loss == 0, 'RSI_14'] = 100.0
    result.loc[(avg_gain == 0) & (avg_loss != 0), 'RSI_14'] = 0.0

    bb_std = close.rolling(window=20, min_periods=20).std()
    result['BB_Middle'] = result['SMA_20']
    result['BB_Upper'] = result['BB_Middle'] + 2.0 * bb_std
    result['BB_Lower'] = result['BB_Middle'] - 2.0 * bb_std
    result['BB_Squeeze_Width'] = (result['BB_Upper'] - result['BB_Lower']).div(result['SMA_20']).replace([np.inf, -np.inf], np.nan)

    price_5d_low = close.rolling(window=5, min_periods=5).min()
    rsi_5d_low = result['RSI_14'].rolling(window=5, min_periods=5).min()
    rsi_5d_prev_low = result['RSI_14'].shift(1).rolling(window=5, min_periods=5).min()
    result['RSI_Divergence'] = (
        (close == price_5d_low)
        & rsi_5d_low.gt(rsi_5d_prev_low)
    )
    result['RSI_Divergence'] = result['RSI_Divergence'].fillna(False).astype(bool)

    pct_change = close.pct_change().fillna(0.0)
    result['VPT'] = (result['Volume'] * pct_change).cumsum()

    feature_columns = [
        'SMA_10',
        'SMA_20',
        'EMA_12',
        'MACD',
        'RSI_14',
        'BB_Middle',
        'BB_Upper',
        'BB_Lower',
        'BB_Squeeze_Width',
        'VPT',
    ]
    logger.debug('Dropping incomplete rows created by rolling windows')
    result = result.dropna(subset=feature_columns).reset_index(drop=True)

    if result.empty:
        logger.error('Feature dataset is empty after rolling indicator generation')
        raise ValueError('No valid rows remain after feature engineering')

    logger.info('Generated engineered dataset with %d rows and %d columns', len(result), len(result.columns))
    return result


def save_processed_csv(df: pd.DataFrame, input_path: Path, output_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    """Save the cleaned dataframe to the processed data directory."""
    ensure_directory(output_dir)
    output_file = output_dir / input_path.name
    df.to_csv(output_file, index=False)
    logger.info('Saved processed CSV to %s', output_file)
    return output_file


def process_stock_file(input_path: Path, output_dir: Path = DEFAULT_PROCESSED_DIR) -> Path:
    """Load, clean, and save a single stock CSV file."""
    try:
        raw_df = load_stock_csv(input_path)
        processed_df = clean_dataframe(raw_df)
        return save_processed_csv(processed_df, input_path, output_dir)
    except Exception as error:
        logger.exception('Failed to process %s: %s', input_path, error)
        raise


def process_all_raw_files(raw_dir: Path = Path('data') / 'raw', output_dir: Path = DEFAULT_PROCESSED_DIR) -> dict[str, Path]:
    """Process every CSV file found in the raw data directory."""
    logger.info('Processing raw files from %s', raw_dir)
    results: dict[str, Path] = {}
    ensure_directory(output_dir)

    for csv_file in sorted(raw_dir.glob('*.csv')):
        logger.info('Processing file: %s', csv_file)
        try:
            saved_file = process_stock_file(csv_file, output_dir=output_dir)
            results[csv_file.name] = saved_file
        except Exception:
            logger.warning('Skipping file after processing failure: %s', csv_file)

    return results


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    processed = process_all_raw_files()
    print('Processed files:')
    for name, path in processed.items():
        print(name, path)
