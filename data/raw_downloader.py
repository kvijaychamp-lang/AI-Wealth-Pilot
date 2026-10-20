"""Download raw historical NSE stock data with retry logic and CSV export."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf
from requests.exceptions import RequestException
from tqdm import tqdm

logger = logging.getLogger(__name__)
DEFAULT_RAW_DIR = Path('data') / 'raw'


def ensure_directory(path: Path) -> None:
    """Create the target directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def fetch_stock_history(
    symbol: str,
    period_years: int = 5,
    interval: str = '1d',
    max_retries: int = 3,
    backoff_seconds: int = 5,
) -> pd.DataFrame:
    """Download OHLCV history for a single NSE symbol, retrying on failure."""
    attempt = 0
    while True:
        attempt += 1
        try:
            logger.info('Downloading %s (attempt %d)', symbol, attempt)
            raw_data = yf.download(symbol, period=f'{period_years}y', interval=interval, progress=False)
            if raw_data.empty:
                raise ValueError('Received empty data frame from yfinance')

            data = raw_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            data.index.name = 'Date'
            data.reset_index(inplace=True)
            return data

        except RequestException as error:
            logger.warning(
                'Network error while downloading %s: %s', symbol, error,
            )
        except Exception as error:
            logger.warning(
                'Failed to download %s on attempt %d: %s', symbol, attempt, error,
            )

        if attempt >= max_retries:
            logger.error('Giving up after %d attempts for %s', attempt, symbol)
            raise

        sleep_time = backoff_seconds * attempt
        logger.info('Retrying %s in %d seconds', symbol, sleep_time)
        time.sleep(sleep_time)


def save_stock_csv(data: pd.DataFrame, symbol: str, output_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Save downloaded stock data to a CSV file under the raw data directory."""
    ensure_directory(output_dir)
    file_name = f'{symbol.replace(".", "_")}.csv'
    output_path = output_dir / file_name
    data.to_csv(output_path, index=False)
    logger.info('Saved %s to %s', symbol, output_path)
    return output_path


def download_nse_historical_data(
    symbols: Iterable[str],
    period_years: int = 5,
    interval: str = '1d',
    output_dir: Path = DEFAULT_RAW_DIR,
    max_retries: int = 3,
    backoff_seconds: int = 5,
) -> dict[str, Path]:
    """Download historical data for multiple NSE symbols and save each as CSV."""
    ensure_directory(output_dir)
    results: dict[str, Path] = {}

    for symbol in tqdm(symbols, desc='Downloading NSE data', unit='symbol'):
        try:
            data_frame = fetch_stock_history(
                symbol,
                period_years=period_years,
                interval=interval,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )
            results[symbol] = save_stock_csv(data_frame, symbol, output_dir)
        except Exception:
            logger.exception('Skipping %s after repeated download failures', symbol)

    return results


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    example_symbols = [
        'RELIANCE.NS',
        'TCS.NS',
        'INFY.NS',
        'HDFCBANK.NS',
        'ICICIBANK.NS',
    ]

    downloaded = download_nse_historical_data(example_symbols)
    print('Downloaded files:')
    for symbol, path in downloaded.items():
        print(symbol, path)
