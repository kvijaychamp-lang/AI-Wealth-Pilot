"""Historical stock data loader for NSE symbols."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging

import pandas as pd
import yfinance as yf

from utils.constants import DATA_DIR, DATE_FORMAT

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Load and cache historical stock data for analysis and training."""

    def __init__(self, symbols: list[str], start_date: str, end_date: str) -> None:
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self._cache_dir = DATA_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._raw_dir = DATA_DIR / 'raw'

    def fetch(self) -> dict[str, pd.DataFrame]:
        """Download historical data for all configured symbols."""
        result: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            result[symbol] = self._fetch_symbol(symbol)
        return result

    def _fetch_symbol(self, symbol: str) -> pd.DataFrame:
        file_path = self._cache_dir / f'{symbol.replace(".", "_")}.csv'

        # Validate cache: must exist, have real rows, and be recent (<= 3 calendar days old)
        if file_path.exists():
            try:
                cached = pd.read_csv(file_path, parse_dates=['Date'])
                if not cached.empty and len(cached) >= 10:
                    last_date = pd.to_datetime(cached['Date']).max()
                    age_days = (pd.Timestamp.now() - last_date).days
                    if age_days <= 3:
                        logger.info(
                            'Cache hit for %s (%d rows, last date %s)',
                            symbol, len(cached), last_date.date(),
                        )
                        return cached
                    logger.info(
                        'Cache for %s is %d days old — refreshing from yfinance',
                        symbol, age_days,
                    )
                else:
                    logger.warning(
                        'Cache for %s is empty or too small (%d rows) — re-fetching',
                        symbol, len(cached),
                    )
            except Exception as exc:
                logger.warning('Could not read cache for %s: %s — re-fetching', symbol, exc)
            file_path.unlink(missing_ok=True)

        # Download fresh from yfinance
        data = self._download_history(symbol)
        if data.empty:
            raise ValueError(
                f'No historical data found for {symbol}. '
                'Check your internet connection or that the symbol is valid (e.g. RELIANCE.NS).'
            )

        # Persist to cache for next load
        data.to_csv(file_path, index=False)
        logger.info('Saved %d rows to cache for %s', len(data), symbol)
        return data

    def _download_history(self, symbol: str) -> pd.DataFrame:
        """Download daily OHLCV history via yfinance with fallback to yf.download."""
        logger.info(
            'Downloading historical data for %s (%s to %s)',
            symbol, self.start_date, self.end_date,
        )
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=self.start_date, end=self.end_date, auto_adjust=True)
            if not data.empty:
                return self._normalise_columns(data)
        except Exception as exc:
            logger.warning('ticker.history() failed for %s: %s', symbol, exc)

        logger.warning('Falling back to yf.download() for %s', symbol)
        try:
            data = yf.download(
                symbol,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True,
            )
            if not data.empty:
                return self._normalise_columns(data)
        except Exception as exc:
            logger.warning('yf.download() also failed for %s: %s', symbol, exc)

        return pd.DataFrame()

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DatetimeIndex becomes a Date column; flatten MultiIndex if present."""
        result = df.copy()

        # Flatten MultiIndex columns produced by yf.download (field, ticker)
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [
                str(col[0]) if isinstance(col, tuple) else str(col)
                for col in result.columns
            ]

        # Promote DatetimeIndex to a plain column
        result = result.reset_index()
        for alias in ('Datetime', 'datetime', 'index'):
            if alias in result.columns and 'Date' not in result.columns:
                result = result.rename(columns={alias: 'Date'})

        # Normalise casing
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume', 'date': 'Date',
        }
        for old, new in rename_map.items():
            if old in result.columns and new not in result.columns:
                result = result.rename(columns={old: new})

        # Strip timezone so downstream tz-naive comparisons work cleanly
        if 'Date' in result.columns:
            result['Date'] = pd.to_datetime(result['Date']).dt.tz_localize(None)

        return result.dropna(subset=['Date', 'Close']).reset_index(drop=True)

    def _load_local_csv(self, symbol: str) -> pd.DataFrame | None:
        """Load a non-empty local CSV from raw directory if available."""
        candidates = [
            self._raw_dir / f'{symbol.replace(".", "_")}.csv',
            Path('data') / f'{symbol.replace(".", "_")}.csv',
        ]
        for path in candidates:
            if path.exists():
                data = pd.read_csv(path, parse_dates=['Date'])
                if not data.empty:
                    logger.info('Loaded local CSV for %s from %s', symbol, path)
                    return data
                logger.warning('Local CSV %s exists but is empty', path)
        return None

    @staticmethod
    def default_date_range(days: int = 90) -> tuple[str, str]:
        now = datetime.now()
        return (
            (now.replace(hour=0, minute=0, second=0) - pd.Timedelta(days=days)).strftime(DATE_FORMAT),
            now.strftime(DATE_FORMAT),
        )
