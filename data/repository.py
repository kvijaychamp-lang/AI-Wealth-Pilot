"""Data repository encapsulates stock data access patterns."""
from typing import Any

import pandas as pd


class StockDataRepository:
    """Provide simple methods to query dataframe data for training and predictions."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self.data = data

    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Return the raw data frame for a given symbol."""
        return self.data.get(symbol, pd.DataFrame())

    def get_latest_close(self, symbol: str) -> float | None:
        """Return the latest closing price for the symbol."""
        df = self.get_symbol_data(symbol)
        if df.empty:
            return None
        return float(df.loc[df.index[-1], 'Close'])

    def add_feature(self, symbol: str, feature_name: str, values: list[Any]) -> None:
        """Attach a new feature column to the symbol dataframe."""
        df = self.get_symbol_data(symbol)
        if not df.empty:
            df[feature_name] = values
