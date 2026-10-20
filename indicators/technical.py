"""Technical indicator calculations for stock analysis."""
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Compute a simple moving average over a rolling window."""
    return series.rolling(window=window, min_periods=1).mean()


def relative_strength_index(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute a basic RSI indicator for a price series."""
    delta = series.diff().fillna(0)
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss.replace(0, 1e-8)
    return 100 - (100 / (1 + rs))
