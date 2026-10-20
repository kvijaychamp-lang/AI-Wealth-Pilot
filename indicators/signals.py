"""Signal generation based on technical indicators."""
import pandas as pd

from indicators.technical import moving_average, relative_strength_index


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add buy and sell signals to the data frame."""
    df['ma_short'] = moving_average(df['Close'], window=5)
    df['ma_long'] = moving_average(df['Close'], window=20)
    df['rsi'] = relative_strength_index(df['Close'])

    df['signal'] = 0
    df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
    df.loc[df['rsi'] > 70, 'signal'] = -1
    return df
