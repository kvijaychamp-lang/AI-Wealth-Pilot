import sys
sys.path.insert(0, '.')
import pandas as pd
from data.loader import HistoricalDataLoader

symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
start, end = HistoricalDataLoader.default_date_range(days=60)
loader = HistoricalDataLoader(symbols=symbols, start_date=start, end_date=end)
data = loader.fetch()

for sym, df in data.items():
    if df.empty:
        print(f'FAIL  {sym}: empty dataframe')
    else:
        lo = df['Close'].min()
        hi = df['Close'].max()
        last_dt = pd.to_datetime(df['Date']).max().date()
        print(f'OK    {sym}: {len(df)} rows | Close Rs{lo:.2f}-Rs{hi:.2f} | last {last_dt}')

import yfinance as yf
for sym in symbols[:2]:
    t = yf.Ticker(sym)
    h = t.history(period='1d', interval='1m', auto_adjust=True)
    if not h.empty:
        last = h.iloc[-1]
        price = float(last['Close'])
        ts = last.name
        print(f'LIVE  {sym}: Rs{price:.2f} at {ts}')
    else:
        print(f'LIVE  {sym}: no intraday data (market may be closed or after hours)')
