import yfinance as yf

symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
start = '2026-02-14'
end = '2026-05-15'

for s in symbols:
    try:
        df = yf.Ticker(s).history(start=start, end=end)
        print(s, df.shape)
        if df.empty:
            print('EMPTY for', s)
        else:
            print(df.head(2).to_string())
    except Exception as e:
        print('ERR', s, e)
