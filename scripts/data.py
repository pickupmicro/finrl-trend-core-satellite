"""Daily close data from Yahoo Finance (Alpaca paper accounts only get IEX data)."""
import pandas as pd
import yfinance as yf


def load_prices(symbols, start="2005-01-01", end=None) -> pd.DataFrame:
    df = yf.download(sorted(set(symbols)), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(symbols[0])
    return df.dropna(how="all")
