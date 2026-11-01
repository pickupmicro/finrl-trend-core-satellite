"""Reproduce the strategy backtest. Usage: python -m scripts.backtest --start 2008-01-01"""
import argparse
import numpy as np
import pandas as pd
from scripts.data import load_prices
from src.strategies.trend_core_satellite import TrendCoreSatelliteStrategy, TrendCoreSatelliteConfig


def run(start="2008-01-01", cost=0.001):
    cfg = TrendCoreSatelliteConfig()
    px = load_prices(cfg.risky_symbols + [cfg.cash_symbol, "SHY"], start="2005-01-01")
    w = TrendCoreSatelliteStrategy(cfg).weight_history(px)
    rets = px.pct_change(fill_method=None)
    rets[cfg.cash_symbol] = rets[cfg.cash_symbol].fillna(rets["SHY"])  # BIL starts 2007-05
    rets = rets.fillna(0.0)
    held = w.shift(1).reindex(rets.index).fillna(0.0)   # decided at close t, earns t+1
    gross = (held * rets[held.columns]).sum(axis=1)
    turn = held.diff().abs().sum(axis=1).fillna(0.0)
    r = (gross - cost * turn).loc[start:]
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    n = len(r) / 252
    print(f"CAGR {eq.iloc[-1] ** (1 / n) - 1:.2%}  Sharpe {r.mean() / r.std() * np.sqrt(252):.2f}  "
          f"Sortino {r.mean() * 252 / (r[r < 0].std() * np.sqrt(252)):.2f}  MDD {dd.min():.2%}")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2008-01-01")
    ap.add_argument("--cost", type=float, default=0.001)
    a = ap.parse_args()
    run(a.start, a.cost).to_csv("results/backtest_returns.csv")
