"""Trend Core-Satellite strategy for FinRL-X.

Core   : SPY, scaled by a 200-day trend filter (full weight above the MA, a floor below it).
Satellite: GLD / IEF / TLT / DBC, each held only above its own 200-day MA,
           inverse-volatility weighted, rebalanced weekly (decided on the first trading day of each week).
Overlay: portfolio volatility target with a no-trade band. Remainder goes to BIL (T-bills).

The weight history is replayed deterministically from prices only, so the live signal for a
date is identical to the backtest signal for that date (no hidden state, no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult


@dataclass
class TrendCoreSatelliteConfig(StrategyConfig):
    name: str = "TrendCoreSatellite"
    core_symbol: str = "SPY"
    satellite_symbols: List[str] = field(default_factory=lambda: ["GLD", "IEF", "TLT", "DBC"])
    cash_symbol: str = "BIL"
    ma_window: int = 200
    core_weight: float = 0.60
    core_floor: float = 0.30          # fraction of core kept when SPY is below its MA
    vol_lookback: int = 60            # days, for inverse-vol satellite weights
    vol_target: Optional[float] = 0.12
    vol_target_lookback: int = 20
    vol_band: float = 0.10            # only change the vol scaler when it moves more than this
    cash_buffer: float = 0.01         # kept as USD cash for fees/rounding

    @property
    def risky_symbols(self) -> List[str]:
        return [self.core_symbol] + list(self.satellite_symbols)


def _week_start_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True on the first trading day of each week. Knowable in real time (unlike 'last day of week')."""
    per = pd.Series(index.to_period("W"), index=index)
    return per != per.shift(1)


class TrendCoreSatelliteStrategy(BaseStrategy):
    def __init__(self, config: Optional[TrendCoreSatelliteConfig] = None):
        super().__init__(config or TrendCoreSatelliteConfig())
        self.cfg: TrendCoreSatelliteConfig = self.config  # type: ignore[assignment]

    # ------------------------------------------------------------------ core logic
    def weight_history(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Target weights decided at each date's close (to be traded on the next session)."""
        c = self.cfg
        px = prices[c.risky_symbols].sort_index().ffill()
        rets = px.pct_change(fill_method=None).fillna(0.0)
        ma = px.rolling(c.ma_window).mean()
        vol = rets.rolling(c.vol_lookback).std() * np.sqrt(252)

        above_core = (px[c.core_symbol] > ma[c.core_symbol]).astype(float)
        core_w = (above_core * (1 - c.core_floor) + c.core_floor) * c.core_weight

        sat = list(c.satellite_symbols)
        inv = 1.0 / vol[sat]
        on = (px[sat] > ma[sat]).astype(float)
        sat_w = (inv * on).div(inv.sum(axis=1), axis=0) * (1 - c.core_weight)
        sat_w = sat_w.where(_week_start_mask(px.index)).ffill()

        w = pd.concat([core_w.rename(c.core_symbol), sat_w], axis=1)
        valid = ma.notna().all(axis=1) & vol.notna().all(axis=1)
        w = w.where(valid).fillna(0.0)

        if c.vol_target:
            held = w.shift(1).fillna(0.0)                 # weights actually earning day t
            port = (held * rets[w.columns]).sum(axis=1)
            rv = port.rolling(c.vol_target_lookback).std() * np.sqrt(252)
            raw = np.minimum(1.0, c.vol_target / rv).fillna(1.0).values
            scale = np.empty_like(raw)
            last = 1.0
            for i, v in enumerate(raw):
                if abs(v - last) > c.vol_band:
                    last = v
                scale[i] = last
            w = w.mul(scale, axis=0)

        w = w * (1 - c.cash_buffer)
        w[c.cash_symbol] = (1 - c.cash_buffer) - w.sum(axis=1)
        return w.where(valid).dropna()

    # ------------------------------------------------------------------ FinRL-X API
    def generate_weights(self, data: Dict[str, pd.DataFrame], target_date: Optional[str] = None) -> StrategyResult:
        prices = data["prices"]
        if target_date is not None:
            prices = prices.loc[: pd.Timestamp(target_date)]
        hist = self.weight_history(prices)
        if hist.empty:
            raise ValueError("Not enough price history to compute weights")
        last = hist.iloc[-1]
        df = pd.DataFrame({"symbol": last.index, "weight": last.values})
        df = df[df.weight > 1e-6].reset_index(drop=True)
        return StrategyResult(
            strategy_name=self.cfg.name,
            weights=df,
            metadata={"signal_date": str(hist.index[-1].date()),
                      "spy_above_ma": bool(prices[self.cfg.core_symbol].iloc[-1]
                                           > prices[self.cfg.core_symbol].rolling(self.cfg.ma_window).mean().iloc[-1])},
        )
