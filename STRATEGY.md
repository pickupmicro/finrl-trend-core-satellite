# Trend Core-Satellite (FinRL-X)

## Idea
Stock picking showed no reliable edge in our tests, so this strategy targets the competition's scoring
(Sharpe/Sortino, max drawdown, stability) with a simple, robust trend-following multi-asset portfolio.

| Sleeve | Assets | Rule |
|---|---|---|
| Core 60% | SPY | 100% of the sleeve when SPY > 200-day MA, 30% when below |
| Satellite 40% | GLD, IEF, TLT, DBC | Each held only above its own 200-day MA; inverse 60-day volatility weights; weekly |
| Overlay | whole book | 12% annual volatility target (20-day), no leverage, changes only when the scaler moves > 10% |
| Cash | BIL | Remainder in T-bills, 1% USD buffer |

Signals use daily closes (Yahoo Finance) and are traded the next session at about 09:45 New York via Alpaca Paper.
Live trades happen only when a target changes or drift exceeds 2%.

## Research process
1. A LightGBM cross-sectional stock selector with a heavy risk overlay failed (2016–2026 Sharpe 0.36, IC 0.0085).
2. FinRL-X Adaptive Rotation re-implemented: original symbol list Sharpe 0.93, but an unbiased sector-ETF version only 0.43 → its edge relies on hindsight symbol choice.
3. Multi-asset trend core-satellite selected.

## Backtest (10 bp per unit turnover, 1-day execution lag)
| Period | CAGR | Sharpe | Max DD |
|---|---|---|---|
| 2008-01 – 2026-10 | 7.0% | 0.98 | -12.0% |
| 2016-01 – 2026-10 | 8.3% | 1.17 | -12.0% |
| SPY 2008 – 2026 | 11.1% | 0.63 | -51.9% |

Robustness: all 144 combinations of MA (100–250), core weight (50–70%), floor (0–50%) and vol target (none–15%)
gave Sharpe 0.78–1.00 (2008–2026), every one above SPY; max drawdown -9% to -21%. 2× costs: Sharpe 0.85; 2-day lag: 0.94.

## Known weaknesses
- Lags SPY in strong bull markets (roughly half SPY's return).
- Over 30-day windows, median Sharpe 1.44 vs SPY 1.59.
- Chosen parameters sit in the upper fifth of the grid; expect Sharpe nearer 0.9 out of sample.

## Reproduce
```
pip install -r requirements-live.txt
python -m scripts.backtest --start 2008-01-01
python -m pytest -q tests
python -m scripts.run_daily --dry-run   # needs APCA_* paper keys
```
Deterministic: no randomness in the strategy; tests use seed 42.
