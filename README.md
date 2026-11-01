# Trend Core-Satellite on FinRL-X — AI4Finance Open Trading Challenge

A deliberately simple, robust multi-asset trend strategy built on the
[FinRL-X / FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading) framework
(weight-centric `BaseStrategy.generate_weights()` + `AlpacaManager` execution), trading an Alpaca Paper account
fully automatically through GitHub Actions. No manual intervention during the live period.

- Strategy & research: [STRATEGY.md](STRATEGY.md)
- Strategy code: [`src/strategies/trend_core_satellite.py`](src/strategies/trend_core_satellite.py)
- Live runner: [`scripts/run_daily.py`](scripts/run_daily.py) · Schedule: [`.github/workflows/trade.yml`](.github/workflows/trade.yml)
- Public logs: [`logs/decisions.csv`](logs/), [`logs/trades.csv`](logs/) · Equity curve: [`results/daily_equity.csv`](results/)
- Change log: [CHANGELOG.md](CHANGELOG.md)

## Portfolio in one table
| Sleeve | Assets | Rule |
|---|---|---|
| Core 60% | SPY | Full when SPY > 200-day MA, 30% of sleeve when below |
| Satellite 40% | GLD, IEF, TLT, DBC | Only above own 200-day MA, inverse-vol weights, weekly |
| Overlay | all | 12% vol target, no leverage |
| Cash | BIL | remainder |

## Backtest (10 bp costs, next-day execution)
2008-01 → 2026-10: CAGR 7.0%, Sharpe 0.98, max drawdown -12.0% (SPY: 11.1%, 0.63, -51.9%).
All 144 parameter variants tested beat SPY on Sharpe. Past results do not guarantee future performance.

## Run it
```bash
pip install -r requirements-live.txt
python -m pytest -q tests
python -m scripts.backtest --start 2008-01-01
APCA_API_KEY=... APCA_API_SECRET=... python -m scripts.run_daily --dry-run
```

## Credits & license
Framework code © AI4Finance Foundation, Apache-2.0 (see LICENSE and README_FinRL-X.md). Strategy, runner and tests added in this repo.
