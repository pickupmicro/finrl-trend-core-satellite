"""Daily live job: compute target weights, rebalance on Alpaca Paper, log trades and equity.

Usage:
  python -m scripts.run_daily --dry-run     # plan only
  python -m scripts.run_daily               # place orders (paper)

Rebalances only when a target changed or drift exceeds DRIFT_BAND, to keep turnover low.
No manual intervention: every run is driven by GitHub Actions and fully logged.
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.data import load_prices
from src.strategies.trend_core_satellite import TrendCoreSatelliteConfig, TrendCoreSatelliteStrategy

DRIFT_BAND = 0.02
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR, RES_DIR = ROOT / "logs", ROOT / "results"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_daily")


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def with_retry(fn, tries=3, wait=20):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network / API hiccups
            log.warning("attempt %d failed: %s", i + 1, e)
            if i == tries - 1:
                raise
            time.sleep(wait)


def snapshot():
    """After-close equity snapshot for the daily equity curve (no trading)."""
    from src.trading.alpaca_manager import AlpacaManager, create_alpaca_account_from_env
    am = AlpacaManager([create_alpaca_account_from_env()])
    info = with_retry(am.get_account_info)
    append_csv(RES_DIR / "daily_equity.csv", {
        "date_utc": datetime.now(timezone.utc).date().isoformat(),
        "equity": info.get("equity"), "cash": info.get("cash"), "last_equity": info.get("last_equity")})


LIVE_START = os.environ.get("LIVE_START", "2026-11-05")  # first live trading day (New York date)


def main(dry_run: bool):
    from zoneinfo import ZoneInfo
    today_ny = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    if today_ny < LIVE_START and not dry_run:
        log.info("before live start %s (today %s): dry run only", LIVE_START, today_ny)
        dry_run = True
    from src.trading.alpaca_manager import AlpacaManager, create_alpaca_account_from_env

    cfg = TrendCoreSatelliteConfig()
    strat = TrendCoreSatelliteStrategy(cfg)
    px = with_retry(lambda: load_prices(cfg.risky_symbols + [cfg.cash_symbol], start="2023-01-01"))
    res = strat.generate_weights({"prices": px})
    target = {r.symbol: round(float(r.weight), 4) for r in res.weights.itertuples()}
    log.info("signal %s target %s", res.metadata, target)

    account = create_alpaca_account_from_env()
    if "paper" not in account.base_url:
        sys.exit("Refusing to run: APCA_BASE_URL is not a paper endpoint")
    am = AlpacaManager([account])

    info = with_retry(am.get_account_info)
    equity = float(info.get("equity") or info.get("portfolio_value"))
    positions = with_retry(am.get_positions)
    current = {p["symbol"]: float(p["market_value"]) / equity for p in positions}
    drift = max(abs(target.get(s, 0) - current.get(s, 0)) for s in set(target) | set(current))
    last_target_file = RES_DIR / "last_target.json"
    last_target = json.loads(last_target_file.read_text()) if last_target_file.exists() else {}
    changed = any(abs(target.get(s, 0) - last_target.get(s, 0)) > 1e-4 for s in set(target) | set(last_target))

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    action = "hold"
    if changed or drift > DRIFT_BAND:
        action = "dry_run" if dry_run else "rebalance"
        out = with_retry(lambda: am.execute_portfolio_rebalance(target, dry_run=dry_run, market_closed_action="skip"))
        for side in ("sell_orders", "buy_orders", "orders"):
            for o in (out or {}).get(side, []) or []:
                o = o if isinstance(o, dict) else getattr(o, "__dict__", {"order": str(o)})
                append_csv(LOG_DIR / "trades.csv", {
                    "time_utc": now, "mode": action, "symbol": o.get("symbol"), "side": o.get("side"),
                    "qty": o.get("quantity", o.get("qty")), "status": o.get("status"), "order_id": o.get("order_id", o.get("id"))})
        if not dry_run:
            last_target_file.write_text(json.dumps(target, indent=1))

    append_csv(LOG_DIR / "decisions.csv", {
        "time_utc": now, "signal_date": res.metadata["signal_date"], "action": action,
        "max_drift": round(drift, 4), "target": json.dumps(target), "spy_above_ma": res.metadata["spy_above_ma"]})
    # equity curve: one row per run date (last value of the day wins when post-processed)
    append_csv(RES_DIR / "equity_curve.csv", {
        "time_utc": now, "equity": equity, "cash": info.get("cash"), "positions": len(positions)})
    log.info("action=%s drift=%.4f equity=%.2f", action, drift, equity)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--snapshot", action="store_true", help="record end-of-day equity only")
    a = ap.parse_args()
    snapshot() if a.snapshot else main(a.dry_run)
