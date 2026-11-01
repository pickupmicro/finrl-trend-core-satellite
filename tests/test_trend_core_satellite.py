import numpy as np
import pandas as pd
import pytest

from src.strategies.trend_core_satellite import TrendCoreSatelliteConfig, TrendCoreSatelliteStrategy


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2018-01-01", periods=800)
    cols = ["SPY", "GLD", "IEF", "TLT", "DBC", "BIL"]
    r = rng.normal(0.0003, 0.01, size=(len(idx), len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)


def test_weights_are_valid(prices):
    w = TrendCoreSatelliteStrategy().weight_history(prices)
    assert (w >= -1e-9).all().all()
    assert np.allclose(w.sum(axis=1), 1 - TrendCoreSatelliteConfig().cash_buffer)
    assert (w["SPY"] <= 0.6 + 1e-9).all()


def test_no_lookahead(prices):
    s = TrendCoreSatelliteStrategy()
    cut = prices.index[600]
    short = s.weight_history(prices.loc[:cut]).iloc[-1]
    full = s.weight_history(prices).loc[cut]
    pd.testing.assert_series_equal(short, full, check_names=False)


def test_generate_weights_api(prices):
    res = TrendCoreSatelliteStrategy().generate_weights({"prices": prices}, target_date=str(prices.index[-1].date()))
    assert set(res.weights.columns) == {"symbol", "weight"}
    assert res.weights.weight.sum() <= 1.0


def test_core_floor_when_below_ma(prices):
    p = prices.copy()
    p.iloc[-60:, p.columns.get_loc("SPY")] *= np.linspace(1, 0.6, 60)  # crash SPY
    cfg = TrendCoreSatelliteConfig(vol_target=None)
    w = TrendCoreSatelliteStrategy(cfg).weight_history(p).iloc[-1]
    assert w["SPY"] == pytest.approx(0.6 * 0.3 * (1 - cfg.cash_buffer))
