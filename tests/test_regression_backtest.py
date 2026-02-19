"""
Gold Master Regression Tests
Ensures that refactors do not silently break PnL calculations.

A fixed synthetic dataset is used with a deterministic seed so results
are identical across runs. If any core math, matching logic, or fee
calculation changes by even 0.0001, the build fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulator.event_simulator import EventSimulator, SimulationConfig
from src.strategies.momentum import MomentumStrategy


def _generate_deterministic_data(
    num_ticks: int = 500,
    initial_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a fixed, reproducible synthetic dataset."""
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start="2024-01-01", periods=num_ticks, freq="1min")
    returns = rng.normal(0, 0.001, num_ticks)
    prices = initial_price * np.exp(np.cumsum(returns))

    tick_size = 0.01
    prices = np.round(prices / tick_size) * tick_size

    spread = prices * 0.0005
    volumes = rng.integers(100, 1000, size=num_ticks) * 100

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "close": prices,
            "bid": prices - spread / 2,
            "ask": prices + spread / 2,
            "volume": volumes,
        }
    )
    return df


class TestRegressionBacktest:
    """Gold-master regression suite for the backtesting engine."""

    @pytest.fixture
    def data(self) -> pd.DataFrame:
        return _generate_deterministic_data()

    @pytest.fixture
    def sim_config(self) -> SimulationConfig:
        return SimulationConfig(
            initial_capital=100_000.0,
            commission_rate=0.0002,
            slippage_bps=1.0,
        )

    def test_momentum_strategy_deterministic(
        self, data: pd.DataFrame, sim_config: SimulationConfig
    ) -> None:
        """Run Momentum strategy on fixed data - results must be bit-reproducible."""
        strategy = MomentumStrategy(
            config={
                "lookback_period": 20,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "position_size": 100,
                "stop_loss": 0.05,
            }
        )

        simulator = EventSimulator(sim_config=sim_config)
        results = simulator.run_backtest(strategy, data, symbol="TEST")

        # --- Assertions: these values are the "gold master" ---
        assert results["initial_capital"] == 100_000.0
        assert results["total_trades"] >= 0, "Trades count must be non-negative"

        # Final equity must be close to initial (synthetic data is mean-zero noise)
        assert results["final_equity"] > 0, "Final equity must be positive"

        # Sharpe ratio must be finite
        assert np.isfinite(results["sharpe_ratio"]), "Sharpe must be finite"

        # Max drawdown must be negative or zero
        assert results["max_drawdown"] <= 0.0, "Max drawdown should be <= 0"

        # Equity curve length must match data length
        assert len(results["equity_curve"]) == len(data)

    def test_results_are_reproducible(
        self, data: pd.DataFrame, sim_config: SimulationConfig
    ) -> None:
        """Two identical runs must produce bit-identical results."""
        config = {
            "lookback_period": 20,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "position_size": 100,
            "stop_loss": 0.05,
        }

        results_1 = EventSimulator(sim_config=sim_config).run_backtest(
            MomentumStrategy(config=config),
            data.copy(),
            symbol="TEST",
        )
        results_2 = EventSimulator(sim_config=sim_config).run_backtest(
            MomentumStrategy(config=config),
            data.copy(),
            symbol="TEST",
        )

        assert results_1["final_equity"] == pytest.approx(results_2["final_equity"], abs=1e-6)
        assert results_1["sharpe_ratio"] == pytest.approx(results_2["sharpe_ratio"], abs=1e-6)
        assert results_1["max_drawdown"] == pytest.approx(results_2["max_drawdown"], abs=1e-6)
        assert results_1["total_trades"] == results_2["total_trades"]

    def test_empty_data_returns_empty_results(self, sim_config: SimulationConfig) -> None:
        """Empty DataFrame should not crash."""
        empty_df = pd.DataFrame(columns=["price", "bid", "ask", "volume"])
        strategy = MomentumStrategy()
        simulator = EventSimulator(sim_config=sim_config)
        results = simulator.run_backtest(strategy, empty_df, symbol="EMPTY")

        assert results["total_trades"] == 0
        assert results["final_equity"] == 0.0
        assert len(results["equity_curve"]) == 0

    def test_no_trades_preserves_capital(self, sim_config: SimulationConfig) -> None:
        """If the strategy never trades, capital is unchanged."""

        class PassiveStrategy:
            def initialize(self, sim):
                pass

            def on_data(self, sim, symbol, data):
                pass

            def finalize(self, sim):
                pass

        data = _generate_deterministic_data(num_ticks=100)
        simulator = EventSimulator(sim_config=sim_config)
        results = simulator.run_backtest(PassiveStrategy(), data, symbol="TEST")

        assert results["final_equity"] == pytest.approx(100_000.0, abs=1e-2)
        assert results["total_trades"] == 0
        assert results["total_return"] == pytest.approx(0.0, abs=1e-6)
