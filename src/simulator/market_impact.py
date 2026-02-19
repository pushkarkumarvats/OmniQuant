"""
Market Impact Models for Next-Generation Backtesting

Implements institutional-grade transaction cost analysis (TCA) and
market-impact models so that backtests account for the *feedback effect*
of a strategy's own trading on prices.

Models implemented:
  1. **Almgren-Chriss (2001)** – Optimal execution with permanent +
     temporary impact.  The gold standard for VWAP/TWAP scheduling.
  2. **Square-Root Impact** – Empirical √(Q/V) model used in
     production TCA engines.
  3. **Propagator (Bouchaud)** – Transient impact kernel that decays
     over time, for realistic intraday simulation.

All models expose a common interface:  ``compute_impact(order) -> ImpactResult``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------

class ImpactSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class MarketState:
    """Snapshot of market conditions at the time of execution."""
    mid_price: float
    spread: float
    daily_volume: float          # ADV
    daily_volatility: float      # σ (annualised)
    intraday_volume_profile: Optional[np.ndarray] = None  # fractional volume per bucket
    tick_size: float = 0.01
    bid_depth: float = 0.0       # total bid-side depth (shares)
    ask_depth: float = 0.0


@dataclass
class ImpactResult:
    """Output of an impact model computation."""
    permanent_impact_bps: float
    temporary_impact_bps: float
    total_impact_bps: float
    execution_cost_usd: float    # dollar TCA
    optimal_schedule: Optional[np.ndarray] = None  # shares per time bucket
    arrival_price: float = 0.0
    expected_avg_price: float = 0.0
    slippage_bps: float = 0.0
    variance_cost: float = 0.0   # execution risk (Almgren-Chriss)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Almgren-Chriss Optimal Execution
# ---------------------------------------------------------------------------

class AlmgrenChrissModel:
    """
    Almgren-Chriss (2001) model for optimal trade execution.

    Given a parent order of *X* shares to execute over *T* periods,
    the model finds the trading trajectory that minimises the sum of
    execution cost and execution risk:

        min_x  E[cost(x)] + λ · Var[cost(x)]

    where λ is the risk-aversion parameter.

    Parameters
    ----------
    eta : float
        Temporary impact coefficient (market-order slippage).
    gamma : float
        Permanent impact coefficient (price drift per share).
    sigma : float
        Per-period return volatility.
    risk_aversion : float
        λ – trade-off between expected cost and variance.

    Reference
    ---------
    Almgren, R. & Chriss, N. (2001).  "Optimal execution of portfolio
    transactions".  *J. Risk*, 3(2), 5–39.
    """

    def __init__(
        self,
        eta: float = 2.5e-6,
        gamma: float = 2.5e-7,
        sigma: float = 0.02,
        risk_aversion: float = 1e-6,
    ) -> None:
        self.eta = eta
        self.gamma = gamma
        self.sigma = sigma
        self.risk_aversion = risk_aversion

    def compute_impact(
        self,
        total_shares: int,
        n_periods: int,
        market: MarketState,
        side: ImpactSide = ImpactSide.BUY,
    ) -> ImpactResult:
        """
        Compute the optimal execution schedule and expected costs.

        Parameters
        ----------
        total_shares : int
            Total shares to execute (always positive; sign from *side*).
        n_periods : int
            Number of discrete time buckets.
        market : MarketState
            Current market conditions.
        side : ImpactSide
            BUY or SELL.

        Returns
        -------
        ImpactResult
            Optimal schedule and cost breakdown.
        """
        X = abs(total_shares)
        T = max(n_periods, 1)
        tau = 1.0 / T  # fraction of horizon per period

        sigma = market.daily_volatility / math.sqrt(252) if market.daily_volatility > 0 else self.sigma
        eta = self.eta
        gamma = self.gamma
        lam = self.risk_aversion

        # Almgren-Chriss κ (characteristic trading rate)
        kappa_sq = lam * sigma ** 2 / (eta * (1 / tau))
        kappa = math.sqrt(max(kappa_sq, 1e-12))

        # Optimal holdings trajectory  x_j = X · sinh(κ(T−j)) / sinh(κT)
        schedule = np.zeros(T)
        holdings = np.zeros(T + 1)
        sinh_kT = math.sinh(kappa * T) if kappa * T < 500 else math.exp(kappa * T) / 2

        for j in range(T + 1):
            holdings[j] = X * math.sinh(kappa * (T - j)) / sinh_kT if sinh_kT != 0 else X * (1 - j / T)
        schedule = -np.diff(holdings)  # shares traded per period (positive)

        # Expected cost components
        permanent_cost = 0.5 * gamma * X ** 2
        temp_cost = eta * np.sum(schedule ** 2 / tau)
        total_cost = permanent_cost + temp_cost

        # Execution risk
        variance = sigma ** 2 * tau * np.sum(holdings[:-1] ** 2)
        utility = total_cost + lam * variance

        # Convert to basis points
        notional = X * market.mid_price
        perm_bps = (permanent_cost / notional * 10_000) if notional > 0 else 0.0
        temp_bps = (temp_cost / notional * 10_000) if notional > 0 else 0.0

        sign = 1.0 if side == ImpactSide.BUY else -1.0
        expected_avg = market.mid_price * (1 + sign * (perm_bps + temp_bps) / 10_000)

        return ImpactResult(
            permanent_impact_bps=perm_bps,
            temporary_impact_bps=temp_bps,
            total_impact_bps=perm_bps + temp_bps,
            execution_cost_usd=total_cost,
            optimal_schedule=schedule,
            arrival_price=market.mid_price,
            expected_avg_price=expected_avg,
            slippage_bps=perm_bps + temp_bps,
            variance_cost=variance,
            metadata={
                "kappa": kappa,
                "utility": utility,
                "eta": eta,
                "gamma": gamma,
                "risk_aversion": lam,
            },
        )

    def adapt_parameters(self, market: MarketState) -> None:
        """
        Adapt η and γ to current market conditions using empirical
        scaling rules.

        - η ∝ σ / √(ADV)
        - γ ∝ σ / ADV
        """
        if market.daily_volume > 0 and market.daily_volatility > 0:
            vol = market.daily_volatility / math.sqrt(252)
            self.eta = vol / math.sqrt(market.daily_volume) * market.mid_price
            self.gamma = vol / market.daily_volume * market.mid_price * 0.1
            logger.debug(f"Adapted AC params: η={self.eta:.2e}, γ={self.gamma:.2e}")


# ---------------------------------------------------------------------------
# 2. Square-Root Impact Model
# ---------------------------------------------------------------------------

class SquareRootImpactModel:
    """
    Empirical square-root impact model:

        ΔP/P = α · σ · √(Q / V)

    where *α* is a universal constant (~1.0), *σ* daily volatility,
    *Q* order size, and *V* average daily volume.

    This model is used by most production TCA systems.
    """

    def __init__(self, alpha: float = 1.0, spread_fraction: float = 0.5) -> None:
        self.alpha = alpha
        self.spread_fraction = spread_fraction  # fraction of spread paid

    def compute_impact(
        self,
        total_shares: int,
        market: MarketState,
        side: ImpactSide = ImpactSide.BUY,
    ) -> ImpactResult:
        Q = abs(total_shares)
        V = max(market.daily_volume, 1)
        sigma = market.daily_volatility / math.sqrt(252) if market.daily_volatility > 0 else 0.01
        participation = Q / V

        # Impact in fractional price
        impact_frac = self.alpha * sigma * math.sqrt(participation)
        impact_bps = impact_frac * 10_000

        # Half spread
        half_spread_bps = (market.spread / 2 / market.mid_price * 10_000) if market.mid_price > 0 else 0
        spread_cost_bps = half_spread_bps * self.spread_fraction

        total_bps = impact_bps + spread_cost_bps
        notional = Q * market.mid_price
        cost_usd = notional * total_bps / 10_000

        sign = 1.0 if side == ImpactSide.BUY else -1.0
        expected_avg = market.mid_price * (1 + sign * total_bps / 10_000)

        return ImpactResult(
            permanent_impact_bps=impact_bps * 0.5,  # roughly half is permanent
            temporary_impact_bps=impact_bps * 0.5 + spread_cost_bps,
            total_impact_bps=total_bps,
            execution_cost_usd=cost_usd,
            arrival_price=market.mid_price,
            expected_avg_price=expected_avg,
            slippage_bps=total_bps,
            metadata={"participation_rate": participation, "alpha": self.alpha},
        )


# ---------------------------------------------------------------------------
# 3. Transient Impact (Propagator / Bouchaud)
# ---------------------------------------------------------------------------

class TransientImpactModel:
    """
    Bouchaud propagator model with power-law decay:

        G(t) = G_0 · t^{-β}

    Impact from a trade at time *t* decays as a power law.
    Typical β ≈ 0.5 (square-root decay), G_0 calibrated to market.

    This gives much more realistic intraday price paths than
    permanent-only models.
    """

    def __init__(self, g0: float = 1e-5, beta: float = 0.5, n_decay_steps: int = 100) -> None:
        self.g0 = g0
        self.beta = beta
        self.n_decay_steps = n_decay_steps

    def kernel(self, t: int) -> float:
        """Propagator kernel G(t)."""
        if t <= 0:
            return self.g0
        return self.g0 * (t ** (-self.beta))

    def simulate_path(
        self,
        trade_schedule: np.ndarray,
        initial_price: float,
        volatility: float = 0.01,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate a price path with transient impact and noise.

        Parameters
        ----------
        trade_schedule : np.ndarray
            Signed trade sizes per time step (positive = buy).
        initial_price : float
            Starting mid-price.
        volatility : float
            Per-step return standard deviation.
        seed : optional int
            RNG seed for reproducibility.

        Returns
        -------
        np.ndarray
            Simulated price path (length = len(trade_schedule) + 1).
        """
        rng = np.random.default_rng(seed)
        n = len(trade_schedule)
        prices = np.zeros(n + 1)
        prices[0] = initial_price

        for t in range(n):
            # Impact from all past trades
            impact = 0.0
            for s in range(t + 1):
                lag = t - s + 1
                impact += self.kernel(lag) * trade_schedule[s]
            noise = volatility * rng.standard_normal()
            prices[t + 1] = prices[t] * (1 + impact + noise)

        return prices

    def compute_impact(
        self,
        total_shares: int,
        n_periods: int,
        market: MarketState,
        side: ImpactSide = ImpactSide.BUY,
    ) -> ImpactResult:
        # Uniform schedule for simplicity
        schedule = np.full(n_periods, total_shares / n_periods)
        path = self.simulate_path(
            schedule,
            market.mid_price,
            market.daily_volatility / math.sqrt(252 * n_periods) if market.daily_volatility > 0 else 0.01,
        )
        avg_exec_price = np.mean(path[1:])
        slippage = (avg_exec_price - market.mid_price) / market.mid_price * 10_000
        if side == ImpactSide.SELL:
            slippage = -slippage
        cost = abs(slippage / 10_000 * total_shares * market.mid_price)
        return ImpactResult(
            permanent_impact_bps=abs(slippage) * 0.3,
            temporary_impact_bps=abs(slippage) * 0.7,
            total_impact_bps=abs(slippage),
            execution_cost_usd=cost,
            optimal_schedule=schedule,
            arrival_price=market.mid_price,
            expected_avg_price=avg_exec_price,
            slippage_bps=abs(slippage),
            metadata={"g0": self.g0, "beta": self.beta},
        )


# ---------------------------------------------------------------------------
# Adversarial Simulation - Predatory HFT Agents
# ---------------------------------------------------------------------------

class PredatoryAgent:
    """
    Simulates predatory HFT behavior that front-runs large orders.

    When a large order is detected (via information leakage / pattern
    detection), the predatory agent:
    1. Front-runs: trades ahead in the same direction
    2. Pushes the price adversely
    3. Liquidates once the large order has moved the market

    This creates a hostile execution environment for realistic
    backtesting of execution algorithms.
    """

    def __init__(
        self,
        detection_threshold: float = 0.01,   # fraction of ADV
        front_run_size: float = 0.002,        # fraction of ADV
        aggression: float = 0.5,              # how much to worsen the price
        decay_period: int = 10,               # periods to unwind
    ) -> None:
        self.detection_threshold = detection_threshold
        self.front_run_size = front_run_size
        self.aggression = aggression
        self.decay_period = decay_period
        self._accumulated_flow = 0.0
        self._position = 0.0
        self._front_run_trades: List[float] = []

    def observe_trade(
        self, signed_qty: float, adv: float
    ) -> float:
        """
        Observe a trade from the target strategy.

        Returns the predatory agent's own trade (same direction)
        or zero if the trade is below the detection threshold.
        """
        participation = abs(signed_qty) / max(adv, 1)
        self._accumulated_flow += signed_qty

        if participation >= self.detection_threshold:
            fr_qty = signed_qty * self.front_run_size / self.detection_threshold
            fr_qty *= self.aggression
            self._position += fr_qty
            self._front_run_trades.append(fr_qty)
            return fr_qty
        return 0.0

    def unwind(self) -> List[float]:
        """
        Generate unwind trades (opposite direction) to capture profit.
        """
        if abs(self._position) < 1:
            return []
        per_period = -self._position / self.decay_period
        unwinds = [per_period] * self.decay_period
        self._position = 0.0
        return unwinds

    def reset(self) -> None:
        self._accumulated_flow = 0.0
        self._position = 0.0
        self._front_run_trades.clear()


class AdversarialSimulator:
    """
    Wraps a standard backtest with adversarial agents that degrade
    execution quality, providing a stress-test for alpha decay and
    execution algorithms.
    """

    def __init__(
        self,
        n_predators: int = 3,
        noise_agents: int = 10,
        detection_threshold: float = 0.01,
    ) -> None:
        self.predators = [
            PredatoryAgent(
                detection_threshold=detection_threshold * (1 + 0.2 * i),
                aggression=0.3 + 0.2 * i,
            )
            for i in range(n_predators)
        ]
        self.noise_agents = noise_agents
        self._rng = np.random.default_rng(42)

    def apply_adversarial_impact(
        self,
        trade_qty: float,
        market: MarketState,
    ) -> Tuple[float, float]:
        """
        Apply adversarial pressure to a trade.

        Returns
        -------
        (additional_impact_bps, total_adversarial_volume)
        """
        adv = market.daily_volume
        total_predatory = 0.0
        for pred in self.predators:
            fr = pred.observe_trade(trade_qty, adv)
            total_predatory += abs(fr)

        # Noise agent volume
        noise_vol = sum(
            abs(self._rng.normal(0, adv * 0.001))
            for _ in range(self.noise_agents)
        )

        # Adversarial impact
        if adv > 0:
            extra_impact = (
                market.daily_volatility / math.sqrt(252)
                * math.sqrt(total_predatory / adv)
                * 10_000
            )
        else:
            extra_impact = 0.0

        return extra_impact, total_predatory + noise_vol

    def get_unwind_pressure(self) -> float:
        """Get the total unwind volume from predators."""
        total = 0.0
        for pred in self.predators:
            for qty in pred.unwind():
                total += abs(qty)
        return total

    def reset(self) -> None:
        for p in self.predators:
            p.reset()


__all__ = [
    "ImpactSide",
    "MarketState",
    "ImpactResult",
    "AlmgrenChrissModel",
    "SquareRootImpactModel",
    "TransientImpactModel",
    "PredatoryAgent",
    "AdversarialSimulator",
]
