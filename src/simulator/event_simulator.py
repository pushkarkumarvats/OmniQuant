"""
Event-Driven Simulator
High-performance backtesting engine using vectorized iteration and
a decoupled ExchangeEmulator for order execution.

Key design decisions:
  * Zero ``iterrows`` - uses ``itertuples`` for ~100x speedup
  * Strategy <-> Exchange are decoupled via Protocol interfaces
  * Optional progress callback for live streaming to WebSockets / UIs
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from loguru import logger

from .exchange_emulator import ExchangeEmulator, Tick
from .matching_engine import MarketConfig
from .orderbook import Side
from .market_impact import (
    AlmgrenChrissModel,
    SquareRootImpactModel,
    AdversarialSimulator,
    MarketState,
    ImpactSide,
)


# ---------------------------------------------------------------------------
# Protocols - contracts between Strategy, Simulator, and Exchange
# ---------------------------------------------------------------------------


@runtime_checkable
class StrategyProtocol(Protocol):
    """Minimal interface a strategy must satisfy to run in the simulator."""

    def initialize(self, ctx: "SimulationContext") -> None: ...
    def on_data(self, ctx: "SimulationContext", symbol: str, data: pd.Series) -> None: ...
    def finalize(self, ctx: "SimulationContext") -> None: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0002  # 2 bps
    slippage_bps: float = 1.0
    enable_short_selling: bool = True
    max_position_size: int = 10_000
    risk_free_rate: float = 0.02
    progress_interval: int = 500  # Emit progress every N ticks

    # --- v2: Market impact & adversarial simulation ---
    enable_market_impact: bool = False
    impact_model: str = "sqrt"          # "almgren_chriss", "sqrt", "none"
    almgren_chriss_risk_aversion: float = 1e-6
    enable_adversarial: bool = False
    adversarial_n_predators: int = 3
    daily_volume_default: float = 1_000_000.0
    daily_volatility_default: float = 0.25


# ---------------------------------------------------------------------------
# Position & Portfolio
# ---------------------------------------------------------------------------


class Position:
    """Trading position tracker."""

    __slots__ = ("symbol", "quantity", "avg_price", "realized_pnl", "unrealized_pnl")

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.quantity: int = 0
        self.avg_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0

    def update(self, quantity: int, price: float) -> None:
        if self.quantity == 0:
            self.avg_price = price
            self.quantity = quantity
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            total_cost = self.avg_price * abs(self.quantity) + price * abs(quantity)
            self.quantity += quantity
            if self.quantity != 0:
                self.avg_price = total_cost / abs(self.quantity)
        else:
            closed_quantity = min(abs(self.quantity), abs(quantity))
            pnl = closed_quantity * (price - self.avg_price) * np.sign(self.quantity)
            self.realized_pnl += float(pnl)
            prev_sign = np.sign(self.quantity)
            self.quantity += quantity
            if self.quantity != 0 and np.sign(self.quantity) != prev_sign:
                self.avg_price = price

    def mark_to_market(self, current_price: float) -> None:
        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_price)
        else:
            self.unrealized_pnl = 0.0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


class Portfolio:
    """Portfolio manager."""

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self.cash: float = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []

    def execute_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        commission: float,
    ) -> None:
        cost = quantity * price + commission
        self.cash -= cost
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        self.positions[symbol].update(quantity, price)
        self.trade_history.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "timestamp": time.time(),
            }
        )

    def mark_to_market(self, prices: Dict[str, float]) -> None:
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.mark_to_market(prices[symbol])

    @property
    def equity(self) -> float:
        return self.cash + sum(pos.total_pnl for pos in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_capital

    def get_position(self, symbol: str) -> int:
        if symbol in self.positions:
            return self.positions[symbol].quantity
        return 0


# ---------------------------------------------------------------------------
# Simulation Context - the "handle" a strategy uses
# ---------------------------------------------------------------------------


class SimulationContext:
    """Facade for strategies to query market state and submit orders."""

    def __init__(
        self,
        exchange: ExchangeEmulator,
        portfolio: Portfolio,
        sim_config: SimulationConfig,
        impact_model: Optional[Any] = None,
        adversarial: Optional[AdversarialSimulator] = None,
    ) -> None:
        self._exchange = exchange
        self._portfolio = portfolio
        self._config = sim_config
        self._impact_model = impact_model
        self._adversarial = adversarial
        self._last_prices: Dict[str, float] = {}
        self.impact_log: List[Dict[str, Any]] = []

    def _apply_impact(
        self, symbol: str, quantity: int, side: ImpactSide
    ) -> float:
        extra_bps = 0.0
        price = self._last_prices.get(symbol, 0.0)
        if price <= 0:
            return 0.0

        market = MarketState(
            mid_price=price,
            spread=price * 0.0002,
            daily_volume=self._config.daily_volume_default,
            daily_volatility=self._config.daily_volatility_default,
        )

        if self._impact_model is not None:
            if isinstance(self._impact_model, AlmgrenChrissModel):
                result = self._impact_model.compute_impact(
                    quantity, 10, market, side
                )
            else:
                result = self._impact_model.compute_impact(quantity, market, side)
            extra_bps += result.total_impact_bps
            self.impact_log.append({
                "symbol": symbol,
                "quantity": quantity,
                "side": side.value,
                "impact_bps": result.total_impact_bps,
                "cost_usd": result.execution_cost_usd,
            })

        if self._adversarial is not None:
            signed_qty = quantity if side == ImpactSide.BUY else -quantity
            adv_bps, _ = self._adversarial.apply_adversarial_impact(
                signed_qty, market
            )
            extra_bps += adv_bps

        return extra_bps

    def _price_with_impact(
        self, symbol: str, base_price: float, quantity: int, is_buy: bool
    ) -> float:
        if not self._config.enable_market_impact:
            return base_price
        side = ImpactSide.BUY if is_buy else ImpactSide.SELL
        bps = self._apply_impact(symbol, quantity, side)
        sign = 1.0 if is_buy else -1.0
        return base_price * (1 + sign * bps / 10_000)

    def buy(self, symbol: str, quantity: int, limit_price: Optional[float] = None) -> bool:
        fill = self._exchange.submit_order(symbol, Side.BID, quantity, limit_price)
        if fill is not None:
            adj_price = self._price_with_impact(symbol, fill.avg_price, quantity, True)
            self._portfolio.execute_trade(symbol, fill.quantity, adj_price, fill.commission)
            return True
        return False

    def sell(self, symbol: str, quantity: int, limit_price: Optional[float] = None) -> bool:
        fill = self._exchange.submit_order(symbol, Side.ASK, quantity, limit_price)
        if fill is not None:
            adj_price = self._price_with_impact(symbol, fill.avg_price, quantity, False)
            self._portfolio.execute_trade(symbol, -fill.quantity, adj_price, fill.commission)
            return True
        return False

    def update_price(self, symbol: str, price: float) -> None:
        self._last_prices[symbol] = price

    def get_position(self, symbol: str) -> int:
        return self._portfolio.get_position(symbol)

    def get_cash(self) -> float:
        return self._portfolio.cash

    def get_equity(self) -> float:
        return self._portfolio.equity


# ---------------------------------------------------------------------------
# Progress event (for WebSocket streaming)
# ---------------------------------------------------------------------------


@dataclass
class ProgressEvent:
    current_step: int
    total_steps: int
    pct_complete: float
    current_equity: float
    current_pnl: float
    timestamp: float = field(default_factory=time.time)


ProgressCallback = Callable[[ProgressEvent], None]


# ---------------------------------------------------------------------------
# Main Simulator
# ---------------------------------------------------------------------------


class EventSimulator:
    """High-performance event-driven backtesting engine using vectorized iteration."""

    def __init__(
        self,
        market_config: Optional[MarketConfig] = None,
        sim_config: Optional[SimulationConfig] = None,
    ) -> None:
        self.market_config = market_config or MarketConfig()
        self.sim_config = sim_config or SimulationConfig()

        self.exchange = ExchangeEmulator(
            market_config=self.market_config,
            commission_rate=self.sim_config.commission_rate,
            slippage_bps=self.sim_config.slippage_bps,
        )
        self.portfolio = Portfolio(self.sim_config.initial_capital)

        # Performance tracking
        self.equity_curve: List[float] = []
        self.timestamps: List[Any] = []

        # v2: Market impact model
        self._impact_model: Optional[Any] = None
        if self.sim_config.enable_market_impact:
            if self.sim_config.impact_model == "almgren_chriss":
                self._impact_model = AlmgrenChrissModel(
                    risk_aversion=self.sim_config.almgren_chriss_risk_aversion
                )
            elif self.sim_config.impact_model == "sqrt":
                self._impact_model = SquareRootImpactModel()

        # v2: Adversarial simulator
        self._adversarial: Optional[AdversarialSimulator] = None
        if self.sim_config.enable_adversarial:
            self._adversarial = AdversarialSimulator(
                n_predators=self.sim_config.adversarial_n_predators
            )

        # Legacy compatibility - strategies call simulator.buy / .sell
        self._ctx: Optional[SimulationContext] = None

    # -- Legacy convenience wrappers (delegate to context) -------------------

    def buy(self, symbol: str, quantity: int, limit_price: Optional[float] = None) -> bool:
        if self._ctx is None:
            return False
        return self._ctx.buy(symbol, quantity, limit_price)

    def sell(self, symbol: str, quantity: int, limit_price: Optional[float] = None) -> bool:
        if self._ctx is None:
            return False
        return self._ctx.sell(symbol, quantity, limit_price)

    def get_position(self, symbol: str) -> int:
        return self.portfolio.get_position(symbol)

    def get_cash(self) -> float:
        return self.portfolio.cash

    def get_equity(self) -> float:
        return self.portfolio.equity

    # -- Main backtest loop --------------------------------------------------

    def run_backtest(
        self,
        strategy: Any,
        data: pd.DataFrame,
        symbol: str = "SYM",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Run a full backtest using itertuples for row iteration."""
        logger.info(f"Starting backtest for {symbol}")
        logger.info(f"Data shape: {data.shape}")

        total_rows = len(data)
        if total_rows == 0:
            return self._empty_results()

        # Build simulation context
        self._ctx = SimulationContext(
            self.exchange, self.portfolio, self.sim_config,
            impact_model=self._impact_model,
            adversarial=self._adversarial,
        )

        # Ensure order book exists
        self.exchange._engine.create_orderbook(symbol)

        # Initialize strategy
        if hasattr(strategy, "initialize"):
            try:
                strategy.initialize(self._ctx)
            except TypeError:
                strategy.initialize(self)

        # Determine column positions from DataFrame for fast access
        columns = list(data.columns)
        col_map = {c: i for i, c in enumerate(columns)}

        has_price = "price" in col_map
        has_close = "close" in col_map
        has_bid = "bid" in col_map
        has_ask = "ask" in col_map
        has_volume = "volume" in col_map
        has_timestamp = "timestamp" in col_map

        progress_interval = self.sim_config.progress_interval

        # -------------------------------------------------------------------
        # FAST ITERATION via itertuples (replaces iterrows)
        # -------------------------------------------------------------------
        for step, row_tuple in enumerate(data.itertuples(index=True)):
            idx = row_tuple[0]  # The DataFrame index

            # Build a lightweight Series for strategy compatibility
            row_values = {columns[i]: row_tuple[i + 1] for i in range(len(columns))}

            # Determine current price
            current_price: float = 0.0
            if has_price:
                current_price = row_values["price"]
            elif has_close:
                current_price = row_values["close"]

            # Update context with latest price (for impact models)
            if current_price > 0 and self._ctx is not None:
                self._ctx.update_price(symbol, current_price)

            # Feed tick to exchange emulator
            if has_bid and has_ask:
                tick = Tick(
                    timestamp=row_values.get("timestamp", idx) if has_timestamp else float(step),
                    symbol=symbol,
                    price=current_price,
                    bid=row_values["bid"],
                    ask=row_values["ask"],
                    volume=int(row_values.get("volume", 100)) if has_volume else 100,
                )
                self.exchange.process_tick(tick)
            elif current_price > 0:
                spread = current_price * 0.0002  # 2 bps synthetic spread
                tick = Tick(
                    timestamp=row_values.get("timestamp", idx) if has_timestamp else float(step),
                    symbol=symbol,
                    price=current_price,
                    bid=current_price - spread / 2,
                    ask=current_price + spread / 2,
                    volume=int(row_values.get("volume", 100)) if has_volume else 100,
                )
                self.exchange.process_tick(tick)

            # Call strategy
            if hasattr(strategy, "on_data"):
                row_series = pd.Series(row_values, name=idx)
                try:
                    strategy.on_data(self._ctx, symbol, row_series)
                except TypeError:
                    strategy.on_data(self, symbol, row_series)

            # Mark to market
            if current_price > 0:
                self.portfolio.mark_to_market({symbol: current_price})

            # Record equity
            self.equity_curve.append(self.portfolio.equity)
            self.timestamps.append(row_values.get("timestamp", idx) if has_timestamp else idx)

            # Emit progress
            if progress_callback and (step % progress_interval == 0 or step == total_rows - 1):
                progress_callback(
                    ProgressEvent(
                        current_step=step + 1,
                        total_steps=total_rows,
                        pct_complete=round((step + 1) / total_rows * 100, 2),
                        current_equity=self.portfolio.equity,
                        current_pnl=self.portfolio.total_pnl,
                    )
                )

        # Finalize strategy
        if hasattr(strategy, "finalize"):
            try:
                strategy.finalize(self._ctx)
            except TypeError:
                strategy.finalize(self)

        # Calculate metrics
        results = self._calculate_metrics()

        # v2: Attach impact log if market impact was enabled
        if self._ctx is not None and self._ctx.impact_log:
            results["impact_log"] = self._ctx.impact_log
            total_impact = sum(e["cost_usd"] for e in self._ctx.impact_log)
            results["total_impact_cost_usd"] = total_impact
            results["avg_impact_bps"] = (
                np.mean([e["impact_bps"] for e in self._ctx.impact_log])
                if self._ctx.impact_log else 0.0
            )

        logger.info("Backtest completed")
        logger.info(f"Final Equity: ${results['final_equity']:,.2f}")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")

        return results

    # -- Metrics calculation -------------------------------------------------

    def _calculate_metrics(self) -> Dict[str, Any]:
        if not self.equity_curve:
            return self._empty_results()

        equity_arr = np.array(self.equity_curve, dtype=np.float64)
        returns = np.diff(equity_arr) / equity_arr[:-1]

        # Total return
        total_return = float(
            (equity_arr[-1] - self.sim_config.initial_capital) / self.sim_config.initial_capital
        )

        # Sharpe ratio (annualised)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = float(np.sqrt(252) * np.mean(returns) / np.std(returns))
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        cummax = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - cummax) / cummax
        max_drawdown = float(np.min(drawdown))

        # Win rate
        trades = self.portfolio.trade_history
        winning_trades = sum(1 for t in trades if t["quantity"] > 0)
        total_trades = len(trades)
        win_rate = winning_trades / max(1, total_trades)

        return {
            "initial_capital": self.sim_config.initial_capital,
            "final_equity": float(equity_arr[-1]),
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "equity_curve": self.equity_curve,
            "timestamps": self.timestamps,
        }

    @staticmethod
    def _empty_results() -> Dict[str, Any]:
        return {
            "initial_capital": 0.0,
            "final_equity": 0.0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "equity_curve": [],
            "timestamps": [],
        }
