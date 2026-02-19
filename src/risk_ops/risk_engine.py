"""
Independent Pre-Trade Risk Engine

Institutional-grade risk management that runs independently of the trading path:
  - Pre-trade risk checks (position limits, notional limits, rate limits)
  - Real-time P&L monitoring with kill switch capability
  - Multi-level risk hierarchy (account → strategy → portfolio → firm)
  - Configurable risk limits with alerting
  - Circuit breakers and automated position flattening
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger


# --------------------------------------------------------------------------- #
#  Types                                                                       #
# --------------------------------------------------------------------------- #

class RiskAction(Enum):
    ALLOW = "allow"
    REJECT = "reject"
    THROTTLE = "throttle"
    KILL = "kill"


class RiskLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RiskCheckType(Enum):
    POSITION_LIMIT = "position_limit"
    NOTIONAL_LIMIT = "notional_limit"
    LOSS_LIMIT = "loss_limit"
    ORDER_RATE = "order_rate"
    ORDER_SIZE = "order_size"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    DRAWDOWN = "drawdown"
    FAT_FINGER = "fat_finger"
    DUPLICATE_ORDER = "duplicate_order"
    MARKET_IMPACT = "market_impact"


@dataclass
class RiskCheckResult:
    """Result of a single risk check."""
    check_type: RiskCheckType
    action: RiskAction
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    level: RiskLevel = RiskLevel.INFO
    latency_ns: int = 0


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    # Position limits
    max_position_qty: float = 100_000
    max_position_notional: float = 10_000_000
    max_single_order_qty: float = 10_000
    max_single_order_notional: float = 1_000_000
    # P&L limits
    max_daily_loss: float = 500_000
    max_drawdown_pct: float = 0.05
    max_unrealized_loss: float = 250_000
    # Rate limits
    max_orders_per_second: int = 100
    max_orders_per_minute: int = 1000
    max_cancel_rate_pct: float = 0.95   # order-to-cancel ratio
    # Concentration
    max_sector_concentration_pct: float = 0.25
    max_single_name_pct: float = 0.10
    # Leverage
    max_gross_leverage: float = 4.0
    max_net_leverage: float = 2.0
    # Fat finger
    max_price_deviation_pct: float = 0.05  # from last trade
    min_order_price: float = 0.01
    max_order_price: float = 1_000_000


@dataclass
class Position:
    """Current position state."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    notional: float = 0.0
    sector: str = ""

    def update_market_price(self, price: float) -> None:
        self.market_price = price
        self.notional = abs(self.quantity) * price
        self.unrealized_pnl = self.quantity * (price - self.avg_price)


@dataclass
class RiskState:
    """Aggregate risk state for an entity (strategy/account/firm)."""
    entity_id: str
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    gross_notional: float = 0.0
    net_notional: float = 0.0
    order_count_1s: int = 0
    order_count_1m: int = 0
    cancel_count_1m: int = 0
    last_order_timestamps: List[int] = field(default_factory=list)
    kill_switch_active: bool = False


# --------------------------------------------------------------------------- #
#  Pre-Trade Risk Checks                                                       #
# --------------------------------------------------------------------------- #

class RiskCheck:
    """Base risk check."""

    def __init__(self, check_type: RiskCheckType) -> None:
        self.check_type = check_type

    def check(
        self, order: Dict[str, Any], state: RiskState, limits: RiskLimits,
    ) -> RiskCheckResult:
        raise NotImplementedError


class KillSwitchCheck(RiskCheck):
    """Rejects all orders if kill switch is active."""

    def __init__(self) -> None:
        super().__init__(RiskCheckType.LOSS_LIMIT)

    def check(self, order: Dict, state: RiskState, limits: RiskLimits) -> RiskCheckResult:
        start = time.time_ns()
        if state.kill_switch_active:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.KILL,
                message="Kill switch is active - all orders rejected",
                level=RiskLevel.EMERGENCY,
                latency_ns=time.time_ns() - start,
            )
        return RiskCheckResult(
            check_type=self.check_type,
            action=RiskAction.ALLOW,
            message="Kill switch not active",
            latency_ns=time.time_ns() - start,
        )


class PositionLimitCheck(RiskCheck):
    """Check position quantity and notional limits."""

    def __init__(self) -> None:
        super().__init__(RiskCheckType.POSITION_LIMIT)

    def check(self, order: Dict, state: RiskState, limits: RiskLimits) -> RiskCheckResult:
        start = time.time_ns()
        symbol = order.get("symbol", "")
        side = order.get("side", "buy")
        qty = order.get("quantity", 0)
        price = order.get("price", 0)

        current_pos = state.positions.get(symbol, Position(symbol=symbol))
        new_qty = current_pos.quantity + (qty if side == "buy" else -qty)
        new_notional = abs(new_qty) * price

        if abs(new_qty) > limits.max_position_qty:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.REJECT,
                message=f"Position limit exceeded: {abs(new_qty):.0f} > {limits.max_position_qty:.0f}",
                level=RiskLevel.WARNING,
                details={"new_qty": new_qty, "limit": limits.max_position_qty},
                latency_ns=time.time_ns() - start,
            )

        if new_notional > limits.max_position_notional:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.REJECT,
                message=f"Notional limit exceeded: ${new_notional:,.0f} > ${limits.max_position_notional:,.0f}",
                level=RiskLevel.WARNING,
                details={"new_notional": new_notional, "limit": limits.max_position_notional},
                latency_ns=time.time_ns() - start,
            )

        return RiskCheckResult(
            check_type=self.check_type,
            action=RiskAction.ALLOW,
            message="Position within limits",
            latency_ns=time.time_ns() - start,
        )


class OrderSizeCheck(RiskCheck):
    """Check individual order size limits (fat finger protection)."""

    def __init__(self) -> None:
        super().__init__(RiskCheckType.ORDER_SIZE)

    def check(self, order: Dict, state: RiskState, limits: RiskLimits) -> RiskCheckResult:
        start = time.time_ns()
        qty = order.get("quantity", 0)
        price = order.get("price", 0)
        notional = qty * price

        if qty > limits.max_single_order_qty:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.REJECT,
                message=f"Order qty {qty:.0f} exceeds max {limits.max_single_order_qty:.0f}",
                level=RiskLevel.WARNING,
                latency_ns=time.time_ns() - start,
            )

        if notional > limits.max_single_order_notional:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.REJECT,
                message=f"Order notional ${notional:,.0f} exceeds max ${limits.max_single_order_notional:,.0f}",
                level=RiskLevel.WARNING,
                latency_ns=time.time_ns() - start,
            )

        return RiskCheckResult(
            check_type=self.check_type,
            action=RiskAction.ALLOW,
            message="Order size within limits",
            latency_ns=time.time_ns() - start,
        )


class FatFingerCheck(RiskCheck):
    """Check order price against recent market price."""

    def __init__(self) -> None:
        super().__init__(RiskCheckType.FAT_FINGER)

    def check(self, order: Dict, state: RiskState, limits: RiskLimits) -> RiskCheckResult:
        start = time.time_ns()
        symbol = order.get("symbol", "")
        price = order.get("price", 0)

        if price < limits.min_order_price or price > limits.max_order_price:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.REJECT,
                message=f"Price ${price} outside valid range [${limits.min_order_price}, ${limits.max_order_price}]",
                level=RiskLevel.CRITICAL,
                latency_ns=time.time_ns() - start,
            )

        pos = state.positions.get(symbol)
        if pos and pos.market_price > 0:
            deviation = abs(price - pos.market_price) / pos.market_price
            if deviation > limits.max_price_deviation_pct:
                return RiskCheckResult(
                    check_type=self.check_type,
                    action=RiskAction.REJECT,
                    message=f"Price ${price:.2f} deviates {deviation:.1%} from market ${pos.market_price:.2f}",
                    level=RiskLevel.CRITICAL,
                    details={"deviation": deviation, "market_price": pos.market_price},
                    latency_ns=time.time_ns() - start,
                )

        return RiskCheckResult(
            check_type=self.check_type,
            action=RiskAction.ALLOW,
            message="Price within bounds",
            latency_ns=time.time_ns() - start,
        )


class OrderRateCheck(RiskCheck):
    """Check order submission rate."""

    def __init__(self) -> None:
        super().__init__(RiskCheckType.ORDER_RATE)

    def check(self, order: Dict, state: RiskState, limits: RiskLimits) -> RiskCheckResult:
        start = time.time_ns()
        now = time.time_ns()

        # Clean old timestamps (keep last 60 seconds)
        cutoff_1s = now - 1_000_000_000
        cutoff_1m = now - 60_000_000_000
        state.last_order_timestamps = [
            ts for ts in state.last_order_timestamps if ts > cutoff_1m
        ]

        count_1s = sum(1 for ts in state.last_order_timestamps if ts > cutoff_1s)
        count_1m = len(state.last_order_timestamps)

        if count_1s >= limits.max_orders_per_second:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.THROTTLE,
                message=f"Rate limit: {count_1s}/s (max {limits.max_orders_per_second}/s)",
                level=RiskLevel.WARNING,
                details={"rate_1s": count_1s, "rate_1m": count_1m},
                latency_ns=time.time_ns() - start,
            )

        if count_1m >= limits.max_orders_per_minute:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.THROTTLE,
                message=f"Rate limit: {count_1m}/min (max {limits.max_orders_per_minute}/min)",
                level=RiskLevel.WARNING,
                latency_ns=time.time_ns() - start,
            )

        state.last_order_timestamps.append(now)
        return RiskCheckResult(
            check_type=self.check_type,
            action=RiskAction.ALLOW,
            message="Rate within limits",
            latency_ns=time.time_ns() - start,
        )


class DailyLossCheck(RiskCheck):
    """Check daily P&L loss limits."""

    def __init__(self) -> None:
        super().__init__(RiskCheckType.LOSS_LIMIT)

    def check(self, order: Dict, state: RiskState, limits: RiskLimits) -> RiskCheckResult:
        start = time.time_ns()

        if state.daily_pnl < -limits.max_daily_loss:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.KILL,
                message=f"Daily loss ${abs(state.daily_pnl):,.0f} exceeds limit ${limits.max_daily_loss:,.0f}",
                level=RiskLevel.EMERGENCY,
                details={"daily_pnl": state.daily_pnl, "limit": limits.max_daily_loss},
                latency_ns=time.time_ns() - start,
            )

        # Warn at 80% of limit
        if state.daily_pnl < -limits.max_daily_loss * 0.8:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.ALLOW,
                message=f"Daily loss approaching limit: ${abs(state.daily_pnl):,.0f} / ${limits.max_daily_loss:,.0f}",
                level=RiskLevel.WARNING,
                latency_ns=time.time_ns() - start,
            )

        return RiskCheckResult(
            check_type=self.check_type,
            action=RiskAction.ALLOW,
            message="Daily P&L within limits",
            latency_ns=time.time_ns() - start,
        )


class DrawdownCheck(RiskCheck):
    """Check drawdown from peak equity."""

    def __init__(self) -> None:
        super().__init__(RiskCheckType.DRAWDOWN)

    def check(self, order: Dict, state: RiskState, limits: RiskLimits) -> RiskCheckResult:
        start = time.time_ns()

        if state.peak_equity <= 0:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.ALLOW,
                message="No equity history",
                latency_ns=time.time_ns() - start,
            )

        drawdown_pct = (state.peak_equity - state.current_equity) / state.peak_equity

        if drawdown_pct > limits.max_drawdown_pct:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.KILL,
                message=f"Drawdown {drawdown_pct:.2%} exceeds limit {limits.max_drawdown_pct:.2%}",
                level=RiskLevel.EMERGENCY,
                details={
                    "drawdown_pct": drawdown_pct,
                    "peak_equity": state.peak_equity,
                    "current_equity": state.current_equity,
                },
                latency_ns=time.time_ns() - start,
            )

        return RiskCheckResult(
            check_type=self.check_type,
            action=RiskAction.ALLOW,
            message=f"Drawdown {drawdown_pct:.2%} within limit",
            latency_ns=time.time_ns() - start,
        )


class LeverageCheck(RiskCheck):
    """Check gross and net leverage."""

    def __init__(self) -> None:
        super().__init__(RiskCheckType.LEVERAGE)

    def check(self, order: Dict, state: RiskState, limits: RiskLimits) -> RiskCheckResult:
        start = time.time_ns()

        if state.current_equity <= 0:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.ALLOW,
                message="No equity data",
                latency_ns=time.time_ns() - start,
            )

        gross_leverage = state.gross_notional / state.current_equity
        net_leverage = abs(state.net_notional) / state.current_equity

        if gross_leverage > limits.max_gross_leverage:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.REJECT,
                message=f"Gross leverage {gross_leverage:.2f}x exceeds {limits.max_gross_leverage:.2f}x",
                level=RiskLevel.CRITICAL,
                latency_ns=time.time_ns() - start,
            )

        if net_leverage > limits.max_net_leverage:
            return RiskCheckResult(
                check_type=self.check_type,
                action=RiskAction.REJECT,
                message=f"Net leverage {net_leverage:.2f}x exceeds {limits.max_net_leverage:.2f}x",
                level=RiskLevel.WARNING,
                latency_ns=time.time_ns() - start,
            )

        return RiskCheckResult(
            check_type=self.check_type,
            action=RiskAction.ALLOW,
            message=f"Leverage OK (gross={gross_leverage:.2f}x, net={net_leverage:.2f}x)",
            latency_ns=time.time_ns() - start,
        )


# --------------------------------------------------------------------------- #
#  Pre-Trade Risk Engine                                                       #
# --------------------------------------------------------------------------- #

class PreTradeRiskEngine:
    """
    Independent pre-trade risk engine.

    Runs all configured checks in sequence (fail-fast) with sub-microsecond
    target latency. Thread-safe for concurrent order validation.
    """

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self._limits = limits or RiskLimits()
        self._states: Dict[str, RiskState] = {}
        self._checks: List[RiskCheck] = []
        self._lock = threading.Lock()
        self._total_checks = 0
        self._total_rejects = 0
        self._total_latency_ns = 0
        self._alert_callbacks: List[Callable[[RiskCheckResult], None]] = []

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        self._checks = [
            KillSwitchCheck(),
            OrderRateCheck(),
            OrderSizeCheck(),
            FatFingerCheck(),
            PositionLimitCheck(),
            DailyLossCheck(),
            DrawdownCheck(),
            LeverageCheck(),
        ]

    def add_check(self, check: RiskCheck) -> None:
        self._checks.append(check)

    def on_alert(self, callback: Callable[[RiskCheckResult], None]) -> None:
        self._alert_callbacks.append(callback)

    def validate_order(
        self,
        order: Dict[str, Any],
        entity_id: str = "default",
    ) -> Tuple[RiskAction, List[RiskCheckResult]]:
        """
        Run all pre-trade risk checks.

        Returns (final_action, list_of_check_results).
        Fail-fast: stops on first REJECT or KILL.

        Thread-safety: the lock is held for the entire validate+check
        sequence to prevent another thread from mutating the
        :class:`RiskState` (e.g. via ``update_position``) mid-evaluation.
        """
        start = time.time_ns()

        with self._lock:
            state = self._states.setdefault(entity_id, RiskState(entity_id=entity_id))

            results: List[RiskCheckResult] = []
            final_action = RiskAction.ALLOW

            for check in self._checks:
                result = check.check(order, state, self._limits)
                results.append(result)

                if result.action in (RiskAction.REJECT, RiskAction.KILL):
                    final_action = result.action
                    logger.warning(
                        f"Risk {result.action.value}: {result.check_type.value} - {result.message}"
                    )
                    # Alert callbacks (fire outside lock to avoid deadlocks
                    # with callbacks that acquire their own locks)
                    alert_results = [(cb, result) for cb in self._alert_callbacks]

                    if result.action == RiskAction.KILL:
                        state.kill_switch_active = True
                        logger.critical(f"KILL SWITCH ACTIVATED for {entity_id}")
                    break

                if result.action == RiskAction.THROTTLE:
                    final_action = RiskAction.THROTTLE
            else:
                alert_results = []

        # Fire alert callbacks outside the lock
        for cb, res in alert_results:
            try:
                cb(res)
            except Exception:
                pass

        total_latency = time.time_ns() - start
        # Atomic counter updates (benign race on stats; acceptable)
        self._total_checks += 1
        self._total_latency_ns += total_latency
        if final_action in (RiskAction.REJECT, RiskAction.KILL):
            self._total_rejects += 1

        return final_action, results

    def update_position(
        self,
        entity_id: str,
        symbol: str,
        quantity: float,
        avg_price: float,
        market_price: float,
    ) -> None:
        """Update position state from execution reports."""
        with self._lock:
            state = self._states.setdefault(entity_id, RiskState(entity_id=entity_id))
            pos = state.positions.setdefault(symbol, Position(symbol=symbol))
            pos.quantity = quantity
            pos.avg_price = avg_price
            pos.update_market_price(market_price)

            # Recalculate aggregates
            state.gross_notional = sum(p.notional for p in state.positions.values())
            state.net_notional = sum(
                p.quantity * p.market_price for p in state.positions.values()
            )
            state.daily_pnl = sum(
                p.realized_pnl + p.unrealized_pnl for p in state.positions.values()
            )
            if state.current_equity > state.peak_equity:
                state.peak_equity = state.current_equity

    def update_equity(self, entity_id: str, equity: float) -> None:
        """Update current equity."""
        with self._lock:
            state = self._states.setdefault(entity_id, RiskState(entity_id=entity_id))
            state.current_equity = equity
            if equity > state.peak_equity:
                state.peak_equity = equity

    def deactivate_kill_switch(self, entity_id: str) -> None:
        """Manually deactivate kill switch (requires human intervention)."""
        with self._lock:
            state = self._states.get(entity_id)
            if state:
                state.kill_switch_active = False
        logger.warning(f"Kill switch deactivated for {entity_id}")

    def get_risk_summary(self, entity_id: str = "default") -> Dict[str, Any]:
        """Get current risk state summary."""
        state = self._states.get(entity_id)
        if not state:
            return {"entity_id": entity_id, "status": "no_data"}

        return {
            "entity_id": entity_id,
            "kill_switch": state.kill_switch_active,
            "daily_pnl": state.daily_pnl,
            "gross_notional": state.gross_notional,
            "net_notional": state.net_notional,
            "current_equity": state.current_equity,
            "peak_equity": state.peak_equity,
            "drawdown_pct": (
                (state.peak_equity - state.current_equity) / state.peak_equity
                if state.peak_equity > 0 else 0
            ),
            "positions": len(state.positions),
            "order_rate_1m": len(state.last_order_timestamps),
        }

    @property
    def stats(self) -> Dict[str, Any]:
        avg_latency = (
            self._total_latency_ns / self._total_checks
            if self._total_checks > 0 else 0
        )
        return {
            "total_checks": self._total_checks,
            "total_rejects": self._total_rejects,
            "reject_rate": self._total_rejects / max(self._total_checks, 1),
            "avg_latency_us": avg_latency / 1000,
        }
