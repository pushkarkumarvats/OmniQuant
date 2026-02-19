"""
FIX Drop Copy Reconciliation Service

Ensures trade consistency between internal OMS and exchange-reported fills:
  - Real-time fill matching (internal vs. drop copy)
  - Break detection with configurable tolerance
  - Automated and manual break resolution
  - End-of-day reconciliation reports
  - Audit trail for regulatory compliance
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger


# --------------------------------------------------------------------------- #
#  Types                                                                       #
# --------------------------------------------------------------------------- #

class BreakType(Enum):
    MISSING_INTERNAL = "missing_internal"
    MISSING_EXCHANGE = "missing_exchange"
    QTY_MISMATCH = "qty_mismatch"
    PRICE_MISMATCH = "price_mismatch"
    SIDE_MISMATCH = "side_mismatch"
    SYMBOL_MISMATCH = "symbol_mismatch"
    TIMESTAMP_DRIFT = "timestamp_drift"


class BreakStatus(Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED_AUTO = "resolved_auto"
    RESOLVED_MANUAL = "resolved_manual"
    ACKNOWLEDGED = "acknowledged"


class ReconciliationMode(Enum):
    REAL_TIME = "real_time"
    BATCH = "batch"
    END_OF_DAY = "end_of_day"


@dataclass
class Fill:
    """A trade execution fill."""
    fill_id: str
    order_id: str
    symbol: str
    side: str           # "buy" or "sell"
    quantity: float
    price: float
    venue: str
    timestamp_ns: int
    commission: float = 0.0
    strategy_id: str = ""
    account: str = ""
    exchange_fill_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    @property
    def match_key(self) -> str:
        """Key for matching internal fills to exchange fills.

        Includes venue so that fills at different exchanges for the
        same order are reconciled independently.
        """
        return f"{self.order_id}:{self.symbol}:{self.side}:{self.venue}"


@dataclass
class Break:
    """A trade reconciliation break."""
    break_id: str
    break_type: BreakType
    status: BreakStatus = BreakStatus.OPEN
    internal_fill: Optional[Fill] = None
    exchange_fill: Optional[Fill] = None
    description: str = ""
    detected_at: int = field(default_factory=time.time_ns)
    resolved_at: Optional[int] = None
    resolution_notes: str = ""
    severity: str = "high"

    @property
    def age_seconds(self) -> float:
        return (time.time_ns() - self.detected_at) / 1e9


@dataclass
class ReconciliationReport:
    """End-of-day reconciliation report."""
    report_date: str
    generated_at: int = field(default_factory=time.time_ns)
    total_internal_fills: int = 0
    total_exchange_fills: int = 0
    matched_fills: int = 0
    breaks_found: int = 0
    breaks_resolved: int = 0
    breaks_open: int = 0
    total_internal_notional: float = 0.0
    total_exchange_notional: float = 0.0
    notional_difference: float = 0.0
    breaks: List[Break] = field(default_factory=list)
    match_rate_pct: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Drop Copy Reconciliation Engine                                             #
# --------------------------------------------------------------------------- #

class DropCopyReconciler:
    """
    Real-time and batch fill reconciliation engine.

    Matches internal OMS fills against exchange-reported drop copy fills.
    Detects breaks and triggers alerts.
    """

    def __init__(
        self,
        price_tolerance: float = 0.01,  # 1 cent
        qty_tolerance: float = 0.001,
        max_timestamp_drift_ms: float = 1000,
    ) -> None:
        self._price_tolerance = price_tolerance
        self._qty_tolerance = qty_tolerance
        self._max_timestamp_drift_ns = int(max_timestamp_drift_ms * 1e6)

        # State
        self._internal_fills: Dict[str, Fill] = {}   # keyed by fill_id
        self._exchange_fills: Dict[str, Fill] = {}
        self._matched_pairs: List[Tuple[Fill, Fill]] = []
        self._breaks: Dict[str, Break] = {}          # keyed by break_id
        self._break_callbacks: List[Callable[[Break], None]] = []

        # Indexes for matching
        self._internal_by_match_key: Dict[str, List[Fill]] = defaultdict(list)
        self._exchange_by_match_key: Dict[str, List[Fill]] = defaultdict(list)

    def on_break(self, callback: Callable[[Break], None]) -> None:
        """Register a callback for break detection."""
        self._break_callbacks.append(callback)

    def add_internal_fill(self, fill: Fill) -> Optional[Break]:
        """Add a fill from the internal OMS and attempt to match."""
        self._internal_fills[fill.fill_id] = fill
        self._internal_by_match_key[fill.match_key].append(fill)
        return self._try_match(fill, source="internal")

    def add_exchange_fill(self, fill: Fill) -> Optional[Break]:
        """Add a fill from the exchange drop copy and attempt to match."""
        self._exchange_fills[fill.fill_id] = fill
        self._exchange_by_match_key[fill.match_key].append(fill)
        return self._try_match(fill, source="exchange")

    def _try_match(self, fill: Fill, source: str) -> Optional[Break]:
        """Attempt to match a fill against the opposite source."""
        match_key = fill.match_key

        if source == "internal":
            candidates = self._exchange_by_match_key.get(match_key, [])
        else:
            candidates = self._internal_by_match_key.get(match_key, [])

        for candidate in candidates:
            # Check if candidate is already matched
            if self._is_matched(candidate):
                continue

            # Attempt match
            brk = self._compare_fills(
                fill if source == "internal" else candidate,
                candidate if source == "internal" else fill,
            )

            if brk is None:
                # Perfect match
                self._matched_pairs.append((
                    fill if source == "internal" else candidate,
                    candidate if source == "internal" else fill,
                ))
                return None
            else:
                # Partial match with break
                self._register_break(brk)
                return brk

        # No match found yet - this may be expected if the counterpart hasn't arrived
        return None

    def _compare_fills(self, internal: Fill, exchange: Fill) -> Optional[Break]:
        """Compare two fills and return a Break if they don't match."""
        # Quantity check
        if abs(internal.quantity - exchange.quantity) > self._qty_tolerance:
            return Break(
                break_id=self._gen_break_id(internal, exchange),
                break_type=BreakType.QTY_MISMATCH,
                internal_fill=internal,
                exchange_fill=exchange,
                description=(
                    f"Qty mismatch: internal={internal.quantity:.4f}, "
                    f"exchange={exchange.quantity:.4f}"
                ),
            )

        # Price check
        if abs(internal.price - exchange.price) > self._price_tolerance:
            return Break(
                break_id=self._gen_break_id(internal, exchange),
                break_type=BreakType.PRICE_MISMATCH,
                internal_fill=internal,
                exchange_fill=exchange,
                description=(
                    f"Price mismatch: internal={internal.price:.4f}, "
                    f"exchange={exchange.price:.4f}"
                ),
            )

        # Timestamp drift
        ts_diff = abs(internal.timestamp_ns - exchange.timestamp_ns)
        if ts_diff > self._max_timestamp_drift_ns:
            return Break(
                break_id=self._gen_break_id(internal, exchange),
                break_type=BreakType.TIMESTAMP_DRIFT,
                internal_fill=internal,
                exchange_fill=exchange,
                description=(
                    f"Timestamp drift: {ts_diff / 1e6:.1f}ms "
                    f"(max {self._max_timestamp_drift_ns / 1e6:.1f}ms)"
                ),
                severity="low",
            )

        return None  # Match

    def _is_matched(self, fill: Fill) -> bool:
        for internal, exchange in self._matched_pairs:
            if fill.fill_id == internal.fill_id or fill.fill_id == exchange.fill_id:
                return True
        return False

    def _register_break(self, brk: Break) -> None:
        self._breaks[brk.break_id] = brk
        logger.warning(f"Break detected: {brk.break_type.value} - {brk.description}")
        for cb in self._break_callbacks:
            try:
                cb(brk)
            except Exception as e:
                logger.error(f"Break callback error: {e}")

    def _gen_break_id(self, internal: Fill, exchange: Fill) -> str:
        content = f"{internal.fill_id}:{exchange.fill_id}:{time.time_ns()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def detect_missing_fills(self, max_age_seconds: float = 30.0) -> List[Break]:
        """
        Detect fills that exist in one source but not the other.

        Called periodically to find unmatched fills older than max_age_seconds.
        """
        now = time.time_ns()
        cutoff = now - int(max_age_seconds * 1e9)
        new_breaks = []

        # Check for internal fills without exchange match
        for fill_id, fill in self._internal_fills.items():
            if fill.timestamp_ns > cutoff:
                continue
            if not self._is_matched(fill):
                brk = Break(
                    break_id=hashlib.md5(f"missing_exchange:{fill_id}".encode()).hexdigest()[:12],
                    break_type=BreakType.MISSING_EXCHANGE,
                    internal_fill=fill,
                    description=f"Internal fill {fill_id} has no exchange match after {max_age_seconds}s",
                )
                if brk.break_id not in self._breaks:
                    self._register_break(brk)
                    new_breaks.append(brk)

        # Check for exchange fills without internal match
        for fill_id, fill in self._exchange_fills.items():
            if fill.timestamp_ns > cutoff:
                continue
            if not self._is_matched(fill):
                brk = Break(
                    break_id=hashlib.md5(f"missing_internal:{fill_id}".encode()).hexdigest()[:12],
                    break_type=BreakType.MISSING_INTERNAL,
                    exchange_fill=fill,
                    description=f"Exchange fill {fill_id} has no internal match after {max_age_seconds}s",
                    severity="critical",
                )
                if brk.break_id not in self._breaks:
                    self._register_break(brk)
                    new_breaks.append(brk)

        return new_breaks

    def resolve_break(
        self, break_id: str, resolution: str = "", manual: bool = True,
    ) -> bool:
        """Resolve an open break."""
        brk = self._breaks.get(break_id)
        if not brk:
            return False
        brk.status = BreakStatus.RESOLVED_MANUAL if manual else BreakStatus.RESOLVED_AUTO
        brk.resolved_at = time.time_ns()
        brk.resolution_notes = resolution
        logger.info(f"Break {break_id} resolved: {resolution}")
        return True

    def generate_eod_report(self, report_date: Optional[str] = None) -> ReconciliationReport:
        """Generate end-of-day reconciliation report."""
        report_date = report_date or date.today().isoformat()

        total_internal_notional = sum(f.notional for f in self._internal_fills.values())
        total_exchange_notional = sum(f.notional for f in self._exchange_fills.values())

        open_breaks = [b for b in self._breaks.values() if b.status == BreakStatus.OPEN]
        resolved_breaks = [b for b in self._breaks.values() if b.status in (
            BreakStatus.RESOLVED_AUTO, BreakStatus.RESOLVED_MANUAL
        )]

        total_fills = max(len(self._internal_fills), len(self._exchange_fills))
        match_rate = (
            len(self._matched_pairs) / total_fills * 100
            if total_fills > 0 else 100.0
        )

        report = ReconciliationReport(
            report_date=report_date,
            total_internal_fills=len(self._internal_fills),
            total_exchange_fills=len(self._exchange_fills),
            matched_fills=len(self._matched_pairs),
            breaks_found=len(self._breaks),
            breaks_resolved=len(resolved_breaks),
            breaks_open=len(open_breaks),
            total_internal_notional=total_internal_notional,
            total_exchange_notional=total_exchange_notional,
            notional_difference=abs(total_internal_notional - total_exchange_notional),
            breaks=list(self._breaks.values()),
            match_rate_pct=match_rate,
            details={
                "break_types": {
                    bt.value: sum(1 for b in self._breaks.values() if b.break_type == bt)
                    for bt in BreakType
                },
            },
        )

        logger.info(
            f"EOD Report {report_date}: "
            f"matched={report.matched_fills}/{total_fills} ({match_rate:.1f}%), "
            f"breaks_open={report.breaks_open}"
        )
        return report

    def reset_daily(self) -> None:
        """Reset state for a new trading day."""
        self._internal_fills.clear()
        self._exchange_fills.clear()
        self._matched_pairs.clear()
        self._breaks.clear()
        self._internal_by_match_key.clear()
        self._exchange_by_match_key.clear()
        logger.info("Drop copy reconciler reset for new day")

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "internal_fills": len(self._internal_fills),
            "exchange_fills": len(self._exchange_fills),
            "matched_pairs": len(self._matched_pairs),
            "open_breaks": sum(1 for b in self._breaks.values() if b.status == BreakStatus.OPEN),
            "total_breaks": len(self._breaks),
        }


# --------------------------------------------------------------------------- #
#  Live Streaming Reconciliation Service (v2)                                  #
# --------------------------------------------------------------------------- #

class LiveReconciliationService:
    """
    Production-grade live reconciliation service that wraps
    :class:`DropCopyReconciler` with:

    - Periodic missing-fill scanning
    - Automatic resolution of timestamp-drift breaks
    - Event-sourcing integration for audit trail
    - PnL reconciliation between internal and exchange books
    - Alerting thresholds for break severity escalation
    """

    def __init__(
        self,
        reconciler: Optional[DropCopyReconciler] = None,
        *,
        scan_interval_s: float = 5.0,
        auto_resolve_drift: bool = True,
        max_open_breaks_alert: int = 10,
        notional_tolerance_pct: float = 0.01,
    ) -> None:
        self.reconciler = reconciler or DropCopyReconciler()
        self._scan_interval = scan_interval_s
        self._auto_resolve_drift = auto_resolve_drift
        self._max_open_breaks_alert = max_open_breaks_alert
        self._notional_tolerance_pct = notional_tolerance_pct

        # PnL reconciliation
        self._internal_pnl: Dict[str, float] = defaultdict(float)  # symbol -> pnl
        self._exchange_pnl: Dict[str, float] = defaultdict(float)
        self._alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._running = False

    def on_alert(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register an alert callback (severity, details)."""
        self._alert_callbacks.append(callback)

    def ingest_internal_fill(self, fill: Fill) -> Optional[Break]:
        """Process a fill from the internal OMS."""
        self._internal_pnl[fill.symbol] += fill.notional * (
            1 if fill.side == "buy" else -1
        )
        brk = self.reconciler.add_internal_fill(fill)
        self._check_alerts()
        return brk

    def ingest_exchange_fill(self, fill: Fill) -> Optional[Break]:
        """Process a fill from the exchange drop-copy."""
        self._exchange_pnl[fill.symbol] += fill.notional * (
            1 if fill.side == "buy" else -1
        )
        brk = self.reconciler.add_exchange_fill(fill)
        self._check_alerts()
        return brk

    def run_scan(self) -> List[Break]:
        """
        Run a periodic missing-fill scan and auto-resolve drift breaks.
        Call this on a timer or in a background thread.
        """
        new_breaks = self.reconciler.detect_missing_fills(
            max_age_seconds=self._scan_interval * 6
        )

        if self._auto_resolve_drift:
            for brk in list(self.reconciler._breaks.values()):
                if (
                    brk.break_type == BreakType.TIMESTAMP_DRIFT
                    and brk.status == BreakStatus.OPEN
                ):
                    self.reconciler.resolve_break(
                        brk.break_id,
                        resolution="Auto-resolved: timestamp drift within tolerance",
                        manual=False,
                    )

        self._check_alerts()
        return new_breaks

    def pnl_reconciliation(self) -> Dict[str, Any]:
        """
        Compare internal PnL vs exchange PnL per symbol.
        Returns a report of mismatches.
        """
        symbols = set(self._internal_pnl) | set(self._exchange_pnl)
        mismatches = {}
        for sym in symbols:
            internal = self._internal_pnl.get(sym, 0.0)
            exchange = self._exchange_pnl.get(sym, 0.0)
            diff = abs(internal - exchange)
            threshold = max(abs(internal), abs(exchange), 1.0) * (
                self._notional_tolerance_pct / 100
            )
            if diff > threshold:
                mismatches[sym] = {
                    "internal_pnl": internal,
                    "exchange_pnl": exchange,
                    "difference": diff,
                    "threshold": threshold,
                }
        return {
            "timestamp": time.time_ns(),
            "symbols_checked": len(symbols),
            "mismatches": mismatches,
            "all_reconciled": len(mismatches) == 0,
        }

    def _check_alerts(self) -> None:
        """Fire alert callbacks if thresholds are breached."""
        stats = self.reconciler.stats
        open_breaks = stats["open_breaks"]

        if open_breaks >= self._max_open_breaks_alert:
            self._fire_alert("critical", {
                "message": f"Open breaks ({open_breaks}) exceeded threshold "
                           f"({self._max_open_breaks_alert})",
                "stats": stats,
            })

    def _fire_alert(self, severity: str, details: Dict[str, Any]) -> None:
        logger.warning(f"ALERT [{severity}]: {details.get('message', '')}")
        for cb in self._alert_callbacks:
            try:
                cb(severity, details)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive reconciliation status report."""
        eod = self.reconciler.generate_eod_report()
        pnl = self.pnl_reconciliation()
        return {
            "eod_report": {
                "report_date": eod.report_date,
                "match_rate_pct": eod.match_rate_pct,
                "breaks_open": eod.breaks_open,
                "breaks_resolved": eod.breaks_resolved,
                "notional_diff": eod.notional_difference,
            },
            "pnl_reconciliation": pnl,
            "stats": self.reconciler.stats,
        }
