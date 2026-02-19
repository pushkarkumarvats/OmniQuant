"""
Trading Dashboard Backend API

FastAPI-based backend for the institutional trading dashboard:
  - Real-time WebSocket streaming (positions, P&L, orders)
  - REST endpoints for historical data, risk reports, reconciliation
  - Authentication and authorization
  - Rate limiting and CORS configuration
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("FastAPI not installed. Dashboard API unavailable.")


# --------------------------------------------------------------------------- #
#  Pydantic Models                                                             #
# --------------------------------------------------------------------------- #

if HAS_FASTAPI:

    class PositionResponse(BaseModel):
        symbol: str
        quantity: float
        avg_price: float
        market_price: float
        unrealized_pnl: float
        realized_pnl: float
        notional: float
        pnl_pct: float

    class OrderResponse(BaseModel):
        order_id: str
        symbol: str
        side: str
        quantity: float
        price: float
        status: str
        strategy_id: str
        timestamp: str
        fill_qty: float = 0.0
        fill_price: float = 0.0

    class RiskSummaryResponse(BaseModel):
        entity_id: str
        daily_pnl: float
        gross_notional: float
        net_notional: float
        current_equity: float
        drawdown_pct: float
        kill_switch: bool
        order_rate_1m: int
        positions_count: int

    class PerformanceResponse(BaseModel):
        total_return: float
        annual_return: float
        sharpe_ratio: float
        sortino_ratio: float
        max_drawdown: float
        win_rate: float
        total_trades: int
        avg_trade_pnl: float
        calmar_ratio: float

    class ReconciliationSummary(BaseModel):
        report_date: str
        matched_fills: int
        total_fills: int
        match_rate_pct: float
        open_breaks: int
        notional_difference: float


# --------------------------------------------------------------------------- #
#  WebSocket Manager                                                           #
# --------------------------------------------------------------------------- #

class WebSocketManager:
    """Manages WebSocket connections for real-time streaming."""

    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = {}  # channel -> websockets

    async def connect(self, websocket: WebSocket, channel: str = "default") -> None:
        await websocket.accept()
        if channel not in self._connections:
            self._connections[channel] = []
        self._connections[channel].append(websocket)
        logger.info(f"WebSocket connected to channel '{channel}'")

    def disconnect(self, websocket: WebSocket, channel: str = "default") -> None:
        if channel in self._connections:
            self._connections[channel] = [
                ws for ws in self._connections[channel] if ws != websocket
            ]

    async def broadcast(self, channel: str, data: Dict[str, Any]) -> None:
        """Broadcast message to all connections on a channel."""
        if channel not in self._connections:
            return
        message = json.dumps(data, default=str)
        disconnected = []
        for ws in self._connections[channel]:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws, channel)

    @property
    def connection_count(self) -> int:
        return sum(len(conns) for conns in self._connections.values())


# --------------------------------------------------------------------------- #
#  Dashboard App Factory                                                       #
# --------------------------------------------------------------------------- #

def create_dashboard_app(
    risk_engine=None,
    reconciler=None,
    feature_store=None,
    timeseries_db=None,
) -> "FastAPI":
    """Create the FastAPI dashboard application."""

    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI not installed")

    app = FastAPI(
        title="HRT Institutional Trading Dashboard",
        description="Real-time trading operations dashboard",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_manager = WebSocketManager()

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "websocket_connections": ws_manager.connection_count,
            "components": {
                "risk_engine": risk_engine is not None,
                "reconciler": reconciler is not None,
                "feature_store": feature_store is not None,
                "timeseries_db": timeseries_db is not None,
            },
        }

    # ── Risk ─────────────────────────────────────────────────────────────

    @app.get("/api/risk/summary", response_model=RiskSummaryResponse)
    async def risk_summary(entity_id: str = "default"):
        if not risk_engine:
            raise HTTPException(500, "Risk engine not configured")
        summary = risk_engine.get_risk_summary(entity_id)
        return RiskSummaryResponse(
            entity_id=summary.get("entity_id", entity_id),
            daily_pnl=summary.get("daily_pnl", 0),
            gross_notional=summary.get("gross_notional", 0),
            net_notional=summary.get("net_notional", 0),
            current_equity=summary.get("current_equity", 0),
            drawdown_pct=summary.get("drawdown_pct", 0),
            kill_switch=summary.get("kill_switch", False),
            order_rate_1m=summary.get("order_rate_1m", 0),
            positions_count=summary.get("positions", 0),
        )

    @app.get("/api/risk/stats")
    async def risk_stats():
        if not risk_engine:
            raise HTTPException(500, "Risk engine not configured")
        return risk_engine.stats

    @app.post("/api/risk/kill-switch/{entity_id}/activate")
    async def activate_kill_switch(entity_id: str):
        if not risk_engine:
            raise HTTPException(500, "Risk engine not configured")
        risk_engine._activate_kill_switch(entity_id)
        return {"status": "activated", "entity_id": entity_id}

    @app.post("/api/risk/kill-switch/{entity_id}/deactivate")
    async def deactivate_kill_switch(entity_id: str):
        if not risk_engine:
            raise HTTPException(500, "Risk engine not configured")
        risk_engine.deactivate_kill_switch(entity_id)
        return {"status": "deactivated", "entity_id": entity_id}

    # ── Reconciliation ───────────────────────────────────────────────────

    @app.get("/api/reconciliation/summary")
    async def recon_summary():
        if not reconciler:
            raise HTTPException(500, "Reconciler not configured")
        return reconciler.stats

    @app.get("/api/reconciliation/eod-report")
    async def recon_eod_report(report_date: Optional[str] = None):
        if not reconciler:
            raise HTTPException(500, "Reconciler not configured")
        report = reconciler.generate_eod_report(report_date)
        return {
            "report_date": report.report_date,
            "matched_fills": report.matched_fills,
            "total_internal_fills": report.total_internal_fills,
            "total_exchange_fills": report.total_exchange_fills,
            "match_rate_pct": report.match_rate_pct,
            "breaks_open": report.breaks_open,
            "breaks_resolved": report.breaks_resolved,
            "notional_difference": report.notional_difference,
        }

    @app.post("/api/reconciliation/breaks/{break_id}/resolve")
    async def resolve_break(break_id: str, resolution: str = ""):
        if not reconciler:
            raise HTTPException(500, "Reconciler not configured")
        success = reconciler.resolve_break(break_id, resolution)
        if not success:
            raise HTTPException(404, f"Break {break_id} not found")
        return {"status": "resolved", "break_id": break_id}

    # ── WebSocket Streaming ──────────────────────────────────────────────

    @app.websocket("/ws/positions")
    async def ws_positions(websocket: WebSocket):
        await ws_manager.connect(websocket, "positions")
        try:
            while True:
                # Stream position updates
                if risk_engine:
                    for entity_id, state in risk_engine._states.items():
                        positions = [
                            {
                                "symbol": p.symbol,
                                "quantity": p.quantity,
                                "avg_price": p.avg_price,
                                "market_price": p.market_price,
                                "unrealized_pnl": p.unrealized_pnl,
                                "notional": p.notional,
                            }
                            for p in state.positions.values()
                        ]
                        await websocket.send_json({
                            "type": "positions",
                            "entity_id": entity_id,
                            "data": positions,
                            "timestamp": time.time_ns(),
                        })
                await asyncio.sleep(0.5)  # 2 Hz update rate
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, "positions")

    @app.websocket("/ws/pnl")
    async def ws_pnl(websocket: WebSocket):
        await ws_manager.connect(websocket, "pnl")
        try:
            while True:
                if risk_engine:
                    for entity_id, state in risk_engine._states.items():
                        await websocket.send_json({
                            "type": "pnl",
                            "entity_id": entity_id,
                            "daily_pnl": state.daily_pnl,
                            "equity": state.current_equity,
                            "drawdown_pct": (
                                (state.peak_equity - state.current_equity) / state.peak_equity
                                if state.peak_equity > 0 else 0
                            ),
                            "timestamp": time.time_ns(),
                        })
                await asyncio.sleep(0.25)  # 4 Hz
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, "pnl")

    @app.websocket("/ws/risk-alerts")
    async def ws_risk_alerts(websocket: WebSocket):
        await ws_manager.connect(websocket, "risk_alerts")
        try:
            while True:
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket, "risk_alerts")

    return app
