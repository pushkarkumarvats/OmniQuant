"""
FastAPI REST API Layer
Exposes OmniQuant functionality via REST endpoints.

Architecture:
  * Backtests run in background via an async job registry (no Celery needed)
  * ``/api/v1/backtest/run`` returns immediately with a ``job_id``
  * ``/api/v1/backtest/status/{job_id}`` polls for results
  * WebSocket ``/ws/backtest/{job_id}`` streams live progress + equity curve
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from src.common.dependency_injection import get_container
from src.common.event_bus import get_event_bus
from src.data_pipeline.ingestion import DataIngestion
from src.portfolio.optimizer import PortfolioOptimizer
from src.simulator.event_simulator import EventSimulator, ProgressEvent

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OmniQuant API",
    description="Quantitative Trading Research & Backtesting Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Async Job Registry (in-process, no external broker needed)
# ---------------------------------------------------------------------------


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRegistry:
    """Simple async job registry for background backtests."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._progress: Dict[str, List[Dict[str, Any]]] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}  # type: ignore[type-arg]

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())[:12]
        self._jobs[job_id] = {"status": JobStatus.PENDING, "result": None, "error": None}
        self._progress[job_id] = []
        self._subscribers[job_id] = []
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)

    def push_progress(self, job_id: str, event: Dict[str, Any]) -> None:
        if job_id in self._progress:
            self._progress[job_id].append(event)
        for q in self._subscribers.get(job_id, []):
            q.put_nowait(event)

    def subscribe(self, job_id: str) -> asyncio.Queue:  # type: ignore[type-arg]
        q: asyncio.Queue = asyncio.Queue()  # type: ignore[type-arg]
        self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:  # type: ignore[type-arg]
        subs = self._subscribers.get(job_id, [])
        if q in subs:
            subs.remove(q)


_job_registry = JobRegistry()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class BacktestRequest(BaseModel):
    strategy_type: str = Field(..., description="Strategy type: momentum, market_maker, arbitrage")
    symbol: str = Field(..., description="Trading symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(100000.0, description="Initial capital")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class BacktestJobResponse(BaseModel):
    job_id: str
    status: str


class BacktestResultResponse(BaseModel):
    job_id: str
    status: str
    initial_capital: Optional[float] = None
    final_equity: Optional[float] = None
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_trades: Optional[int] = None
    win_rate: Optional[float] = None
    error: Optional[str] = None


class PortfolioOptimizationRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols")
    method: str = Field("mean_variance", description="Optimization method")
    start_date: str
    end_date: str
    constraints: Dict[str, Any] = Field(default_factory=dict)


class PortfolioOptimizationResponse(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


class MarketDataRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    interval: str = "1d"


class FeatureRequest(BaseModel):
    symbol: str
    feature_types: List[str] = Field(["technical", "microstructure"])
    start_date: str
    end_date: str


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ---------------------------------------------------------------------------
# Market Data Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/data/fetch")
async def fetch_market_data(request: MarketDataRequest) -> Dict[str, Any]:
    try:
        container = get_container()
        ingestion = container.resolve(DataIngestion)
        df = ingestion.fetch_yahoo_finance(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
        )
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        return {"symbol": request.symbol, "records": len(df), "data": df.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/data/generate_synthetic")
async def generate_synthetic_data(
    num_ticks: int = 10000,
    initial_price: float = 100.0,
    volatility: float = 0.02,
) -> Dict[str, Any]:
    try:
        container = get_container()
        ingestion = container.resolve(DataIngestion)
        df = ingestion.generate_synthetic_tick_data(
            num_ticks=num_ticks, initial_price=initial_price, volatility=volatility,
        )
        return {"records": len(df), "data": df.head(100).to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


@app.post("/api/v1/features/generate")
async def generate_features(request: FeatureRequest) -> Dict[str, Any]:
    try:
        from src.feature_engineering.technical_features import TechnicalFeatures
        from src.feature_engineering.microstructure_features import MicrostructureFeatures

        container = get_container()
        ingestion = container.resolve(DataIngestion)
        df = ingestion.fetch_yahoo_finance(
            symbol=request.symbol, start_date=request.start_date, end_date=request.end_date,
        )
        if "technical" in request.feature_types:
            tech = TechnicalFeatures()
            df = tech.generate_all_features(df)
        if "microstructure" in request.feature_types:
            micro = MicrostructureFeatures()
            df = micro.generate_all_features(df)
        return {
            "symbol": request.symbol,
            "features": len(df.columns),
            "feature_names": df.columns.tolist(),
            "records": len(df),
        }
    except Exception as e:
        logger.error(f"Error generating features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Backtesting Endpoints (Async Job Queue)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dynamic strategy class resolver
# ---------------------------------------------------------------------------

# Maps short names to fully-qualified ``module_path:ClassName`` strings so
# new strategies only need a one-line entry (or can be specified as a full
# dotted path in the request).  This replaces the previous closed dict that
# required a code change for every new strategy.
_STRATEGY_ALIASES: Dict[str, str] = {
    "momentum":     "src.strategies.momentum:MomentumStrategy",
    "market_maker": "src.strategies.market_maker:MarketMakerStrategy",
    "arbitrage":    "src.strategies.arbitrage:ArbitrageStrategy",
}


def _resolve_strategy_class(name: str) -> type:
    """Resolve a strategy name or dotted path to its class object."""
    import importlib

    target = _STRATEGY_ALIASES.get(name, name)

    if ":" in target:
        module_path, class_name = target.rsplit(":", 1)
    elif "." in target:
        module_path, class_name = target.rsplit(".", 1)
    else:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(_STRATEGY_ALIASES)}, "
            f"or pass a fully-qualified 'module.path:ClassName'."
        )

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Module '{module_path}' has no attribute '{class_name}'")
    return cls


def _run_backtest_sync(job_id: str, request: BacktestRequest) -> None:
    """Worker function executed in a background thread."""
    try:
        _job_registry.update_job(job_id, status=JobStatus.RUNNING)

        strategy_cls = _resolve_strategy_class(request.strategy_type)
        strategy = strategy_cls(config=request.parameters)

        container = get_container()
        ingestion = container.resolve(DataIngestion)
        df = ingestion.fetch_yahoo_finance(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        from src.simulator.event_simulator import SimulationConfig

        sim_config = SimulationConfig(
            initial_capital=request.initial_capital,
            progress_interval=200,
        )
        simulator = EventSimulator(sim_config=sim_config)

        def on_progress(evt: ProgressEvent) -> None:
            _job_registry.push_progress(job_id, {
                "progress": evt.pct_complete,
                "current_equity": evt.current_equity,
                "current_pnl": evt.current_pnl,
                "step": evt.current_step,
                "total": evt.total_steps,
            })

        results = simulator.run_backtest(
            strategy, df, symbol=request.symbol, progress_callback=on_progress,
        )

        _job_registry.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            result={
                "initial_capital": results["initial_capital"],
                "final_equity": results["final_equity"],
                "total_return": results["total_return"],
                "sharpe_ratio": results["sharpe_ratio"],
                "max_drawdown": results["max_drawdown"],
                "total_trades": results["total_trades"],
                "win_rate": results["win_rate"],
            },
        )
        _job_registry.push_progress(job_id, {"status": "completed"})

    except Exception as e:
        logger.error(f"Backtest job {job_id} failed: {e}")
        _job_registry.update_job(job_id, status=JobStatus.FAILED, error=str(e))
        _job_registry.push_progress(job_id, {"status": "failed", "error": str(e)})


@app.post("/api/v1/backtest/run", response_model=BacktestJobResponse)
async def run_backtest(request: BacktestRequest) -> BacktestJobResponse:
    """
    Submit a backtest job. Returns immediately with a job_id.
    Use /api/v1/backtest/status/{job_id} to poll for results,
    or connect to /ws/backtest/{job_id} to stream live progress.
    """
    job_id = _job_registry.create_job()
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_backtest_sync, job_id, request)
    return BacktestJobResponse(job_id=job_id, status=JobStatus.PENDING)


@app.get("/api/v1/backtest/status/{job_id}", response_model=BacktestResultResponse)
async def get_backtest_status(job_id: str) -> BacktestResultResponse:
    """Poll for backtest results."""
    job = _job_registry.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    result = job.get("result") or {}
    return BacktestResultResponse(
        job_id=job_id,
        status=job["status"],
        initial_capital=result.get("initial_capital"),
        final_equity=result.get("final_equity"),
        total_return=result.get("total_return"),
        sharpe_ratio=result.get("sharpe_ratio"),
        max_drawdown=result.get("max_drawdown"),
        total_trades=result.get("total_trades"),
        win_rate=result.get("win_rate"),
        error=job.get("error"),
    )


# ---------------------------------------------------------------------------
# WebSocket - live backtest progress & equity curve streaming
# ---------------------------------------------------------------------------


@app.websocket("/ws/backtest/{job_id}")
async def backtest_progress_stream(websocket: WebSocket, job_id: str) -> None:
    """Stream live backtest progress updates over WebSocket."""
    await websocket.accept()
    job = _job_registry.get_job(job_id)
    if job is None:
        await websocket.send_json({"error": f"Job {job_id} not found"})
        await websocket.close()
        return

    queue = _job_registry.subscribe(job_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(event)
                if event.get("status") in ("completed", "failed"):
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"heartbeat": True})
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    finally:
        _job_registry.unsubscribe(job_id, queue)


@app.websocket("/ws/market_data/{symbol}")
async def market_data_stream(websocket: WebSocket, symbol: str) -> None:
    """WebSocket endpoint for real-time market data."""
    await websocket.accept()
    logger.info(f"WebSocket connected for {symbol}")
    try:
        event_bus = get_event_bus()

        async def send_market_data(event: Any) -> None:
            if event.data.get("symbol") == symbol:
                await websocket.send_json(event.to_dict())

        event_bus.subscribe("market_data", send_market_data)
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ---------------------------------------------------------------------------
# Portfolio Optimization
# ---------------------------------------------------------------------------


@app.post("/api/v1/portfolio/optimize", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(
    request: PortfolioOptimizationRequest,
) -> PortfolioOptimizationResponse:
    try:
        import numpy as np

        container = get_container()
        ingestion = container.resolve(DataIngestion)
        returns_list = []
        for symbol in request.symbols:
            df = ingestion.fetch_yahoo_finance(
                symbol=symbol, start_date=request.start_date, end_date=request.end_date,
            )
            returns = df["close"].pct_change().dropna()
            returns_list.append(returns)
        returns_df = pd.concat(returns_list, axis=1)
        returns_df.columns = request.symbols
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        optimizer = PortfolioOptimizer()
        if request.method == "mean_variance":
            weights = optimizer.mean_variance_optimization(expected_returns, cov_matrix)
        elif request.method == "risk_parity":
            weights = optimizer.risk_parity(cov_matrix)
        elif request.method == "hierarchical":
            weights = optimizer.hierarchical_risk_parity(returns_df)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        portfolio_return = float(np.dot(weights, expected_returns))
        portfolio_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0
        return PortfolioOptimizationResponse(
            weights={s: float(w) for s, w in zip(request.symbols, weights)},
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@app.get("/api/v1/stats")
async def get_statistics() -> Dict[str, Any]:
    event_bus = get_event_bus()
    return {
        "event_subscribers": {
            "market_data": event_bus.get_subscriber_count("market_data"),
            "trade": event_bus.get_subscriber_count("trade"),
            "signal": event_bus.get_subscriber_count("signal"),
        },
        "event_history_size": len(event_bus.get_history()),
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from src.common.dependency_injection import configure_services

    configure_services()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
