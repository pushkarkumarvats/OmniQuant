"""
FastAPI REST API Layer
Exposes OmniQuant functionality via REST endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
from loguru import logger

from src.common.dependency_injection import get_container
from src.data_pipeline.ingestion import DataIngestion
from src.simulator.event_simulator import EventSimulator
from src.portfolio.optimizer import PortfolioOptimizer
from src.common.event_bus import get_event_bus, MarketDataEvent

# Initialize FastAPI app
app = FastAPI(
    title="OmniQuant API",
    description="Quantitative Trading Research & Backtesting Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class BacktestRequest(BaseModel):
    """Backtest request model"""
    strategy_type: str = Field(..., description="Strategy type: momentum, market_maker, arbitrage")
    symbol: str = Field(..., description="Trading symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(100000.0, description="Initial capital")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class BacktestResponse(BaseModel):
    """Backtest response model"""
    backtest_id: str
    status: str
    initial_capital: float
    final_equity: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float


class PortfolioOptimizationRequest(BaseModel):
    """Portfolio optimization request"""
    symbols: List[str] = Field(..., description="List of symbols")
    method: str = Field("mean_variance", description="Optimization method")
    start_date: str
    end_date: str
    constraints: Dict[str, Any] = Field(default_factory=dict)


class PortfolioOptimizationResponse(BaseModel):
    """Portfolio optimization response"""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


class MarketDataRequest(BaseModel):
    """Market data request"""
    symbol: str
    start_date: str
    end_date: str
    interval: str = "1d"


class FeatureRequest(BaseModel):
    """Feature generation request"""
    symbol: str
    feature_types: List[str] = Field(["technical", "microstructure"])
    start_date: str
    end_date: str


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Market Data Endpoints
@app.post("/api/v1/data/fetch")
async def fetch_market_data(request: MarketDataRequest):
    """
    Fetch market data
    
    Args:
        request: Market data request
        
    Returns:
        Market data DataFrame as JSON
    """
    try:
        container = get_container()
        ingestion = container.resolve(DataIngestion)
        
        df = ingestion.fetch_yahoo_finance(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        return {
            "symbol": request.symbol,
            "records": len(df),
            "data": df.to_dict(orient="records")
        }
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/data/generate_synthetic")
async def generate_synthetic_data(
    num_ticks: int = 10000,
    initial_price: float = 100.0,
    volatility: float = 0.02
):
    """Generate synthetic market data"""
    try:
        container = get_container()
        ingestion = container.resolve(DataIngestion)
        
        df = ingestion.generate_synthetic_tick_data(
            num_ticks=num_ticks,
            initial_price=initial_price,
            volatility=volatility
        )
        
        return {
            "records": len(df),
            "data": df.head(100).to_dict(orient="records")  # Return first 100
        }
    
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Feature Engineering Endpoints
@app.post("/api/v1/features/generate")
async def generate_features(request: FeatureRequest):
    """Generate features for given symbol"""
    try:
        from src.feature_engineering.technical_features import TechnicalFeatures
        from src.feature_engineering.microstructure_features import MicrostructureFeatures
        
        # Fetch data
        container = get_container()
        ingestion = container.resolve(DataIngestion)
        df = ingestion.fetch_yahoo_finance(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Generate features
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
            "records": len(df)
        }
    
    except Exception as e:
        logger.error(f"Error generating features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Backtesting Endpoints
@app.post("/api/v1/backtest/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run a backtest
    
    Args:
        request: Backtest configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Backtest results
    """
    try:
        from src.strategies.momentum import MomentumStrategy
        from src.strategies.market_maker import MarketMakerStrategy
        from src.strategies.arbitrage import ArbitrageStrategy
        
        # Select strategy
        strategy_map = {
            "momentum": MomentumStrategy,
            "market_maker": MarketMakerStrategy,
            "arbitrage": ArbitrageStrategy
        }
        
        if request.strategy_type not in strategy_map:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy: {request.strategy_type}"
            )
        
        strategy_cls = strategy_map[request.strategy_type]
        strategy = strategy_cls(config=request.parameters)
        
        # Fetch data
        container = get_container()
        ingestion = container.resolve(DataIngestion)
        df = ingestion.fetch_yahoo_finance(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Run backtest
        simulator = EventSimulator()
        results = simulator.run_backtest(strategy, df, symbol=request.symbol)
        
        return BacktestResponse(
            backtest_id=f"BT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status="completed",
            initial_capital=results['initial_capital'],
            final_equity=results['final_equity'],
            total_return=results['total_return'],
            sharpe_ratio=results['sharpe_ratio'],
            max_drawdown=results['max_drawdown'],
            total_trades=results['total_trades'],
            win_rate=results['win_rate']
        )
    
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Portfolio Optimization Endpoints
@app.post("/api/v1/portfolio/optimize", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """
    Optimize portfolio weights
    
    Args:
        request: Optimization request
        
    Returns:
        Optimal weights and statistics
    """
    try:
        import numpy as np
        
        # Fetch data for all symbols
        container = get_container()
        ingestion = container.resolve(DataIngestion)
        
        returns_list = []
        for symbol in request.symbols:
            df = ingestion.fetch_yahoo_finance(
                symbol=symbol,
                start_date=request.start_date,
                end_date=request.end_date
            )
            returns = df['close'].pct_change().dropna()
            returns_list.append(returns)
        
        # Calculate statistics
        returns_df = pd.concat(returns_list, axis=1)
        returns_df.columns = request.symbols
        
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # Optimize
        optimizer = PortfolioOptimizer()
        
        if request.method == "mean_variance":
            weights = optimizer.mean_variance_optimization(expected_returns, cov_matrix)
        elif request.method == "risk_parity":
            weights = optimizer.risk_parity(cov_matrix)
        elif request.method == "hierarchical":
            weights = optimizer.hierarchical_risk_parity(returns_df)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        # Calculate portfolio statistics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        return PortfolioOptimizationResponse(
            weights={symbol: float(w) for symbol, w in zip(request.symbols, weights)},
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe)
        )
    
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Real-time streaming endpoint (WebSocket)
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

@app.websocket("/ws/market_data/{symbol}")
async def market_data_stream(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time market data
    
    Args:
        websocket: WebSocket connection
        symbol: Trading symbol
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for {symbol}")
    
    try:
        # Subscribe to event bus
        event_bus = get_event_bus()
        
        async def send_market_data(event):
            if event.data.get('symbol') == symbol:
                await websocket.send_json(event.to_dict())
        
        event_bus.subscribe("market_data", send_market_data)
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Echo back
            await websocket.send_text(f"Received: {data}")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {symbol}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# Model endpoints
@app.post("/api/v1/models/train")
async def train_model(
    model_type: str,
    symbol: str,
    start_date: str,
    end_date: str,
    background_tasks: BackgroundTasks
):
    """Train an alpha model"""
    try:
        from src.alpha_models.boosting_model import BoostingAlphaModel
        from src.alpha_models.lstm_model import LSTMAlphaModel
        
        # This would be done in background for long-running tasks
        def train_task():
            # Fetch and prepare data
            container = get_container()
            ingestion = container.resolve(DataIngestion)
            df = ingestion.fetch_yahoo_finance(symbol, start_date, end_date)
            
            # Generate features
            from src.feature_engineering.technical_features import TechnicalFeatures
            tech = TechnicalFeatures()
            df = tech.generate_all_features(df)
            
            # Prepare training data
            # ... (feature selection, train/test split)
            
            # Train model
            if model_type == "xgboost":
                model = BoostingAlphaModel(model_type="xgboost")
            elif model_type == "lstm":
                model = LSTMAlphaModel()
            
            # model.train(X_train, y_train, X_val, y_val)
            logger.info(f"Model training started for {symbol}")
        
        background_tasks.add_task(train_task)
        
        return {
            "status": "training_started",
            "model_type": model_type,
            "symbol": symbol
        }
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics endpoint
@app.get("/api/v1/stats")
async def get_statistics():
    """Get system statistics"""
    event_bus = get_event_bus()
    
    return {
        "event_subscribers": {
            "market_data": event_bus.get_subscriber_count("market_data"),
            "trade": event_bus.get_subscriber_count("trade"),
            "signal": event_bus.get_subscriber_count("signal")
        },
        "event_history_size": len(event_bus.get_history()),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    # Configure services
    from src.common.dependency_injection import configure_services
    configure_services()
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
