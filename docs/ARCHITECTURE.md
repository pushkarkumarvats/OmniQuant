# OmniQuant Architecture Documentation

## System Overview

OmniQuant is designed as a modular, production-ready quantitative trading research platform that emulates the complete workflow inside a professional trading firm.

```
┌─────────────────────────────────────────────────────────────┐
│                     OmniQuant System                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    Data     │───▶│   Feature    │───▶│    Alpha     │  │
│  │  Pipeline   │    │ Engineering  │    │   Models     │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │          │
│         ▼                   ▼                    ▼          │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Market     │◀───│  Strategy    │◀───│  Portfolio   │  │
│  │ Simulator   │    │   Engine     │    │  Optimizer   │  │
│  └─────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │          │
│         └───────────────────┴────────────────────┘          │
│                             ▼                                │
│                    ┌──────────────┐                         │
│                    │  Monitoring  │                         │
│                    │ & Dashboard  │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. Data Pipeline (`src/data_pipeline/`)

**Purpose**: Ingest, clean, and prepare market data

**Components**:
- **Ingestion** (`ingestion.py`): Load data from multiple sources
  - CSV, Parquet, databases
  - Yahoo Finance API
  - Synthetic data generation
- **Cleaning** (`cleaning.py`): Data quality and preprocessing
  - Outlier detection (Z-score, IQR)
  - Missing value handling
  - Data validation
- **Alignment** (`alignment.py`): Synchronize multi-source data
  - Time alignment
  - Trading calendar filtering
  - Regular grid creation

**Key Features**:
- Multi-format support
- Efficient columnar storage (Parquet)
- Data quality checks
- Synthetic data generation for testing

### 2. Feature Engineering (`src/feature_engineering/`)

**Purpose**: Generate predictive features from raw data

**Components**:
- **Microstructure Features** (`microstructure_features.py`)
  - Order Flow Imbalance (OFI)
  - Bid-Ask Spread
  - Order Book Depth
  - Trade Intensity
  - Price Impact
  
- **Technical Features** (`technical_features.py`)
  - Momentum indicators
  - Moving averages (SMA, EMA)
  - Volatility measures
  - RSI, MACD, Bollinger Bands
  - VWAP and deviations
  
- **Causal Features** (`causal_features.py`)
  - Granger causality testing
  - Mutual information
  - Feature interactions
  - Lagged features

**Performance Optimization**:
- Numba JIT compilation for critical paths
- Vectorized operations
- Caching of intermediate results

### 3. Alpha Models (`src/alpha_models/`)

**Purpose**: Predict future returns using machine learning

**Components**:
- **LSTM Model** (`lstm_model.py`)
  - Deep learning for sequence prediction
  - Bidirectional LSTM support
  - Attention mechanisms (optional)
  
- **Boosting Models** (`boosting_model.py`)
  - XGBoost, LightGBM, CatBoost
  - Hyperparameter optimization (Optuna)
  - Feature importance analysis
  
- **Statistical Models** (`statistical_model.py`)
  - ARIMA-GARCH for volatility
  - Kalman filters for state estimation
  - Cointegration testing
  - Regime detection (HMM)
  
- **Ensemble Models** (`ensemble_model.py`)
  - Stacking, blending
  - Weighted averaging
  - Bayesian model averaging

**Model Pipeline**:
1. Feature selection
2. Cross-validation
3. Hyperparameter tuning
4. Model training
5. Performance evaluation
6. Feature importance analysis

### 4. Market Simulator (`src/simulator/`)

**Purpose**: Event-driven backtesting with realistic market microstructure

**Components**:
- **Order Book** (`orderbook.py`)
  - Price-time priority matching
  - Limit and market orders
  - Bid/ask depth tracking
  
- **Matching Engine** (`matching_engine.py`)
  - Order routing and execution
  - Latency simulation
  - Price impact modeling
  - Commission calculation
  
- **Event Simulator** (`event_simulator.py`)
  - Event-driven backtesting
  - Portfolio tracking
  - Performance metrics

**Market Microstructure**:
- Realistic order book dynamics
- Configurable latency (mean ± std)
- Multiple impact models (linear, sqrt, power)
- Slippage modeling

### 5. Trading Strategies (`src/strategies/`)

**Purpose**: Implement trading logic

**Components**:
- **Base Strategy** (`base_strategy.py`)
  - Abstract base class
  - Common utilities
  
- **Market Maker** (`market_maker.py`)
  - Avellaneda-Stoikov framework
  - Inventory management
  - Spread optimization
  
- **Momentum** (`momentum.py`)
  - Trend following
  - Statistical signals
  - Stop loss management
  
- **Arbitrage** (`arbitrage.py`)
  - Statistical arbitrage
  - Pairs trading
  - Mean reversion

**Strategy Lifecycle**:
1. Initialize
2. On data (event handler)
3. Generate signals
4. Execute orders
5. Manage risk
6. Finalize

### 6. Portfolio Management (`src/portfolio/`)

**Purpose**: Multi-strategy allocation and risk management

**Components**:
- **Optimizer** (`optimizer.py`)
  - Mean-variance optimization
  - Risk parity
  - Hierarchical Risk Parity (HRP)
  - Black-Litterman
  - Maximum diversification
  
- **Risk Manager** (`risk_manager.py`)
  - Position limits
  - VaR/CVaR calculation
  - Drawdown monitoring
  - Concentration limits
  
- **Regime Detector** (`regime_detector.py`)
  - Hidden Markov Models
  - K-means clustering
  - Regime-dependent allocation

**Optimization Methods**:
- **Mean-Variance**: Classic Markowitz
- **Risk Parity**: Equal risk contribution
- **HRP**: Hierarchical clustering + risk parity
- **Black-Litterman**: Bayesian approach with views

### 7. Dashboard (`src/dashboard/`)

**Purpose**: Interactive visualization and monitoring

**Technology**: Streamlit

**Features**:
- Real-time PnL tracking
- Feature importance visualization
- Portfolio allocation charts
- Risk metrics dashboard
- Backtest visualization
- Regime detection plots

## Data Flow

```
Raw Data → Cleaning → Feature Engineering → Model Training
                              ↓
                         Predictions
                              ↓
         Trading Signals ← Strategy Logic
                              ↓
         Order Execution ← Simulator
                              ↓
         Portfolio Updates → Risk Management
                              ↓
         Performance Tracking → Dashboard
```

## Design Principles

### 1. Modularity
Each component is independent and can be used standalone

### 2. Extensibility
Easy to add new:
- Data sources
- Features
- Models
- Strategies
- Optimization methods

### 3. Performance
- Vectorized operations
- JIT compilation (Numba)
- Efficient data structures
- Caching

### 4. Testability
- Unit tests for all modules
- Integration tests
- Synthetic data for testing

### 5. Production-Ready
- Logging (loguru)
- Configuration management (YAML)
- Error handling
- Type hints

## Configuration

All modules are configurable via:
1. Config files (`configs/config.yaml`)
2. Constructor parameters
3. Environment variables

## Performance Optimization

### Data Layer
- Parquet for columnar storage
- DuckDB for fast analytics
- Lazy loading

### Computation
- Numba for hot paths
- NumPy vectorization
- Parallel processing (Ray/Dask)

### Memory
- Chunking for large datasets
- Generator patterns
- Memory profiling

## Deployment

### Development
```bash
python -m src.dashboard.app
```

### Production
```bash
docker-compose up -d
```

### Scaling
- Horizontal: Multiple workers (Ray)
- Vertical: GPU acceleration (CUDA)
- Distributed: Kubernetes cluster

## Future Enhancements

1. **Real Data Connectors**
   - Interactive Brokers
   - Alpaca
   - Polygon.io

2. **Advanced Models**
   - Transformers
   - GANs for synthetic data
   - Reinforcement learning

3. **Live Trading**
   - Real-time data feeds
   - Order management system
   - Risk controls

4. **Alternative Data**
   - Sentiment analysis
   - Satellite imagery
   - Web scraping

## References

- **Market Microstructure**: Harris (2003)
- **Portfolio Optimization**: Markowitz (1952), Lopez de Prado (2016)
- **Alpha Research**: Advances in Financial Machine Learning
- **Execution**: Almgren & Chriss (2001)
