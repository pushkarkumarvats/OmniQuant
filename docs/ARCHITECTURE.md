# Architecture

## Overview

OmniQuant follows a **Hybrid Architecture**, leveraging Python for research flexibility and Rust for execution performance.

```
┌─────────────────────────────────────────────────────────────┐
│                     OmniQuant System                         │
├─────────────────────────────────────────────────────────────┤
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│   │  Frontend   │   │  Ray Dist.  │   │  Native OMS │       │
│   │  (Next.js)  │   │  Cluster    │   │   (Rust)    │       │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│          │                 │                 │              │
│          ▼                 ▼                 ▼              │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │    Data     │───▶│   Feature    │───▶│    Alpha     │    │
│  │  Platform   │    │ Engineering  │    │   Models     │    │
│  └─────────────┘    └──────────────┘    └──────────────┘    │
│         │                   │                    │          │
│         ▼                   ▼                    ▼          │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  Market     │◀───│  Strategy    │◀───│  Portfolio   │    │
│  │ Simulator   │    │   Engine     │    │  Optimizer   │    │
│  └─────────────┘    └──────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. Data Platform (`src/data_platform/` & `src/data_pipeline/`)

**Purpose**: Institutional-grade data management
- **Feature Store**: Point-in-time correct feature serving for training and inference.
- **Timeseries DB**: Abstractions for high-frequency data storage.
- **Data Pipeline**: Ingestion (`ingestion.py`), Cleaning (`cleaning.py`), and Alignment.

### 2. Native Execution (`native/oms-core/`)

**Purpose**: Low-latency Order Management System (OMS) and Matching Engine
- Written in **Rust** for microsecond-latency simulation.
- **Matching Engine**: Limit Order Book (LOB) with price-time priority.
- **Lock-free Data Structures**: Uses `crossbeam` for high-throughput messaging.

### 3. Distributed Research (`src/distributed/`)

**Purpose**: Scaling backtests and training
- Built on **Ray**.
- Parallel backtesting of parameter grids.
- Distributed training for RL agents and Alpha models.

### 4. Strategy & Alpha (`src/strategies/` & `src/alpha_models/`)

**Purpose**: Signal generation and decision making

- **Alpha Models**: 
  - **Deep Learning**: LSTM, Transformer (`transformer_model.py`).
  - **Gradient Boosting**: XGBoost, LightGBM.
  - **Statistical**: ARIMA-GARCH.
  
- **Strategies**:
  - **Reinforcement Learning**: PPO, SAC agents (`rl_agents.py`).
  - **Market Making**: Avellaneda-Stoikov.
  - **Momentum**: Trend following.

### 5. Feature Engineering (`src/feature_engineering/`)
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

### 5. Production-Readiness
- Logging (loguru)
- Configuration management (YAML)
- Error handling
- Type hints

These make the codebase easier to work with but don't constitute production hardening on their own.

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

## Possible future work

1. **Real data connectors** — Interactive Brokers, Alpaca, Polygon.io
2. **More models** — GANs for synthetic data, reinforcement learning
3. **Live trading** — real-time feeds, OMS, risk controls
4. **Alternative data** — sentiment, satellite imagery, web scraping

## References

- Harris, *Trading and Exchanges* (2003)
- Markowitz (1952); Lopez de Prado, *Building Diversified Portfolios* (2016)
- de Prado, *Advances in Financial Machine Learning* (2018)
- Almgren & Chriss, *Optimal Execution of Portfolio Transactions* (2001)
