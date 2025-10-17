# OmniQuant — Quantitative Research & Trading Simulation Framework

**A comprehensive research and simulation platform** for algorithmic trading strategy development and backtesting. Designed as an educational and research tool that demonstrates professional quantitative trading workflows.

⚠️ **Note**: This is a research/simulation framework, not production trading software. It lacks the robustness, security, real-time data feeds, and broker integrations required for live trading.

OmniQuant emulates the complete quant research pipeline inside a trading firm:

```
Data Ingestion → Feature Engineering → Alpha Discovery → Strategy Simulation → Portfolio Optimization → Visualization
```

## 🎯 Key Features

- **Multi-Layer Architecture**: Data, Feature/Alpha, Strategy, Execution, Portfolio, and Monitoring layers
- **Hybrid Performance**: Python for research/modeling, C++ for high-speed simulation
- **Multi-Agent System**: Market Maker, Momentum Trader, and Arbitrageur agents
- **Advanced ML/RL**: LSTM, Transformers, Gradient Boosting, RL-based execution
- **Causal Inference**: Feature causality and regime dependency analysis
- **Real-Time Dashboard**: Interactive PnL, feature importance, and order book visualization

## 📊 Architecture

```
+------------------------------------------------------------+
|                        OmniQuant                           |
|                                                            |
|  [Data Layer] → [Feature/Alpha Layer] → [Strategy Engine]  |
|         ↓                        ↓                ↓         |
|  Ingestion, LOB sim         ML models         Execution sim |
|                                                            |
|  [Portfolio Manager] ← [Risk/Regime Model] ← [Monitoring]  |
+------------------------------------------------------------+
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Build C++ simulator (optional for high-performance mode)
cd src/simulator
mkdir build && cd build
cmake ..
make
cd ../../..
```

### Run Sample Backtest

```python
from src.strategies.momentum import MomentumStrategy
from src.simulator.interface import EventSimulator
from src.portfolio.optimizer import PortfolioOptimizer

# Initialize components
simulator = EventSimulator()
strategy = MomentumStrategy()
portfolio = PortfolioOptimizer()

# Run backtest
results = simulator.run_backtest(strategy, data='data/processed/sample.parquet')
print(f"Sharpe Ratio: {results.sharpe:.2f}")
```

### Launch Dashboard

```bash
streamlit run src/dashboard/app.py
```

## 📁 Project Structure

```
OmniQuant/
├── data/
│   ├── raw/                    # Raw tick/bar data
│   └── processed/              # Cleaned and aligned data
├── src/
│   ├── data_pipeline/          # Data ingestion and cleaning
│   ├── feature_engineering/    # Alpha feature generation
│   ├── alpha_models/           # ML/statistical models
│   ├── simulator/              # C++ event-driven simulator
│   ├── strategies/             # Trading strategies (MM, momentum, arb)
│   ├── execution/              # Order execution algorithms
│   ├── portfolio/              # Portfolio optimization and risk
│   ├── monitoring/             # Regime detection and metrics
│   └── dashboard/              # Visualization and reporting
├── notebooks/                  # Research notebooks
├── tests/                      # Unit and integration tests
├── configs/                    # Configuration files
├── docker/                     # Docker setup
└── docs/                       # Documentation
```

## 🧮 Components

### 1. Data Pipeline
- Tick and LOB data ingestion
- Multi-source data alignment
- Efficient storage (Parquet, DuckDB)

### 2. Feature Engineering
- **Microstructure Features**: OFI, order book imbalance, spread, volume clustering
- **Technical Features**: Momentum, volatility, VWAP deviations
- **Causal Features**: Granger causality, feature interaction graphs

### 3. Alpha Models
- **ML Models**: LSTM, Transformer, XGBoost, LightGBM
- **Statistical Models**: ARIMA-GARCH, Kalman filters, cointegration
- **Feature Selection**: Mutual information, SHAP, Granger causality

### 🎯 Project Overview

OmniQuant is an **educational research platform** that demonstrates quantitative trading workflows:

- **Data Pipeline**: Synthetic data generation and multi-format ingestion (CSV, Parquet, APIs)
- **Feature Engineering**: Technical indicators, microstructure features, and causal analysis
- **Alpha Models**: Machine learning implementations (LSTM, XGBoost, LightGBM, statistical models)
- **Market Simulator**: Event-driven backtesting with simulated order book matching
- **Trading Strategies**: Example implementations (Market Making, Momentum, Pairs Trading)
- **Portfolio Management**: Optimization algorithms (Mean-Variance, Risk Parity, HRP)
- **Visualization**: Interactive Streamlit dashboard for analysis

**Use Cases**: Learning quantitative finance, strategy research, backtesting experiments, portfolio projectsation (HMM)

### 4. Market Simulator
- C++ event-driven order book engine
- Realistic latency and slippage modeling
- Python bindings via pybind11

### 5. Multi-Agent Strategies
- **Market Maker**: Inventory control + spread optimization (RL-based)
- **Momentum Trader**: Statistical prediction with adaptive sizing
- **Arbitrageur**: Cross-market and statistical arbitrage

### 6. Execution Optimization
- TWAP, VWAP, POV, Implementation Shortfall
- RL-based adaptive execution

### 7. Portfolio Management
- Bayesian model averaging
- Risk parity and volatility targeting
- Regime-dependent allocation (HMM)

### 8. Risk & Monitoring
- Real-time PnL tracking
- Regime detection (HMM, clustering)
- Drawdown analysis

## 📈 Performance Metrics

| Category | Metrics |
|----------|---------|
| **Performance** | Annualized return, Sharpe, Sortino, max drawdown, turnover |
| **Market Making** | Inventory variance, spread PnL, quote fill ratio |
| **Execution** | Slippage, participation rate, latency-adjusted PnL |
| **Modeling** | Feature importance, predictive power (AUC, MI) |
| **Portfolio** | Correlation, diversification ratio, risk contribution |

## 🔬 Research Notebooks

- `notebooks/AlphaResearch.ipynb` - Feature exploration and alpha discovery
- `notebooks/BacktestReport.ipynb` - Strategy backtesting and analysis
- `notebooks/RiskAnalysis.ipynb` - Portfolio risk decomposition
- `notebooks/RegimeAnalysis.ipynb` - Market regime detection

## 🛠️ Technology Stack

- **Languages**: Python 3.9+, C++17
- **Data Processing**: pandas, polars, pyarrow, duckdb
- **ML/AI**: PyTorch, scikit-learn, xgboost, lightgbm, stable-baselines3
- **Optimization**: cvxpy, pymoo, optuna
- **Causal Modeling**: dowhy, econml
- **Visualization**: plotly, dash, streamlit
- **Infrastructure**: Docker, Ray, PostgreSQL

## 🎓 Use Cases

1. **Algorithmic Trading Research**: Test new alpha ideas and strategies
2. **Market Microstructure Analysis**: Study order book dynamics
3. **Execution Optimization**: Minimize trading costs
4. **Portfolio Construction**: Multi-strategy allocation
5. **Educational**: Learn quantitative finance and algorithmic trading

## 📚 Documentation

See `docs/` for detailed documentation:
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Strategy Development Guide](docs/strategy_guide.md)
- [Performance Optimization](docs/optimization.md)

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This project draws inspiration from:
- Academic research in market microstructure
- Open-source backtesting frameworks (Backtrader, Zipline)
- Industry best practices in quantitative finance

## 📧 Contact

For questions or collaboration: [Your Contact Info]

---

**Disclaimer**: This framework is for research and educational purposes only. Past performance does not guarantee future results. Trading involves substantial risk.
