# OmniQuant Quick Start Guide

Get up and running with OmniQuant in minutes!

## Installation

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/omniquant.git
cd omniquant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install OmniQuant
pip install -e .
```

### Option 2: Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access dashboard at http://localhost:8501
```

## Quick Examples

### 1. Generate Synthetic Data

```python
from src.data_pipeline.ingestion import DataIngestion

# Create data ingestion instance
ingestion = DataIngestion()

# Generate synthetic tick data
tick_data = ingestion.generate_synthetic_tick_data(num_ticks=10000, seed=42)

# Generate order book snapshots
orderbook_data = ingestion.generate_synthetic_orderbook(num_snapshots=1000, seed=42)

# Save data
ingestion.save_to_parquet(tick_data, "sample_ticks.parquet")
```

### 2. Feature Engineering

```python
from src.feature_engineering.microstructure_features import MicrostructureFeatures
from src.feature_engineering.technical_features import TechnicalFeatures

# Microstructure features
micro_gen = MicrostructureFeatures()
data_with_micro = micro_gen.generate_all_features(tick_data)

# Technical features
tech_gen = TechnicalFeatures()
data_with_tech = tech_gen.generate_all_features(df, price_col='close')
```

### 3. Train Alpha Model

```python
from src.alpha_models.boosting_model import BoostingAlphaModel

# Create and train model
model = BoostingAlphaModel(model_type='xgboost')
model.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance(top_n=10)
```

### 4. Run Backtest

```python
from src.simulator.event_simulator import EventSimulator
from src.strategies.momentum import MomentumStrategy

# Create strategy
strategy = MomentumStrategy(config={
    'lookback_period': 20,
    'entry_threshold': 2.0,
    'position_size': 100
})

# Run backtest
simulator = EventSimulator()
results = simulator.run_backtest(strategy, data, symbol="TEST")

# View results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

### 5. Portfolio Optimization

```python
from src.portfolio.optimizer import PortfolioOptimizer
import numpy as np

# Create optimizer
optimizer = PortfolioOptimizer()

# Mean-variance optimization
weights = optimizer.mean_variance_optimization(
    expected_returns=returns_mean,
    cov_matrix=returns_cov,
    risk_aversion=1.0
)

# Risk parity
weights_rp = optimizer.risk_parity(cov_matrix=returns_cov)

# Get portfolio statistics
stats = optimizer.get_portfolio_stats(weights, expected_returns, cov_matrix)
```

### 6. Launch Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/dashboard/app.py
```

Navigate to `http://localhost:8501` to access the interactive dashboard.

## Directory Structure

```
OmniQuant/
├── src/
│   ├── data_pipeline/       # Data ingestion and cleaning
│   ├── feature_engineering/ # Feature generation
│   ├── alpha_models/        # ML models
│   ├── simulator/           # Market simulator
│   ├── strategies/          # Trading strategies
│   ├── portfolio/           # Portfolio optimization
│   └── dashboard/           # Web dashboard
├── data/                    # Data storage
├── notebooks/               # Research notebooks
├── examples/                # Example scripts
├── tests/                   # Unit tests
└── configs/                 # Configuration files
```

## Running Examples

```bash
# Alpha research example
python notebooks/AlphaResearch_Example.py

# Simple backtest
python examples/simple_backtest.py
```

## Configuration

Edit `configs/config.yaml` to customize:
- Data sources and parameters
- Feature engineering settings
- Model hyperparameters
- Risk limits
- And more...

## Next Steps

1. **Read the Documentation**: Check out the full documentation in `docs/`
2. **Explore Notebooks**: Review research notebooks in `notebooks/`
3. **Customize Strategies**: Create your own strategies in `src/strategies/`
4. **Train Models**: Experiment with different alpha models
5. **Optimize Portfolio**: Try different optimization methods

## Troubleshooting

### Common Issues

**Import Errors**: Make sure you installed OmniQuant with `pip install -e .`

**Missing Dependencies**: Run `pip install -r requirements.txt` again

**Data Not Found**: Generate sample data first using `DataIngestion.generate_synthetic_tick_data()`

## Support

- **Documentation**: See `docs/` folder
- **Issues**: Report bugs on GitHub
- **Examples**: Check `examples/` folder

## What's Next?

- Implement real data connectors
- Add more alpha models (Transformers, GANs)
- Integrate live trading capabilities
- Build execution optimization algorithms
- Add alternative data sources

Happy trading! 🚀
