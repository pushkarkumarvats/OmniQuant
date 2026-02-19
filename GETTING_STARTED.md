# Getting Started

Set up OmniQuant and run your first backtest.

## Prerequisites

- **Python 3.10 or higher** - Check with `python --version`
- **Rust 1.70+ & Cargo** - For the native OMS engine. Check with `cargo --version`
- **Node.js 18+** - For the frontend dashboard. Check with `node --version`
- **pip** - Python package manager
- **Git** (optional) - For cloning the repository
- **8GB RAM minimum** - For running simulations
- **Basic Python knowledge** - Understanding of pandas, numpy

## Step 1: Installation

### Option A: Clone from Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/omniquant.git
cd omniquant

# 1. Python Setup
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Rust Native Extensions Setup
cd native/oms-core
cargo build --release
cd ../..

# 3. Frontend Setup (Optional)
cd frontend
npm install
cd ..
```

### Option B: Docker (Recommended for Full Stack)

```bash
docker-compose up -d
```

### Verify Installation

```bash
# Test imports (Python & Rust binding)
python -c "from src.data_pipeline.ingestion import DataIngestion; import oms_core; print('Core Systems: OK')"
```

## Step 2: Generate Sample Data

Since we don't have real market data yet, let's generate synthetic data for testing:

```python
# save as: generate_data.py
from src.data_pipeline.ingestion import DataIngestion
from loguru import logger

# Create data ingestion instance
ingestion = DataIngestion()

# Generate 10,000 tick records
logger.info("Generating tick data...")
tick_data = ingestion.generate_synthetic_tick_data(
    num_ticks=10000,
    initial_price=100.0,
    volatility=0.02,
    seed=42  # For reproducibility
)

# Save to file
ingestion.save_to_parquet(tick_data, "sample_ticks.parquet")
logger.info(f"Generated and saved {len(tick_data)} ticks")

# Preview the data
print("\nFirst 5 rows:")
print(tick_data.head())
print("\nData shape:", tick_data.shape)
print("\nColumns:", tick_data.columns.tolist())
```

Run it:
```bash
python generate_data.py
```

**Expected Output:**
```
2024-01-01 10:00:00 | INFO  | Generating 10000 synthetic ticks
2024-01-01 10:00:01 | INFO  | Generated synthetic tick data with 10000 rows
2024-01-01 10:00:01 | INFO  | Saved 10000 rows to data/raw/sample_ticks.parquet
✓ Generated and saved 10000 ticks

First 5 rows:
                 timestamp      price  ...
0 2024-01-01 09:00:00.000  100.01000  ...
...
```

## Step 3: Feature Engineering

Now let's add technical indicators to our data:

```python
# save as: create_features.py
import pandas as pd
from src.data_pipeline.ingestion import DataIngestion
from src.feature_engineering.technical_features import TechnicalFeatures
from loguru import logger

# Load the data we generated
ingestion = DataIngestion()
tick_data = ingestion.load_parquet("data/raw/sample_ticks.parquet")

# Resample to 1-minute bars (required for most features)
logger.info("Resampling to 1-minute bars...")
df_bars = tick_data.set_index('timestamp').resample('1min').agg({
    'price': 'last',
    'volume': 'sum',
    'bid': 'last',
    'ask': 'last'
}).dropna().reset_index()

logger.info(f"Created {len(df_bars)} bars")

# Generate features
tech = TechnicalFeatures()
logger.info("Generating technical features...")
df_with_features = tech.generate_all_features(df_bars, price_col='price')

# Remove rows with NaN (from rolling calculations)
df_clean = df_with_features.dropna()

logger.info(f"✓ Features created: {len(df_with_features.columns)} columns")
logger.info(f"✓ Clean data: {len(df_clean)} rows")

# Save for later use
ingestion.save_to_parquet(df_clean, "data_with_features.parquet")

# Show feature names
print("\nAvailable features:")
for i, col in enumerate(df_clean.columns, 1):
    print(f"{i:2d}. {col}")
```

Run it:
```bash
python create_features.py
```

## Step 4: Your First Backtest

Let's run a simple momentum strategy:

```python
# save as: my_first_backtest.py
from src.data_pipeline.ingestion import DataIngestion
from src.simulator.event_simulator import EventSimulator, SimulationConfig
from src.strategies.momentum import MomentumStrategy
from loguru import logger

# Load data with features
ingestion = DataIngestion()
df = ingestion.load_parquet("data/raw/data_with_features.parquet")

logger.info(f"Loaded {len(df)} bars for backtesting")

# Create strategy with parameters
strategy = MomentumStrategy(config={
    'lookback_period': 20,      # Look at 20 bars of history
    'entry_threshold': 1.5,     # Enter when z-score > 1.5
    'exit_threshold': 0.5,      # Exit when z-score < 0.5
    'position_size': 100,       # Trade 100 shares
    'stop_loss': 0.05           # Stop loss at 5%
})

# Create simulator
sim_config = SimulationConfig(
    initial_capital=100000.0,
    commission_rate=0.0002,     # 2 basis points
    slippage_bps=1.0
)

simulator = EventSimulator(sim_config=sim_config)

# Run backtest
logger.info("Starting backtest...")
results = simulator.run_backtest(
    strategy=strategy,
    data=df,
    symbol="TEST"
)

# Display results
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"Initial Capital:    ${results['initial_capital']:,.2f}")
print(f"Final Equity:       ${results['final_equity']:,.2f}")
print(f"Total Return:       {results['total_return']:>8.2%}")
print(f"Sharpe Ratio:       {results['sharpe_ratio']:>8.2f}")
print(f"Max Drawdown:       {results['max_drawdown']:>8.2%}")
print(f"Total Trades:       {results['total_trades']:>8d}")
print(f"Win Rate:           {results['win_rate']:>8.2%}")
print("="*60)
```

Run it:
```bash
python my_first_backtest.py
```

**Expected Output:**
```
=================================================================
BACKTEST RESULTS
============================================================
Initial Capital:    $100,000.00
Final Equity:       $102,450.00
Total Return:            2.45%
Sharpe Ratio:            0.85
Max Drawdown:           -4.20%
Total Trades:              42
Win Rate:              57.14%
============================================================
```

## Step 5: Visualize Results (Optional)

Create a simple equity curve visualization:

```python
# save as: visualize.py
import pandas as pd
import matplotlib.pyplot as plt
from src.data_pipeline.ingestion import DataIngestion
from src.simulator.event_simulator import EventSimulator
from src.strategies.momentum import MomentumStrategy

# Run backtest (same as before)
ingestion = DataIngestion()
df = ingestion.load_parquet("data/raw/data_with_features.parquet")

strategy = MomentumStrategy(config={'lookback_period': 20, 'entry_threshold': 1.5})
simulator = EventSimulator()
results = simulator.run_backtest(strategy, df, symbol="TEST")

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.plot(results['equity_curve'])
plt.title('Equity Curve - Momentum Strategy')
plt.xlabel('Time Steps')
plt.ylabel('Equity ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('equity_curve.png', dpi=300)
print("Saved equity_curve.png")
```

## Step 6: Launch the Dashboard

Start the interactive web dashboard:

```bash
streamlit run src/dashboard/app.py
```

Then open your browser to `http://localhost:8501`

## Common Issues and Solutions

### Issue: Module not found

```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Make sure you installed in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### Issue: No data found

```
FileNotFoundError: data/raw/sample_ticks.parquet
```

**Solution:**
```bash
# Make sure you ran the data generation script first
python generate_data.py
```

### Issue: Import errors for optional dependencies

```
ImportError: yfinance not installed
```

**Solution:**
```bash
# Install missing package
pip install yfinance

# Or install all optional dependencies
pip install -r requirements.txt
```

## Next Steps

Now that you have the basics working, try:

### 1. Customize the Strategy

Modify the momentum strategy parameters:
```python
strategy = MomentumStrategy(config={
    'lookback_period': 30,      # Try different periods
    'entry_threshold': 2.0,     # More conservative entry
    'position_size': 200,       # Larger positions
})
```

### 2. Try Different Strategies

```python
from src.strategies.market_maker import MarketMakerStrategy

strategy = MarketMakerStrategy(config={
    'spread_bps': 10,
    'inventory_limit': 1000
})
```

### 3. Add More Features

```python
from src.feature_engineering.causal_features import CausalFeatures

causal = CausalFeatures()
df_causal = causal.create_lagged_features(df, ['price', 'volume'], lags=[1, 2, 5])
```

### 4. Train a Machine Learning Model

```python
from src.alpha_models.boosting_model import BoostingAlphaModel

model = BoostingAlphaModel(model_type='xgboost')
model.train(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

### 5. Optimize Your Portfolio

```python
from src.portfolio.optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()
weights = optimizer.risk_parity(cov_matrix)
```

## Learning Resources

- **Examples**: Check `examples/` directory for more scripts
- **Notebooks**: See `notebooks/` for research workflows
- **Documentation**: Read `docs/ARCHITECTURE.md` for system design
- **Tests**: Look at `tests/` for usage examples

## Getting Help

- **Issues**: Check existing issues on GitHub
- **Documentation**: Read the full README.md
- **Examples**: Run example scripts in `examples/` directory
- **Community**: Join discussions (if available)

## Project Structure Quick Reference

```
omniquant/
├── src/
│   ├── data_pipeline/      # Load and clean data
│   ├── feature_engineering/# Create features
│   ├── alpha_models/       # Train ML models
│   ├── simulator/          # Run backtests
│   ├── strategies/         # Trading strategies
│   ├── portfolio/          # Portfolio optimization
│   └── dashboard/          # Web interface
├── data/
│   ├── raw/                # Your generated data goes here
│   └── processed/          # Processed data
├── examples/               # Example scripts
├── notebooks/              # Research notebooks
└── tests/                  # Unit tests
```

## Best Practices

1. **Always use a virtual environment** - Keeps dependencies isolated
2. **Start with synthetic data** - Test your strategies before using real data
3. **Check for lookahead bias** - Make sure features don't use future data
4. **Monitor performance** - Track metrics like Sharpe ratio and drawdown
5. **Test thoroughly** - Run unit tests before deploying strategies
6. **Version control** - Use git to track changes
7. **Document your work** - Add comments and docstrings

## What's Next?

You now have a working OmniQuant installation! Here's a suggested learning path:

1. Run all example scripts in `examples/`
2. Read the architecture docs (`docs/ARCHITECTURE.md`)
3. Experiment with different strategies
4. Train an ML model
5. Build a custom strategy
6. Optimise a multi-asset portfolio
