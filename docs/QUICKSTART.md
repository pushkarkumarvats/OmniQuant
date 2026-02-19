# OmniQuant — Quickstart Guide

Get a backtest running in **under 5 minutes**.

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Rust | 1.70+ |
| Node.js | 18+ |
| pip | latest |
| Git | any |

## 1. Clone & Install

```bash
git clone https://github.com/yourusername/omniquant.git
cd omniquant

# 1. Python Environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

# 2. Build Native Engine (Rust)
cd native/oms-core
cargo build --release
cd ../..
```

## 2. Run Your First Backtest

Create a file called `my_backtest.py` in the project root:

```python
import pandas as pd
import numpy as np
from src.strategies.momentum import MomentumStrategy
from src.simulator.event_simulator import EventSimulator, SimulationConfig

# --- Generate synthetic price data ---
np.random.seed(42)
n = 1000
prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))

data = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
    "price": prices,
    "close": prices,
    "bid": prices * 0.9999,
    "ask": prices * 1.0001,
    "volume": np.random.randint(100, 5000, n),
})

# --- Configure and run ---
config = SimulationConfig(initial_capital=100_000, commission_rate=0.0002)
sim = EventSimulator(sim_config=config)

strategy = MomentumStrategy(config={
    "lookback_period": 20,
    "entry_threshold": 2.0,
    "position_size": 100,
})

results = sim.run_backtest(strategy, data, symbol="SYNTH")

print(f"Final Equity : ${results['final_equity']:,.2f}")
print(f"Total Return : {results['total_return']:.2%}")
print(f"Sharpe Ratio : {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown : {results['max_drawdown']:.2%}")
print(f"Total Trades : {results['total_trades']}")
```

Run it:

```bash
python my_backtest.py
```

## 3. Use the API & Dashboard

```bash
# Start the Backend API
uvicorn src.api.main:app --reload

# Start the Frontend Dashboard (in a new terminal)
cd frontend
npm run dev
# Dashboard available at http://localhost:3000
```

Or connect via **WebSocket** for live progress:

```
ws://localhost:8000/ws/backtest/<job_id>
```

## 4. Create a Custom Strategy

```bash
python -m src.cli new-strategy --name MeanReversion
```

This generates `src/strategies/mean_reversion.py` with boilerplate you can fill in.

## 5. Run Tests

```bash
pytest tests/ -v
```

## 6. Project Layout (key files)

| Path | Purpose |
|------|---------|
| `src/simulator/event_simulator.py` | Core backtest engine |
| `src/simulator/exchange_emulator.py` | Order execution / matching |
| `src/strategies/` | Trading strategy implementations |
| `src/api/main.py` | FastAPI REST + WebSocket layer |
| `src/common/dependency_injection.py` | IoC container |
| `tests/` | Regression, fuzz, and unit tests |
| `Makefile` | Developer shortcuts |

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for a deeper dive.
- Explore `notebooks/AlphaResearch_Example.py` for feature engineering.
- Check `src/strategies/` for more strategy examples.
