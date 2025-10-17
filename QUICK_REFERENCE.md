# OmniQuant Quick Reference Guide

## 🚀 All New Features at a Glance

---

## 1. Real-Time Data Connectors

### Alpaca Markets (Paper & Live Trading)
```python
from src.data_pipeline.real_time_connectors import create_connector

connector = create_connector('alpaca', ['AAPL', 'GOOGL'], config={
    'api_key': 'YOUR_API_KEY',
    'secret_key': 'YOUR_SECRET_KEY',
    'base_url': 'https://paper-api.alpaca.markets'
})

await connector.connect()
```

### Polygon.io (Market Data)
```python
connector = create_connector('polygon', ['AAPL'], config={
    'api_key': 'YOUR_POLYGON_KEY'
})
await connector.connect()
```

### Simulated (Testing)
```python
connector = create_connector('simulated', ['AAPL', 'GOOGL'], config={
    'update_interval': 1.0,  # seconds
    'volatility': 0.02
})
await connector.connect()
```

---

## 2. Advanced Feature Engineering

### Fractional Differentiation
```python
from src.feature_engineering.advanced_features import AdvancedFeatures

adv = AdvancedFeatures()

# Make series stationary while preserving memory
frac_diff = adv.fractional_differentiation(series, d=0.5)
```

### Wavelet Analysis
```python
# Multi-frequency decomposition
wavelet_features = adv.wavelet_features(df, price_col='close', levels=3)

# Returns: wavelet_detail_1, wavelet_detail_2, wavelet_detail_3, wavelet_approx
```

### Time Series Decomposition
```python
# Separate trend, seasonal, residual
decomp = adv.time_series_decomposition(series, period=50)
trend = decomp['trend']
seasonal = decomp['seasonal']
residual = decomp['residual']
```

### Hurst Exponent (Trend Detection)
```python
hurst = adv.hurst_exponent(series)
# H < 0.5: Mean-reverting
# H = 0.5: Random walk
# H > 0.5: Trending
```

### Spectral Features
```python
spectral = adv.spectral_features(series)
# Returns: centroid, spread, skewness, kurtosis, entropy
```

---

## 3. Transformer Model (NEW)

### Initialize & Train
```python
from src.alpha_models.transformer_model import TransformerAlphaModel

model = TransformerAlphaModel(
    input_dim=50,
    config={
        'd_model': 128,      # Model dimension
        'nhead': 8,          # Attention heads
        'num_layers': 3,     # Transformer layers
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 100
    }
)

# Train
model.train(X_train, y_train, X_val, y_val, seq_len=50)

# Predict
predictions = model.predict(X_test, seq_len=50)

# Save/Load
model.save_model('models/transformer.pth')
model.load_model('models/transformer.pth')
```

---

## 4. Dependency Injection

### Configure Services
```python
from src.common.dependency_injection import get_container, configure_services

# One-time configuration
configure_services()

# Resolve services
container = get_container()
ingestion = container.resolve(DataIngestion)
optimizer = container.resolve(PortfolioOptimizer)
risk_mgr = container.resolve(RiskManager)
```

### Register Custom Services
```python
# Singleton (one instance)
container.register_singleton(MyService, MyServiceImpl)

# Transient (new instance each time)
container.register_transient(MyStrategy)

# Factory
container.register_factory(ComplexService, factory_function)

# Existing instance
container.register_instance(Logger, logger_instance)
```

---

## 5. Event Bus (Pub/Sub)

### Subscribe to Events
```python
from src.common.event_bus import get_event_bus, MarketDataEvent

bus = get_event_bus()

def on_market_data(event):
    print(f"Price: {event.data['symbol']} @ {event.data['price']}")

def on_trade(event):
    print(f"Trade: {event.data}")

bus.subscribe("market_data", on_market_data)
bus.subscribe("trade", on_trade)
bus.subscribe("*", lambda e: print(f"Any event: {e.event_type}"))  # Wildcard
```

### Publish Events
```python
# Market data
event = MarketDataEvent(
    symbol="AAPL",
    price=150.0,
    volume=1000,
    source="my_feed"
)
bus.publish(event)

# Custom event
from src.common.event_bus import Event
custom = Event(
    event_type="custom_signal",
    data={'signal': 'buy', 'strength': 0.8}
)
bus.publish(custom)
```

### Event History
```python
# Get last 100 events
history = bus.get_history(limit=100)

# Filter by type
market_events = bus.get_history(event_type="market_data", limit=50)
```

### Redis-Backed Events (Distributed)
```python
from src.common.event_bus import RedisEventBus

redis_bus = RedisEventBus(redis_url="redis://localhost:6379")
redis_bus.subscribe_redis("market_data")
redis_bus.listen()  # Blocks and processes messages
```

---

## 6. FastAPI REST API

### Start Server
```bash
# Development
uvicorn src.api.main:app --reload --port 8000

# Production
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Generate Synthetic Data
```bash
curl -X POST "http://localhost:8000/api/v1/data/generate_synthetic?num_ticks=1000"
```

#### Fetch Market Data
```bash
curl -X POST http://localhost:8000/api/v1/data/fetch \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "interval": "1d"
  }'
```

#### Generate Features
```bash
curl -X POST http://localhost:8000/api/v1/features/generate \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "feature_types": ["technical", "microstructure"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  }'
```

#### Run Backtest
```bash
curl -X POST http://localhost:8000/api/v1/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_type": "momentum",
    "symbol": "AAPL",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000,
    "parameters": {
      "lookback_period": 20,
      "entry_threshold": 2.0
    }
  }'
```

#### Optimize Portfolio
```bash
curl -X POST http://localhost:8000/api/v1/portfolio/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "method": "risk_parity",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  }'
```

#### WebSocket (Real-Time Streaming)
```python
import asyncio
import websockets

async def stream_data():
    uri = "ws://localhost:8000/ws/market_data/AAPL"
    async with websockets.connect(uri) as ws:
        while True:
            data = await ws.recv()
            print(f"Received: {data}")

asyncio.run(stream_data())
```

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 7. Performance Tracker (Separated)

### Comprehensive Metrics
```python
from src.simulator.performance_tracker import PerformanceTracker

tracker = PerformanceTracker(initial_capital=100000)

# Update equity
for timestamp, equity in equity_curve:
    tracker.update(equity, timestamp)

# Record trades
for trade_pnl in trade_pnls:
    tracker.record_trade(trade_pnl)

# Get all metrics
metrics = tracker.get_metrics()
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"Max DD: {metrics.max_drawdown:.2%}")
print(f"Win Rate: {metrics.win_rate:.2%}")

# Print full summary
tracker.print_summary()
```

---

## 8. Testing

### Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Specific test file
pytest tests/test_advanced_features.py -v

# Specific test
pytest tests/test_orderbook.py::TestOrderBook::test_spread_calculation -v

# Parallel execution
pytest tests/ -n auto
```

### Run Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Individual checks
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

---

## 9. CI/CD Pipeline

### GitHub Actions Workflow
Automatically runs on push/PR:
1. **Test**: Multi-OS (Ubuntu, Windows, macOS), Multi-Python (3.9-3.11)
2. **Security**: Bandit, safety scans
3. **Docs**: Sphinx build
4. **Build**: Package creation
5. **Docker**: Container build and push

### Manual Trigger
```bash
# Push to main branch
git push origin main

# Create pull request
gh pr create --base main
```

### Local Simulation
```bash
# Run what CI will run
black --check src tests
isort --check src tests
flake8 src tests
mypy src
pytest tests/ --cov=src
```

---

## 10. Docker Deployment

### Build Image
```bash
docker build -t omniquant:latest .
```

### Run Container
```bash
# Dashboard
docker run -p 8501:8501 omniquant:latest

# API
docker run -p 8000:8000 omniquant:latest uvicorn src.api.main:app --host 0.0.0.0

# With volumes
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models omniquant:latest
```

### Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 11. Advanced Portfolio Optimization

### CVaR Optimization (NEW)
```python
from src.portfolio.optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer()

# Conditional Value at Risk optimization
weights = optimizer.cvar_optimization(
    returns=returns_df,
    alpha=0.05,  # 5% tail risk
    target_return=0.12
)
```

### Transaction Cost-Aware (NEW)
```python
# Optimization considering rebalancing costs
weights = optimizer.transaction_cost_aware_optimization(
    expected_returns=returns,
    cov_matrix=cov,
    current_weights=current_portfolio,
    transaction_cost=0.001  # 10 bps
)
```

---

## 12. Configuration

### Update Config File
Edit `configs/config.yaml`:
```yaml
data_pipeline:
  tick_size: 0.01
  cache_enabled: true

simulator:
  commission_rate: 0.0002
  slippage_bps: 1.0
  latency_mean_ms: 10
  latency_std_ms: 2

strategies:
  momentum:
    lookback_period: 20
    entry_threshold: 2.0
```

### Environment Variables
Create `.env`:
```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
POLYGON_API_KEY=your_polygon_key
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/omniquant
```

Load with:
```python
from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv('ALPACA_API_KEY')
```

---

## 13. Common Workflows

### Complete Research Pipeline
```python
# 1. Get data
from src.data_pipeline.ingestion import DataIngestion
ingestion = DataIngestion()
df = ingestion.fetch_yahoo_finance('AAPL', '2024-01-01', '2024-12-31')

# 2. Generate features
from src.feature_engineering.technical_features import TechnicalFeatures
from src.feature_engineering.advanced_features import AdvancedFeatures

tech = TechnicalFeatures()
df = tech.generate_all_features(df)

adv = AdvancedFeatures()
df['frac_diff'] = adv.fractional_differentiation(df['close'], d=0.5)
df['hurst'] = adv.hurst_exponent(df['close'].rolling(100))

# 3. Train model
from src.alpha_models.transformer_model import TransformerAlphaModel
model = TransformerAlphaModel(input_dim=len(df.columns))
model.train(X_train, y_train, X_val, y_val)

# 4. Backtest
from src.simulator.event_simulator import EventSimulator
from src.strategies.momentum import MomentumStrategy

strategy = MomentumStrategy()
simulator = EventSimulator()
results = simulator.run_backtest(strategy, df, 'AAPL')

# 5. Optimize portfolio
from src.portfolio.optimizer import PortfolioOptimizer
optimizer = PortfolioOptimizer()
weights = optimizer.hierarchical_risk_parity(returns_df)
```

---

## 14. Performance Tips

### Numba Acceleration
Features already use Numba where possible. For custom functions:
```python
from numba import jit

@jit(nopython=True)
def fast_calculation(data):
    # Your computation
    return result
```

### Parallel Processing
```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(process_symbol)(symbol) 
    for symbol in symbols
)
```

### GPU Acceleration (Transformer)
```python
# Automatically uses GPU if available
model = TransformerAlphaModel(input_dim=50)
# Check device
print(model.device)  # cuda or cpu
```

---

## 15. Troubleshooting

### Common Issues

**Import Error**: Module not found
```bash
pip install -e .
# or
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Missing Dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**API Connection Error**
```python
# Check API keys in .env file
# Verify broker API status
# Check network connectivity
```

**Test Failures**
```bash
# Clear cache
pytest --cache-clear

# Update dependencies
pip install --upgrade -r requirements.txt
```

---

## 📚 Documentation Links

- **Main README**: `README.md`
- **Getting Started**: `GETTING_STARTED.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Contributing**: `CONTRIBUTING.md`
- **Improvements**: `IMPROVEMENTS_COMPLETE.md`
- **Final Status**: `FINAL_STATUS.md`
- **API Docs**: http://localhost:8000/docs (when server running)

---

## 🆘 Getting Help

1. Check documentation files
2. Review examples in `examples/` directory
3. Run example scripts in `notebooks/`
4. Check test files for usage examples
5. Access API documentation at `/docs` endpoint

---

**Quick Reference Version**: 1.0  
**Last Updated**: 2025-01-17  
**Status**: Complete & Production-Ready ✅
