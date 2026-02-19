# OmniQuant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust 2021](https://img.shields.io/badge/rust-2021_edition-orange.svg)](https://www.rust-lang.org/)
[![TypeScript](https://img.shields.io/badge/typescript-5.3-blue.svg)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Quantitative trading research & simulation platform** — a hybrid Python / Rust / TypeScript system for strategy research, alpha generation, backtesting, and execution simulation.

The platform includes a native **Rust Order Management System** with a BTreeMap limit-order book, ITCH 5.0 parser, and append-only event journal, bridged to a Python ML/alpha layer and a Next.js monitoring dashboard.

> **Disclaimer:** This is a research and simulation platform. Do not deploy against live markets without independent validation, additional hardening, and proper regulatory compliance.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Next.js / React Dashboard                        │
│              (TypeScript · TanStack Query · Tailwind)                │
├──────────────────────────────────────────────────────────────────────┤
│                     FastAPI Gateway  (REST + WebSocket)               │
│         /api/v1/backtest · /api/v1/portfolio · /ws/market_data       │
├──────────────────────────────────────────────────────────────────────┤
│                      Message Transport Layer                         │
│  InMemory (default)  |  Aeron (opt-in)  |  Kafka (opt-in)          │
├──────────────────────────────────────────────────────────────────────┤
│              Rust Native OMS  (oms-core · cdylib FFI)                │
│  ┌───────────┐ ┌────────────┐ ┌───────────┐ ┌───────────────────┐  │
│  │ Matching   │ │ ITCH 5.0   │ │ Event     │ │ Software Risk     │  │
│  │ Engine     │ │ Parser     │ │ Journal   │ │ Gate (SW ref.     │  │
│  │ (BTreeMap) │ │            │ │ (WAL)     │ │ implementation)   │  │
│  └───────────┘ └────────────┘ └───────────┘ └───────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│                   Python Alpha & Research Layer                      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────┐   │
│  │ ML Alpha   │ │ Strategies │ │ Portfolio  │ │ Risk Engine    │   │
│  │ Models     │ │            │ │ Optimizer  │ │                │   │
│  │ LSTM       │ │ Momentum   │ │ MVO / HRP  │ │ 11 pre-trade  │   │
│  │ Transformer│ │ Mkt Making │ │ Risk Parity│ │ checks, kill   │   │
│  │ XGBoost    │ │ StatArb    │ │ BL / CVaR  │ │ switch, drop   │   │
│  │ Ensemble   │ │ RL (DQN)   │ │            │ │ copy reconcil. │   │
│  │ ARIMA-GARCH│ │            │ │            │ │                │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────────┘   │
├──────────────────────────────────────────────────────────────────────┤
│  Feature Engineering · Timeseries DB (DuckDB) · Data Pipeline       │
├──────────────────────────────────────────────────────────────────────┤
│  Infrastructure: Docker · Kubernetes · Prometheus                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What's Implemented vs. Planned

The table below distinguishes **working** functionality from items that are **planned / API-only**.

| Component | Status | Notes |
|---|---|---|
| Rust OMS — BTreeMap LOB, matching, BBO | ✅ Working | `native/oms-core` |
| Rust OMS — ITCH 5.0 parser | ✅ Working | 20 message types |
| Rust OMS — Append-only event journal | ✅ Working | WAL + snapshots |
| Rust OMS — FPGA risk gate | ⚠️ Software reference | PCIe DMA integration planned |
| Rust OMS — DPDK / io_uring | 🔲 Planned | Feature-gated, not implemented |
| Python — Momentum strategy | ✅ Working | Z-score, configurable thresholds |
| Python — Market Making (A-S framework) | ✅ Working | Reservation price, order tracking |
| Python — Statistical Arbitrage | ✅ Working | Pairs trading, spread Z-score |
| Python — RL agent (DQN) | ⚠️ Prototype | Random-env training; no real data loop |
| ML — LSTM, Transformer, XGBoost, Ensemble, ARIMA-GARCH | ✅ Working | Training + prediction pipelines |
| Portfolio — MVO, HRP, Risk Parity, BL | ✅ Working | `src/portfolio/optimizer.py` |
| Risk Engine — 11 pre-trade checks | ✅ Working | `src/risk_ops/risk_engine.py` |
| Risk — Drop copy reconciliation | ✅ Working | `src/risk_ops/` |
| Execution — TWAP, VWAP, POV, IS | ✅ Working | Almgren-Chriss trajectory |
| FIX Protocol — 4.2/4.4/5.0SP2 | ✅ Working | Session lifecycle, TCP+TLS |
| Data Pipeline — Yahoo Finance, CSV, synthetic | ✅ Working | `src/data_pipeline/ingestion.py` |
| Data Pipeline — Alpaca connector | ⚠️ API skeleton | Requires `alpaca-py` + API key |
| Data Pipeline — Polygon connector | ⚠️ API skeleton | Requires `websockets` + API key |
| Feature Engineering — Technical, Microstructure, Causal | ✅ Working | Transfer entropy, Granger causality |
| Timeseries DB — DuckDB backend | ✅ Working | Tick/bar/book tables |
| Timeseries DB — ClickHouse backend | ⚠️ Needs `clickhouse-driver` | Code exists, dep not bundled |
| Timeseries DB — kdb+ / ArcticDB | 🔲 Planned | Enum defined, no implementation |
| Feature Store — Offline + Online | ✅ Working | In-memory online store |
| Alt Data — FinBERT sentiment | ⚠️ Optional | Fallback to rule-based if no `transformers` |
| Alt Data — NewsAPI connector | ✅ Working | Requires API key |
| Alt Data — SEC EDGAR connector | ✅ Working | Full-text search API |
| Alt Data — Options flow | ⚠️ API skeleton | Needs a paid data provider |
| Messaging — InMemory transport | ✅ Working | Default for dev/testing |
| Messaging — Aeron transport | ⚠️ Opt-in | Requires `aeron-python` — raises on missing dep |
| Messaging — Kafka transport | ⚠️ Opt-in | Requires `confluent-kafka` — raises on missing dep |
| Serialization — MsgPack, JSON | ✅ Working | MsgPack is wire default |
| Serialization — Compact Binary | ✅ Working | Custom offset-table format |
| Serialization — Protobuf | 🔲 Planned | Falls back to MsgPack |
| Frontend — Next.js dashboard | ⚠️ Minimal | Single page with basic tables |
| Distributed — Ray / Dask | ⚠️ Opt-in | Code exists, deps not bundled |
| GPU Training — DDP + Optuna | ⚠️ Opt-in | Requires multi-GPU PyTorch |
| Docker / K8s / Prometheus | ✅ Working | `docker-compose.yml`, `k8s/` |

**Legend:** ✅ Working | ⚠️ Partial / needs optional deps | 🔲 Planned

---

## Key Capabilities

### Native Rust OMS (`oms-core`)
- **BTreeMap limit order book** with price-time priority matching and BBO tracking
- **10+ order types**: Market, Limit, Stop, StopLimit, IOC, FOK, GTC, Iceberg, Peg, Trailing Stop
- **ITCH 5.0 feed parser** (20 message types) and **OUCH 4.2** order entry protocol
- **Append-only Event Journal** with write-ahead semantics and snapshot replay
- **Software risk gate** that mirrors intended FPGA register semantics (PCIe DMA integration planned)
- **Feature flags**: `dpdk`, `io_uring`, `fpga` — API surface defined, hardware integration not yet implemented
- **Release profile**: `opt-level = 3`, LTO, single codegen unit, abort-on-panic
- **Python bridge** (`oms_bridge.py`): ctypes FFI with nanosecond latency tracking; includes a pure-Python `ReferenceOMS` fallback

### FIX Protocol Engine
- Full **FIX 4.2 / 4.4 / 5.0SP2** session lifecycle: logon, logout, heartbeat, sequence number management, gap-fill recovery
- **TCP + TLS** transport with automatic reconnection
- NewOrderSingle, OrderCancelRequest, execution report callbacks

### Execution Algorithms
- **TWAP** — time-weighted slicing
- **VWAP** — volume-weighted participation
- **POV** — percentage of volume with cap
- **Implementation Shortfall** — Almgren-Chriss optimal execution trajectory
- **Adaptive Execution** — dynamic strategy selection based on urgency, volatility, liquidity

### Pre-Trade Risk Engine
- **11 sequential fail-fast checks**: Kill Switch, Position Limit, Order Size, Fat Finger, Order Rate, Cancel Rate, Daily Loss, Drawdown, Leverage, Sector Concentration, Single-Name Exposure
- **Drop Copy Reconciliation**: internal-vs-exchange fill matching, break detection, auto-resolution

### ML Alpha Models
| Model | Framework |
|---|---|
| **LSTM** | PyTorch (bidirectional, 3 FC layers) |
| **Transformer** | PyTorch (encoder-decoder, multi-head attention) |
| **Gradient Boosting** | XGBoost / LightGBM / CatBoost |
| **Ensemble** | scikit-learn (stacking, blending, weighted average) |
| **Statistical** | statsmodels / arch (ARIMA-GARCH, Kalman Filter, cointegration) |

### Trading Strategies
- **Momentum** — Z-score entry/exit with configurable thresholds and stop loss
- **Market Making** — Avellaneda-Stoikov reservation price with inventory-aware skewing and EMA volatility
- **Statistical Arbitrage** — Pairs trading with spread Z-score and dynamic hedge ratios
- **Reinforcement Learning** — DQN with experience replay (prototype; environment uses synthetic data)

### Data Platform
- **Feature Store**: point-in-time correct, streaming + batch + on-demand compute, dependency graph, TTL, versioning
- **Timeseries DB**: DuckDB (working), ClickHouse (code present, needs dep), kdb+/ArcticDB (planned)
- **Data Reconciliation**: gap/duplicate/outlier detection, cross-source reconciliation

### Alternative Data
- **Sentiment**: FinBERT model when `transformers` installed, rule-based fallback otherwise
- **NewsAPI connector**: fetches articles via REST, computes sentiment per symbol
- **SEC EDGAR connector**: full-text search for 10-K/10-Q/8-K filings
- **Options flow**: detection framework in place, requires a paid data source to be useful

### Messaging & Transport
- **InMemory** transport (default) — zero-dependency, for development and testing
- **Aeron** shared-memory IPC — opt-in, requires `aeron-python` (raises on missing dep)
- **Kafka / Redpanda** persistent streaming — opt-in, requires `confluent-kafka` (raises on missing dep)
- Serialization: MsgPack (default), JSON, compact binary format

### Dashboard
- **Next.js 14** / React / TypeScript — single-page dashboard with API integration
- Basic metric cards, positions table; charting components planned

### Infrastructure
- **Event Bus**: sync + async pub/sub with priority levels, wildcard subscriptions
- **Event Sourcing**: 20+ event types, append-only journal with checksums, snapshots
- **Security**: JWT (HS256), bcrypt, API key management, Fernet encryption, rate limiting
- **Monitoring**: Prometheus metrics, health checks, alerting (Slack, email, PagerDuty)
- **Docker / Kubernetes**: multi-service compose, K8s deployment with HPA

---

## Quick Start

### Prerequisites
- Python 3.10+
- Rust toolchain (2021 Edition) & Cargo
- Node.js 18+ & npm
- Docker & Docker Compose

### Installation

```bash
git clone https://github.com/yourusername/omniquant.git
cd omniquant

# 1. Python environment
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Build the Rust native OMS (release profile: LTO + opt-level 3)
cd native/oms-core
cargo build --release
cd ../..

# 3. Frontend
cd frontend
npm install
cd ..
```

### Run the System

```bash
# Full stack (API + Postgres + Redis)
docker-compose up -d

# Or individually:
uvicorn src.api.main:app --reload          # FastAPI backend
cd frontend && npm run dev                  # Next.js trading terminal
```

### Run a Backtest

```python
from src.strategies.momentum import MomentumStrategy
from src.simulator.event_simulator import EventSimulator, SimulationConfig

config = SimulationConfig(initial_capital=100_000, commission_rate=0.0002)
sim = EventSimulator(sim_config=config)
strategy = MomentumStrategy(config={"lookback_period": 20, "entry_threshold": 2.0})

results = sim.run_backtest(strategy, symbol="SYNTH")
print(f"Sharpe: {results['sharpe_ratio']:.2f}  Return: {results['total_return']:.2%}")
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/deployment.yaml
# Deploys to omniquant-prod namespace:
#   - 3 replicas (HPA scales 3→10 on CPU/memory)
#   - 50Gi fast-SSD PVC
#   - LoadBalancer service (port 80 → 8000, 9090 metrics)
#   - Liveness/readiness probes on /health
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for a full walkthrough.

---

## Project Layout

```
native/
  oms-core/               Rust OMS — matching engine, ITCH/OUCH parser, event journal, FPGA stubs
frontend/                 Next.js 14 / React / TypeScript trading terminal
src/
  integration.py          Top-level TradingSystem wiring (hot path + cold path)
  cli.py                  Typer CLI (strategy scaffolding)
  alpha_models/           LSTM, Transformer, XGBoost/LightGBM/CatBoost, Ensemble, ARIMA-GARCH
  api/                    FastAPI REST + WebSocket gateway (backtest jobs, portfolio, market data)
  common/                 Event Bus, Event Sourcing, Monitoring, Security, DI Container
  data_pipeline/          Ingestion, Cleaning, Alignment (Arrow/Parquet)
  data_platform/          Feature Store (streaming + batch), Timeseries DB, Data Reconciliation
  execution/              TWAP/VWAP/POV/IS/Adaptive algos, FIX engine, ITCH/OUCH feed handlers, Rust bridge
  feature_engineering/    Technical, Microstructure, Causal (DoWhy/EconML) features
  messaging/              Aeron / Kafka / InMemory transport, MsgPack/FlatBuffer serialization
  portfolio/              MVO, Risk Parity, HRP, Black-Litterman, CVaR optimization & risk management
  research/               Distributed backtesting (Ray/Dask), GPU training (DDP/Optuna), alt data pipelines
  risk_ops/               Pre-trade risk engine (11 checks), drop copy reconciliation
  simulator/              Event-driven backtest engine, order book, matching engine, market impact models
  strategies/             Momentum, Market Making, StatArb, RL agents (DQN)
k8s/                      Kubernetes manifests (Deployment, HPA, PVC, Service)
configs/                  YAML configuration (features, models, strategies, execution, risk)
notebooks/                Research notebooks
```

## Technology Stack

| Layer | Technologies |
|---|---|
| **Languages** | Python 3.10+, Rust (2021 Edition), TypeScript 5.3 |
| **Native OMS** | crossbeam, tokio, memmap2, quanta, ringbuf, prometheus |
| **ML / AI** | PyTorch, Optuna (opt-in), XGBoost, LightGBM, CatBoost, scikit-learn, statsmodels, arch |
| **NLP** | FinBERT via `transformers` (optional, rule-based fallback) |
| **Data** | Pandas 2.0, Polars, Apache Arrow/Parquet, DuckDB (default TSDB) |
| **Messaging** | InMemory (default), Aeron (opt-in), Kafka (opt-in); MsgPack, JSON |
| **Web** | FastAPI, Next.js 14, React 18, TanStack Query, Tailwind CSS |
| **Infrastructure** | Docker, Kubernetes, Prometheus |
| **Protocols** | FIX 4.2/4.4/5.0SP2, ITCH 5.0, OUCH 4.2, WebSocket |

## Development

```bash
# Python tests
pytest tests/ -v

# Rust tests
cd native/oms-core && cargo test

# Linting
make lint
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Quickstart](docs/QUICKSTART.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

This software is provided for research and simulation purposes. Trading financial instruments involves substantial risk of loss. Past backtest performance does not guarantee future results. Do not deploy against live markets without independent validation, additional hardening, and compliance review. Use at your own risk.

## Contact

https://www.linkedin.com/in/pushkar-kumar-vats/
