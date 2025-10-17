"""
Simple Backtest Example
Demonstrates how to run a backtest with OmniQuant
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.ingestion import DataIngestion
from src.simulator.event_simulator import EventSimulator
from src.strategies.momentum import MomentumStrategy
from loguru import logger

# Generate sample data
logger.info("Generating sample data...")
ingestion = DataIngestion()
tick_data = ingestion.generate_synthetic_tick_data(num_ticks=5000, seed=42)

# Resample to bars
df = tick_data.set_index('timestamp').resample('1min').agg({
    'price': 'last',
    'bid': 'last',
    'ask': 'last',
    'volume': 'sum'
}).dropna()

logger.info(f"Data shape: {df.shape}")

# Create strategy
strategy = MomentumStrategy(config={
    'lookback_period': 20,
    'entry_threshold': 2.0,
    'position_size': 100
})

# Run backtest
simulator = EventSimulator()
results = simulator.run_backtest(strategy, df, symbol="TEST")

# Display results
logger.info("\n" + "="*60)
logger.info("BACKTEST RESULTS")
logger.info("="*60)
logger.info(f"Initial Capital: ${results['initial_capital']:,.2f}")
logger.info(f"Final Equity: ${results['final_equity']:,.2f}")
logger.info(f"Total Return: {results['total_return']:.2%}")
logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
logger.info(f"Total Trades: {results['total_trades']}")
logger.info(f"Win Rate: {results['win_rate']:.2%}")
