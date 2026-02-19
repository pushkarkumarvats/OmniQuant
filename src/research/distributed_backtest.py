"""
Distributed Backtesting Engine

Scales the existing EventSimulator across a Ray / Dask cluster:
  - Partitions universe by symbol or time range
  - Runs parameter sweeps in parallel
  - Aggregates per-partition PnL into a unified report
  - Supports walk-forward optimization and combinatorial purged CV
"""

from __future__ import annotations

import hashlib
import itertools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
from loguru import logger


# --------------------------------------------------------------------------- #
#  Types                                                                       #
# --------------------------------------------------------------------------- #

class PartitionStrategy(Enum):
    BY_SYMBOL = "by_symbol"
    BY_TIME = "by_time"
    BY_SYMBOL_TIME = "by_symbol_time"


class BacktestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestConfig:
    strategy_class: str
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 1_000_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 0.5
    data_frequency: str = "1m"
    benchmark: str = "SPY"
    max_leverage: float = 2.0
    # Walk-forward
    walk_forward: bool = False
    train_window_days: int = 252
    test_window_days: int = 63
    # Combinatorial purged CV
    purged_cv: bool = False
    n_splits: int = 5
    embargo_days: int = 5


@dataclass
class BacktestPartition:
    partition_id: str
    symbols: List[str]
    start_date: str
    end_date: str
    config: BacktestConfig
    status: BacktestStatus = BacktestStatus.PENDING
    worker_id: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class BacktestResult:
    run_id: str
    config: BacktestConfig
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    daily_returns: Optional[pd.Series] = None
    equity_curve: Optional[pd.Series] = None
    trade_log: Optional[pd.DataFrame] = None
    partitions: List[BacktestPartition] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Local Backtest Runner                                                       #
# --------------------------------------------------------------------------- #

class LocalBacktestRunner:
    """Single-process backtest runner wrapping EventSimulator."""

    def __init__(self) -> None:
        self._results_cache: Dict[str, BacktestResult] = {}

    async def run(self, partition: BacktestPartition) -> Dict[str, Any]:
        """Run a single partition backtest."""
        partition.status = BacktestStatus.RUNNING
        partition.started_at = time.time()

        try:
            # Import the existing simulator
            from src.simulator.event_simulator import EventSimulator

            config_dict = {
                "symbols": partition.symbols,
                "start_date": partition.start_date,
                "end_date": partition.end_date,
                "initial_capital": partition.config.initial_capital,
                "commission_rate": partition.config.commission_bps / 10000,
                "slippage_rate": partition.config.slippage_bps / 10000,
            }

            simulator = EventSimulator(config_dict)
            result = await simulator.run()

            partition.status = BacktestStatus.COMPLETED
            partition.finished_at = time.time()
            partition.result = result

            return result

        except Exception as e:
            partition.status = BacktestStatus.FAILED
            partition.finished_at = time.time()
            partition.error = str(e)
            logger.error(f"Partition {partition.partition_id} failed: {e}")
            return {"error": str(e)}

    def run_sync(self, partition: BacktestPartition) -> Dict[str, Any]:
        """Synchronous version for Ray/Dask remote functions."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run(partition))
        finally:
            loop.close()


# --------------------------------------------------------------------------- #
#  Partitioner                                                                  #
# --------------------------------------------------------------------------- #

class BacktestPartitioner:
    """Splits a backtest config into parallelizable partitions."""

    @staticmethod
    def partition(
        config: BacktestConfig,
        strategy: PartitionStrategy = PartitionStrategy.BY_SYMBOL,
        max_symbols_per_partition: int = 10,
        time_chunk_days: int = 90,
    ) -> List[BacktestPartition]:
        """Create partitions from a backtest config."""
        partitions: List[BacktestPartition] = []

        if strategy == PartitionStrategy.BY_SYMBOL:
            # Chunk symbols into groups
            symbols = config.symbols or ["SPY"]
            for i in range(0, len(symbols), max_symbols_per_partition):
                chunk = symbols[i:i + max_symbols_per_partition]
                pid = hashlib.md5(
                    f"{config.strategy_class}:{','.join(chunk)}:{config.start_date}".encode()
                ).hexdigest()[:8]
                partitions.append(BacktestPartition(
                    partition_id=pid,
                    symbols=chunk,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    config=config,
                ))

        elif strategy == PartitionStrategy.BY_TIME:
            # Chunk time range
            start = datetime.strptime(config.start_date, "%Y-%m-%d")
            end = datetime.strptime(config.end_date, "%Y-%m-%d")
            current = start
            from datetime import timedelta
            while current < end:
                chunk_end = min(current + timedelta(days=time_chunk_days), end)
                pid = hashlib.md5(
                    f"{config.strategy_class}:{current.isoformat()}:{chunk_end.isoformat()}".encode()
                ).hexdigest()[:8]
                partitions.append(BacktestPartition(
                    partition_id=pid,
                    symbols=config.symbols,
                    start_date=current.strftime("%Y-%m-%d"),
                    end_date=chunk_end.strftime("%Y-%m-%d"),
                    config=config,
                ))
                current = chunk_end

        elif strategy == PartitionStrategy.BY_SYMBOL_TIME:
            # Cross-product: each symbol × each time chunk
            symbol_partitions = BacktestPartitioner.partition(
                config, PartitionStrategy.BY_SYMBOL, max_symbols_per_partition
            )
            time_partitions = BacktestPartitioner.partition(
                config, PartitionStrategy.BY_TIME, time_chunk_days=time_chunk_days
            )
            for sp in symbol_partitions:
                for tp in time_partitions:
                    pid = hashlib.md5(
                        f"{','.join(sp.symbols)}:{tp.start_date}:{tp.end_date}".encode()
                    ).hexdigest()[:8]
                    partitions.append(BacktestPartition(
                        partition_id=pid,
                        symbols=sp.symbols,
                        start_date=tp.start_date,
                        end_date=tp.end_date,
                        config=config,
                    ))

        logger.info(f"Created {len(partitions)} partitions ({strategy.value})")
        return partitions


# --------------------------------------------------------------------------- #
#  Parameter Sweep                                                             #
# --------------------------------------------------------------------------- #

class ParameterSweep:
    """Parameter grid generation for strategy optimization."""

    @staticmethod
    def grid(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Full Cartesian product of parameter values."""
        keys = list(params.keys())
        values = list(params.values())
        combos = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combos]

    @staticmethod
    def random(
        params: Dict[str, Tuple[float, float]],
        n_samples: int = 100,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Random sampling from parameter ranges."""
        rng = np.random.RandomState(seed)
        results = []
        for _ in range(n_samples):
            sample = {}
            for key, (lo, hi) in params.items():
                sample[key] = rng.uniform(lo, hi)
            results.append(sample)
        return results


# --------------------------------------------------------------------------- #
#  Walk-Forward Optimizer                                                       #
# --------------------------------------------------------------------------- #

class WalkForwardOptimizer:
    """Walk-forward analysis with anchored or rolling train/test windows."""

    def __init__(
        self,
        train_days: int = 252,
        test_days: int = 63,
        anchored: bool = False,
    ) -> None:
        self._train_days = train_days
        self._test_days = test_days
        self._anchored = anchored

    def generate_windows(
        self, start_date: str, end_date: str,
    ) -> List[Tuple[str, str, str, str]]:
        """Generate (train_start, train_end, test_start, test_end) windows."""
        from datetime import timedelta
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        windows = []
        anchor = start
        current = start

        while current + timedelta(days=self._train_days + self._test_days) <= end:
            train_start = anchor if self._anchored else current
            train_end = current + timedelta(days=self._train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self._test_days)

            windows.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))

            current = test_end

        logger.info(f"Walk-forward: {len(windows)} windows "
                     f"(train={self._train_days}d, test={self._test_days}d)")
        return windows


# --------------------------------------------------------------------------- #
#  Combinatorial Purged Cross-Validation                                       #
# --------------------------------------------------------------------------- #

class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation with embargo to prevent look-ahead bias."""

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        self._n_splits = n_splits
        self._embargo_pct = embargo_pct

    def split(
        self, n_samples: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test index splits."""
        indices = np.arange(n_samples)
        group_size = n_samples // self._n_splits
        embargo_size = int(n_samples * self._embargo_pct)

        groups = []
        for i in range(self._n_splits):
            start = i * group_size
            end = start + group_size if i < self._n_splits - 1 else n_samples
            groups.append(indices[start:end])

        splits = []
        for test_idx in range(self._n_splits):
            test = groups[test_idx]
            train_groups = [g for i, g in enumerate(groups) if i != test_idx]

            # Apply embargo: remove samples near test boundaries
            test_start = test[0]
            test_end = test[-1]

            train = np.concatenate(train_groups)
            # Purge: remove train samples that overlap with test
            mask = np.ones(len(train), dtype=bool)
            for j, idx in enumerate(train):
                if test_start - embargo_size <= idx <= test_end + embargo_size:
                    mask[j] = False
            train = train[mask]

            splits.append((train, test))

        return splits


# --------------------------------------------------------------------------- #
#  Distributed Orchestrator                                                    #
# --------------------------------------------------------------------------- #

class DistributedBacktestOrchestrator:
    """Distributes backtests across Ray/Dask or falls back to local execution."""

    def __init__(self, backend: str = "local") -> None:
        self._backend = backend
        self._runner = LocalBacktestRunner()

    async def run_sweep(
        self,
        base_config: BacktestConfig,
        param_grid: List[Dict[str, Any]],
        partition_strategy: PartitionStrategy = PartitionStrategy.BY_SYMBOL,
    ) -> List[BacktestResult]:
        """Run a parameter sweep across configurations."""
        results: List[BacktestResult] = []
        total = len(param_grid)

        for i, params in enumerate(param_grid):
            config = BacktestConfig(
                strategy_class=base_config.strategy_class,
                strategy_params={**base_config.strategy_params, **params},
                symbols=base_config.symbols,
                start_date=base_config.start_date,
                end_date=base_config.end_date,
                initial_capital=base_config.initial_capital,
                commission_bps=base_config.commission_bps,
                slippage_bps=base_config.slippage_bps,
            )
            logger.info(f"Sweep {i + 1}/{total}: {params}")
            result = await self.run(config, partition_strategy)
            results.append(result)

        # Sort by Sharpe
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results

    async def run(
        self,
        config: BacktestConfig,
        partition_strategy: PartitionStrategy = PartitionStrategy.BY_SYMBOL,
    ) -> BacktestResult:
        """Run a single backtest (potentially distributed)."""
        run_id = hashlib.md5(
            f"{config.strategy_class}:{time.time_ns()}".encode()
        ).hexdigest()[:12]

        start_time = time.time()
        partitions = BacktestPartitioner.partition(config, partition_strategy)

        if self._backend == "ray":
            partition_results = await self._run_ray(partitions)
        elif self._backend == "dask":
            partition_results = await self._run_dask(partitions)
        else:
            partition_results = await self._run_local(partitions)

        # Aggregate
        result = self._aggregate_results(run_id, config, partitions, partition_results)
        result.duration_seconds = time.time() - start_time

        logger.info(
            f"Backtest {run_id} complete: "
            f"Sharpe={result.sharpe_ratio:.3f}, "
            f"Return={result.total_return:.2%}, "
            f"MaxDD={result.max_drawdown:.2%}, "
            f"Trades={result.total_trades}, "
            f"Duration={result.duration_seconds:.1f}s"
        )
        return result

    async def _run_local(self, partitions: List[BacktestPartition]) -> List[Dict]:
        results = []
        for p in partitions:
            r = await self._runner.run(p)
            results.append(r)
        return results

    async def _run_ray(self, partitions: List[BacktestPartition]) -> List[Dict]:
        try:
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            @ray.remote
            def run_partition(partition: BacktestPartition) -> Dict:
                runner = LocalBacktestRunner()
                return runner.run_sync(partition)

            futures = [run_partition.remote(p) for p in partitions]
            results = ray.get(futures)
            return results

        except ImportError:
            logger.warning("Ray not available, falling back to local execution")
            return await self._run_local(partitions)

    async def _run_dask(self, partitions: List[BacktestPartition]) -> List[Dict]:
        try:
            from dask.distributed import Client, get_client

            try:
                client = get_client()
            except ValueError:
                client = Client(processes=True)

            def run_partition(partition: BacktestPartition) -> Dict:
                runner = LocalBacktestRunner()
                return runner.run_sync(partition)

            futures = client.map(run_partition, partitions)
            results = client.gather(futures)
            return results

        except ImportError:
            logger.warning("Dask not available, falling back to local execution")
            return await self._run_local(partitions)

    def _aggregate_results(
        self,
        run_id: str,
        config: BacktestConfig,
        partitions: List[BacktestPartition],
        partition_results: List[Dict],
    ) -> BacktestResult:
        result = BacktestResult(
            run_id=run_id,
            config=config,
            partitions=partitions,
        )

        # Collect equity curves and trades from partitions
        all_returns = []
        total_pnl = 0.0
        total_trades = 0

        for pr in partition_results:
            if isinstance(pr, dict) and "error" not in pr:
                pnl = pr.get("total_pnl", 0.0)
                trades = pr.get("total_trades", 0)
                total_pnl += pnl
                total_trades += trades

                if "daily_returns" in pr and pr["daily_returns"] is not None:
                    all_returns.append(pr["daily_returns"])

        result.total_trades = total_trades
        result.total_return = total_pnl / config.initial_capital if config.initial_capital > 0 else 0

        # Calculate risk metrics from combined daily returns
        if all_returns:
            combined = pd.concat(all_returns, axis=1).sum(axis=1)
            result.daily_returns = combined

            annual_factor = np.sqrt(252)
            mean_ret = combined.mean()
            std_ret = combined.std()

            result.annual_return = mean_ret * 252
            result.sharpe_ratio = (mean_ret / std_ret * annual_factor) if std_ret > 0 else 0
            result.max_drawdown = self._calculate_max_drawdown(combined)
            result.calmar_ratio = (
                result.annual_return / abs(result.max_drawdown)
                if result.max_drawdown != 0 else 0
            )

            # Sortino ratio
            downside = combined[combined < 0]
            downside_std = downside.std() if len(downside) > 0 else 0
            result.sortino_ratio = (
                mean_ret / downside_std * annual_factor
                if downside_std > 0 else 0
            )

        return result

    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min()) if len(drawdown) > 0 else 0.0
