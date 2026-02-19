"""
Institutional Feature Store

Real-time and batch feature computation, storage, and serving:
  - Online store: sub-ms serving for inference (Redis/DynamoDB)
  - Offline store: batch feature computation (Spark/DuckDB)
  - Streaming: real-time feature pipelines (Flink-style windowed aggregations)
  - Point-in-time correct joins for training data
  - Feature registry with lineage tracking
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


# --------------------------------------------------------------------------- #
#  Feature Definitions & Registry                                              #
# --------------------------------------------------------------------------- #

class FeatureType(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"
    CATEGORICAL = "categorical"


class ComputeMode(Enum):
    BATCH = "batch"
    STREAMING = "streaming"
    ON_DEMAND = "on_demand"


@dataclass
class FeatureDefinition:
    name: str
    group: str
    description: str
    dtype: str = "float64"
    feature_type: FeatureType = FeatureType.SCALAR
    compute_mode: ComputeMode = ComputeMode.BATCH
    dependencies: List[str] = field(default_factory=list)
    window_seconds: int = 0
    ttl_seconds: int = 86400
    version: int = 1
    owner: str = "research"
    tags: List[str] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        return f"{self.group}/{self.name}:v{self.version}"

    @property
    def fingerprint(self) -> str:
        """Content-based hash for cache invalidation."""
        content = f"{self.qualified_name}:{self.dtype}:{self.dependencies}:{self.window_seconds}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class FeatureRegistry:
    """Central catalog of all feature definitions with lineage."""

    def __init__(self) -> None:
        self._features: Dict[str, FeatureDefinition] = {}
        self._groups: Dict[str, Set[str]] = defaultdict(set)
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)

    def register(self, feature: FeatureDefinition) -> None:
        key = feature.qualified_name
        self._features[key] = feature
        self._groups[feature.group].add(key)
        for dep in feature.dependencies:
            self._dependency_graph[key].add(dep)
        logger.debug(f"Registered feature: {key}")

    def get(self, qualified_name: str) -> Optional[FeatureDefinition]:
        return self._features.get(qualified_name)

    def get_group(self, group: str) -> List[FeatureDefinition]:
        return [self._features[k] for k in self._groups.get(group, set())]

    def get_dependency_order(self, feature_names: List[str]) -> List[str]:
        """Topological sort of features by dependencies."""
        visited: Set[str] = set()
        order: List[str] = []

        def dfs(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in self._dependency_graph.get(name, set()):
                dfs(dep)
            order.append(name)

        for name in feature_names:
            dfs(name)
        return order

    def list_all(self) -> List[FeatureDefinition]:
        return list(self._features.values())

    def search(self, query: str) -> List[FeatureDefinition]:
        query_lower = query.lower()
        return [
            f for f in self._features.values()
            if query_lower in f.name.lower()
            or query_lower in f.description.lower()
            or any(query_lower in t.lower() for t in f.tags)
        ]


# --------------------------------------------------------------------------- #
#  Streaming Feature Engine (Flink-style)                                      #
# --------------------------------------------------------------------------- #

@dataclass
class WindowState:
    values: Deque[Tuple[int, float]] = field(default_factory=deque)  # (ts_ns, value)
    window_ns: int = 60_000_000_000  # 60s default
    _sum: float = 0.0
    _count: int = 0
    _min: float = float("inf")
    _max: float = float("-inf")

    def add(self, ts_ns: int, value: float) -> None:
        self.values.append((ts_ns, value))
        self._sum += value
        self._count += 1
        self._min = min(self._min, value)
        self._max = max(self._max, value)
        self._evict(ts_ns)

    def _evict(self, current_ts: int) -> None:
        cutoff = current_ts - self.window_ns
        while self.values and self.values[0][0] < cutoff:
            old_ts, old_val = self.values.popleft()
            self._sum -= old_val
            self._count -= 1
        # Recalc min/max after eviction (lazy)
        if self._count > 0 and self.values:
            vals = [v for _, v in self.values]
            self._min = min(vals)
            self._max = max(vals)
        else:
            self._min = float("inf")
            self._max = float("-inf")

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def count(self) -> int:
        return self._count

    @property
    def min(self) -> float:
        return self._min if self._count > 0 else 0.0

    @property
    def max(self) -> float:
        return self._max if self._count > 0 else 0.0

    @property
    def std(self) -> float:
        if self._count < 2:
            return 0.0
        vals = [v for _, v in self.values]
        return float(np.std(vals, ddof=1))


class StreamingFeatureEngine:
    """Real-time feature computation with windowed aggregations per symbol."""

    def __init__(self, registry: FeatureRegistry) -> None:
        self._registry = registry
        self._windows: Dict[str, Dict[str, WindowState]] = defaultdict(dict)
        self._computed: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._processors: Dict[str, Callable] = {}
        self._update_count = 0

    def register_processor(
        self, feature_name: str, processor: Callable[[str, Dict[str, Any]], float],
    ) -> None:
        """Register a custom feature processor function."""
        self._processors[feature_name] = processor

    def process_tick(self, symbol: str, price: float, size: float,
                     side: str, ts_ns: int) -> Dict[str, float]:
        """Process a tick event and return updated features for this symbol."""
        entity = symbol
        event = {"price": price, "size": size, "side": side, "ts_ns": ts_ns}

        # Built-in windowed features
        self._update_window(entity, "price", ts_ns, price)
        self._update_window(entity, "size", ts_ns, size)
        self._update_window(entity, "trade_value", ts_ns, price * size)

        # Volume imbalance
        if side == "B":
            self._update_window(entity, "buy_volume", ts_ns, size)
        else:
            self._update_window(entity, "sell_volume", ts_ns, size)

        # Compute derived features
        features = self._compute_derived(entity, event)
        self._computed[entity].update(features)
        self._update_count += 1

        return self._computed[entity].copy()

    def process_bar(self, symbol: str, open_: float, high: float,
                    low: float, close: float, volume: float, ts_ns: int) -> Dict[str, float]:
        """Process a bar event."""
        entity = symbol
        event = {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "ts_ns": ts_ns}

        self._update_window(entity, "bar_close", ts_ns, close)
        self._update_window(entity, "bar_volume", ts_ns, volume)
        self._update_window(entity, "bar_range", ts_ns, (high - low) / close if close > 0 else 0)

        features = self._compute_derived(entity, event)
        self._computed[entity].update(features)

        return self._computed[entity].copy()

    def get_features(self, symbol: str) -> Dict[str, float]:
        return self._computed.get(symbol, {}).copy()

    def get_all_features(self) -> Dict[str, Dict[str, float]]:
        return {k: v.copy() for k, v in self._computed.items()}

    def _update_window(self, entity: str, feature: str, ts_ns: int, value: float) -> None:
        if feature not in self._windows[entity]:
            self._windows[entity][feature] = WindowState(window_ns=60_000_000_000)
        self._windows[entity][feature].add(ts_ns, value)

    def _compute_derived(self, entity: str, event: Dict[str, Any]) -> Dict[str, float]:
        features: Dict[str, float] = {}
        windows = self._windows.get(entity, {})

        # Price statistics
        if "price" in windows:
            w = windows["price"]
            features["price_mean_60s"] = w.mean
            features["price_std_60s"] = w.std
            features["price_min_60s"] = w.min
            features["price_max_60s"] = w.max
            features["tick_count_60s"] = float(w.count)

        # Volume statistics
        if "size" in windows:
            features["volume_sum_60s"] = windows["size"].sum
            features["avg_trade_size_60s"] = windows["size"].mean

        # VWAP
        if "trade_value" in windows and "size" in windows:
            total_val = windows["trade_value"].sum
            total_vol = windows["size"].sum
            features["vwap_60s"] = total_val / total_vol if total_vol > 0 else 0.0

        # Volume imbalance
        buy_vol = windows.get("buy_volume", WindowState())
        sell_vol = windows.get("sell_volume", WindowState())
        total = buy_vol.sum + sell_vol.sum
        if total > 0:
            features["volume_imbalance_60s"] = (buy_vol.sum - sell_vol.sum) / total
        else:
            features["volume_imbalance_60s"] = 0.0

        # Bar-based features
        if "bar_close" in windows:
            features["bar_close_mean"] = windows["bar_close"].mean
            features["bar_close_std"] = windows["bar_close"].std
        if "bar_range" in windows:
            features["avg_bar_range"] = windows["bar_range"].mean

        # Custom processors
        for feat_name, proc in self._processors.items():
            try:
                features[feat_name] = proc(entity, event)
            except Exception as e:
                logger.warning(f"Feature processor {feat_name} failed: {e}")

        return features


# --------------------------------------------------------------------------- #
#  Offline Feature Store (batch)                                                #
# --------------------------------------------------------------------------- #

class OfflineFeatureStore:
    """Batch feature computation with point-in-time correct joins."""

    def __init__(self, db_path: str = "data/feature_store.duckdb") -> None:
        self._db_path = db_path
        self._conn = None

    async def connect(self) -> None:
        import duckdb
        self._conn = duckdb.connect(self._db_path)
        await self._create_tables()
        logger.info(f"OfflineFeatureStore connected: {self._db_path}")

    async def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_values (
                entity_id VARCHAR,
                feature_name VARCHAR,
                timestamp BIGINT,
                value DOUBLE,
                created_at BIGINT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fv_entity_ts
            ON feature_values(entity_id, feature_name, timestamp)
        """)

    async def write_features(
        self, entity_id: str, features: Dict[str, float], timestamp_ns: int,
    ) -> None:
        now = time.time_ns()
        data = [
            (entity_id, name, timestamp_ns, value, now)
            for name, value in features.items()
        ]
        self._conn.executemany(
            "INSERT INTO feature_values VALUES (?, ?, ?, ?, ?)", data
        )

    async def write_features_batch(
        self, records: List[Tuple[str, Dict[str, float], int]],
    ) -> int:
        """Bulk write features: [(entity_id, features_dict, timestamp_ns), ...]"""
        now = time.time_ns()
        data = []
        for entity_id, features, ts in records:
            for name, value in features.items():
                data.append((entity_id, name, ts, value, now))
        if data:
            self._conn.executemany(
                "INSERT INTO feature_values VALUES (?, ?, ?, ?, ?)", data
            )
        return len(data)

    async def read_features(
        self,
        entity_id: str,
        feature_names: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Read features for an entity over a time range."""
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)
        placeholders = ", ".join(["?"] * len(feature_names))
        query = f"""
            SELECT entity_id, feature_name, timestamp, value
            FROM feature_values
            WHERE entity_id = ?
              AND feature_name IN ({placeholders})
              AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        params = [entity_id] + feature_names + [start_ns, end_ns]
        return self._conn.execute(query, params).fetchdf()

    async def point_in_time_join(
        self,
        entities_df: pd.DataFrame,
        feature_names: List[str],
        entity_col: str = "symbol",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Join the latest feature value as-of each row's timestamp (no look-ahead bias)."""
        self._conn.register("entities_view", entities_df)

        feature_dfs = []
        for feat_name in feature_names:
            query = f"""
                SELECT
                    e.{entity_col},
                    e.{timestamp_col},
                    (
                        SELECT fv.value
                        FROM feature_values fv
                        WHERE fv.entity_id = e.{entity_col}
                          AND fv.feature_name = '{feat_name}'
                          AND fv.timestamp <= e.{timestamp_col}
                        ORDER BY fv.timestamp DESC
                        LIMIT 1
                    ) as {feat_name}
                FROM entities_view e
            """
            df = self._conn.execute(query).fetchdf()
            feature_dfs.append(df[[feat_name]])

        result = entities_df.copy()
        for fdf in feature_dfs:
            result = pd.concat([result.reset_index(drop=True), fdf.reset_index(drop=True)], axis=1)

        self._conn.unregister("entities_view")
        return result

    async def get_training_dataset(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        label_name: str,
        start: datetime,
        end: datetime,
        sample_interval_ns: int = 60_000_000_000,  # 1 minute
    ) -> pd.DataFrame:
        """Build a training dataset with point-in-time correct features on a time grid."""
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)

        # Build time grid
        timestamps = list(range(start_ns, end_ns, sample_interval_ns))
        rows = []
        for entity in entity_ids:
            for ts in timestamps:
                rows.append({"symbol": entity, "timestamp": ts})

        grid_df = pd.DataFrame(rows)

        # Point-in-time join all features + label
        all_features = feature_names + [label_name]
        result = await self.point_in_time_join(grid_df, all_features)

        return result.dropna()

    async def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# --------------------------------------------------------------------------- #
#  Online Feature Store (real-time serving)                                    #
# --------------------------------------------------------------------------- #

class OnlineFeatureStore:
    """Low-latency feature serving with TTL-based eviction."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._store: Dict[str, Dict[str, Tuple[float, int]]] = defaultdict(dict)
        self._ttl_ns = ttl_seconds * 1_000_000_000

    def write(self, entity_id: str, features: Dict[str, float]) -> None:
        now = time.time_ns()
        for name, value in features.items():
            self._store[entity_id][name] = (value, now)

    def read(self, entity_id: str, feature_names: List[str]) -> Dict[str, Optional[float]]:
        """Read features with TTL-based staleness check."""
        now = time.time_ns()
        result: Dict[str, Optional[float]] = {}
        entity_data = self._store.get(entity_id, {})

        for name in feature_names:
            if name in entity_data:
                value, written_at = entity_data[name]
                if now - written_at <= self._ttl_ns:
                    result[name] = value
                else:
                    result[name] = None  # Stale
            else:
                result[name] = None

        return result

    def read_vector(self, entity_id: str, feature_names: List[str]) -> np.ndarray:
        """Read features as a numpy vector (for model inference)."""
        features = self.read(entity_id, feature_names)
        return np.array([features.get(n, 0.0) or 0.0 for n in feature_names], dtype=np.float64)

    def evict_stale(self) -> int:
        """Evict expired entries. Returns count evicted."""
        now = time.time_ns()
        evicted = 0
        for entity_id in list(self._store.keys()):
            entity_data = self._store[entity_id]
            stale_keys = [k for k, (_, ts) in entity_data.items() if now - ts > self._ttl_ns]
            for k in stale_keys:
                del entity_data[k]
                evicted += 1
            if not entity_data:
                del self._store[entity_id]
        return evicted

    def stats(self) -> Dict[str, int]:
        total_features = sum(len(v) for v in self._store.values())
        return {
            "entities": len(self._store),
            "total_features": total_features,
        }
