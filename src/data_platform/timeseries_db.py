"""
Institutional Time-Series Database Abstraction

Pluggable backend for tick/bar/book storage & retrieval:
  - ClickHouse: Column-oriented OLAP for analytics workloads
  - kdb+/q:    Ultra-low-latency in-memory time-series (HDB/RDB)
  - ArcticDB:  DataFrame-native versioned store (S3/LMDB)
  - DuckDB:    Embedded OLAP for research notebooks (fallback)

All backends implement TimeSeriesDB protocol for seamless swap.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


# --------------------------------------------------------------------------- #
#  Schema & Config                                                             #
# --------------------------------------------------------------------------- #

class TimeSeriesBackend(Enum):
    CLICKHOUSE = "clickhouse"
    KDB = "kdb"
    ARCTICDB = "arcticdb"
    DUCKDB = "duckdb"


class DataGranularity(Enum):
    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    DAILY = "1d"


@dataclass
class TimeSeriesConfig:
    backend: TimeSeriesBackend = TimeSeriesBackend.DUCKDB
    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_database: str = "hrt_market_data"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_cluster: str = ""
    # kdb+
    kdb_host: str = "localhost"
    kdb_port: int = 5000
    kdb_user: str = ""
    kdb_password: str = ""
    # ArcticDB
    arctic_uri: str = "lmdb://data/arcticdb"
    arctic_library: str = "market_data"
    # DuckDB
    duckdb_path: str = "data/timeseries.duckdb"
    # Common
    batch_size: int = 10_000
    flush_interval_seconds: float = 1.0
    retention_days: int = 365 * 5
    partitioning_key: str = "toYYYYMM(timestamp)"


@dataclass
class TickRecord:
    timestamp: int          # epoch nanoseconds
    symbol: str
    venue: str
    price: float
    size: float
    side: str               # "B" | "A" | "T"
    conditions: str = ""
    sequence_number: int = 0


@dataclass
class BarRecord:
    timestamp: int
    symbol: str
    venue: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0
    trade_count: int = 0
    granularity: str = "1m"


@dataclass
class BookSnapshotRecord:
    timestamp: int
    symbol: str
    venue: str
    bid_prices: List[float] = field(default_factory=list)
    bid_sizes: List[float] = field(default_factory=list)
    ask_prices: List[float] = field(default_factory=list)
    ask_sizes: List[float] = field(default_factory=list)
    depth: int = 10


# --------------------------------------------------------------------------- #
#  Abstract base                                                               #
# --------------------------------------------------------------------------- #

class TimeSeriesDB(abc.ABC):
    """Protocol for pluggable time-series backends."""

    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def disconnect(self) -> None: ...

    @abc.abstractmethod
    async def create_tables(self) -> None: ...

    @abc.abstractmethod
    async def insert_ticks(self, ticks: Sequence[TickRecord]) -> int: ...

    @abc.abstractmethod
    async def insert_bars(self, bars: Sequence[BarRecord]) -> int: ...

    @abc.abstractmethod
    async def insert_book_snapshots(self, snaps: Sequence[BookSnapshotRecord]) -> int: ...

    @abc.abstractmethod
    async def query_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        venue: Optional[str] = None,
    ) -> pd.DataFrame: ...

    @abc.abstractmethod
    async def query_bars(
        self,
        symbol: str,
        granularity: DataGranularity,
        start: datetime,
        end: datetime,
        venue: Optional[str] = None,
    ) -> pd.DataFrame: ...

    @abc.abstractmethod
    async def query_book_snapshots(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        venue: Optional[str] = None,
        depth: int = 10,
    ) -> pd.DataFrame: ...

    @abc.abstractmethod
    async def get_symbols(self) -> List[str]: ...

    @abc.abstractmethod
    async def get_date_range(self, symbol: str) -> Tuple[datetime, datetime]: ...


# --------------------------------------------------------------------------- #
#  DuckDB implementation (fallback / research)                                 #
# --------------------------------------------------------------------------- #

class DuckDBTimeSeriesDB(TimeSeriesDB):
    """Embedded DuckDB backend - perfect for local research and tests."""

    def __init__(self, config: TimeSeriesConfig) -> None:
        self._config = config
        self._conn = None

    async def connect(self) -> None:
        import duckdb
        self._conn = duckdb.connect(self._config.duckdb_path)
        logger.info(f"DuckDB connected: {self._config.duckdb_path}")

    async def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    async def create_tables(self) -> None:
        assert self._conn is not None
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                timestamp BIGINT,
                symbol VARCHAR,
                venue VARCHAR,
                price DOUBLE,
                size DOUBLE,
                side VARCHAR(1),
                conditions VARCHAR,
                sequence_number BIGINT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS bars (
                timestamp BIGINT,
                symbol VARCHAR,
                venue VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                vwap DOUBLE,
                trade_count INTEGER,
                granularity VARCHAR(4)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS book_snapshots (
                timestamp BIGINT,
                symbol VARCHAR,
                venue VARCHAR,
                bid_prices DOUBLE[],
                bid_sizes DOUBLE[],
                ask_prices DOUBLE[],
                ask_sizes DOUBLE[],
                depth INTEGER
            )
        """)
        # Indexes
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_sym_ts ON ticks(symbol, timestamp)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_bars_sym_ts ON bars(symbol, granularity, timestamp)")
        logger.info("DuckDB tables created")

    async def insert_ticks(self, ticks: Sequence[TickRecord]) -> int:
        if not ticks:
            return 0
        data = [
            (t.timestamp, t.symbol, t.venue, t.price, t.size, t.side, t.conditions, t.sequence_number)
            for t in ticks
        ]
        self._conn.executemany(
            "INSERT INTO ticks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", data
        )
        return len(data)

    async def insert_bars(self, bars: Sequence[BarRecord]) -> int:
        if not bars:
            return 0
        data = [
            (b.timestamp, b.symbol, b.venue, b.open, b.high, b.low, b.close,
             b.volume, b.vwap, b.trade_count, b.granularity)
            for b in bars
        ]
        self._conn.executemany(
            "INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data
        )
        return len(data)

    async def insert_book_snapshots(self, snaps: Sequence[BookSnapshotRecord]) -> int:
        if not snaps:
            return 0
        data = [
            (s.timestamp, s.symbol, s.venue, s.bid_prices, s.bid_sizes,
             s.ask_prices, s.ask_sizes, s.depth)
            for s in snaps
        ]
        self._conn.executemany(
            "INSERT INTO book_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)", data
        )
        return len(data)

    async def query_ticks(
        self, symbol: str, start: datetime, end: datetime,
        venue: Optional[str] = None,
    ) -> pd.DataFrame:
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)
        query = "SELECT * FROM ticks WHERE symbol = ? AND timestamp BETWEEN ? AND ?"
        params: list = [symbol, start_ns, end_ns]
        if venue:
            query += " AND venue = ?"
            params.append(venue)
        query += " ORDER BY timestamp"
        return self._conn.execute(query, params).fetchdf()

    async def query_bars(
        self, symbol: str, granularity: DataGranularity,
        start: datetime, end: datetime, venue: Optional[str] = None,
    ) -> pd.DataFrame:
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)
        query = "SELECT * FROM bars WHERE symbol = ? AND granularity = ? AND timestamp BETWEEN ? AND ?"
        params: list = [symbol, granularity.value, start_ns, end_ns]
        if venue:
            query += " AND venue = ?"
            params.append(venue)
        query += " ORDER BY timestamp"
        return self._conn.execute(query, params).fetchdf()

    async def query_book_snapshots(
        self, symbol: str, start: datetime, end: datetime,
        venue: Optional[str] = None, depth: int = 10,
    ) -> pd.DataFrame:
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)
        query = "SELECT * FROM book_snapshots WHERE symbol = ? AND timestamp BETWEEN ? AND ?"
        params: list = [symbol, start_ns, end_ns]
        if venue:
            query += " AND venue = ?"
            params.append(venue)
        query += " ORDER BY timestamp"
        return self._conn.execute(query, params).fetchdf()

    async def get_symbols(self) -> List[str]:
        result = self._conn.execute("SELECT DISTINCT symbol FROM ticks ORDER BY symbol").fetchall()
        return [r[0] for r in result]

    async def get_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        row = self._conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM ticks WHERE symbol = ?",
            [symbol],
        ).fetchone()
        if row and row[0]:
            return (
                datetime.fromtimestamp(row[0] / 1e9, tz=timezone.utc),
                datetime.fromtimestamp(row[1] / 1e9, tz=timezone.utc),
            )
        return datetime.now(tz=timezone.utc), datetime.now(tz=timezone.utc)


# --------------------------------------------------------------------------- #
#  ClickHouse implementation                                                   #
# --------------------------------------------------------------------------- #

class ClickHouseTimeSeriesDB(TimeSeriesDB):
    """ClickHouse backend for production analytics workloads."""

    def __init__(self, config: TimeSeriesConfig) -> None:
        self._config = config
        self._client = None

    async def connect(self) -> None:
        try:
            from clickhouse_driver import Client
            self._client = Client(
                host=self._config.clickhouse_host,
                port=self._config.clickhouse_port,
                database=self._config.clickhouse_database,
                user=self._config.clickhouse_user,
                password=self._config.clickhouse_password,
            )
            # Ensure database exists
            self._client.execute(
                f"CREATE DATABASE IF NOT EXISTS {self._config.clickhouse_database}"
            )
            logger.info(f"ClickHouse connected: {self._config.clickhouse_host}:{self._config.clickhouse_port}")
        except ImportError:
            raise RuntimeError("clickhouse-driver not installed. pip install clickhouse-driver")

    async def disconnect(self) -> None:
        if self._client:
            self._client.disconnect()
            self._client = None

    async def create_tables(self) -> None:
        self._client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._config.clickhouse_database}.ticks (
                timestamp UInt64,
                symbol LowCardinality(String),
                venue LowCardinality(String),
                price Float64,
                size Float64,
                side LowCardinality(String),
                conditions String,
                sequence_number UInt64
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(toDateTime(timestamp / 1000000000))
            ORDER BY (symbol, timestamp)
            TTL toDateTime(timestamp / 1000000000) + INTERVAL {self._config.retention_days} DAY
        """)
        self._client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._config.clickhouse_database}.bars (
                timestamp UInt64,
                symbol LowCardinality(String),
                venue LowCardinality(String),
                open Float64,
                high Float64,
                low Float64,
                close Float64,
                volume Float64,
                vwap Float64,
                trade_count UInt32,
                granularity LowCardinality(String)
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(toDateTime(timestamp / 1000000000))
            ORDER BY (symbol, granularity, timestamp)
        """)
        self._client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._config.clickhouse_database}.book_snapshots (
                timestamp UInt64,
                symbol LowCardinality(String),
                venue LowCardinality(String),
                bid_prices Array(Float64),
                bid_sizes Array(Float64),
                ask_prices Array(Float64),
                ask_sizes Array(Float64),
                depth UInt16
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(toDateTime(timestamp / 1000000000))
            ORDER BY (symbol, timestamp)
        """)
        logger.info("ClickHouse tables created")

    async def insert_ticks(self, ticks: Sequence[TickRecord]) -> int:
        if not ticks:
            return 0
        data = [
            (t.timestamp, t.symbol, t.venue, t.price, t.size, t.side, t.conditions, t.sequence_number)
            for t in ticks
        ]
        self._client.execute(
            f"INSERT INTO {self._config.clickhouse_database}.ticks VALUES",
            data,
        )
        return len(data)

    async def insert_bars(self, bars: Sequence[BarRecord]) -> int:
        if not bars:
            return 0
        data = [
            (b.timestamp, b.symbol, b.venue, b.open, b.high, b.low, b.close,
             b.volume, b.vwap, b.trade_count, b.granularity)
            for b in bars
        ]
        self._client.execute(
            f"INSERT INTO {self._config.clickhouse_database}.bars VALUES",
            data,
        )
        return len(data)

    async def insert_book_snapshots(self, snaps: Sequence[BookSnapshotRecord]) -> int:
        if not snaps:
            return 0
        data = [
            (s.timestamp, s.symbol, s.venue, s.bid_prices, s.bid_sizes,
             s.ask_prices, s.ask_sizes, s.depth)
            for s in snaps
        ]
        self._client.execute(
            f"INSERT INTO {self._config.clickhouse_database}.book_snapshots VALUES",
            data,
        )
        return len(data)

    async def query_ticks(
        self, symbol: str, start: datetime, end: datetime,
        venue: Optional[str] = None,
    ) -> pd.DataFrame:
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)
        venue_clause = f"AND venue = '{venue}'" if venue else ""
        query = f"""
            SELECT * FROM {self._config.clickhouse_database}.ticks
            WHERE symbol = '{symbol}'
              AND timestamp BETWEEN {start_ns} AND {end_ns}
              {venue_clause}
            ORDER BY timestamp
        """
        result, columns = self._client.execute(query, with_column_types=True)
        col_names = [c[0] for c in columns]
        return pd.DataFrame(result, columns=col_names)

    async def query_bars(
        self, symbol: str, granularity: DataGranularity,
        start: datetime, end: datetime, venue: Optional[str] = None,
    ) -> pd.DataFrame:
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)
        venue_clause = f"AND venue = '{venue}'" if venue else ""
        query = f"""
            SELECT * FROM {self._config.clickhouse_database}.bars
            WHERE symbol = '{symbol}'
              AND granularity = '{granularity.value}'
              AND timestamp BETWEEN {start_ns} AND {end_ns}
              {venue_clause}
            ORDER BY timestamp
        """
        result, columns = self._client.execute(query, with_column_types=True)
        col_names = [c[0] for c in columns]
        return pd.DataFrame(result, columns=col_names)

    async def query_book_snapshots(
        self, symbol: str, start: datetime, end: datetime,
        venue: Optional[str] = None, depth: int = 10,
    ) -> pd.DataFrame:
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)
        venue_clause = f"AND venue = '{venue}'" if venue else ""
        query = f"""
            SELECT * FROM {self._config.clickhouse_database}.book_snapshots
            WHERE symbol = '{symbol}'
              AND timestamp BETWEEN {start_ns} AND {end_ns}
              {venue_clause}
            ORDER BY timestamp
        """
        result, columns = self._client.execute(query, with_column_types=True)
        col_names = [c[0] for c in columns]
        return pd.DataFrame(result, columns=col_names)

    async def get_symbols(self) -> List[str]:
        result = self._client.execute(
            f"SELECT DISTINCT symbol FROM {self._config.clickhouse_database}.ticks ORDER BY symbol"
        )
        return [r[0] for r in result]

    async def get_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        row = self._client.execute(f"""
            SELECT min(timestamp), max(timestamp)
            FROM {self._config.clickhouse_database}.ticks
            WHERE symbol = '{symbol}'
        """)
        if row and row[0][0]:
            return (
                datetime.fromtimestamp(row[0][0] / 1e9, tz=timezone.utc),
                datetime.fromtimestamp(row[0][1] / 1e9, tz=timezone.utc),
            )
        return datetime.now(tz=timezone.utc), datetime.now(tz=timezone.utc)


# --------------------------------------------------------------------------- #
#  ArcticDB implementation                                                     #
# --------------------------------------------------------------------------- #

class ArcticDBTimeSeriesDB(TimeSeriesDB):
    """ArcticDB backend - DataFrame-native versioned time-series store."""

    def __init__(self, config: TimeSeriesConfig) -> None:
        self._config = config
        self._lib = None

    async def connect(self) -> None:
        try:
            import arcticdb as adb
            store = adb.Arctic(self._config.arctic_uri)
            self._lib = store.get_library(self._config.arctic_library, create_if_missing=True)
            logger.info(f"ArcticDB connected: {self._config.arctic_uri}")
        except ImportError:
            raise RuntimeError("arcticdb not installed. pip install arcticdb")

    async def disconnect(self) -> None:
        self._lib = None

    async def create_tables(self) -> None:
        # ArcticDB is schema-on-write, no explicit table creation needed
        logger.info("ArcticDB: schema-on-write, no table creation needed")

    async def insert_ticks(self, ticks: Sequence[TickRecord]) -> int:
        if not ticks:
            return 0
        records = [
            {"timestamp": t.timestamp, "symbol": t.symbol, "venue": t.venue,
             "price": t.price, "size": t.size, "side": t.side}
            for t in ticks
        ]
        df = pd.DataFrame(records)
        for symbol in df["symbol"].unique():
            sym_df = df[df["symbol"] == symbol].set_index("timestamp")
            key = f"ticks/{symbol}"
            self._lib.append(key, sym_df, upsert=True)
        return len(ticks)

    async def insert_bars(self, bars: Sequence[BarRecord]) -> int:
        if not bars:
            return 0
        records = [
            {"timestamp": b.timestamp, "symbol": b.symbol, "venue": b.venue,
             "open": b.open, "high": b.high, "low": b.low, "close": b.close,
             "volume": b.volume, "vwap": b.vwap, "granularity": b.granularity}
            for b in bars
        ]
        df = pd.DataFrame(records)
        for symbol in df["symbol"].unique():
            sym_df = df[df["symbol"] == symbol].set_index("timestamp")
            gran = sym_df["granularity"].iloc[0]
            key = f"bars/{symbol}/{gran}"
            self._lib.append(key, sym_df, upsert=True)
        return len(bars)

    async def insert_book_snapshots(self, snaps: Sequence[BookSnapshotRecord]) -> int:
        if not snaps:
            return 0
        # Flatten for columnar storage
        records = []
        for s in snaps:
            rec = {"timestamp": s.timestamp, "symbol": s.symbol, "venue": s.venue}
            for i in range(min(len(s.bid_prices), s.depth)):
                rec[f"bid_p{i}"] = s.bid_prices[i]
                rec[f"bid_s{i}"] = s.bid_sizes[i]
            for i in range(min(len(s.ask_prices), s.depth)):
                rec[f"ask_p{i}"] = s.ask_prices[i]
                rec[f"ask_s{i}"] = s.ask_sizes[i]
            records.append(rec)
        df = pd.DataFrame(records)
        for symbol in df["symbol"].unique():
            sym_df = df[df["symbol"] == symbol].set_index("timestamp")
            key = f"book/{symbol}"
            self._lib.append(key, sym_df, upsert=True)
        return len(snaps)

    async def query_ticks(
        self, symbol: str, start: datetime, end: datetime,
        venue: Optional[str] = None,
    ) -> pd.DataFrame:
        from arcticdb import QueryBuilder
        key = f"ticks/{symbol}"
        try:
            vit = self._lib.read(key)
            df = vit.data
            start_ns = int(start.timestamp() * 1e9)
            end_ns = int(end.timestamp() * 1e9)
            df = df[(df.index >= start_ns) & (df.index <= end_ns)]
            if venue:
                df = df[df["venue"] == venue]
            return df.reset_index()
        except Exception:
            return pd.DataFrame()

    async def query_bars(
        self, symbol: str, granularity: DataGranularity,
        start: datetime, end: datetime, venue: Optional[str] = None,
    ) -> pd.DataFrame:
        key = f"bars/{symbol}/{granularity.value}"
        try:
            vit = self._lib.read(key)
            df = vit.data
            start_ns = int(start.timestamp() * 1e9)
            end_ns = int(end.timestamp() * 1e9)
            df = df[(df.index >= start_ns) & (df.index <= end_ns)]
            if venue:
                df = df[df["venue"] == venue]
            return df.reset_index()
        except Exception:
            return pd.DataFrame()

    async def query_book_snapshots(
        self, symbol: str, start: datetime, end: datetime,
        venue: Optional[str] = None, depth: int = 10,
    ) -> pd.DataFrame:
        key = f"book/{symbol}"
        try:
            vit = self._lib.read(key)
            df = vit.data
            start_ns = int(start.timestamp() * 1e9)
            end_ns = int(end.timestamp() * 1e9)
            df = df[(df.index >= start_ns) & (df.index <= end_ns)]
            if venue:
                df = df[df["venue"] == venue]
            return df.reset_index()
        except Exception:
            return pd.DataFrame()

    async def get_symbols(self) -> List[str]:
        symbols = set()
        for key in self._lib.list_symbols():
            if key.startswith("ticks/"):
                symbols.add(key.split("/")[1])
        return sorted(symbols)

    async def get_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        key = f"ticks/{symbol}"
        try:
            vit = self._lib.read(key)
            idx = vit.data.index
            return (
                datetime.fromtimestamp(idx.min() / 1e9, tz=timezone.utc),
                datetime.fromtimestamp(idx.max() / 1e9, tz=timezone.utc),
            )
        except Exception:
            return datetime.now(tz=timezone.utc), datetime.now(tz=timezone.utc)


# --------------------------------------------------------------------------- #
#  Factory                                                                     #
# --------------------------------------------------------------------------- #

def create_timeseries_db(config: TimeSeriesConfig) -> TimeSeriesDB:
    """Factory function to create the appropriate backend."""
    backend_map = {
        TimeSeriesBackend.DUCKDB: DuckDBTimeSeriesDB,
        TimeSeriesBackend.CLICKHOUSE: ClickHouseTimeSeriesDB,
        TimeSeriesBackend.ARCTICDB: ArcticDBTimeSeriesDB,
    }
    cls = backend_map.get(config.backend)
    if cls is None:
        raise ValueError(f"Unsupported backend: {config.backend}")
    return cls(config)
