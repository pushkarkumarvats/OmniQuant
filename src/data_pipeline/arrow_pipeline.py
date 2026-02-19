"""
Arrow-Native Data Pipeline - Zero-Copy Columnar Processing

Replaces Pandas-centric ingestion with Apache Arrow / Polars-first pipelines
for 10-100x faster OHLCV and tick data processing. Supports:
  - Zero-copy Arrow IPC between Python, Rust, and GPU
  - Polars lazy evaluation with predicate pushdown
  - DuckDB integration for out-of-core SQL on Parquet
  - Optional cuDF/RAPIDS GPU DataFrames for massive datasets
  - Memory-mapped Parquet for low-latency replay
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ArrowPipelineConfig:
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    chunk_size: int = 1_000_000          # rows per chunk for streaming
    use_memory_map: bool = True          # mmap Parquet files
    use_gpu: bool = False                # Enable cuDF acceleration
    compression: str = "zstd"            # Arrow IPC compression
    max_partitions: int = 256
    parquet_row_group_size: int = 500_000
    enable_statistics: bool = True


# ---------------------------------------------------------------------------
# Arrow Schema Registry
# ---------------------------------------------------------------------------

class SchemaRegistry:
    """Canonical Arrow schemas for all market data types."""

    TICK = pa.schema([
        pa.field("timestamp_ns", pa.uint64()),
        pa.field("symbol", pa.utf8()),
        pa.field("price", pa.float64()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("bid_size", pa.int64()),
        pa.field("ask_size", pa.int64()),
        pa.field("volume", pa.int64()),
        pa.field("side", pa.utf8()),
        pa.field("exchange", pa.utf8()),
    ])

    OHLCV = pa.schema([
        pa.field("timestamp", pa.timestamp("ns")),
        pa.field("symbol", pa.utf8()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("vwap", pa.float64()),
        pa.field("trade_count", pa.int32()),
    ])

    LOB_SNAPSHOT = pa.schema([
        pa.field("timestamp_ns", pa.uint64()),
        pa.field("symbol", pa.utf8()),
        pa.field("bid_prices", pa.list_(pa.float64())),
        pa.field("bid_sizes", pa.list_(pa.int64())),
        pa.field("ask_prices", pa.list_(pa.float64())),
        pa.field("ask_sizes", pa.list_(pa.int64())),
    ])

    TRADE = pa.schema([
        pa.field("timestamp_ns", pa.uint64()),
        pa.field("symbol", pa.utf8()),
        pa.field("price", pa.float64()),
        pa.field("quantity", pa.int64()),
        pa.field("side", pa.utf8()),
        pa.field("trade_id", pa.utf8()),
        pa.field("exchange", pa.utf8()),
    ])


# ---------------------------------------------------------------------------
# Polars-First Ingestion Engine
# ---------------------------------------------------------------------------

class PolarsIngestionEngine:
    """Polars-backed ingestion engine returning LazyFrames for deferred execution."""

    def __init__(self, config: Optional[ArrowPipelineConfig] = None):
        self.config = config or ArrowPipelineConfig()
        self._data_dir = Path(self.config.data_dir)
        self._processed_dir = Path(self.config.processed_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    def scan_parquet(
        self, path: Union[str, Path], *, hive_partitioning: bool = True
    ) -> pl.LazyFrame:
        """Lazy-scan a Parquet file/directory with predicate pushdown."""
        logger.info(f"Lazy-scanning Parquet: {path}")
        return pl.scan_parquet(
            str(path),
            hive_partitioning=hive_partitioning,
            low_memory=False,
        )

    def scan_csv(
        self, path: Union[str, Path], *, separator: str = ","
    ) -> pl.LazyFrame:
        """Lazy-scan a CSV with automatic type inference."""
        logger.info(f"Lazy-scanning CSV: {path}")
        return pl.scan_csv(str(path), separator=separator)

    def read_ticks_optimized(
        self,
        path: Union[str, Path],
        symbol_filter: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
    ) -> pl.LazyFrame:
        """Read ticks with predicate pushdown on symbol and time range."""
        lf = self.scan_parquet(path)
        if symbol_filter:
            lf = lf.filter(pl.col("symbol") == symbol_filter)
        if start_ns is not None:
            lf = lf.filter(pl.col("timestamp_ns") >= start_ns)
        if end_ns is not None:
            lf = lf.filter(pl.col("timestamp_ns") <= end_ns)
        return lf

    def aggregate_ticks_to_bars(
        self,
        ticks: pl.LazyFrame,
        interval: str = "1m",
    ) -> pl.LazyFrame:
        """Aggregate ticks into OHLCV bars via Polars group_by_dynamic."""
        return (
            ticks.sort("timestamp_ns")
            .with_columns(
                pl.col("timestamp_ns")
                .cast(pl.Datetime("ns"))
                .alias("datetime")
            )
            .group_by_dynamic("datetime", every=interval, by="symbol")
            .agg([
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                (pl.col("price") * pl.col("volume")).sum().alias("_pv"),
                pl.col("price").count().alias("trade_count"),
            ])
            .with_columns(
                (pl.col("_pv") / pl.col("volume")).alias("vwap")
            )
            .drop("_pv")
        )

    def compute_features_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Add common microstructure features in a single lazy pass."""
        return lf.with_columns([
            ((pl.col("ask") - pl.col("bid")) / pl.col("price") * 10_000)
                .alias("spread_bps"),
            ((pl.col("bid_size") - pl.col("ask_size"))
             / (pl.col("bid_size") + pl.col("ask_size")))
                .alias("order_imbalance"),
            pl.col("price").pct_change().alias("return_1tick"),
            pl.col("price").rolling_std(window_size=20).alias("realized_vol_20"),
            pl.col("volume").rolling_mean(window_size=50).alias("avg_volume_50"),
        ])

    def write_partitioned_parquet(
        self,
        df: pl.DataFrame,
        path: Union[str, Path],
        partition_by: str = "symbol",
    ) -> None:
        """Write a partitioned Parquet dataset using Hive partitioning."""
        arrow_table = df.to_arrow()
        pq.write_to_dataset(
            arrow_table,
            str(path),
            partition_cols=[partition_by],
            compression=self.config.compression,
        )
        logger.info(f"Wrote partitioned Parquet to {path} (by {partition_by})")


# ---------------------------------------------------------------------------
# Arrow Zero-Copy IPC
# ---------------------------------------------------------------------------

class ArrowIPCBridge:
    """Zero-copy Arrow IPC for cross-process RecordBatch exchange."""

    @staticmethod
    def serialize_batch(batch: pa.RecordBatch, *, compress: bool = True) -> bytes:
        """Serialize a RecordBatch to Arrow IPC bytes."""
        sink = pa.BufferOutputStream()
        options = ipc.IpcWriteOptions(
            compression=pa.Codec("zstd") if compress else None,
        )
        writer = ipc.new_stream(sink, batch.schema, options=options)
        writer.write_batch(batch)
        writer.close()
        return sink.getvalue().to_pybytes()

    @staticmethod
    def deserialize_batch(data: bytes) -> pa.RecordBatch:
        """Deserialize Arrow IPC bytes to a RecordBatch."""
        reader = ipc.open_stream(data)
        return reader.read_all().to_batches()[0]

    @staticmethod
    def table_to_polars(table: pa.Table) -> pl.DataFrame:
        """Zero-copy convert Arrow Table to Polars DataFrame."""
        return pl.from_arrow(table)

    @staticmethod
    def polars_to_arrow(df: pl.DataFrame) -> pa.Table:
        """Zero-copy convert Polars DataFrame to Arrow Table."""
        return df.to_arrow()

    @staticmethod
    def write_ipc_file(
        table: pa.Table, path: Union[str, Path], *, compress: bool = True
    ) -> None:
        """Write an Arrow IPC (Feather v2) file."""
        import pyarrow.feather as feather
        feather.write_feather(
            table, str(path),
            compression="zstd" if compress else "uncompressed",
        )

    @staticmethod
    def read_ipc_file(path: Union[str, Path]) -> pa.Table:
        """Read an Arrow IPC (Feather v2) file with memory-mapping."""
        import pyarrow.feather as feather
        return feather.read_table(str(path), memory_map=True)


# ---------------------------------------------------------------------------
# GPU DataFrame Acceleration (cuDF / RAPIDS)
# ---------------------------------------------------------------------------

class GPUDataFrameEngine:
    """cuDF/RAPIDS GPU acceleration with transparent Polars CPU fallback."""

    def __init__(self) -> None:
        self._has_cudf = False
        try:
            import cudf  # type: ignore
            self._has_cudf = True
            logger.info("cuDF available – GPU acceleration enabled")
        except ImportError:
            logger.info("cuDF not available – falling back to Polars CPU")

    @property
    def gpu_available(self) -> bool:
        return self._has_cudf

    def to_gpu(self, df: pl.DataFrame) -> Any:
        """Transfer a Polars DataFrame to GPU memory via Arrow."""
        if self._has_cudf:
            import cudf  # type: ignore
            return cudf.DataFrame.from_arrow(df.to_arrow())
        logger.warning("cuDF not available; returning Polars DataFrame on CPU")
        return df

    def from_gpu(self, gdf: Any) -> pl.DataFrame:
        """Transfer a cuDF DataFrame back to CPU as Polars."""
        if self._has_cudf:
            return pl.from_arrow(gdf.to_arrow())
        return gdf  # already Polars

    def gpu_rolling_features(
        self, gdf: Any, windows: Sequence[int] = (5, 20, 60)
    ) -> Any:
        """Rolling mean/std on GPU or CPU fallback."""
        if self._has_cudf:
            import cudf  # type: ignore
            for w in windows:
                gdf[f"close_ma_{w}"] = gdf["close"].rolling(w).mean()
                gdf[f"close_std_{w}"] = gdf["close"].rolling(w).std()
                gdf[f"volume_ma_{w}"] = gdf["volume"].rolling(w).mean()
            return gdf

        # Polars CPU fallback
        if isinstance(gdf, pl.DataFrame):
            exprs: list[pl.Expr] = []
            for w in windows:
                exprs.append(pl.col("close").rolling_mean(window_size=w).alias(f"close_ma_{w}"))
                exprs.append(pl.col("close").rolling_std(window_size=w).alias(f"close_std_{w}"))
                exprs.append(pl.col("volume").rolling_mean(window_size=w).alias(f"volume_ma_{w}"))
            return gdf.with_columns(exprs)
        return gdf

    def gpu_correlation_matrix(self, gdf: Any, columns: List[str]) -> np.ndarray:
        """Compute correlation matrix on GPU or CPU."""
        if self._has_cudf:
            sub = gdf[columns]
            return sub.corr().values  # cuDF corr returns cuDF DataFrame
        if isinstance(gdf, pl.DataFrame):
            return gdf.select(columns).to_pandas().corr().values
        return np.eye(len(columns))


# ---------------------------------------------------------------------------
# Streaming Tick Replay (memory-mapped)
# ---------------------------------------------------------------------------

class MmapTickReplayer:
    """Memory-mapped binary tick replayer (41 bytes/tick: ts, price, bid, ask, vol, side)."""

    RECORD_SIZE = 8 + 8 + 8 + 8 + 8 + 1  # 41 bytes

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self._size = 0
        self._offset = 0

    def open(self) -> None:
        fsize = self._path.stat().st_size
        self._fd = os.open(str(self._path), os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, fsize, access=mmap.ACCESS_READ)
        self._size = fsize
        self._offset = 0
        logger.info(f"Opened mmap tick file: {self._path} ({fsize} bytes, "
                     f"{fsize // self.RECORD_SIZE} ticks)")

    def close(self) -> None:
        if self._mm:
            self._mm.close()
        if self._fd is not None:
            os.close(self._fd)

    def __enter__(self):
        self.open(); return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self) -> Iterator[Tuple[int, float, float, float, int, int]]:
        while self._offset + self.RECORD_SIZE <= self._size:
            raw = self._mm[self._offset : self._offset + self.RECORD_SIZE]  # type: ignore
            ts, price, bid, ask, vol = struct.unpack("<Qddddq", raw[:40])
            side = raw[40]
            self._offset += self.RECORD_SIZE
            yield ts, price, bid, ask, vol, side

    def seek_ns(self, target_ns: int) -> None:
        """Binary-search to a target timestamp (requires sorted file)."""
        lo, hi = 0, (self._size // self.RECORD_SIZE) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            off = mid * self.RECORD_SIZE
            ts = struct.unpack("<Q", self._mm[off:off + 8])[0]  # type: ignore
            if ts < target_ns:
                lo = mid + 1
            else:
                hi = mid - 1
        self._offset = lo * self.RECORD_SIZE

    @property
    def tick_count(self) -> int:
        return self._size // self.RECORD_SIZE


# ---------------------------------------------------------------------------
# DuckDB Integration
# ---------------------------------------------------------------------------

class DuckDBAnalytics:
    """Out-of-core SQL analytics on Parquet via DuckDB."""

    def __init__(self, db_path: Optional[str] = None):
        import duckdb
        self._conn = duckdb.connect(db_path or ":memory:")

    def query(self, sql: str) -> pl.DataFrame:
        """Execute SQL and return result as Polars DataFrame."""
        result = self._conn.execute(sql)
        return pl.from_arrow(result.arrow())

    def register_parquet(self, name: str, path: str) -> None:
        """Register a Parquet file as a virtual table."""
        self._conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
        logger.info(f"Registered Parquet view: {name} -> {path}")

    def register_arrow(self, name: str, table: pa.Table) -> None:
        """Register an Arrow table as a virtual table (zero-copy)."""
        self._conn.register(name, table)
        logger.info(f"Registered Arrow table: {name}")

    def ohlcv_from_ticks(self, ticks_path: str, interval: str = "1 minute") -> pl.DataFrame:
        """Aggregate ticks to OHLCV bars using DuckDB SQL."""
        sql = f"""
            SELECT
                time_bucket(INTERVAL '{interval}', timestamp_ns::TIMESTAMP) AS bar_time,
                symbol,
                first(price) AS open,
                max(price) AS high,
                min(price) AS low,
                last(price) AS close,
                sum(volume) AS volume,
                count(*) AS trade_count
            FROM read_parquet('{ticks_path}')
            GROUP BY bar_time, symbol
            ORDER BY bar_time
        """
        return self.query(sql)

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Convenience: unified pipeline facade
# ---------------------------------------------------------------------------

class ArrowDataPipeline:
    """Unified facade over Polars, Arrow IPC, GPU, mmap replay, and DuckDB."""

    def __init__(self, config: Optional[ArrowPipelineConfig] = None):
        self.config = config or ArrowPipelineConfig()
        self.ingestion = PolarsIngestionEngine(self.config)
        self.ipc = ArrowIPCBridge()
        self.gpu = GPUDataFrameEngine()
        self.duckdb = DuckDBAnalytics()
        logger.info("ArrowDataPipeline initialised "
                     f"(GPU={'ON' if self.gpu.gpu_available else 'OFF'})")

    def load_ticks(
        self,
        path: Union[str, Path],
        symbol: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        with_features: bool = False,
    ) -> pl.LazyFrame:
        """Load tick data with optional filtering and feature computation."""
        lf = self.ingestion.read_ticks_optimized(path, symbol, start_ns, end_ns)
        if with_features:
            lf = self.ingestion.compute_features_lazy(lf)
        return lf

    def ticks_to_bars(
        self, ticks_path: Union[str, Path], interval: str = "1m"
    ) -> pl.DataFrame:
        """Aggregate ticks into OHLCV bars."""
        lf = self.ingestion.scan_parquet(ticks_path)
        return self.ingestion.aggregate_ticks_to_bars(lf, interval).collect()

    def to_arrow(self, df: pl.DataFrame) -> pa.Table:
        return self.ipc.polars_to_arrow(df)

    def from_arrow(self, table: pa.Table) -> pl.DataFrame:
        return self.ipc.table_to_polars(table)


__all__ = [
    "ArrowPipelineConfig",
    "SchemaRegistry",
    "PolarsIngestionEngine",
    "ArrowIPCBridge",
    "GPUDataFrameEngine",
    "MmapTickReplayer",
    "DuckDBAnalytics",
    "ArrowDataPipeline",
]
