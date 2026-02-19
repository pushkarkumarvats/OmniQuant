"""
Data Ingestion Module
Handles loading data from various sources (CSV, Parquet, databases, APIs)
"""

import pandas as pd
import polars as pl
import duckdb
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from loguru import logger
import yfinance as yf
from datetime import datetime, timedelta


class DataIngestion:
    """Loads market data from CSV, Parquet, DuckDB, and Yahoo Finance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_path = Path(self.config.get("data_path", "data/raw"))
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def load_csv(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Read a CSV into a DataFrame."""
        logger.info(f"Loading CSV from {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows from CSV")
        return df
    
    def load_parquet(self, filepath: Union[str, Path], engine: str = "pyarrow") -> pd.DataFrame:
        logger.info(f"Loading Parquet from {filepath}")
        df = pd.read_parquet(filepath, engine=engine)
        logger.info(f"Loaded {len(df)} rows from Parquet")
        return df
    
    def load_parquet_polars(self, filepath: Union[str, Path]) -> pl.DataFrame:
        """Polars-based Parquet reader, faster for large files."""
        logger.info(f"Loading Parquet with Polars from {filepath}")
        df = pl.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} rows from Parquet")
        return df
    
    def load_from_duckdb(self, query: str, db_path: Optional[str] = None) -> pd.DataFrame:
        """Run a SQL query against DuckDB and return results as a DataFrame."""
        logger.info(f"Executing DuckDB query")
        conn = duckdb.connect(db_path)
        df = conn.execute(query).df()
        conn.close()
        logger.info(f"Query returned {len(df)} rows")
        return df
    
    def fetch_yahoo_finance(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d',
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance with retry logic."""
        try:
            import yfinance as yf
        except ImportError as e:
            logger.error("yfinance not installed. Install with: pip install yfinance")
            raise ImportError("yfinance package required for Yahoo Finance data") from e
        
        # Validate inputs
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        import time
        from datetime import datetime
        
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}") from e
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()
                
                df = df.reset_index()
                df.columns = df.columns.str.lower()
                
                logger.info(f"Fetched {len(df)} records for {symbol}")
                return df
                
            except ConnectionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error for {symbol}, retry {attempt + 1}/{max_retries}: {e}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {symbol} after {max_retries} attempts")
                    raise ConnectionError(f"Unable to fetch data for {symbol} after {max_retries} retries") from e
            
            except Exception as e:
                logger.error(f"Unexpected error fetching {symbol}: {type(e).__name__}: {e}")
                raise
                
        if data_frames:
            result = pd.concat(data_frames)
            return result
        else:
            return pd.DataFrame()
    
    def generate_synthetic_tick_data(
        self,
        num_ticks: int = 10000,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        tick_size: float = 0.01,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate fake tick data for testing and development."""
        import numpy as np
        
        if seed is not None:
            np.random.seed(seed)
            
        logger.info(f"Generating {num_ticks} synthetic ticks")
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=1),
            periods=num_ticks,
            freq='100ms'
        )
        
        # Generate price changes
        returns = np.random.normal(0, volatility / np.sqrt(252 * 6.5 * 3600 / 0.1), num_ticks)
        prices = initial_price * np.exp(np.cumsum(returns))
        prices = (prices / tick_size).round() * tick_size
        
        # Generate volumes
        volumes = np.random.lognormal(mean=4, sigma=1, size=num_ticks).astype(int) * 100
        
        # Generate bid-ask spread
        spread_ticks = np.random.choice([1, 2, 3], size=num_ticks, p=[0.6, 0.3, 0.1])
        spreads = spread_ticks * tick_size
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'bid': prices - spreads / 2,
            'ask': prices + spreads / 2,
            'bid_size': volumes,
            'ask_size': volumes * np.random.uniform(0.8, 1.2, num_ticks),
            'volume': volumes,
            'side': np.random.choice(['buy', 'sell'], num_ticks)
        })
        
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']
        
        logger.info(f"Generated synthetic tick data with {len(df)} rows")
        return df
    
    def generate_synthetic_orderbook(
        self,
        num_snapshots: int = 1000,
        depth: int = 10,
        initial_price: float = 100.0,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Create synthetic LOB snapshots for testing."""
        import numpy as np
        
        if seed is not None:
            np.random.seed(seed)
            
        logger.info(f"Generating {num_snapshots} order book snapshots with depth {depth}")
        
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=1),
            periods=num_snapshots,
            freq='1s'
        )
        
        snapshots = []
        current_price = initial_price
        
        for ts in timestamps:
            # Random walk for mid price
            current_price *= (1 + np.random.normal(0, 0.0001))
            
            snapshot = {'timestamp': ts, 'mid_price': current_price}
            
            # Generate bid side
            for i in range(depth):
                level_price = current_price - (i + 1) * 0.01
                level_size = np.random.exponential(1000) * (depth - i)
                snapshot[f'bid_price_{i+1}'] = level_price
                snapshot[f'bid_size_{i+1}'] = level_size
            
            # Generate ask side
            for i in range(depth):
                level_price = current_price + (i + 1) * 0.01
                level_size = np.random.exponential(1000) * (depth - i)
                snapshot[f'ask_price_{i+1}'] = level_price
                snapshot[f'ask_size_{i+1}'] = level_size
                
            snapshots.append(snapshot)
        
        df = pd.DataFrame(snapshots)
        logger.info(f"Generated {len(df)} order book snapshots")
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str, compression: str = 'snappy'):
        filepath = self.data_path / filename
        df.to_parquet(filepath, compression=compression, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        filepath = self.data_path / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")


if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion()
    
    # Generate and save synthetic data
    tick_data = ingestion.generate_synthetic_tick_data(num_ticks=10000, seed=42)
    ingestion.save_to_parquet(tick_data, "synthetic_ticks.parquet")
    
    orderbook_data = ingestion.generate_synthetic_orderbook(num_snapshots=1000, seed=42)
    ingestion.save_to_parquet(orderbook_data, "synthetic_orderbook.parquet")
    
    logger.info("Data ingestion example completed")
