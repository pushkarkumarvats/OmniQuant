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
    """
    Main class for data ingestion from multiple sources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataIngestion
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_path = Path(self.config.get("data_path", "data/raw"))
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def load_csv(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading CSV from {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded {len(df)} rows from CSV")
        return df
    
    def load_parquet(self, filepath: Union[str, Path], engine: str = "pyarrow") -> pd.DataFrame:
        """
        Load data from Parquet file
        
        Args:
            filepath: Path to Parquet file
            engine: Engine to use ('pyarrow' or 'fastparquet')
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading Parquet from {filepath}")
        df = pd.read_parquet(filepath, engine=engine)
        logger.info(f"Loaded {len(df)} rows from Parquet")
        return df
    
    def load_parquet_polars(self, filepath: Union[str, Path]) -> pl.DataFrame:
        """
        Load data from Parquet using Polars (faster for large files)
        
        Args:
            filepath: Path to Parquet file
            
        Returns:
            Polars DataFrame
        """
        logger.info(f"Loading Parquet with Polars from {filepath}")
        df = pl.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} rows from Parquet")
        return df
    
    def load_from_duckdb(self, query: str, db_path: Optional[str] = None) -> pd.DataFrame:
        """
        Query data using DuckDB
        
        Args:
            query: SQL query string
            db_path: Path to DuckDB database (None for in-memory)
            
        Returns:
            DataFrame with query results
        """
        logger.info(f"Executing DuckDB query")
        conn = duckdb.connect(db_path)
        df = conn.execute(query).df()
        conn.close()
        logger.info(f"Query returned {len(df)} rows")
        return df
    
    def fetch_yahoo_finance(
        self, 
        tickers: Union[str, List[str]], 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch market data from Yahoo Finance
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1m', '5m', '1h', '1d', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        if isinstance(tickers, str):
            tickers = [tickers]
            
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"Fetching data for {tickers} from {start_date} to {end_date}")
        
        data_frames = []
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
                df['ticker'] = ticker
                data_frames.append(df)
                logger.info(f"Fetched {len(df)} rows for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                
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
        """
        Generate synthetic tick data for testing
        
        Args:
            num_ticks: Number of ticks to generate
            initial_price: Starting price
            volatility: Price volatility
            tick_size: Minimum price increment
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic tick data
        """
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
        """
        Generate synthetic order book snapshots
        
        Args:
            num_snapshots: Number of snapshots
            depth: Order book depth (levels)
            initial_price: Mid price
            seed: Random seed
            
        Returns:
            DataFrame with order book snapshots
        """
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
        """
        Save DataFrame to Parquet format
        
        Args:
            df: DataFrame to save
            filename: Output filename
            compression: Compression algorithm
        """
        filepath = self.data_path / filename
        df.to_parquet(filepath, compression=compression, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame to CSV format
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
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
