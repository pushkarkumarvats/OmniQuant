"""
Data Alignment Module
Handles synchronization and alignment of multiple data sources
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from loguru import logger


class DataAligner:
    """
    Align and synchronize data from multiple sources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataAligner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def align_by_time(
        self,
        dfs: List[pd.DataFrame],
        timestamp_col: str = 'timestamp',
        method: str = 'outer'
    ) -> pd.DataFrame:
        """
        Align multiple DataFrames by timestamp
        
        Args:
            dfs: List of DataFrames to align
            timestamp_col: Name of timestamp column
            method: 'inner', 'outer', 'left', 'right'
            
        Returns:
            Aligned DataFrame
        """
        logger.info(f"Aligning {len(dfs)} DataFrames by time using {method} join")
        
        if not dfs:
            return pd.DataFrame()
        
        # Set timestamp as index for all DataFrames
        dfs_indexed = []
        for i, df in enumerate(dfs):
            df_copy = df.copy()
            if timestamp_col in df_copy.columns:
                df_copy = df_copy.set_index(timestamp_col)
            dfs_indexed.append(df_copy)
        
        # Merge all DataFrames
        result = dfs_indexed[0]
        for i, df in enumerate(dfs_indexed[1:], 1):
            result = result.join(df, how=method, rsuffix=f'_{i}')
        
        logger.info(f"Aligned data shape: {result.shape}")
        return result.reset_index()
    
    def forward_fill_gaps(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        max_gap: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Forward fill gaps in data with optional max gap limit
        
        Args:
            df: Input DataFrame
            columns: Columns to fill (None for all)
            max_gap: Maximum number of consecutive NaNs to fill
            
        Returns:
            DataFrame with gaps filled
        """
        df_filled = df.copy()
        cols = columns if columns else df.columns
        
        if max_gap is not None:
            df_filled[cols] = df_filled[cols].fillna(method='ffill', limit=max_gap)
        else:
            df_filled[cols] = df_filled[cols].fillna(method='ffill')
        
        logger.info(f"Forward filled gaps in {len(cols)} columns")
        return df_filled
    
    def interpolate_gaps(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'linear'
    ) -> pd.DataFrame:
        """
        Interpolate missing values
        
        Args:
            df: Input DataFrame
            columns: Columns to interpolate
            method: Interpolation method ('linear', 'quadratic', 'cubic', 'time')
            
        Returns:
            DataFrame with interpolated values
        """
        df_interp = df.copy()
        cols = columns if columns else df.columns
        
        df_interp[cols] = df_interp[cols].interpolate(method=method)
        
        logger.info(f"Interpolated {len(cols)} columns using {method} method")
        return df_interp
    
    def create_regular_grid(
        self,
        df: pd.DataFrame,
        freq: str,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Create regular time grid and align data
        
        Args:
            df: Input DataFrame
            freq: Desired frequency ('1s', '1min', etc.)
            timestamp_col: Timestamp column name
            
        Returns:
            DataFrame with regular time grid
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Column {timestamp_col} not found")
        
        # Create regular time grid
        start_time = df[timestamp_col].min()
        end_time = df[timestamp_col].max()
        regular_grid = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # Create new DataFrame with regular grid
        df_regular = pd.DataFrame({timestamp_col: regular_grid})
        
        # Merge with original data
        df_indexed = df.set_index(timestamp_col)
        df_regular = df_regular.set_index(timestamp_col)
        df_result = df_regular.join(df_indexed, how='left')
        
        logger.info(f"Created regular grid with {len(df_result)} rows at {freq} frequency")
        return df_result.reset_index()
    
    def align_to_trading_calendar(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        market: str = 'NYSE',
        trading_hours_only: bool = True
    ) -> pd.DataFrame:
        """
        Filter data to trading calendar
        
        Args:
            df: Input DataFrame
            timestamp_col: Timestamp column name
            market: Market name ('NYSE', 'NASDAQ', etc.)
            trading_hours_only: Keep only trading hours
            
        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()
        
        if timestamp_col not in df.columns:
            raise ValueError(f"Column {timestamp_col} not found")
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df_filtered[timestamp_col]):
            df_filtered[timestamp_col] = pd.to_datetime(df_filtered[timestamp_col])
        
        # Remove weekends
        df_filtered = df_filtered[df_filtered[timestamp_col].dt.dayofweek < 5]
        
        if trading_hours_only:
            # Filter to trading hours (9:30 AM - 4:00 PM ET for US markets)
            df_filtered = df_filtered[
                (df_filtered[timestamp_col].dt.time >= pd.Timestamp('09:30').time()) &
                (df_filtered[timestamp_col].dt.time <= pd.Timestamp('16:00').time())
            ]
        
        removed = len(df) - len(df_filtered)
        logger.info(f"Filtered to trading calendar, removed {removed} rows")
        return df_filtered
    
    def synchronize_tick_data(
        self,
        df: pd.DataFrame,
        window: str = '1s',
        agg_func: str = 'last'
    ) -> pd.DataFrame:
        """
        Synchronize tick data to regular intervals
        
        Args:
            df: Tick data DataFrame
            window: Synchronization window
            agg_func: Aggregation function
            
        Returns:
            Synchronized DataFrame
        """
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        df_sync = df.set_index('timestamp')
        
        if agg_func == 'last':
            df_result = df_sync.resample(window).last()
        elif agg_func == 'first':
            df_result = df_sync.resample(window).first()
        elif agg_func == 'mean':
            df_result = df_sync.resample(window).mean()
        elif agg_func == 'ohlc':
            # For price columns, use OHLC
            price_cols = [col for col in df_sync.columns if 'price' in col.lower()]
            if price_cols:
                df_result = df_sync[price_cols[0]].resample(window).ohlc()
                for col in df_sync.columns:
                    if col not in price_cols:
                        df_result[col] = df_sync[col].resample(window).last()
        else:
            df_result = df_sync.resample(window).agg(agg_func)
        
        logger.info(f"Synchronized tick data to {window} intervals")
        return df_result.reset_index()


if __name__ == "__main__":
    # Example usage
    aligner = DataAligner()
    
    # Create sample data
    dates1 = pd.date_range('2024-01-01', periods=100, freq='1min')
    df1 = pd.DataFrame({
        'timestamp': dates1,
        'price_a': 100 + np.random.randn(100).cumsum()
    })
    
    dates2 = pd.date_range('2024-01-01 00:00:30', periods=100, freq='1min')
    df2 = pd.DataFrame({
        'timestamp': dates2,
        'price_b': 200 + np.random.randn(100).cumsum()
    })
    
    # Align DataFrames
    aligned = aligner.align_by_time([df1, df2])
    
    logger.info("Data alignment example completed")
