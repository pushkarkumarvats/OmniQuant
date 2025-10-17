"""
Data Cleaning Module
Handles data quality checks, outlier detection, and data preprocessing
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from loguru import logger
from scipy import stats


class DataCleaner:
    """
    Data cleaning and preprocessing utilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataCleaner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates(subset=subset)
        removed = initial_rows - len(df_clean)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows ({removed/initial_rows*100:.2f}%)")
        
        return df_clean
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        method: str = 'forward_fill',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values using various strategies
        
        Args:
            df: Input DataFrame
            method: 'drop', 'forward_fill', 'backward_fill', 'interpolate', 'mean', 'median'
            columns: Specific columns to process (None for all)
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        cols = columns if columns else df.columns
        
        missing_count = df_clean[cols].isna().sum().sum()
        if missing_count == 0:
            logger.info("No missing values found")
            return df_clean
        
        logger.info(f"Handling {missing_count} missing values using method: {method}")
        
        if method == 'drop':
            df_clean = df_clean.dropna(subset=cols)
        elif method == 'forward_fill':
            df_clean[cols] = df_clean[cols].fillna(method='ffill')
        elif method == 'backward_fill':
            df_clean[cols] = df_clean[cols].fillna(method='bfill')
        elif method == 'interpolate':
            df_clean[cols] = df_clean[cols].interpolate(method='linear')
        elif method == 'mean':
            df_clean[cols] = df_clean[cols].fillna(df_clean[cols].mean())
        elif method == 'median':
            df_clean[cols] = df_clean[cols].fillna(df_clean[cols].median())
        else:
            raise ValueError(f"Unknown method: {method}")
        
        remaining_missing = df_clean[cols].isna().sum().sum()
        logger.info(f"Remaining missing values: {remaining_missing}")
        
        return df_clean
    
    def detect_outliers_zscore(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outliers using Z-score method
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            threshold: Z-score threshold
            
        Returns:
            Boolean DataFrame indicating outliers
        """
        logger.info(f"Detecting outliers using Z-score (threshold={threshold})")
        
        outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_mask = z_scores > threshold
                outliers.loc[df[col].notna(), col] = outlier_mask
                
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    logger.info(f"  {col}: {outlier_count} outliers detected")
        
        return outliers
    
    def detect_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: List[str],
        multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            multiplier: IQR multiplier (typically 1.5 or 3.0)
            
        Returns:
            Boolean DataFrame indicating outliers
        """
        logger.info(f"Detecting outliers using IQR method (multiplier={multiplier})")
        
        outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers[col] = outlier_mask
                
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    logger.info(f"  {col}: {outlier_count} outliers detected")
        
        return outliers
    
    def cap_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        **kwargs
    ) -> pd.DataFrame:
        """
        Cap outliers at threshold values (winsorization)
        
        Args:
            df: Input DataFrame
            columns: Columns to process
            method: 'iqr' or 'quantile'
            **kwargs: Additional arguments for outlier detection
            
        Returns:
            DataFrame with capped outliers
        """
        df_clean = df.copy()
        
        if method == 'iqr':
            multiplier = kwargs.get('multiplier', 1.5)
            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    
        elif method == 'quantile':
            lower_q = kwargs.get('lower_quantile', 0.01)
            upper_q = kwargs.get('upper_quantile', 0.99)
            
            for col in columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    lower_bound = df[col].quantile(lower_q)
                    upper_bound = df[col].quantile(upper_q)
                    
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Capped outliers using {method} method")
        return df_clean
    
    def validate_price_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate price data for common issues
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for negative prices
        price_cols = [col for col in df.columns if 'price' in col.lower() or col in ['open', 'high', 'low', 'close', 'bid', 'ask']]
        for col in price_cols:
            if col in df.columns:
                if (df[col] < 0).any():
                    issues.append(f"Negative prices found in {col}")
        
        # Check for zero volumes
        if 'volume' in df.columns:
            zero_volume_pct = (df['volume'] == 0).sum() / len(df) * 100
            if zero_volume_pct > 10:
                issues.append(f"High percentage of zero volume: {zero_volume_pct:.2f}%")
        
        # Check for reversed bid-ask
        if 'bid' in df.columns and 'ask' in df.columns:
            reversed_count = (df['bid'] > df['ask']).sum()
            if reversed_count > 0:
                issues.append(f"Bid > Ask in {reversed_count} rows")
        
        # Check for OHLC consistency
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()
            if invalid_ohlc > 0:
                issues.append(f"Invalid OHLC relationships in {invalid_ohlc} rows")
        
        # Check for large price jumps
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            extreme_moves = (np.abs(returns) > 0.2).sum()  # 20% moves
            if extreme_moves > 0:
                issues.append(f"Extreme price moves (>20%) in {extreme_moves} periods")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Price data validation passed")
        else:
            logger.warning(f"Price data validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def resample_data(
        self,
        df: pd.DataFrame,
        freq: str,
        agg_dict: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Resample time series data to different frequency
        
        Args:
            df: Input DataFrame with datetime index
            freq: Target frequency ('1min', '5min', '1h', etc.)
            agg_dict: Aggregation rules for each column
            
        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for resampling")
        
        if agg_dict is None:
            # Default aggregation rules
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'price': 'last',
            }
        
        # Filter to only columns that exist
        agg_dict_filtered = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        df_resampled = df.resample(freq).agg(agg_dict_filtered)
        
        logger.info(f"Resampled data from {len(df)} to {len(df_resampled)} rows at {freq} frequency")
        return df_resampled
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Normalize data using various methods
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: 'zscore', 'minmax', or 'robust'
            
        Returns:
            DataFrame with normalized columns
        """
        df_norm = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'zscore':
                df_norm[col] = (df[col] - df[col].mean()) / df[col].std()
            elif method == 'minmax':
                df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            elif method == 'robust':
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                df_norm[col] = (df[col] - median) / iqr
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Normalized {len(columns)} columns using {method} method")
        return df_norm


if __name__ == "__main__":
    # Example usage
    cleaner = DataCleaner()
    
    # Create sample data with issues
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'price': 100 + np.random.randn(1000).cumsum(),
        'volume': np.random.randint(100, 10000, 1000)
    })
    
    # Add some outliers and missing values
    df.loc[50, 'price'] = 200  # Outlier
    df.loc[100:105, 'volume'] = np.nan  # Missing values
    
    # Clean data
    df_clean = cleaner.handle_missing_values(df, method='forward_fill')
    outliers = cleaner.detect_outliers_zscore(df_clean, ['price'], threshold=3.0)
    
    logger.info("Data cleaning example completed")
