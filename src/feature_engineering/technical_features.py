"""
Technical Features
Traditional technical analysis features for alpha discovery
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from loguru import logger
from scipy import stats


class TechnicalFeatures:
    """
    Generate technical analysis features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TechnicalFeatures
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
    def returns(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        periods: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        Calculate returns over multiple periods
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            periods: List of periods for returns
            
        Returns:
            DataFrame with return columns
        """
        df_returns = pd.DataFrame(index=df.index)
        
        for period in periods:
            df_returns[f'return_{period}'] = df[price_col].pct_change(periods=period)
            df_returns[f'log_return_{period}'] = np.log(df[price_col] / df[price_col].shift(period))
        
        logger.debug(f"Calculated returns for periods: {periods}")
        return df_returns
    
    def momentum(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        windows: List[int] = [10, 20, 50, 100]
    ) -> pd.DataFrame:
        """
        Calculate momentum indicators
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            windows: List of lookback windows
            
        Returns:
            DataFrame with momentum features
        """
        df_momentum = pd.DataFrame(index=df.index)
        
        for window in windows:
            # Simple momentum
            df_momentum[f'momentum_{window}'] = df[price_col].pct_change(periods=window)
            
            # Rate of change
            df_momentum[f'roc_{window}'] = (
                (df[price_col] - df[price_col].shift(window)) / df[price_col].shift(window) * 100
            )
        
        logger.debug(f"Calculated momentum for windows: {windows}")
        return df_momentum
    
    def moving_averages(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        windows: List[int] = [5, 10, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Calculate moving averages and crossovers
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            windows: List of MA windows
            
        Returns:
            DataFrame with MA features
        """
        df_ma = pd.DataFrame(index=df.index)
        
        for window in windows:
            # Simple Moving Average
            df_ma[f'sma_{window}'] = df[price_col].rolling(window=window).mean()
            
            # Exponential Moving Average
            df_ma[f'ema_{window}'] = df[price_col].ewm(span=window, adjust=False).mean()
            
            # Distance from MA (normalized)
            df_ma[f'dist_sma_{window}'] = (df[price_col] - df_ma[f'sma_{window}']) / df_ma[f'sma_{window}']
        
        # MA crossovers
        if 50 in windows and 200 in windows:
            df_ma['golden_cross'] = (df_ma['sma_50'] > df_ma['sma_200']).astype(int)
        
        logger.debug(f"Calculated moving averages for windows: {windows}")
        return df_ma
    
    def volatility(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        windows: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """
        Calculate volatility measures
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            windows: List of volatility windows
            
        Returns:
            DataFrame with volatility features
        """
        df_vol = pd.DataFrame(index=df.index)
        
        # Calculate returns for volatility
        returns = df[price_col].pct_change()
        
        for window in windows:
            # Historical volatility (annualized)
            df_vol[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
            
            # Parkinson volatility (using high-low)
            if 'high' in df.columns and 'low' in df.columns:
                hl_ratio = np.log(df['high'] / df['low'])
                df_vol[f'parkinson_vol_{window}'] = (
                    hl_ratio.rolling(window=window).apply(lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))))
                )
            
            # Rolling standard deviation of returns
            df_vol[f'return_std_{window}'] = returns.rolling(window=window).std()
        
        logger.debug(f"Calculated volatility for windows: {windows}")
        return df_vol
    
    def bollinger_bands(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        window: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            window: Rolling window
            num_std: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Band features
        """
        df_bb = pd.DataFrame(index=df.index)
        
        # Middle band (SMA)
        sma = df[price_col].rolling(window=window).mean()
        std = df[price_col].rolling(window=window).std()
        
        # Upper and lower bands
        df_bb[f'bb_upper_{window}'] = sma + (std * num_std)
        df_bb[f'bb_lower_{window}'] = sma - (std * num_std)
        df_bb[f'bb_middle_{window}'] = sma
        
        # Bandwidth
        df_bb[f'bb_width_{window}'] = (df_bb[f'bb_upper_{window}'] - df_bb[f'bb_lower_{window}']) / sma
        
        # %B (position within bands)
        df_bb[f'bb_percent_{window}'] = (df[price_col] - df_bb[f'bb_lower_{window}']) / \
                                        (df_bb[f'bb_upper_{window}'] - df_bb[f'bb_lower_{window}'])
        
        return df_bb
    
    def rsi(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            window: RSI period
            
        Returns:
            Series with RSI values
        """
        # Calculate price changes
        delta = df[price_col].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with MACD features
        """
        df_macd = pd.DataFrame(index=df.index)
        
        # Calculate EMAs
        ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
        
        # MACD line
        df_macd['macd'] = ema_fast - ema_slow
        
        # Signal line
        df_macd['macd_signal'] = df_macd['macd'].ewm(span=signal, adjust=False).mean()
        
        # Histogram
        df_macd['macd_hist'] = df_macd['macd'] - df_macd['macd_signal']
        
        return df_macd
    
    def atr(
        self,
        df: pd.DataFrame,
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: DataFrame with high, low, close
            window: ATR period
            
        Returns:
            Series with ATR values
        """
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            raise ValueError("DataFrame must have 'high', 'low', 'close' columns")
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def vwap(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        volume_col: str = 'volume',
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP)
        
        Args:
            df: DataFrame with price and volume
            price_col: Price column
            volume_col: Volume column
            window: Rolling window (None for cumulative)
            
        Returns:
            Series with VWAP values
        """
        typical_price = df[price_col]
        if 'high' in df.columns and 'low' in df.columns:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        pv = typical_price * df[volume_col]
        
        if window is None:
            # Cumulative VWAP
            vwap = pv.cumsum() / df[volume_col].cumsum()
        else:
            # Rolling VWAP
            vwap = pv.rolling(window=window).sum() / df[volume_col].rolling(window=window).sum()
        
        return vwap
    
    def vwap_deviation(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        volume_col: str = 'volume',
        window: int = 20
    ) -> pd.Series:
        """
        Calculate deviation from VWAP
        
        Args:
            df: DataFrame with price and volume
            price_col: Price column
            volume_col: Volume column
            window: VWAP window
            
        Returns:
            Series with VWAP deviation
        """
        vwap_values = self.vwap(df, price_col, volume_col, window)
        deviation = (df[price_col] - vwap_values) / vwap_values
        
        return deviation
    
    def volume_profile(
        self,
        df: pd.DataFrame,
        volume_col: str = 'volume',
        windows: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """
        Calculate volume profile features
        
        Args:
            df: DataFrame with volume data
            volume_col: Volume column
            windows: List of windows
            
        Returns:
            DataFrame with volume features
        """
        df_vol_profile = pd.DataFrame(index=df.index)
        
        for window in windows:
            # Average volume
            df_vol_profile[f'avg_volume_{window}'] = df[volume_col].rolling(window=window).mean()
            
            # Volume ratio (current / average)
            df_vol_profile[f'volume_ratio_{window}'] = df[volume_col] / df_vol_profile[f'avg_volume_{window}']
            
            # Volume standard deviation
            df_vol_profile[f'volume_std_{window}'] = df[volume_col].rolling(window=window).std()
        
        return df_vol_profile
    
    def generate_all_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        feature_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate all technical features
        
        Args:
            df: Input DataFrame
            price_col: Price column name
            feature_config: Feature configuration
            
        Returns:
            DataFrame with all technical features
        """
        logger.info("Generating all technical features")
        
        df_features = df.copy()
        config = feature_config or {}
        
        # Returns
        returns_df = self.returns(df, price_col, periods=config.get('return_periods', [1, 5, 10, 20]))
        df_features = pd.concat([df_features, returns_df], axis=1)
        
        # Momentum
        momentum_df = self.momentum(df, price_col, windows=config.get('momentum_windows', [10, 20, 50]))
        df_features = pd.concat([df_features, momentum_df], axis=1)
        
        # Moving averages
        ma_df = self.moving_averages(df, price_col, windows=config.get('ma_windows', [5, 10, 20, 50]))
        df_features = pd.concat([df_features, ma_df], axis=1)
        
        # Volatility
        vol_df = self.volatility(df, price_col, windows=config.get('vol_windows', [10, 20, 50]))
        df_features = pd.concat([df_features, vol_df], axis=1)
        
        # Bollinger Bands
        bb_df = self.bollinger_bands(df, price_col, window=20)
        df_features = pd.concat([df_features, bb_df], axis=1)
        
        # RSI
        df_features['rsi_14'] = self.rsi(df, price_col, window=14)
        
        # MACD
        macd_df = self.macd(df, price_col)
        df_features = pd.concat([df_features, macd_df], axis=1)
        
        # ATR
        if all(col in df.columns for col in ['high', 'low']):
            df_features['atr_14'] = self.atr(df, window=14)
        
        # VWAP
        if 'volume' in df.columns:
            df_features['vwap_20'] = self.vwap(df, price_col, 'volume', window=20)
            df_features['vwap_deviation'] = self.vwap_deviation(df, price_col, 'volume', window=20)
            
            vol_profile_df = self.volume_profile(df, 'volume')
            df_features = pd.concat([df_features, vol_profile_df], axis=1)
        
        logger.info(f"Generated {len(df_features.columns) - len(df.columns)} technical features")
        return df_features


if __name__ == "__main__":
    # Example usage
    from src.data_pipeline.ingestion import DataIngestion
    
    # Generate synthetic data
    ingestion = DataIngestion()
    tick_data = ingestion.generate_synthetic_tick_data(num_ticks=5000, seed=42)
    
    # Convert to OHLCV format
    df = tick_data.set_index('timestamp').resample('1min').agg({
        'price': 'ohlc',
        'volume': 'sum'
    })
    df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    df = df.rename(columns={'price_open': 'open', 'price_high': 'high', 
                             'price_low': 'low', 'price_close': 'close'})
    df = df.reset_index()
    
    # Generate features
    feature_gen = TechnicalFeatures()
    df_with_features = feature_gen.generate_all_features(df, price_col='close')
    
    logger.info(f"Generated features shape: {df_with_features.shape}")
    logger.info("Technical features example completed")
