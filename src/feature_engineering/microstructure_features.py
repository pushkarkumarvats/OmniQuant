"""
Microstructure Features
Market microstructure features including order flow imbalance, spread, depth, etc.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from loguru import logger
import numba
from numba import jit


class MicrostructureFeatures:
    """Market microstructure features from tick and order book data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    def order_flow_imbalance(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """Rolling OFI: (bid_size - ask_size) / total, smoothed over window."""
        if 'bid_size' not in df.columns or 'ask_size' not in df.columns:
            raise ValueError("DataFrame must contain 'bid_size' and 'ask_size' columns")
        
        # OFI = (bid_size - ask_size) / (bid_size + ask_size)
        total_size = df['bid_size'] + df['ask_size']
        ofi = (df['bid_size'] - df['ask_size']) / total_size.replace(0, np.nan)
        
        # Rolling average
        ofi_rolling = ofi.rolling(window=window, min_periods=1).mean()
        
        logger.debug(f"Calculated OFI with window={window}")
        return ofi_rolling
    
    def volume_order_imbalance(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """Volume-weighted order imbalance incorporating both size and price changes."""
        # Track changes in bid and ask sizes at best prices
        bid_delta = df['bid_size'].diff()
        ask_delta = df['ask_size'].diff()
        
        # VOI considers both size changes and price changes
        price_mid = (df['bid_price'] + df['ask_price']) / 2
        price_delta = price_mid.diff()
        
        # Buy pressure when bid size increases or ask size decreases
        buy_pressure = np.where(price_delta >= 0, bid_delta, 0) - np.where(price_delta <= 0, ask_delta, 0)
        voi = pd.Series(buy_pressure, index=df.index).rolling(window=window).sum()
        
        logger.debug(f"Calculated VOI with window={window}")
        return voi
    
    def bid_ask_spread(self, df: pd.DataFrame, relative: bool = True) -> pd.Series:
        """Bid-ask spread, in basis points if relative=True."""
        if 'bid_price' not in df.columns or 'ask_price' not in df.columns:
            raise ValueError("DataFrame must contain 'bid_price' and 'ask_price'")
        
        spread = df['ask_price'] - df['bid_price']
        
        if relative:
            mid_price = (df['bid_price'] + df['ask_price']) / 2
            spread = (spread / mid_price) * 10000  # in basis points
        
        return spread
    
    def effective_spread(
        self,
        df: pd.DataFrame,
        relative: bool = True
    ) -> pd.Series:
        """Effective spread (actual transaction cost) as 2 * |price - mid|."""
        mid_price = (df['bid_price'] + df['ask_price']) / 2
        effective_spread = 2 * np.abs(df['price'] - mid_price)
        
        if relative:
            effective_spread = (effective_spread / mid_price) * 10000
        
        return effective_spread
    
    def orderbook_slope(
        self,
        df: pd.DataFrame,
        depth: int = 5
    ) -> Dict[str, pd.Series]:
        """Price impact slope across order book levels (price change per unit volume)."""
        bid_slopes = []
        ask_slopes = []
        
        for idx in df.index:
            # Extract bid side
            bid_prices = []
            bid_sizes = []
            for i in range(1, depth + 1):
                if f'bid_price_{i}' in df.columns and f'bid_size_{i}' in df.columns:
                    bid_prices.append(df.loc[idx, f'bid_price_{i}'])
                    bid_sizes.append(df.loc[idx, f'bid_size_{i}'])
            
            # Calculate cumulative size vs price
            if bid_prices and bid_sizes:
                cumsum_sizes = np.cumsum(bid_sizes)
                # Slope: price change per unit volume
                if len(bid_prices) > 1:
                    bid_slope = (bid_prices[0] - bid_prices[-1]) / cumsum_sizes[-1]
                else:
                    bid_slope = 0
                bid_slopes.append(bid_slope)
            else:
                bid_slopes.append(np.nan)
            
            # Extract ask side
            ask_prices = []
            ask_sizes = []
            for i in range(1, depth + 1):
                if f'ask_price_{i}' in df.columns and f'ask_size_{i}' in df.columns:
                    ask_prices.append(df.loc[idx, f'ask_price_{i}'])
                    ask_sizes.append(df.loc[idx, f'ask_size_{i}'])
            
            if ask_prices and ask_sizes:
                cumsum_sizes = np.cumsum(ask_sizes)
                if len(ask_prices) > 1:
                    ask_slope = (ask_prices[-1] - ask_prices[0]) / cumsum_sizes[-1]
                else:
                    ask_slope = 0
                ask_slopes.append(ask_slope)
            else:
                ask_slopes.append(np.nan)
        
        return {
            'bid_slope': pd.Series(bid_slopes, index=df.index),
            'ask_slope': pd.Series(ask_slopes, index=df.index)
        }
    
    def orderbook_depth(
        self,
        df: pd.DataFrame,
        depth: int = 5,
        distance_bps: Optional[float] = None
    ) -> Dict[str, pd.Series]:
        """Aggregate size across book levels, optionally filtered by distance in bps."""
        bid_depths = []
        ask_depths = []
        
        for idx in df.index:
            bid_depth = 0
            ask_depth = 0
            
            mid_price = df.loc[idx, 'mid_price'] if 'mid_price' in df.columns else \
                       (df.loc[idx, 'bid_price_1'] + df.loc[idx, 'ask_price_1']) / 2
            
            for i in range(1, depth + 1):
                # Bid side
                if f'bid_size_{i}' in df.columns:
                    if distance_bps is None:
                        bid_depth += df.loc[idx, f'bid_size_{i}']
                    else:
                        bid_price = df.loc[idx, f'bid_price_{i}']
                        price_distance = abs(bid_price - mid_price) / mid_price * 10000
                        if price_distance <= distance_bps:
                            bid_depth += df.loc[idx, f'bid_size_{i}']
                
                # Ask side
                if f'ask_size_{i}' in df.columns:
                    if distance_bps is None:
                        ask_depth += df.loc[idx, f'ask_size_{i}']
                    else:
                        ask_price = df.loc[idx, f'ask_price_{i}']
                        price_distance = abs(ask_price - mid_price) / mid_price * 10000
                        if price_distance <= distance_bps:
                            ask_depth += df.loc[idx, f'ask_size_{i}']
            
            bid_depths.append(bid_depth)
            ask_depths.append(ask_depth)
        
        return {
            'bid_depth': pd.Series(bid_depths, index=df.index),
            'ask_depth': pd.Series(ask_depths, index=df.index),
            'total_depth': pd.Series(np.array(bid_depths) + np.array(ask_depths), index=df.index)
        }
    
    def trade_intensity(
        self,
        df: pd.DataFrame,
        window: int = 60
    ) -> pd.Series:
        """Trade count in a rolling time window (seconds)."""
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Count trades in rolling window
        df_indexed = df.set_index('timestamp')
        trade_counts = df_indexed.rolling(f'{window}s').size()
        
        return trade_counts
    
    def price_impact(
        self,
        df: pd.DataFrame,
        window: int = 10
    ) -> pd.Series:
        """Estimated price impact: price change per sqrt(volume), rolling averaged."""
        if 'price' not in df.columns or 'volume' not in df.columns:
            raise ValueError("DataFrame must have 'price' and 'volume' columns")
        
        # Price change per unit volume
        price_change = df['price'].diff()
        
        # Normalize by volume
        impact = price_change / np.sqrt(df['volume'].replace(0, np.nan))
        
        # Rolling average
        impact_rolling = impact.rolling(window=window).mean()
        
        return impact_rolling
    
    def volatility_clustering(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """Realized volatility (annualized) over a rolling window."""
        if 'price' not in df.columns:
            raise ValueError("DataFrame must have 'price' column")
        
        # Calculate returns
        returns = df['price'].pct_change()
        
        # Realized volatility
        realized_vol = returns.rolling(window=window).std() * np.sqrt(252 * 6.5 * 3600)
        
        return realized_vol
    
    def generate_all_features(
        self,
        df: pd.DataFrame,
        feature_config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Run all microstructure feature generators and merge results."""
        logger.info("Generating all microstructure features")
        
        df_features = df.copy()
        config = feature_config or {}
        
        # Order flow imbalance
        if 'bid_size' in df.columns and 'ask_size' in df.columns:
            for window in config.get('ofi_windows', [10, 20, 50]):
                df_features[f'ofi_{window}'] = self.order_flow_imbalance(df, window=window)
                df_features[f'voi_{window}'] = self.volume_order_imbalance(df, window=window)
        
        # Spread features
        if 'bid_price' in df.columns and 'ask_price' in df.columns:
            df_features['spread_bps'] = self.bid_ask_spread(df, relative=True)
            df_features['spread_abs'] = self.bid_ask_spread(df, relative=False)
            
            if 'price' in df.columns:
                df_features['effective_spread'] = self.effective_spread(df)
        
        # Order book features
        if any(f'bid_price_1' in df.columns for _ in range(1)):
            depth_features = self.orderbook_depth(df, depth=5)
            for key, series in depth_features.items():
                df_features[key] = series
            
            slope_features = self.orderbook_slope(df, depth=5)
            for key, series in slope_features.items():
                df_features[key] = series
        
        # Trade intensity
        if 'timestamp' in df.columns:
            for window in config.get('intensity_windows', [30, 60, 300]):
                df_features[f'trade_intensity_{window}'] = self.trade_intensity(df, window=window)
        
        # Price impact and volatility
        if 'price' in df.columns and 'volume' in df.columns:
            df_features['price_impact'] = self.price_impact(df)
            df_features['volatility'] = self.volatility_clustering(df)
        
        logger.info(f"Generated {len(df_features.columns) - len(df.columns)} microstructure features")
        return df_features


# Numba-accelerated functions for performance
@jit(nopython=True)
def calculate_ofi_fast(bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> np.ndarray:
    """Numba-accelerated OFI calculation."""
    total_sizes = bid_sizes + ask_sizes
    ofi = np.zeros_like(bid_sizes, dtype=np.float64)
    
    for i in range(len(bid_sizes)):
        if total_sizes[i] > 0:
            ofi[i] = (bid_sizes[i] - ask_sizes[i]) / total_sizes[i]
        else:
            ofi[i] = 0.0
    
    return ofi


if __name__ == "__main__":
    # Example usage
    from src.data_pipeline.ingestion import DataIngestion
    
    # Generate synthetic data
    ingestion = DataIngestion()
    df = ingestion.generate_synthetic_orderbook(num_snapshots=1000, seed=42)
    
    # Generate features
    feature_gen = MicrostructureFeatures()
    
    # Individual features
    ofi = feature_gen.order_flow_imbalance(df)
    spread = feature_gen.bid_ask_spread(df)
    
    # All features
    df_with_features = feature_gen.generate_all_features(df)
    
    logger.info(f"Generated features shape: {df_with_features.shape}")
    logger.info("Microstructure features example completed")
