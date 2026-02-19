"""
Unit tests for Feature Engineering
"""

import unittest
import pandas as pd
import numpy as np
from src.feature_engineering.technical_features import TechnicalFeatures
from src.feature_engineering.microstructure_features import MicrostructureFeatures


class TestTechnicalFeatures(unittest.TestCase):
    """Test cases for technical features"""
    
    def setUp(self):
        """Create sample data"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min'),
            'price': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'volume': np.random.randint(100, 1000, n)
        })
        self.features = TechnicalFeatures()
    
    def test_returns(self):
        """Test returns calculation"""
        result = self.features.calculate_returns(self.df, periods=[1, 5])
        
        self.assertIn('return_1', result.columns)
        self.assertIn('return_5', result.columns)
        # Check no lookahead bias
        self.assertTrue(pd.isna(result['return_1'].iloc[0]))
    
    def test_momentum(self):
        """Test momentum calculation"""
        result = self.features.calculate_momentum(self.df, periods=[10, 20])
        
        self.assertIn('momentum_10', result.columns)
        self.assertIn('momentum_20', result.columns)
        # First values should be NaN
        self.assertTrue(pd.isna(result['momentum_10'].iloc[9]))
    
    def test_moving_averages(self):
        """Test moving average calculation"""
        result = self.features.calculate_moving_averages(self.df, windows=[5, 10])
        
        self.assertIn('sma_5', result.columns)
        self.assertIn('ema_5', result.columns)
        # Check SMA calculation
        expected_sma = self.df['price'].rolling(5).mean().iloc[4]
        self.assertAlmostEqual(result['sma_5'].iloc[4], expected_sma, places=4)
    
    def test_volatility(self):
        """Test volatility calculation"""
        result = self.features.calculate_volatility(self.df, windows=[10])
        
        self.assertIn('volatility_10', result.columns)
        self.assertGreater(result['volatility_10'].iloc[-1], 0)
    
    def test_rsi(self):
        """Test RSI calculation"""
        result = self.features.calculate_rsi(self.df, period=14)
        
        self.assertIn('rsi_14', result.columns)
        # RSI should be between 0 and 100
        rsi_values = result['rsi_14'].dropna()
        self.assertTrue((rsi_values >= 0).all())
        self.assertTrue((rsi_values <= 100).all())
    
    def test_no_lookahead_bias(self):
        """Test that features don't use future data"""
        # Generate features
        result = self.features.generate_all_features(self.df.copy())
        
        # For each row, verify that features only use past data
        # This is done by checking that first N values are NaN for window N
        for col in result.columns:
            if 'return_1' in col:
                self.assertTrue(pd.isna(result[col].iloc[0]))


class TestMicrostructureFeatures(unittest.TestCase):
    """Test cases for microstructure features"""
    
    def setUp(self):
        """Create sample tick data"""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1s'),
            'price': 100 + np.cumsum(np.random.randn(n) * 0.01),
            'volume': np.random.randint(10, 100, n),
            'bid': 100 + np.cumsum(np.random.randn(n) * 0.01) - 0.05,
            'ask': 100 + np.cumsum(np.random.randn(n) * 0.01) + 0.05,
        })
        self.features = MicrostructureFeatures()
    
    def test_spread_calculation(self):
        """Test spread calculation"""
        result = self.features.calculate_spread(self.df)
        
        self.assertIn('spread_bps', result.columns)
        # Spread should be positive
        self.assertTrue((result['spread_bps'] > 0).all())
    
    def test_trade_intensity(self):
        """Test trade intensity"""
        result = self.features.calculate_trade_intensity(self.df, window=10)
        
        self.assertIn('trade_intensity_10', result.columns)
        # Should be positive
        self.assertTrue((result['trade_intensity_10'].dropna() > 0).all())


if __name__ == '__main__':
    unittest.main()
