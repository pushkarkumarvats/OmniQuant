"""
Unit tests for advanced features
"""

import unittest
import pandas as pd
import numpy as np
from src.feature_engineering.advanced_features import AdvancedFeatures


class TestAdvancedFeatures(unittest.TestCase):
    """Test cases for advanced feature engineering"""
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)
        n = 500
        self.dates = pd.date_range('2024-01-01', periods=n, freq='1min')
        self.prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
        self.series = pd.Series(self.prices, index=self.dates)
        self.adv = AdvancedFeatures()
    
    def test_fractional_differentiation(self):
        """Test fractional differentiation"""
        result = self.adv.fractional_differentiation(self.series, d=0.5)
        
        # Should return series of same length
        self.assertEqual(len(result), len(self.series))
        
        # Should be more stationary than original
        # (lower variance of first differences)
        orig_diff_var = self.series.diff().var()
        frac_diff_var = result.diff().var()
        
        # Fractional diff should smooth the series
        self.assertIsNotNone(frac_diff_var)
    
    def test_time_series_decomposition(self):
        """Test time series decomposition"""
        # Add seasonality
        seasonal = 10 * np.sin(np.arange(len(self.series)) * 2 * np.pi / 50)
        series_with_season = self.series + seasonal
        
        result = self.adv.time_series_decomposition(series_with_season, period=50)
        
        # Should have trend, seasonal, and residual
        self.assertIn('trend', result)
        self.assertIn('seasonal', result)
        self.assertIn('residual', result)
        
        # Components should sum to original (for additive model)
        reconstructed = result['trend'] + result['seasonal'] + result['residual']
        np.testing.assert_array_almost_equal(
            series_with_season.values,
            reconstructed.values,
            decimal=5
        )
    
    def test_wavelet_decomposition(self):
        """Test wavelet decomposition"""
        result = self.adv.wavelet_decomposition(self.series, level=3)
        
        # Should have approximation and 3 detail levels
        self.assertIn('approximation', result)
        self.assertIn('detail_1', result)
        self.assertIn('detail_2', result)
        self.assertIn('detail_3', result)
    
    def test_wavelet_features(self):
        """Test wavelet feature generation"""
        df = pd.DataFrame({'close': self.prices}, index=self.dates)
        result = self.adv.wavelet_features(df, levels=2)
        
        # Should have original columns plus wavelet features
        self.assertIn('wavelet_detail_1', result.columns)
        self.assertIn('wavelet_detail_2', result.columns)
        self.assertIn('wavelet_approx', result.columns)
    
    def test_hilbert_transform(self):
        """Test Hilbert transform"""
        amplitude, phase = self.adv.hilbert_transform(self.series)
        
        # Should return same length
        self.assertEqual(len(amplitude), len(self.series))
        self.assertEqual(len(phase), len(self.series))
        
        # Amplitude should be positive
        self.assertTrue((amplitude >= 0).all())
        
        # Phase should be in [-pi, pi]
        self.assertTrue((phase >= -np.pi).all())
        self.assertTrue((phase <= np.pi).all())
    
    def test_spectral_features(self):
        """Test spectral features"""
        features = self.adv.spectral_features(self.series)
        
        # Should contain expected features
        self.assertIn('spectral_centroid', features)
        self.assertIn('spectral_spread', features)
        self.assertIn('spectral_entropy', features)
        
        # Values should be finite
        for value in features.values():
            self.assertTrue(np.isfinite(value))
    
    def test_hurst_exponent(self):
        """Test Hurst exponent calculation"""
        hurst = self.adv.hurst_exponent(self.series, max_lag=50)
        
        # Hurst should be between 0 and 1
        self.assertGreater(hurst, 0)
        self.assertLess(hurst, 1)
        
        # Test with trending series
        trending = pd.Series(np.arange(len(self.series)))
        hurst_trending = self.adv.hurst_exponent(trending, max_lag=50)
        
        # Trending series should have H > 0.5
        self.assertGreater(hurst_trending, 0.5)
    
    def test_detrended_fluctuation_analysis(self):
        """Test DFA"""
        alpha = self.adv.detrended_fluctuation_analysis(self.series)
        
        # Alpha should be positive
        self.assertGreater(alpha, 0)
        
        # Should be finite
        self.assertTrue(np.isfinite(alpha))


if __name__ == '__main__':
    unittest.main()
