"""
Advanced Feature Engineering
Fractional differentiation, wavelets, time series decomposition
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from loguru import logger
from scipy import signal
import pywt


class AdvancedFeatures:
    """Advanced feature engineering techniques"""
    
    def __init__(self):
        self.config = {}
    
    def fractional_differentiation(
        self,
        series: pd.Series,
        d: float = 0.5,
        threshold: float = 1e-5
    ) -> pd.Series:
        """Apply fractional differentiation to preserve memory while achieving stationarity."""
        # Calculate weights
        weights = [1.0]
        for k in range(1, len(series)):
            weight = -weights[-1] * (d - k + 1) / k
            if abs(weight) < threshold:
                break
            weights.append(weight)
        
        weights = np.array(weights[::-1])
        
        # Apply convolution
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(weights), len(series)):
            result.iloc[i] = np.dot(weights, series.iloc[i-len(weights)+1:i+1])
        
        logger.debug(f"Applied fractional differentiation with d={d}")
        return result
    
    def time_series_decomposition(
        self,
        series: pd.Series,
        period: Optional[int] = None,
        model: str = 'additive'
    ) -> Dict[str, pd.Series]:
        """Decompose into trend, seasonal, and residual. Estimates period via FFT if not given."""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        if period is None:
            # Estimate period using FFT
            fft = np.fft.fft(series.values)
            freq = np.fft.fftfreq(len(series))
            power = np.abs(fft) ** 2
            period = int(1 / abs(freq[np.argmax(power[1:]) + 1]))
        
        decomposition = seasonal_decompose(
            series,
            model=model,
            period=period,
            extrapolate_trend='freq'
        )
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
    
    def wavelet_decomposition(
        self,
        series: pd.Series,
        wavelet: str = 'db4',
        level: int = 3
    ) -> Dict[str, np.ndarray]:
        """Multi-level wavelet decomposition returning approximation and detail coefficients."""
        coeffs = pywt.wavedec(series.values, wavelet, level=level)
        
        result = {'approximation': coeffs[0]}
        for i, detail in enumerate(coeffs[1:], 1):
            result[f'detail_{i}'] = detail
        
        logger.debug(f"Wavelet decomposition completed with {level} levels")
        return result
    
    def wavelet_features(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        levels: int = 3
    ) -> pd.DataFrame:
        """Reconstruct wavelet detail and approximation signals as DataFrame columns."""
        result = df.copy()
        
        # Perform wavelet decomposition
        decomp = self.wavelet_decomposition(df[price_col], level=levels)
        
        # Reconstruct signals at different levels
        for i in range(1, levels + 1):
            # High-frequency component
            detail_signal = pywt.upcoef('d', decomp[f'detail_{i}'], 'db4', level=i, take=len(df))
            result[f'wavelet_detail_{i}'] = detail_signal[:len(df)]
        
        # Low-frequency component
        approx_signal = pywt.upcoef('a', decomp['approximation'], 'db4', level=levels, take=len(df))
        result['wavelet_approx'] = approx_signal[:len(df)]
        
        return result
    
    def hilbert_transform(
        self,
        series: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Returns (amplitude, phase) via Hilbert transform."""
        analytic_signal = signal.hilbert(series.values)
        amplitude = np.abs(analytic_signal)
        phase = np.angle(analytic_signal)
        
        return (
            pd.Series(amplitude, index=series.index),
            pd.Series(phase, index=series.index)
        )
    
    def empirical_mode_decomposition(
        self,
        series: pd.Series,
        max_imf: int = 5
    ) -> List[pd.Series]:
        """Extract intrinsic mode functions via EMD. Requires PyEMD."""
        try:
            from PyEMD import EMD
            
            emd = EMD()
            imfs = emd.emd(series.values, max_imf=max_imf)
            
            result = []
            for i, imf in enumerate(imfs):
                result.append(pd.Series(imf, index=series.index, name=f'IMF_{i+1}'))
            
            logger.debug(f"EMD completed with {len(result)} IMFs")
            return result
        
        except ImportError:
            logger.error("PyEMD not installed. Install with: pip install EMD-signal")
            return []
    
    def spectral_features(
        self,
        series: pd.Series,
        sample_rate: float = 1.0
    ) -> Dict[str, float]:
        """FFT-based spectral features: centroid, spread, skewness, kurtosis, entropy."""
        # Compute FFT
        fft_vals = np.fft.fft(series.values)
        fft_freq = np.fft.fftfreq(len(series), 1/sample_rate)
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Positive frequencies only
        pos_mask = fft_freq > 0
        fft_freq = fft_freq[pos_mask]
        power_spectrum = power_spectrum[pos_mask]
        
        # Calculate features
        total_power = np.sum(power_spectrum)
        
        return {
            'spectral_centroid': np.sum(fft_freq * power_spectrum) / total_power,
            'spectral_spread': np.sqrt(np.sum(((fft_freq - np.mean(fft_freq)) ** 2) * power_spectrum) / total_power),
            'spectral_skewness': np.sum(((fft_freq - np.mean(fft_freq)) ** 3) * power_spectrum) / (total_power * np.std(fft_freq) ** 3),
            'spectral_kurtosis': np.sum(((fft_freq - np.mean(fft_freq)) ** 4) * power_spectrum) / (total_power * np.std(fft_freq) ** 4),
            'spectral_entropy': -np.sum((power_spectrum / total_power) * np.log(power_spectrum / total_power + 1e-10))
        }
    
    def hurst_exponent(
        self,
        series: pd.Series,
        max_lag: int = 100
    ) -> float:
        """Hurst exponent via R/S analysis. H < 0.5 mean-reverting, H > 0.5 trending."""
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        
        # Linear fit in log-log space
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0]
        
        return hurst
    
    def detrended_fluctuation_analysis(
        self,
        series: pd.Series,
        scales: Optional[List[int]] = None
    ) -> float:
        """DFA exponent for long-range correlation detection."""
        if scales is None:
            scales = [4, 8, 16, 32, 64, 128]
        
        # Cumulative sum
        y = np.cumsum(series - np.mean(series))
        
        fluctuations = []
        for scale in scales:
            # Divide into segments
            n_segments = len(y) // scale
            
            segment_fluct = []
            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]
                # Fit polynomial trend
                t = np.arange(len(segment))
                coeffs = np.polyfit(t, segment, 1)
                trend = np.polyval(coeffs, t)
                # Calculate fluctuation
                fluct = np.sqrt(np.mean((segment - trend) ** 2))
                segment_fluct.append(fluct)
            
            fluctuations.append(np.mean(segment_fluct))
        
        # Log-log fit
        coeffs = np.polyfit(np.log(scales), np.log(fluctuations), 1)
        alpha = coeffs[0]
        
        return alpha


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.1)
    series = pd.Series(prices, index=dates)
    
    adv = AdvancedFeatures()
    
    # Fractional differentiation
    frac_diff = adv.fractional_differentiation(series, d=0.5)
    print(f"Fractional diff: {frac_diff.head()}")
    
    # Wavelet decomposition
    df = pd.DataFrame({'close': prices}, index=dates)
    wavelet_df = adv.wavelet_features(df)
    print(f"Wavelet features: {wavelet_df.columns.tolist()}")
    
    # Hurst exponent
    hurst = adv.hurst_exponent(series)
    print(f"Hurst exponent: {hurst:.4f}")
