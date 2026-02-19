"""
Statistical Alpha Model
Classical statistical models: ARIMA-GARCH, Kalman Filter, Cointegration
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from loguru import logger
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.stattools import coint, adfuller
from pykalman import KalmanFilter
import warnings
warnings.filterwarnings('ignore')


class StatisticalAlphaModel:
    """Classical statistical models: ARIMA-GARCH, Kalman filter, cointegration, regime switching."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models = {}
        
    def fit_arima_garch(
        self,
        returns: np.ndarray,
        arima_order: Tuple[int, int, int] = (1, 0, 1),
        garch_p: int = 1,
        garch_q: int = 1
    ) -> Dict[str, Any]:
        """Fit joint ARIMA (mean) + GARCH (volatility) model on a return series."""
        logger.info(f"Fitting ARIMA{arima_order}-GARCH({garch_p},{garch_q}) model")
        
        # Fit ARIMA for mean
        arima_model = ARIMA(returns, order=arima_order)
        arima_fit = arima_model.fit()
        
        # Get residuals
        residuals = arima_fit.resid
        
        # Fit GARCH for volatility
        garch_model = arch_model(
            residuals,
            vol='Garch',
            p=garch_p,
            q=garch_q
        )
        garch_fit = garch_model.fit(disp='off')
        
        # Store models
        self.models['arima'] = arima_fit
        self.models['garch'] = garch_fit
        
        logger.info("ARIMA-GARCH model fitted successfully")
        
        return {
            'arima_model': arima_fit,
            'garch_model': garch_fit,
            'arima_aic': arima_fit.aic,
            'arima_bic': arima_fit.bic,
            'garch_aic': garch_fit.aic,
            'garch_bic': garch_fit.bic
        }
    
    def predict_arima_garch(
        self,
        steps: int = 1
    ) -> Dict[str, np.ndarray]:
        """Multi-step ahead forecast of mean return and volatility."""
        if 'arima' not in self.models or 'garch' not in self.models:
            raise ValueError("ARIMA-GARCH models not fitted. Call fit_arima_garch() first.")
        
        # Forecast mean
        mean_forecast = self.models['arima'].forecast(steps=steps)
        
        # Forecast volatility
        vol_forecast = self.models['garch'].forecast(horizon=steps)
        
        return {
            'mean': np.array(mean_forecast),
            'volatility': np.sqrt(vol_forecast.variance.values[-1])
        }
    
    def fit_kalman_filter(
        self,
        observations: np.ndarray,
        n_states: int = 2
    ) -> KalmanFilter:
        """Fit a Kalman filter via EM on the observation series."""
        logger.info(f"Fitting Kalman Filter with {n_states} states")
        
        # Initialize Kalman Filter
        kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=n_states,
            initial_state_mean=np.zeros(n_states),
            initial_state_covariance=np.eye(n_states),
            transition_matrices=np.eye(n_states),
            observation_matrices=np.ones((1, n_states)) / n_states
        )
        
        # Fit using EM algorithm
        kf = kf.em(observations.reshape(-1, 1), n_iter=10)
        
        self.models['kalman'] = kf
        
        logger.info("Kalman Filter fitted successfully")
        return kf
    
    def kalman_smooth(
        self,
        observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if 'kalman' not in self.models:
            raise ValueError("Kalman Filter not fitted. Call fit_kalman_filter() first.")
        
        kf = self.models['kalman']
        
        # Smooth
        smoothed_state_means, smoothed_state_covariances = kf.smooth(observations.reshape(-1, 1))
        
        return smoothed_state_means, smoothed_state_covariances
    
    def test_cointegration(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        significance: float = 0.05
    ) -> Dict[str, Any]:
        """Engle-Granger cointegration test between two series."""
        logger.info("Testing for cointegration")
        
        # Cointegration test
        score, pvalue, _ = coint(series1, series2)
        
        is_cointegrated = pvalue < significance
        
        result = {
            'test_statistic': score,
            'p_value': pvalue,
            'is_cointegrated': is_cointegrated,
            'significance': significance
        }
        
        if is_cointegrated:
            logger.info(f"Series are cointegrated (p-value: {pvalue:.4f})")
        else:
            logger.info(f"Series are NOT cointegrated (p-value: {pvalue:.4f})")
        
        return result
    
    def find_cointegrated_pairs(
        self,
        price_data: pd.DataFrame,
        significance: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Brute-force pairwise cointegration scan across all columns."""
        logger.info(f"Searching for cointegrated pairs in {len(price_data.columns)} series")
        
        columns = price_data.columns
        pairs = []
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                series1 = price_data[columns[i]].dropna()
                series2 = price_data[columns[j]].dropna()
                
                # Align series
                common_idx = series1.index.intersection(series2.index)
                if len(common_idx) < 30:  # Need enough data
                    continue
                
                series1 = series1.loc[common_idx]
                series2 = series2.loc[common_idx]
                
                result = self.test_cointegration(series1.values, series2.values, significance)
                
                if result['is_cointegrated']:
                    pairs.append({
                        'asset1': columns[i],
                        'asset2': columns[j],
                        'p_value': result['p_value'],
                        'test_statistic': result['test_statistic']
                    })
        
        logger.info(f"Found {len(pairs)} cointegrated pairs")
        return pairs
    
    def calculate_spread(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        hedge_ratio: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """Compute the spread; estimates hedge ratio by OLS if not provided."""
        if hedge_ratio is None:
            # Estimate hedge ratio using linear regression
            from sklearn.linear_model import LinearRegression
            
            lr = LinearRegression()
            lr.fit(series2.reshape(-1, 1), series1)
            hedge_ratio = lr.coef_[0]
        
        spread = series1 - hedge_ratio * series2
        
        return spread, hedge_ratio
    
    def test_stationarity(
        self,
        series: np.ndarray,
        significance: float = 0.05
    ) -> Dict[str, Any]:
        """ADF stationarity test."""
        # ADF test
        adf_result = adfuller(series)
        
        is_stationary = adf_result[1] < significance
        
        result = {
            'test_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': is_stationary
        }
        
        return result
    
    def mean_reversion_signal(
        self,
        spread: np.ndarray,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ) -> np.ndarray:
        """Z-score based mean-reversion signals: +1 (long), -1 (short), 0 (flat)."""
        # Calculate z-score
        mean = np.mean(spread)
        std = np.std(spread)
        z_score = (spread - mean) / std
        
        # Generate signals
        signals = np.zeros_like(spread)
        position = 0
        
        for i in range(len(z_score)):
            if position == 0:
                # Enter long if spread is low
                if z_score[i] < -entry_threshold:
                    position = 1
                # Enter short if spread is high
                elif z_score[i] > entry_threshold:
                    position = -1
            else:
                # Exit if spread reverts
                if abs(z_score[i]) < exit_threshold:
                    position = 0
            
            signals[i] = position
        
        return signals
    
    def fit_regime_switching(
        self,
        returns: np.ndarray,
        n_regimes: int = 2
    ) -> Dict[str, Any]:
        """Fit a Gaussian HMM for regime detection."""
        from hmmlearn import hmm
        
        logger.info(f"Fitting regime switching model with {n_regimes} regimes")
        
        # Prepare data
        X = returns.reshape(-1, 1)
        
        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        model.fit(X)
        
        # Predict states
        states = model.predict(X)
        
        self.models['regime'] = model
        
        logger.info("Regime switching model fitted successfully")
        
        return {
            'model': model,
            'states': states,
            'means': model.means_,
            'covars': model.covars_,
            'transition_matrix': model.transmat_
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic return series
    n = 500
    returns = np.random.randn(n) * 0.01
    
    # Initialize model
    stat_model = StatisticalAlphaModel()
    
    # 1. ARIMA-GARCH
    logger.info("\n" + "="*50)
    logger.info("ARIMA-GARCH Example")
    logger.info("="*50)
    
    arima_garch_result = stat_model.fit_arima_garch(returns, arima_order=(1, 0, 1))
    forecast = stat_model.predict_arima_garch(steps=5)
    logger.info(f"Mean forecast: {forecast['mean']}")
    logger.info(f"Volatility forecast: {forecast['volatility']}")
    
    # 2. Kalman Filter
    logger.info("\n" + "="*50)
    logger.info("Kalman Filter Example")
    logger.info("="*50)
    
    observations = 100 + np.cumsum(np.random.randn(n) * 0.5)
    kf = stat_model.fit_kalman_filter(observations)
    smoothed, _ = stat_model.kalman_smooth(observations)
    logger.info(f"Smoothed state shape: {smoothed.shape}")
    
    # 3. Cointegration
    logger.info("\n" + "="*50)
    logger.info("Cointegration Example")
    logger.info("="*50)
    
    # Create cointegrated series
    series1 = np.cumsum(np.random.randn(n))
    series2 = series1 + np.random.randn(n) * 0.5  # Cointegrated with series1
    
    coint_result = stat_model.test_cointegration(series1, series2)
    spread, hedge_ratio = stat_model.calculate_spread(series1, series2)
    logger.info(f"Hedge ratio: {hedge_ratio:.4f}")
    logger.info(f"Spread mean: {np.mean(spread):.4f}, std: {np.std(spread):.4f}")
    
    # Mean reversion signals
    signals = stat_model.mean_reversion_signal(spread)
    logger.info(f"Generated {len(signals)} trading signals")
