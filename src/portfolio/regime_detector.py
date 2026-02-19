"""
Regime Detector
Detect market regimes using HMM and clustering
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from loguru import logger
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """HMM / K-Means regime detector for adaptive allocation."""
    
    def __init__(self, n_regimes: int = 3, method: str = 'hmm'):
        self.n_regimes = n_regimes
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.current_regime = None
        
    def fit(self, features: pd.DataFrame):
        """Fit the regime model on scaled features."""
        logger.info(f"Fitting {self.method} regime detector with {self.n_regimes} regimes")
        
        # Normalize features
        X = self.scaler.fit_transform(features.values)
        
        if self.method == 'hmm':
            # Hidden Markov Model
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            self.model.fit(X)
            
        elif self.method == 'clustering':
            # K-Means clustering
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init=10
            )
            self.model.fit(X)
        
        logger.info("Regime detector fitted successfully")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict regime labels."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.scaler.transform(features.values)
        
        if self.method == 'hmm':
            regimes = self.model.predict(X)
        else:
            regimes = self.model.predict(X)
        
        return regimes
    
    def get_current_regime(self, features: pd.Series) -> int:
        X = self.scaler.transform(features.values.reshape(1, -1))
        
        if self.method == 'hmm':
            regime = self.model.predict(X)[0]
        else:
            regime = self.model.predict(X)[0]
        
        self.current_regime = regime
        return regime
    
    def get_regime_statistics(
        self,
        features: pd.DataFrame,
        returns: pd.Series
    ) -> Dict[int, Dict[str, float]]:
        """Per-regime return stats (mean, vol, Sharpe, max DD)."""
        regimes = self.predict(features)
        
        stats = {}
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask]
            
            stats[regime] = {
                'count': regime_mask.sum(),
                'frequency': regime_mask.mean(),
                'mean_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(regime_returns)
            }
        
        return stats
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def get_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        """Empirical transition probabilities between regimes."""
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)
        
        return transition_matrix
    
    def create_regime_features(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.DataFrame:
        """Rolling vol, mean, skew, kurtosis, autocorr, trend as regime features."""
        features = pd.DataFrame(index=returns.index)
        
        # Volatility
        features['volatility'] = returns.rolling(window=window).std()
        
        # Mean return
        features['mean_return'] = returns.rolling(window=window).mean()
        
        # Skewness
        features['skewness'] = returns.rolling(window=window).skew()
        
        # Kurtosis
        features['kurtosis'] = returns.rolling(window=window).kurt()
        
        # Autocorrelation
        features['autocorr'] = returns.rolling(window=window).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0
        )
        
        # Trend (linear regression slope)
        features['trend'] = returns.rolling(window=window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # Drop NaN
        features = features.dropna()
        
        return features


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic return data with regimes
    n_periods = 500
    
    # Regime 1: Low vol
    regime1_returns = np.random.normal(0.001, 0.005, 150)
    
    # Regime 2: High vol
    regime2_returns = np.random.normal(-0.002, 0.02, 200)
    
    # Regime 3: Trending
    regime3_returns = np.random.normal(0.003, 0.01, 150)
    
    returns = pd.Series(np.concatenate([regime1_returns, regime2_returns, regime3_returns]))
    
    # Create regime detector
    detector = RegimeDetector(n_regimes=3, method='hmm')
    
    # Create features
    features = detector.create_regime_features(returns, window=20)
    
    # Fit model
    detector.fit(features)
    
    # Predict regimes
    predicted_regimes = detector.predict(features)
    
    # Get statistics
    stats = detector.get_regime_statistics(features, returns.loc[features.index])
    
    logger.info("\nRegime Statistics:")
    for regime, regime_stats in stats.items():
        logger.info(f"\nRegime {regime}:")
        for key, value in regime_stats.items():
            logger.info(f"  {key}: {value}")
    
    # Transition matrix
    transition_matrix = detector.get_transition_matrix(predicted_regimes)
    logger.info(f"\nTransition Matrix:\n{transition_matrix}")
