"""
Portfolio Optimizer
Implements various portfolio optimization techniques
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger
import cvxpy as cp
from scipy.optimize import minimize


class PortfolioOptimizer:
    """Portfolio optimization — MVO, risk parity, HRP, Black-Litterman."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.weights = None
        
    def mean_variance_optimization(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        constraints: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Solve the classic Markowitz problem via cvxpy. Returns weight vector."""
        logger.info("Running mean-variance optimization")
        
        n_assets = len(expected_returns)
        
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Objective: maximize return - risk_aversion * variance
        portfolio_return = expected_returns @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints_list = [
            cp.sum(w) == 1,  # Weights sum to 1
        ]
        
        if constraints:
            # Long-only constraint
            if constraints.get('long_only', True):
                constraints_list.append(w >= 0)
            
            # Max weight per asset
            if 'max_weight' in constraints:
                constraints_list.append(w <= constraints['max_weight'])
            
            # Min weight per asset
            if 'min_weight' in constraints:
                constraints_list.append(w >= constraints['min_weight'])
            
            # Max leverage
            if 'max_leverage' in constraints:
                constraints_list.append(cp.norm(w, 1) <= constraints['max_leverage'])
        else:
            constraints_list.append(w >= 0)  # Default: long-only
        
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status == 'optimal':
            self.weights = w.value
            logger.info(f"Optimization successful - Portfolio return: {portfolio_return.value:.4f}")
            return self.weights
        else:
            logger.error(f"Optimization failed with status: {problem.status}")
            return np.zeros(n_assets)
    
    def risk_parity(
        self,
        cov_matrix: np.ndarray,
        target_risk: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Equalise risk contributions across assets. Uses SLSQP."""
        logger.info("Running risk parity optimization")
        
        n_assets = cov_matrix.shape[0]
        
        if target_risk is None:
            target_risk = np.ones(n_assets) / n_assets
        
        def objective(w):
            portfolio_vol = np.sqrt(w @ cov_matrix @ w)
            marginal_contrib = cov_matrix @ w
            risk_contrib = w * marginal_contrib / portfolio_vol
            risk_contrib = risk_contrib / risk_contrib.sum()
            
            # Distance from target
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
        ]
        
        bounds = [(0, 1) for _ in range(n_assets)]  # Long-only
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            self.weights = result.x
            logger.info("Risk parity optimization successful")
            return self.weights
        else:
            logger.error("Risk parity optimization failed")
            return np.ones(n_assets) / n_assets
    
    def hierarchical_risk_parity(
        self,
        returns: pd.DataFrame
    ) -> np.ndarray:
        """Lopez de Prado's HRP via single-linkage clustering."""
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        logger.info("Running Hierarchical Risk Parity optimization")
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Calculate distance matrix
        distance_matrix = np.sqrt((1 - corr_matrix) / 2)
        
        # Hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method='single')
        
        # Get quasi-diagonalization
        sort_idx = self._get_quasi_diag(linkage_matrix)
        sorted_returns = returns.iloc[:, sort_idx]
        
        # Calculate weights recursively
        weights = self._recursive_bisection(sorted_returns)
        
        # Reorder to original
        weights_original = np.zeros(len(weights))
        for i, idx in enumerate(sort_idx):
            weights_original[idx] = weights[i]
        
        self.weights = weights_original
        logger.info("HRP optimization successful")
        return self.weights
    
    def _get_quasi_diag(self, linkage_matrix):
        from scipy.cluster.hierarchy import dendrogram
        
        dend = dendrogram(linkage_matrix, no_plot=True)
        return dend['leaves']
    
    def _recursive_bisection(self, returns: pd.DataFrame) -> np.ndarray:
        cov_matrix = returns.cov()
        
        # Base case
        if len(returns.columns) == 1:
            return np.array([1.0])
        
        # Split into two clusters
        mid = len(returns.columns) // 2
        left_returns = returns.iloc[:, :mid]
        right_returns = returns.iloc[:, mid:]
        
        # Calculate cluster variances
        left_var = self._cluster_variance(left_returns)
        right_var = self._cluster_variance(right_returns)
        
        # Allocate weight between clusters
        alpha = 1 - left_var / (left_var + right_var)
        
        # Recurse
        left_weights = self._recursive_bisection(left_returns)
        right_weights = self._recursive_bisection(right_returns)
        
        # Combine
        weights = np.concatenate([left_weights * (1 - alpha), right_weights * alpha])
        return weights
    
    def _cluster_variance(self, returns: pd.DataFrame) -> float:
        cov_matrix = returns.cov()
        weights = np.ones(len(returns.columns)) / len(returns.columns)
        return weights @ cov_matrix @ weights
    
    def black_litterman(
        self,
        market_caps: np.ndarray,
        cov_matrix: np.ndarray,
        views: Optional[List[Tuple[np.ndarray, float, float]]] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05
    ) -> np.ndarray:
        """Combine equilibrium priors with investor views. Returns posterior weights."""
        logger.info("Running Black-Litterman optimization")
        
        n_assets = len(market_caps)
        
        # Market equilibrium weights
        w_mkt = market_caps / market_caps.sum()
        
        # Implied equilibrium returns (reverse optimization)
        pi = risk_aversion * cov_matrix @ w_mkt
        
        # Incorporate views if provided
        if views:
            # Views matrix P and vector Q
            P = np.array([v[0] for v in views])
            Q = np.array([v[1] for v in views])
            omega = np.diag([v[2] for v in views])
            
            # Posterior returns
            tau_cov = tau * cov_matrix
            M = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(omega) @ P)
            mu_bl = M @ (np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(omega) @ Q)
        else:
            mu_bl = pi
        
        # Optimize with posterior returns
        self.weights = self.mean_variance_optimization(mu_bl, cov_matrix, risk_aversion)
        
        return self.weights
    
    def maximum_diversification(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """Maximise the diversification ratio (weighted vols / portfolio vol)."""
        logger.info("Running maximum diversification optimization")
        
        n_assets = len(expected_returns)
        
        # Standard deviations
        std_devs = np.sqrt(np.diag(cov_matrix))
        
        def objective(w):
            portfolio_vol = np.sqrt(w @ cov_matrix @ w)
            weighted_vol = w @ std_devs
            return -weighted_vol / portfolio_vol  # Negative for minimization
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        bounds = [(0, 1) for _ in range(n_assets)]
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            self.weights = result.x
            logger.info("Maximum diversification optimization successful")
            return self.weights
        else:
            logger.error("Optimization failed")
            return np.ones(n_assets) / n_assets
    
    def volatility_targeting(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        target_vol: float = 0.15
    ) -> np.ndarray:
        """Rescale weights so portfolio annualised vol hits target_vol."""
        current_vol = np.sqrt(weights @ cov_matrix @ weights)
        scale = target_vol / current_vol
        
        scaled_weights = weights * scale
        logger.info(f"Volatility targeting: {current_vol:.2%} -> {target_vol:.2%} (scale: {scale:.2f})")
        
        return scaled_weights
    
    def get_portfolio_stats(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Return expected return, vol, Sharpe for the given weights."""
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'weights': weights
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic data
    n_assets = 5
    n_periods = 252
    
    # Random returns
    returns = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.01,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Calculate statistics
    expected_returns = returns.mean().values * 252  # Annualized
    cov_matrix = returns.cov().values * 252
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Test different methods
    logger.info("\n" + "="*60)
    logger.info("Mean-Variance Optimization")
    logger.info("="*60)
    weights_mv = optimizer.mean_variance_optimization(expected_returns, cov_matrix)
    stats_mv = optimizer.get_portfolio_stats(weights_mv, expected_returns, cov_matrix)
    logger.info(f"Weights: {weights_mv}")
    logger.info(f"Stats: {stats_mv}")
    
    logger.info("\n" + "="*60)
    logger.info("Risk Parity")
    logger.info("="*60)
    weights_rp = optimizer.risk_parity(cov_matrix)
    stats_rp = optimizer.get_portfolio_stats(weights_rp, expected_returns, cov_matrix)
    logger.info(f"Weights: {weights_rp}")
    logger.info(f"Stats: {stats_rp}")
    
    logger.info("\n" + "="*60)
    logger.info("Hierarchical Risk Parity")
    logger.info("="*60)
    weights_hrp = optimizer.hierarchical_risk_parity(returns)
    stats_hrp = optimizer.get_portfolio_stats(weights_hrp, expected_returns, cov_matrix)
    logger.info(f"Weights: {weights_hrp}")
    logger.info(f"Stats: {stats_hrp}")
