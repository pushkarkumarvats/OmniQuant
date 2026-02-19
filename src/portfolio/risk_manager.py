"""
Risk Manager
Real-time risk monitoring and position limits
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from loguru import logger
from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_position_size: int = 10000
    max_portfolio_value: float = 1000000.0
    max_concentration: float = 0.25  # Max % in single position
    max_leverage: float = 1.0
    max_drawdown: float = 0.20
    var_confidence: float = 0.95
    stop_loss_pct: float = 0.05


class RiskManager:
    """Real-time risk monitoring and position limits."""
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        
        # State
        self.positions: Dict[str, int] = {}
        self.prices: Dict[str, float] = {}
        self.peak_equity = 0.0
        self.current_equity = 0.0
        
        # Risk metrics
        self.var = 0.0
        self.cvar = 0.0
        self.current_drawdown = 0.0
        
        # Alerts
        self.alerts: List[Dict[str, Any]] = []
        
    def update_positions(self, positions: Dict[str, int], prices: Dict[str, float], equity: float):
        """Refresh position/price/equity state and recalculate drawdown."""
        self.positions = positions
        self.prices = prices
        self.current_equity = equity
        
        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        
    def check_position_limit(self, symbol: str, quantity: int) -> bool:
        """Returns False and alerts if abs(quantity) > limit."""
        if abs(quantity) > self.limits.max_position_size:
            self._create_alert('POSITION_LIMIT', f"Position size {abs(quantity)} exceeds limit {self.limits.max_position_size}")
            return False
        return True
    
    def check_concentration_limit(self, symbol: str) -> bool:
        if symbol not in self.positions or symbol not in self.prices:
            return True
        
        position_value = abs(self.positions[symbol] * self.prices[symbol])
        concentration = position_value / max(self.current_equity, 1.0)
        
        if concentration > self.limits.max_concentration:
            self._create_alert('CONCENTRATION_LIMIT', f"Concentration {concentration:.2%} exceeds limit {self.limits.max_concentration:.2%}")
            return False
        return True
    
    def check_leverage_limit(self) -> bool:
        total_gross_exposure = sum(abs(pos * self.prices.get(sym, 0)) for sym, pos in self.positions.items())
        leverage = total_gross_exposure / max(self.current_equity, 1.0)
        
        if leverage > self.limits.max_leverage:
            self._create_alert('LEVERAGE_LIMIT', f"Leverage {leverage:.2f} exceeds limit {self.limits.max_leverage:.2f}")
            return False
        return True
    
    def check_drawdown_limit(self) -> bool:
        if self.current_drawdown > self.limits.max_drawdown:
            self._create_alert('DRAWDOWN_LIMIT', f"Drawdown {self.current_drawdown:.2%} exceeds limit {self.limits.max_drawdown:.2%}")
            return False
        return True
    
    def calculate_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence: Optional[float] = None
    ) -> float:
        """Historical VaR at the given confidence level."""
        if confidence is None:
            confidence = self.limits.var_confidence
        
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Historical VaR
        var = np.percentile(portfolio_returns, (1 - confidence) * 100)
        self.var = abs(var)
        
        return self.var
    
    def calculate_cvar(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence: Optional[float] = None
    ) -> float:
        """Expected shortfall (CVaR) at the given confidence level."""
        if confidence is None:
            confidence = self.limits.var_confidence
        
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # CVaR (expected shortfall)
        var_threshold = np.percentile(portfolio_returns, (1 - confidence) * 100)
        cvar = portfolio_returns[portfolio_returns <= var_threshold].mean()
        self.cvar = abs(cvar)
        
        return self.cvar
    
    def calculate_risk_metrics(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Compute vol, downside dev, VaR, CVaR, max drawdown for the weighted portfolio."""
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Volatility
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Downside deviation
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # VaR and CVaR
        var_95 = self.calculate_var(returns, weights, 0.95)
        cvar_95 = self.calculate_cvar(returns, weights, 0.95)
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': abs(max_dd),
            'current_drawdown': self.current_drawdown
        }
    
    def check_stop_loss(self, symbol: str, entry_price: float, current_price: float, side: str = 'long') -> bool:
        """True if unrealized loss exceeds stop_loss_pct."""
        if side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        if pnl_pct < -self.limits.stop_loss_pct:
            self._create_alert('STOP_LOSS', f"{symbol} stop loss triggered: {pnl_pct:.2%}")
            return True
        
        return False
    
    def _create_alert(self, alert_type: str, message: str):
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': pd.Timestamp.now()
        }
        self.alerts.append(alert)
        logger.warning(f"RISK ALERT [{alert_type}]: {message}")
    
    def get_alerts(self, recent_only: bool = True, n: int = 10) -> List[Dict[str, Any]]:
        """Return recent risk alerts."""
        if recent_only:
            return self.alerts[-n:]
        return self.alerts
    
    def clear_alerts(self):
        self.alerts.clear()
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Snapshot of current risk state."""
        total_gross_exposure = sum(abs(pos * self.prices.get(sym, 0)) for sym, pos in self.positions.items())
        leverage = total_gross_exposure / max(self.current_equity, 1.0)
        
        # Calculate concentration
        concentrations = {}
        for sym, pos in self.positions.items():
            if sym in self.prices:
                position_value = abs(pos * self.prices[sym])
                concentrations[sym] = position_value / max(self.current_equity, 1.0)
        
        return {
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'current_drawdown': self.current_drawdown,
            'leverage': leverage,
            'num_positions': len([p for p in self.positions.values() if p != 0]),
            'var_95': self.var,
            'cvar_95': self.cvar,
            'concentrations': concentrations,
            'alerts_count': len(self.alerts)
        }


if __name__ == "__main__":
    # Example usage
    risk_manager = RiskManager()
    
    # Simulate positions
    positions = {
        'AAPL': 100,
        'GOOGL': 50,
        'MSFT': -30
    }
    
    prices = {
        'AAPL': 150.0,
        'GOOGL': 2800.0,
        'MSFT': 300.0
    }
    
    equity = 500000.0
    
    # Update risk state
    risk_manager.update_positions(positions, prices, equity)
    
    # Check limits
    logger.info("Checking risk limits...")
    risk_manager.check_position_limit('AAPL', 100)
    risk_manager.check_concentration_limit('GOOGL')
    risk_manager.check_leverage_limit()
    risk_manager.check_drawdown_limit()
    
    # Get risk summary
    summary = risk_manager.get_risk_summary()
    logger.info(f"\nRisk Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    # Get alerts
    alerts = risk_manager.get_alerts()
    logger.info(f"\nAlerts: {len(alerts)}")
