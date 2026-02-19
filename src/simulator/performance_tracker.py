"""
Performance Tracker
Separate class for calculating backtest performance metrics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'avg_trade': self.avg_trade,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses
        }


class PerformanceTracker:
    """Tracks equity curve and computes backtest performance metrics."""
    
    def __init__(self, initial_capital: float, risk_free_rate: float = 0.02):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Track equity over time
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = []
        
        # Track trades
        self.trade_pnls: List[float] = []
        
        # Peak tracking for drawdown
        self.peak_equity = initial_capital
        self.drawdowns: List[float] = []
    
    def update(self, equity: float, timestamp: Optional[datetime] = None):
        self.equity_curve.append(equity)
        if timestamp:
            self.timestamps.append(timestamp)
        
        # Update peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        drawdown = (self.peak_equity - equity) / self.peak_equity
        self.drawdowns.append(drawdown)
    
    def record_trade(self, pnl: float):
        self.trade_pnls.append(pnl)
    
    def calculate_returns(self) -> pd.Series:
        """Calculate returns series from equity curve"""
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        return returns
    
    def calculate_total_return(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
    
    def calculate_annualized_return(self, trading_days: int = 252) -> float:
        """Annualized return based on equity curve length."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        total_return = self.calculate_total_return()
        num_periods = len(self.equity_curve) - 1
        years = num_periods / trading_days
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def calculate_sharpe_ratio(self, trading_days: int = 252) -> float:
        returns = self.calculate_returns()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / trading_days)
        sharpe = np.sqrt(trading_days) * excess_returns.mean() / returns.std()
        
        return sharpe
    
    def calculate_sortino_ratio(self, trading_days: int = 252) -> float:
        """Sortino ratio using downside deviation only."""
        returns = self.calculate_returns()
        
        if len(returns) == 0:
            return 0.0
        
        # Downside returns only
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / trading_days)
        sortino = np.sqrt(trading_days) * excess_returns.mean() / downside_returns.std()
        
        return sortino
    
    def calculate_max_drawdown(self) -> float:
        if len(self.drawdowns) == 0:
            return 0.0
        return max(self.drawdowns)
    
    def calculate_calmar_ratio(self) -> float:
        ann_return = self.calculate_annualized_return()
        max_dd = self.calculate_max_drawdown()
        
        if max_dd == 0:
            return 0.0
        
        return ann_return / max_dd
    
    def calculate_win_rate(self) -> float:
        if len(self.trade_pnls) == 0:
            return 0.0
        
        winning_trades = sum(1 for pnl in self.trade_pnls if pnl > 0)
        return winning_trades / len(self.trade_pnls)
    
    def calculate_profit_factor(self) -> float:
        """Gross profit divided by gross loss."""
        if len(self.trade_pnls) == 0:
            return 0.0
        
        gross_profit = sum(pnl for pnl in self.trade_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in self.trade_pnls if pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_trade_statistics(self) -> Dict[str, Any]:
        if len(self.trade_pnls) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_trade': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        winning_pnls = [p for p in self.trade_pnls if p > 0]
        losing_pnls = [p for p in self.trade_pnls if p < 0]
        
        # Calculate consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        last_was_win = None
        
        for pnl in self.trade_pnls:
            is_win = pnl > 0
            if last_was_win is None or last_was_win == is_win:
                current_streak += 1
            else:
                if last_was_win:
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                else:
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
                current_streak = 1
            last_was_win = is_win
        
        # Final streak
        if last_was_win:
            max_consecutive_wins = max(max_consecutive_wins, current_streak)
        elif last_was_win is False:
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        
        return {
            'total_trades': len(self.trade_pnls),
            'winning_trades': len(winning_pnls),
            'losing_trades': len(losing_pnls),
            'avg_win': np.mean(winning_pnls) if winning_pnls else 0.0,
            'avg_loss': np.mean(losing_pnls) if losing_pnls else 0.0,
            'avg_trade': np.mean(self.trade_pnls),
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def get_metrics(self) -> PerformanceMetrics:
        """Bundle all metrics into a PerformanceMetrics snapshot."""
        trade_stats = self.calculate_trade_statistics()
        
        return PerformanceMetrics(
            total_return=self.calculate_total_return(),
            annualized_return=self.calculate_annualized_return(),
            sharpe_ratio=self.calculate_sharpe_ratio(),
            sortino_ratio=self.calculate_sortino_ratio(),
            max_drawdown=self.calculate_max_drawdown(),
            calmar_ratio=self.calculate_calmar_ratio(),
            win_rate=self.calculate_win_rate(),
            profit_factor=self.calculate_profit_factor(),
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            avg_trade=trade_stats['avg_trade'],
            max_consecutive_wins=trade_stats['max_consecutive_wins'],
            max_consecutive_losses=trade_stats['max_consecutive_losses']
        )
    
    def get_equity_series(self) -> pd.Series:
        """Get equity curve as pandas Series"""
        if self.timestamps:
            return pd.Series(self.equity_curve, index=self.timestamps)
        return pd.Series(self.equity_curve)
    
    def print_summary(self):
        metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Initial Capital:        ${self.initial_capital:,.2f}")
        print(f"Final Equity:           ${self.equity_curve[-1]:,.2f}")
        print(f"Total Return:           {metrics.total_return:,.2%}")
        print(f"Annualized Return:      {metrics.annualized_return:,.2%}")
        print("="*60)
        print(f"Sharpe Ratio:           {metrics.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:          {metrics.sortino_ratio:.2f}")
        print(f"Calmar Ratio:           {metrics.calmar_ratio:.2f}")
        print(f"Max Drawdown:           {metrics.max_drawdown:.2%}")
        print("="*60)
        print(f"Total Trades:           {metrics.total_trades}")
        print(f"Win Rate:               {metrics.win_rate:.2%}")
        print(f"Profit Factor:          {metrics.profit_factor:.2f}")
        print(f"Average Win:            ${metrics.avg_win:,.2f}")
        print(f"Average Loss:           ${metrics.avg_loss:,.2f}")
        print(f"Max Consecutive Wins:   {metrics.max_consecutive_wins}")
        print(f"Max Consecutive Losses: {metrics.max_consecutive_losses}")
        print("="*60)


if __name__ == "__main__":
    # Example usage
    tracker = PerformanceTracker(initial_capital=100000)
    
    # Simulate equity curve
    np.random.seed(42)
    for i in range(252):
        daily_return = np.random.normal(0.0005, 0.01)
        new_equity = tracker.equity_curve[-1] * (1 + daily_return)
        tracker.update(new_equity)
        
        # Simulate some trades
        if i % 10 == 0:
            trade_pnl = np.random.normal(100, 500)
            tracker.record_trade(trade_pnl)
    
    # Print summary
    tracker.print_summary()
