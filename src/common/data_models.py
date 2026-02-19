"""
Data Models
Standardized data structures to prevent errors and ensure consistency
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np


class AssetType(Enum):
    EQUITY = "equity"
    FUTURE = "future"
    OPTION = "option"
    FOREX = "forex"
    CRYPTO = "crypto"


@dataclass
class TickData:
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    
    def __post_init__(self):
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative, got {self.volume}")
        if self.bid and self.ask and self.bid > self.ask:
            raise ValueError(f"Bid {self.bid} cannot be greater than ask {self.ask}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TickData':
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size
        }


@dataclass
class BarData:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    
    def __post_init__(self):
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"Invalid OHLC: open={self.open}, high={self.high}, low={self.low}")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"Invalid OHLC: close={self.close}, high={self.high}, low={self.low}")
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative: {self.volume}")


@dataclass
class OrderBookSnapshot:
    timestamp: datetime
    symbol: str
    bids: List[tuple[float, int]]  # (price, quantity)
    asks: List[tuple[float, int]]
    
    def __post_init__(self):
        if self.bids and self.asks:
            best_bid = self.bids[0][0] if self.bids else 0
            best_ask = self.asks[0][0] if self.asks else float('inf')
            if best_bid >= best_ask:
                raise ValueError(f"Best bid {best_bid} >= best ask {best_ask}")
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


@dataclass
class FeatureVector:
    timestamp: datetime
    symbol: str
    features: Dict[str, float]
    target: Optional[float] = None
    
    def to_array(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        if feature_names is None:
            feature_names = sorted(self.features.keys())
        return np.array([self.features.get(name, np.nan) for name in feature_names])
    
    @classmethod
    def from_series(cls, series: pd.Series, timestamp_col: str = 'timestamp', 
                    symbol: str = 'UNKNOWN') -> 'FeatureVector':
        timestamp = series[timestamp_col] if timestamp_col in series.index else datetime.now()
        features = {k: v for k, v in series.items() if k not in [timestamp_col, 'symbol', 'target']}
        target = series.get('target', None)
        return cls(timestamp=timestamp, symbol=symbol, features=features, target=target)


@dataclass
class PredictionResult:
    timestamp: datetime
    symbol: str
    prediction: float
    confidence: Optional[float] = None
    model_name: Optional[str] = None
    features_used: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'model_name': self.model_name,
            'features_used': self.features_used
        }


@dataclass
class TradeRecord:
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    strategy_name: Optional[str] = None
    
    def __post_init__(self):
        if self.side not in ['BUY', 'SELL']:
            raise ValueError(f"Side must be 'BUY' or 'SELL', got {self.side}")
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")
    
    @property
    def notional(self) -> float:
        return abs(self.quantity * self.price)
    
    @property
    def total_cost(self) -> float:
        return self.notional + self.commission + abs(self.slippage)


@dataclass
class PositionState:
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.avg_entry_price)
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float = 0.0002
    slippage_bps: float = 1.0
    enable_short_selling: bool = True
    max_position_size: int = 10000
    risk_free_rate: float = 0.02
    
    def __post_init__(self):
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if self.commission_rate < 0 or self.commission_rate > 1:
            raise ValueError("Commission rate must be between 0 and 1")


@dataclass
class BacktestResult:
    config: BacktestConfig
    initial_capital: float
    final_equity: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    equity_curve: pd.Series
    trades: List[TradeRecord] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.final_equity,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor
        }
    
    def summary(self) -> str:
        """Generate summary report"""
        return f"""
Backtest Results Summary
{'='*60}
Period: {self.config.start_date} to {self.config.end_date}
Initial Capital: ${self.initial_capital:,.2f}
Final Equity: ${self.final_equity:,.2f}
Total Return: {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}
{'='*60}
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.2%}
Average Win: ${self.avg_win:,.2f}
Average Loss: ${self.avg_loss:,.2f}
Profit Factor: {self.profit_factor:.2f}
{'='*60}
"""
