"""
Event-Driven Simulator
Simulates trading strategies with realistic market conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from loguru import logger
import time
from collections import defaultdict

from .orderbook import Side, OrderType
from .matching_engine import MatchingEngine, MarketConfig


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    initial_capital: float = 1000000.0
    commission_rate: float = 0.0002  # 2 bps
    slippage_bps: float = 1.0
    enable_short_selling: bool = True
    max_position_size: int = 10000
    risk_free_rate: float = 0.02


class Position:
    """Trading position"""
    
    def __init__(self, symbol: str):
        """
        Initialize position
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.quantity = 0
        self.avg_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
    
    def update(self, quantity: int, price: float):
        """
        Update position with new trade
        
        Args:
            quantity: Trade quantity (positive for buy, negative for sell)
            price: Trade price
        """
        if self.quantity == 0:
            self.avg_price = price
            self.quantity = quantity
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # Increasing position
            total_cost = self.avg_price * abs(self.quantity) + price * abs(quantity)
            self.quantity += quantity
            self.avg_price = total_cost / abs(self.quantity)
        else:
            # Reducing or reversing position
            closed_quantity = min(abs(self.quantity), abs(quantity))
            pnl = closed_quantity * (price - self.avg_price) * np.sign(self.quantity)
            self.realized_pnl += pnl
            self.quantity += quantity
            
            if self.quantity != 0 and np.sign(self.quantity) != np.sign(self.quantity - quantity):
                # Position reversed
                self.avg_price = price
    
    def mark_to_market(self, current_price: float):
        """Update unrealized PnL"""
        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_price)
    
    @property
    def total_pnl(self) -> float:
        """Total PnL"""
        return self.realized_pnl + self.unrealized_pnl


class Portfolio:
    """Portfolio manager"""
    
    def __init__(self, initial_capital: float):
        """
        Initialize portfolio
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
    
    def execute_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        commission: float
    ):
        """
        Execute trade and update portfolio
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Execution price
            commission: Commission paid
        """
        # Update cash
        cost = quantity * price + commission
        self.cash -= cost
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        self.positions[symbol].update(quantity, price)
        
        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'timestamp': time.time()
        })
    
    def mark_to_market(self, prices: Dict[str, float]):
        """
        Mark positions to market
        
        Args:
            prices: Current prices for each symbol
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.mark_to_market(prices[symbol])
    
    @property
    def equity(self) -> float:
        """Total portfolio equity"""
        return self.cash + sum(pos.total_pnl for pos in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Total PnL"""
        return self.equity - self.initial_capital
    
    def get_position(self, symbol: str) -> int:
        """Get current position quantity"""
        if symbol in self.positions:
            return self.positions[symbol].quantity
        return 0


class EventSimulator:
    """
    Event-driven backtesting simulator
    """
    
    def __init__(
        self,
        market_config: Optional[MarketConfig] = None,
        sim_config: Optional[SimulationConfig] = None
    ):
        """
        Initialize event simulator
        
        Args:
            market_config: Market configuration
            sim_config: Simulation configuration
        """
        self.market_config = market_config or MarketConfig()
        self.sim_config = sim_config or SimulationConfig()
        
        self.engine = MatchingEngine(self.market_config)
        self.portfolio = Portfolio(self.sim_config.initial_capital)
        
        # Performance tracking
        self.equity_curve: List[float] = []
        self.timestamps: List[float] = []
        
    def run_backtest(
        self,
        strategy: Any,
        data: pd.DataFrame,
        symbol: str = "SYM"
    ) -> Dict[str, Any]:
        """
        Run backtest with a strategy
        
        Args:
            strategy: Strategy object with on_data() method
            data: Market data DataFrame
            symbol: Trading symbol
            
        Returns:
            Backtest results
        """
        logger.info(f"Starting backtest for {symbol}")
        logger.info(f"Data shape: {data.shape}")
        
        # Create order book
        self.engine.create_orderbook(symbol)
        
        # Initialize strategy
        if hasattr(strategy, 'initialize'):
            strategy.initialize(self)
        
        # Event loop
        for idx, row in data.iterrows():
            timestamp = row.get('timestamp', idx)
            
            # Update market data (simulate order book)
            if 'price' in row:
                self._update_market(symbol, row)
            
            # Call strategy
            if hasattr(strategy, 'on_data'):
                strategy.on_data(self, symbol, row)
            
            # Mark to market
            current_price = row.get('price', row.get('close', 0))
            self.portfolio.mark_to_market({symbol: current_price})
            
            # Record equity
            self.equity_curve.append(self.portfolio.equity)
            self.timestamps.append(timestamp)
        
        # Finalize strategy
        if hasattr(strategy, 'finalize'):
            strategy.finalize(self)
        
        # Calculate metrics
        results = self._calculate_metrics()
        
        logger.info(f"Backtest completed")
        logger.info(f"Final Equity: ${results['final_equity']:,.2f}")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        return results
    
    def _update_market(self, symbol: str, market_data: pd.Series):
        """
        Update market with new data
        
        Args:
            symbol: Trading symbol
            market_data: Market data for this timestamp
        """
        # Simplified: add orders to maintain bid-ask spread
        if 'bid' in market_data and 'ask' in market_data:
            bid_price = market_data['bid']
            ask_price = market_data['ask']
            quantity = market_data.get('volume', 100)
            
            # Add passive orders
            self.engine.submit_order(symbol, Side.BID, bid_price, quantity)
            self.engine.submit_order(symbol, Side.ASK, ask_price, quantity)
    
    def buy(
        self,
        symbol: str,
        quantity: int,
        limit_price: Optional[float] = None
    ) -> bool:
        """
        Buy order
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to buy
            limit_price: Limit price (None for market order)
            
        Returns:
            True if order executed
        """
        return self._submit_order(symbol, Side.BID, quantity, limit_price)
    
    def sell(
        self,
        symbol: str,
        quantity: int,
        limit_price: Optional[float] = None
    ) -> bool:
        """
        Sell order
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to sell
            limit_price: Limit price (None for market order)
            
        Returns:
            True if order executed
        """
        return self._submit_order(symbol, Side.ASK, quantity, limit_price)
    
    def _submit_order(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        limit_price: Optional[float]
    ) -> bool:
        """Submit order to market"""
        # Get current price
        book = self.engine.get_orderbook(symbol)
        if book is None:
            return False
        
        mid_price = book.get_mid_price()
        if mid_price is None:
            return False
        
        # Determine order type and price
        if limit_price is None:
            order_type = OrderType.MARKET
            price = mid_price
        else:
            order_type = OrderType.LIMIT
            price = limit_price
        
        # Submit order
        order, trades = self.engine.submit_order(
            symbol, side, price, quantity, order_type
        )
        
        # Process fills
        if trades:
            total_qty = sum(t.quantity for t in trades)
            avg_price = sum(t.price * t.quantity for t in trades) / total_qty
            
            # Calculate commission
            commission = total_qty * avg_price * self.sim_config.commission_rate
            
            # Update portfolio
            trade_qty = total_qty if side == Side.BID else -total_qty
            self.portfolio.execute_trade(symbol, trade_qty, avg_price, commission)
            
            return True
        
        return False
    
    def get_position(self, symbol: str) -> int:
        """Get current position"""
        return self.portfolio.get_position(symbol)
    
    def get_cash(self) -> float:
        """Get available cash"""
        return self.portfolio.cash
    
    def get_equity(self) -> float:
        """Get total equity"""
        return self.portfolio.equity
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Total return
        total_return = (equity_series.iloc[-1] - self.sim_config.initial_capital) / self.sim_config.initial_capital
        
        # Sharpe ratio
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = sum(1 for t in self.portfolio.trade_history if t['quantity'] > 0)
        total_trades = len(self.portfolio.trade_history)
        win_rate = winning_trades / max(1, total_trades)
        
        return {
            'initial_capital': self.sim_config.initial_capital,
            'final_equity': equity_series.iloc[-1],
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'equity_curve': self.equity_curve,
            'timestamps': self.timestamps
        }


if __name__ == "__main__":
    # Example strategy
    class SimpleMovingAverageCrossover:
        """Simple MA crossover strategy"""
        
        def __init__(self, fast_period=10, slow_period=20):
            self.fast_period = fast_period
            self.slow_period = slow_period
            self.position = 0
        
        def on_data(self, simulator, symbol, data):
            # This is a simplified example
            # In practice, would calculate MAs from historical data
            pass
    
    # Generate synthetic data
    from src.data_pipeline.ingestion import DataIngestion
    
    ingestion = DataIngestion()
    tick_data = ingestion.generate_synthetic_tick_data(num_ticks=1000, seed=42)
    
    # Resample to bars
    df = tick_data.set_index('timestamp').resample('1min').agg({
        'price': 'last',
        'bid': 'last',
        'ask': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Run backtest
    simulator = EventSimulator()
    strategy = SimpleMovingAverageCrossover()
    
    results = simulator.run_backtest(strategy, df, symbol="TEST")
    
    logger.info(f"\nBacktest Results:")
    for key, value in results.items():
        if key not in ['equity_curve', 'timestamps']:
            logger.info(f"  {key}: {value}")
