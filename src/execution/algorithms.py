"""
Execution Algorithms - TWAP, VWAP, POV, Implementation Shortfall
Production-grade execution strategies for optimal order routing
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
import asyncio


@dataclass
class Order:
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    target_price: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    urgency: float = 0.5  # 0 = patient, 1 = aggressive


@dataclass
class ChildOrder:
    quantity: float
    timestamp: datetime
    limit_price: Optional[float] = None
    order_type: str = 'limit'  # 'market', 'limit'


class ExecutionAlgorithm:
    """Base class for execution algorithms"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def schedule(self, order: Order, market_data: pd.DataFrame) -> List[ChildOrder]:
        raise NotImplementedError
    
    async def execute(
        self,
        order: Order,
        market_data: pd.DataFrame,
        execution_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run the scheduling algorithm and execute resulting child orders."""
        child_orders = self.schedule(order, market_data)
        
        executions = []
        total_executed = 0
        total_cost = 0
        
        for child in child_orders:
            if execution_callback:
                result = await execution_callback(child)
                executions.append(result)
                total_executed += result.get('filled_quantity', 0)
                total_cost += result.get('cost', 0)
            else:
                # Simulate execution
                executions.append({
                    'quantity': child.quantity,
                    'price': child.limit_price,
                    'timestamp': child.timestamp
                })
                total_executed += child.quantity
        
        avg_price = total_cost / total_executed if total_executed > 0 else 0
        
        summary = {
            'algorithm': self.__class__.__name__,
            'total_quantity': order.quantity,
            'executed_quantity': total_executed,
            'fill_rate': total_executed / order.quantity,
            'average_price': avg_price,
            'child_orders': len(child_orders),
            'executions': executions
        }
        
        self.execution_history.append(summary)
        return summary


class TWAP(ExecutionAlgorithm):
    """Time-Weighted Average Price -- equal slices at regular intervals."""
    
    def schedule(self, order: Order, market_data: pd.DataFrame) -> List[ChildOrder]:
        num_slices = self.config.get('num_slices', 10)
        interval_seconds = self.config.get('interval_seconds', 60)
        
        slice_size = order.quantity / num_slices
        child_orders = []
        
        start_time = order.start_time or datetime.now()
        
        for i in range(num_slices):
            timestamp = start_time + timedelta(seconds=i * interval_seconds)
            
            child = ChildOrder(
                quantity=slice_size,
                timestamp=timestamp,
                limit_price=order.target_price,
                order_type='limit' if order.target_price else 'market'
            )
            child_orders.append(child)
        
        logger.info(f"TWAP: Scheduled {num_slices} slices of {slice_size:.2f} each")
        return child_orders


class VWAP(ExecutionAlgorithm):
    """Volume-Weighted Average Price -- slices proportional to historical volume."""
    
    def schedule(self, order: Order, market_data: pd.DataFrame) -> List[ChildOrder]:
        if 'volume' not in market_data.columns:
            logger.warning("No volume data, falling back to TWAP")
            return TWAP(self.config).schedule(order, market_data)
        
        num_slices = self.config.get('num_slices', 10)
        
        # Calculate volume profile
        volume_profile = self._calculate_volume_profile(market_data, num_slices)
        
        child_orders = []
        start_time = order.start_time or datetime.now()
        interval_seconds = self.config.get('interval_seconds', 60)
        
        for i, volume_pct in enumerate(volume_profile):
            slice_size = order.quantity * volume_pct
            timestamp = start_time + timedelta(seconds=i * interval_seconds)
            
            child = ChildOrder(
                quantity=slice_size,
                timestamp=timestamp,
                limit_price=order.target_price,
                order_type='limit' if order.target_price else 'market'
            )
            child_orders.append(child)
        
        logger.info(f"VWAP: Scheduled {num_slices} volume-weighted slices")
        return child_orders
    
    def _calculate_volume_profile(
        self,
        market_data: pd.DataFrame,
        num_buckets: int
    ) -> np.ndarray:
        # Group by time bucket
        market_data['bucket'] = pd.cut(
            range(len(market_data)),
            bins=num_buckets,
            labels=False
        )
        
        volume_by_bucket = market_data.groupby('bucket')['volume'].sum()
        total_volume = volume_by_bucket.sum()
        
        # Return as percentage of total
        return (volume_by_bucket / total_volume).values


class POV(ExecutionAlgorithm):
    """Percentage of Volume -- maintains target participation rate."""
    
    def schedule(self, order: Order, market_data: pd.DataFrame) -> List[ChildOrder]:
        target_pov = self.config.get('target_pov', 0.10)  # 10% participation
        max_pov = self.config.get('max_pov', 0.30)  # Don't exceed 30%
        
        if 'volume' not in market_data.columns:
            logger.warning("No volume data for POV")
            return TWAP(self.config).schedule(order, market_data)
        
        child_orders = []
        remaining_qty = order.quantity
        start_time = order.start_time or datetime.now()
        interval_seconds = self.config.get('interval_seconds', 60)
        
        # Estimate market volume per interval
        avg_volume_per_interval = market_data['volume'].mean()
        
        i = 0
        while remaining_qty > 0 and i < 100:  # Max 100 slices
            # Calculate slice size based on POV target
            market_volume_estimate = avg_volume_per_interval
            slice_size = min(
                remaining_qty,
                market_volume_estimate * target_pov,
                market_volume_estimate * max_pov
            )
            
            timestamp = start_time + timedelta(seconds=i * interval_seconds)
            
            child = ChildOrder(
                quantity=slice_size,
                timestamp=timestamp,
                limit_price=order.target_price,
                order_type='limit' if order.target_price else 'market'
            )
            child_orders.append(child)
            
            remaining_qty -= slice_size
            i += 1
        
        logger.info(f"POV: Scheduled {len(child_orders)} slices at {target_pov*100}% participation")
        return child_orders


class ImplementationShortfall(ExecutionAlgorithm):
    """Almgren-Chriss optimal execution minimizing expected cost + risk."""
    
    def schedule(self, order: Order, market_data: pd.DataFrame) -> List[ChildOrder]:
        # Model parameters
        T = self.config.get('time_horizon', 3600)  # 1 hour in seconds
        risk_aversion = self.config.get('risk_aversion', 1e-6)
        
        # Estimate market impact parameters
        sigma = market_data['price'].pct_change().std() if 'price' in market_data.columns else 0.01
        permanent_impact = self.config.get('permanent_impact', 0.1)
        temporary_impact = self.config.get('temporary_impact', 0.01)
        
        num_slices = self.config.get('num_slices', 10)
        dt = T / num_slices
        
        # Almgren-Chriss optimal trajectory
        # kappa = sqrt(lambda * sigma^2 / eta) controls urgency
        kappa = np.sqrt(risk_aversion * sigma**2 / temporary_impact)
        tau = T
        
        child_orders = []
        start_time = order.start_time or datetime.now()
        
        for i in range(num_slices):
            t = i * dt
            
            # Optimal trading rate from the closed-form solution
            optimal_rate = order.quantity * (
                np.sinh(kappa * (tau - t)) / np.sinh(kappa * tau)
            )
            slice_size = optimal_rate * dt / T
            
            timestamp = start_time + timedelta(seconds=int(t))
            
            child = ChildOrder(
                quantity=slice_size,
                timestamp=timestamp,
                limit_price=order.target_price,
                order_type='limit' if order.target_price else 'market'
            )
            child_orders.append(child)
        
        logger.info(f"IS: Scheduled {num_slices} slices using Almgren-Chriss")
        return child_orders


class AdaptiveExecution(ExecutionAlgorithm):
    """Dynamically selects execution strategy based on market conditions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.twap = TWAP(config)
        self.vwap = VWAP(config)
        self.pov = POV(config)
        self.is_algo = ImplementationShortfall(config)
    
    def schedule(self, order: Order, market_data: pd.DataFrame) -> List[ChildOrder]:
        # Analyze market conditions
        volatility = market_data['price'].pct_change().std() if 'price' in market_data.columns else 0
        avg_volume = market_data['volume'].mean() if 'volume' in market_data.columns else 0
        
        # Decision logic
        if order.urgency > 0.8:
            # High urgency: aggressive execution
            logger.info("Adaptive: High urgency, using aggressive TWAP")
            self.config['num_slices'] = 5
            return self.twap.schedule(order, market_data)
        
        elif volatility > self.config.get('high_vol_threshold', 0.02):
            # High volatility: use IS to minimize risk
            logger.info("Adaptive: High volatility, using Implementation Shortfall")
            return self.is_algo.schedule(order, market_data)
        
        elif avg_volume > order.quantity * 10:
            # High liquidity: use POV
            logger.info("Adaptive: High liquidity, using POV")
            return self.pov.schedule(order, market_data)
        
        else:
            # Default: VWAP
            logger.info("Adaptive: Normal conditions, using VWAP")
            return self.vwap.schedule(order, market_data)


class ExecutionManager:
    """Manages multiple concurrent executions"""
    
    def __init__(self):
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.algorithms = {
            'twap': TWAP,
            'vwap': VWAP,
            'pov': POV,
            'is': ImplementationShortfall,
            'adaptive': AdaptiveExecution
        }
    
    async def execute_order(
        self,
        order: Order,
        algorithm: str,
        market_data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        algo_class = self.algorithms[algorithm]
        algo_instance = algo_class(config)
        
        execution_id = f"{order.symbol}_{datetime.now().timestamp()}"
        self.active_executions[execution_id] = {
            'order': order,
            'algorithm': algorithm,
            'status': 'running'
        }
        
        try:
            result = await algo_instance.execute(order, market_data)
            self.active_executions[execution_id]['status'] = 'completed'
            self.active_executions[execution_id]['result'] = result
            return result
        
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self.active_executions[execution_id]['status'] = 'failed'
            self.active_executions[execution_id]['error'] = str(e)
            raise
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        return self.active_executions.get(execution_id, {})
    
    def cancel_execution(self, execution_id: str):
        if execution_id in self.active_executions:
            self.active_executions[execution_id]['status'] = 'cancelled'
            logger.info(f"Execution {execution_id} cancelled")


if __name__ == "__main__":
    # Test execution algorithms
    import asyncio
    
    async def test_algorithms():
        # Generate sample market data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01 09:30', periods=390, freq='1min')
        market_data = pd.DataFrame({
            'timestamp': dates,
            'price': 100 + np.cumsum(np.random.randn(390) * 0.1),
            'volume': np.random.randint(1000, 10000, 390)
        })
        
        # Create test order
        order = Order(
            symbol='AAPL',
            side='buy',
            quantity=10000,
            target_price=100.50,
            start_time=datetime.now(),
            urgency=0.5
        )
        
        # Test each algorithm
        manager = ExecutionManager()
        
        for algo_name in ['twap', 'vwap', 'pov', 'is', 'adaptive']:
            print(f"\n{'='*60}")
            print(f"Testing {algo_name.upper()} Algorithm")
            print(f"{'='*60}")
            
            result = await manager.execute_order(
                order=order,
                algorithm=algo_name,
                market_data=market_data,
                config={'num_slices': 10, 'interval_seconds': 60}
            )
            
            print(f"Algorithm: {result['algorithm']}")
            print(f"Child Orders: {result['child_orders']}")
            print(f"Fill Rate: {result['fill_rate']:.2%}")
        
        print(f"\n{'='*60}")
        print("All execution algorithms tested successfully!")
    
    asyncio.run(test_algorithms())
