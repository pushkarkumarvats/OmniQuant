"""
Production Monitoring, Metrics, and Alerting
Prometheus metrics, health checks, and alert notifications
"""

from typing import Optional, Dict, Any, Callable
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from loguru import logger
import time
import functools
from datetime import datetime
import asyncio
import aiohttp


# Prometheus Metrics
# Trading Metrics
trades_total = Counter('trading_trades_total', 'Total number of trades', ['strategy', 'symbol', 'side'])
trades_pnl = Histogram('trading_pnl', 'Trade PnL distribution', ['strategy'])
position_value = Gauge('trading_position_value', 'Current position value', ['symbol'])
portfolio_equity = Gauge('trading_portfolio_equity', 'Total portfolio equity')
daily_pnl = Gauge('trading_daily_pnl', 'Daily PnL')

# Order Metrics
orders_submitted = Counter('trading_orders_submitted', 'Orders submitted', ['type', 'side'])
orders_filled = Counter('trading_orders_filled', 'Orders filled', ['type'])
orders_rejected = Counter('trading_orders_rejected', 'Orders rejected', ['reason'])
order_latency = Histogram('trading_order_latency_seconds', 'Order execution latency')

# Market Data Metrics
market_data_messages = Counter('market_data_messages_total', 'Market data messages received', ['source'])
market_data_lag = Gauge('market_data_lag_seconds', 'Market data lag', ['source'])

# System Metrics
api_requests = Counter('api_requests_total', 'API requests', ['method', 'endpoint', 'status'])
api_latency = Histogram('api_latency_seconds', 'API request latency', ['endpoint'])
event_processing_time = Histogram('event_processing_seconds', 'Event processing time', ['event_type'])
error_count = Counter('errors_total', 'Total errors', ['component', 'error_type'])

# Model Metrics
model_predictions = Counter('model_predictions_total', 'Model predictions', ['model_name'])
model_latency = Histogram('model_prediction_latency_seconds', 'Model prediction latency', ['model_name'])
model_accuracy = Gauge('model_accuracy', 'Model accuracy', ['model_name'])


class MetricsCollector:
    """Centralized metrics collection"""
    
    def __init__(self):
        self.start_time = time.time()
        self.prometheus_port = 9090
    
    def start_prometheus_server(self, port: int = 9090):
        try:
            start_http_server(port)
            self.prometheus_port = port
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def record_trade(self, strategy: str, symbol: str, side: str, pnl: float):
        trades_total.labels(strategy=strategy, symbol=symbol, side=side).inc()
        trades_pnl.labels(strategy=strategy).observe(pnl)
    
    def update_position(self, symbol: str, value: float):
        position_value.labels(symbol=symbol).set(value)
    
    def update_portfolio(self, equity: float, daily_pnl_value: float):
        portfolio_equity.set(equity)
        daily_pnl.set(daily_pnl_value)
    
    def record_order(self, order_type: str, side: str, status: str, latency: float):
        orders_submitted.labels(type=order_type, side=side).inc()
        if status == "filled":
            orders_filled.labels(type=order_type).inc()
        elif status == "rejected":
            orders_rejected.labels(reason="unknown").inc()
        order_latency.observe(latency)
    
    def record_market_data(self, source: str, lag_seconds: float = 0):
        market_data_messages.labels(source=source).inc()
        if lag_seconds > 0:
            market_data_lag.labels(source=source).set(lag_seconds)
    
    def record_error(self, component: str, error_type: str):
        error_count.labels(component=component, error_type=error_type).inc()


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_time(metric: Histogram):
    """Decorator to track function execution time"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metric.observe(duration)
        return wrapper
    return decorator


def track_async_time(metric: Histogram):
    """Decorator to track async function execution time"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                metric.observe(duration)
        return wrapper
    return decorator


class HealthCheck:
    """System health check"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_check_time: Optional[datetime] = None
        self.last_status: Dict[str, bool] = {}
    
    def register_check(self, name: str, check_func: Callable):
        self.checks[name] = check_func
        logger.debug(f"Registered health check: {name}")
    
    async def run_checks(self) -> Dict[str, Any]:
        """Execute all registered checks and return a status summary."""
        results = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    is_healthy = await check_func()
                else:
                    is_healthy = check_func()
                
                results["checks"][name] = {
                    "status": "pass" if is_healthy else "fail",
                    "healthy": is_healthy
                }
                self.last_status[name] = is_healthy
                
                if not is_healthy:
                    results["status"] = "degraded"
            
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results["checks"][name] = {
                    "status": "fail",
                    "healthy": False,
                    "error": str(e)
                }
                results["status"] = "unhealthy"
                self.last_status[name] = False
        
        self.last_check_time = datetime.now()
        return results
    
    def get_status(self) -> str:
        if not self.last_status:
            return "unknown"
        
        if all(self.last_status.values()):
            return "healthy"
        elif any(self.last_status.values()):
            return "degraded"
        else:
            return "unhealthy"


# Global health check
_health_check: Optional[HealthCheck] = None


def get_health_check() -> HealthCheck:
    global _health_check
    if _health_check is None:
        _health_check = HealthCheck()
    return _health_check


class AlertManager:
    """Alert notification manager"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.alert_history: list = []
        self.max_history = 100
    
    async def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Push an alert to all configured notification channels."""
        alert = {
            "level": level,
            "title": title,
            "message": message,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # Log alert
        log_func = getattr(logger, level, logger.info)
        log_func(f"ALERT: {title} - {message}")
        
        # Send to configured channels
        try:
            if "slack_webhook" in self.config:
                await self._send_slack(alert)
            
            if "email" in self.config:
                await self._send_email(alert)
            
            if "pagerduty" in self.config and level in ["error", "critical"]:
                await self._send_pagerduty(alert)
        
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    async def _send_slack(self, alert: Dict[str, Any]):
        webhook_url = self.config.get("slack_webhook")
        if not webhook_url:
            return
        
        color_map = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "error": "#ff0000",
            "critical": "#8b0000"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert["level"], "#808080"),
                "title": alert["title"],
                "text": alert["message"],
                "footer": "OmniQuant",
                "ts": int(time.time())
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Slack alert failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def _send_email(self, alert: Dict[str, Any]):
        # Implementation would use SMTP or email service API
        logger.debug(f"Email alert: {alert['title']}")
    
    async def _send_pagerduty(self, alert: Dict[str, Any]):
        # Implementation would use PagerDuty Events API
        logger.debug(f"PagerDuty alert: {alert['title']}")
    
    def get_recent_alerts(self, limit: int = 10) -> list:
        return self.alert_history[-limit:]


# Global alert manager
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# Predefined alert functions
async def alert_trading_error(error: Exception, context: Dict[str, Any]):
    manager = get_alert_manager()
    await manager.send_alert(
        level="error",
        title="Trading Error",
        message=f"Error during trading: {str(error)}",
        metadata=context
    )


async def alert_risk_breach(metric: str, value: float, limit: float):
    manager = get_alert_manager()
    await manager.send_alert(
        level="critical",
        title="Risk Limit Breach",
        message=f"{metric} = {value:.2f} exceeds limit of {limit:.2f}",
        metadata={"metric": metric, "value": value, "limit": limit}
    )


async def alert_system_degraded(component: str, reason: str):
    manager = get_alert_manager()
    await manager.send_alert(
        level="warning",
        title="System Degraded",
        message=f"Component {component} is degraded: {reason}",
        metadata={"component": component, "reason": reason}
    )


if __name__ == "__main__":
    # Test monitoring
    import asyncio
    
    async def test():
        # Start Prometheus
        collector = get_metrics_collector()
        collector.start_prometheus_server(9090)
        
        # Record some metrics
        collector.record_trade("momentum", "AAPL", "buy", 150.50)
        collector.update_portfolio(100000, 500)
        
        # Health check
        health = get_health_check()
        health.register_check("database", lambda: True)
        health.register_check("redis", lambda: True)
        status = await health.run_checks()
        print(f"Health: {status}")
        
        # Alert
        alerts = get_alert_manager()
        await alerts.send_alert("info", "Test Alert", "System is running")
        
        print("Monitoring system initialized")
    
    asyncio.run(test())
