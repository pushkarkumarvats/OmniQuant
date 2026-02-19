"""
Production Configuration Management
Handles all system configuration with validation and environment-based settings
"""

from typing import Optional, List
from pydantic import BaseSettings, Field, validator, SecretStr
from pathlib import Path
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseConfig(BaseSettings):
    host: str = Field("localhost", env="DB_HOST")
    port: int = Field(5432, env="DB_PORT")
    database: str = Field("omniquant", env="DB_NAME")
    username: str = Field("postgres", env="DB_USER")
    password: SecretStr = Field(..., env="DB_PASSWORD")
    pool_size: int = Field(10, env="DB_POOL_SIZE")
    max_overflow: int = Field(20, env="DB_MAX_OVERFLOW")
    
    @property
    def connection_url(self) -> str:
        return f"postgresql://{self.username}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "DB_"


class RedisConfig(BaseSettings):
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    password: Optional[SecretStr] = Field(None, env="REDIS_PASSWORD")
    db: int = Field(0, env="REDIS_DB")
    max_connections: int = Field(50, env="REDIS_MAX_CONNECTIONS")
    
    @property
    def connection_url(self) -> str:
        if self.password:
            return f"redis://:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"


class BrokerConfig(BaseSettings):
    # Alpaca
    alpaca_api_key: Optional[SecretStr] = Field(None, env="ALPACA_API_KEY")
    alpaca_secret_key: Optional[SecretStr] = Field(None, env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field("https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    
    # Interactive Brokers
    ib_host: str = Field("127.0.0.1", env="IB_HOST")
    ib_port: int = Field(7497, env="IB_PORT")
    ib_client_id: int = Field(1, env="IB_CLIENT_ID")
    
    # Polygon
    polygon_api_key: Optional[SecretStr] = Field(None, env="POLYGON_API_KEY")
    
    class Config:
        env_prefix = "BROKER_"


class TradingConfig(BaseSettings):
    max_position_size: float = Field(100000.0, env="MAX_POSITION_SIZE")
    max_leverage: float = Field(2.0, env="MAX_LEVERAGE")
    max_drawdown_pct: float = Field(0.20, env="MAX_DRAWDOWN_PCT")
    commission_rate: float = Field(0.0002, env="COMMISSION_RATE")
    slippage_bps: float = Field(1.0, env="SLIPPAGE_BPS")
    risk_free_rate: float = Field(0.02, env="RISK_FREE_RATE")
    
    # Order execution
    order_timeout_seconds: int = Field(30, env="ORDER_TIMEOUT_SECONDS")
    max_retry_attempts: int = Field(3, env="MAX_RETRY_ATTEMPTS")
    
    # Risk limits
    daily_loss_limit: float = Field(5000.0, env="DAILY_LOSS_LIMIT")
    position_concentration_limit: float = Field(0.25, env="POSITION_CONCENTRATION_LIMIT")
    
    class Config:
        env_prefix = "TRADING_"


class SecurityConfig(BaseSettings):
    secret_key: SecretStr = Field(..., env="SECRET_KEY")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # API rate limiting
    rate_limit_per_minute: int = Field(100, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(20, env="RATE_LIMIT_BURST")
    
    # CORS
    allowed_origins: List[str] = Field(["http://localhost:3000"], env="ALLOWED_ORIGINS")
    
    class Config:
        env_prefix = "SECURITY_"


class MessagingConfig(BaseSettings):
    backend: str = Field("in_memory", env="MESSAGING_BACKEND")  # aeron | kafka | redpanda | in_memory
    kafka_bootstrap: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP")
    serialization_format: str = Field("msgpack", env="SERIALIZATION_FORMAT")  # json | msgpack | flatbuffer | protobuf

    class Config:
        env_prefix = "MESSAGING_"


class TimeSeriesDBConfig(BaseSettings):
    backend: str = Field("duckdb", env="TSDB_BACKEND")  # duckdb | clickhouse | arcticdb
    duckdb_path: str = Field("data/timeseries.duckdb", env="TSDB_DUCKDB_PATH")
    clickhouse_host: str = Field("localhost", env="TSDB_CLICKHOUSE_HOST")
    clickhouse_port: int = Field(9000, env="TSDB_CLICKHOUSE_PORT")
    clickhouse_database: str = Field("hrt", env="TSDB_CLICKHOUSE_DB")
    arcticdb_uri: str = Field("lmdb://data/arcticdb", env="TSDB_ARCTICDB_URI")

    class Config:
        env_prefix = "TSDB_"


class FeatureStoreConfig(BaseSettings):
    offline_db_path: str = Field("data/feature_store.duckdb", env="FEATURE_STORE_PATH")
    online_ttl_seconds: int = Field(3600, env="FEATURE_ONLINE_TTL")

    class Config:
        env_prefix = "FEATURE_"


class RiskConfig(BaseSettings):
    max_position_qty: int = Field(100_000, env="RISK_MAX_POS_QTY")
    max_position_notional: float = Field(10_000_000, env="RISK_MAX_POS_NOTIONAL")
    max_order_qty: int = Field(10_000, env="RISK_MAX_ORDER_QTY")
    max_order_notional: float = Field(1_000_000, env="RISK_MAX_ORDER_NOTIONAL")
    max_daily_loss: float = Field(500_000, env="RISK_MAX_DAILY_LOSS")
    max_drawdown_pct: float = Field(0.05, env="RISK_MAX_DRAWDOWN")
    max_orders_per_second: int = Field(100, env="RISK_MAX_OPS")
    max_gross_leverage: float = Field(4.0, env="RISK_MAX_GROSS_LEV")
    fat_finger_deviation: float = Field(0.10, env="RISK_FAT_FINGER_DEV")

    class Config:
        env_prefix = "RISK_"


class OMSConfig(BaseSettings):
    use_native: bool = Field(False, env="OMS_USE_NATIVE")
    native_lib_path: Optional[str] = Field(None, env="OMS_NATIVE_LIB")

    class Config:
        env_prefix = "OMS_"


class DashboardConfig(BaseSettings):
    host: str = Field("0.0.0.0", env="DASHBOARD_HOST")
    port: int = Field(8080, env="DASHBOARD_PORT")
    ws_position_hz: int = Field(2, env="DASHBOARD_WS_POS_HZ")
    ws_pnl_hz: int = Field(4, env="DASHBOARD_WS_PNL_HZ")

    class Config:
        env_prefix = "DASHBOARD_"


class MonitoringConfig(BaseSettings):
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    log_file: Optional[Path] = Field(None, env="LOG_FILE")
    
    # Metrics
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
    
    # Alerting
    alert_email: Optional[str] = Field(None, env="ALERT_EMAIL")
    slack_webhook_url: Optional[SecretStr] = Field(None, env="SLACK_WEBHOOK_URL")
    pagerduty_api_key: Optional[SecretStr] = Field(None, env="PAGERDUTY_API_KEY")
    
    # Health checks
    health_check_interval_seconds: int = Field(60, env="HEALTH_CHECK_INTERVAL")
    
    class Config:
        env_prefix = "MONITORING_"


class APIConfig(BaseSettings):
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    workers: int = Field(4, env="API_WORKERS")
    timeout: int = Field(60, env="API_TIMEOUT")
    max_connections: int = Field(1000, env="API_MAX_CONNECTIONS")
    
    # WebSocket
    websocket_ping_interval: int = Field(20, env="WS_PING_INTERVAL")
    websocket_ping_timeout: int = Field(20, env="WS_PING_TIMEOUT")
    
    class Config:
        env_prefix = "API_"


class Config(BaseSettings):
    # Environment
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # Application
    app_name: str = Field("OmniQuant", env="APP_NAME")
    app_version: str = Field("2.0.0", env="APP_VERSION")
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Phase 2.0 sub-configurations
    messaging: MessagingConfig = Field(default_factory=MessagingConfig)
    timeseries_db: TimeSeriesDBConfig = Field(default_factory=TimeSeriesDBConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    oms: OMSConfig = Field(default_factory=OMSConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    def validate_production_config(self) -> List[str]:
        """Check production-required settings and return any violations."""
        errors = []
        
        if self.is_production:
            # Database password required
            if not self.database.password:
                errors.append("Database password required in production")
            
            # Secret key required
            if not self.security.secret_key:
                errors.append("Secret key required in production")
            
            # Debug should be off
            if self.debug:
                errors.append("Debug mode should be disabled in production")
            
            # Broker credentials
            if not self.broker.alpaca_api_key and not self.broker.ib_host:
                errors.append("At least one broker connection required")
            
            # Monitoring
            if not self.monitoring.alert_email and not self.monitoring.slack_webhook_url:
                errors.append("Alert notification method required in production")
        
        return errors
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
        
        # Validate production config
        if _config.is_production:
            errors = _config.validate_production_config()
            if errors:
                raise ValueError(f"Production configuration errors: {', '.join(errors)}")
    
    return _config


def reload_config():
    global _config
    _config = None
    return get_config()


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(f"Environment: {config.environment}")
    print(f"Database URL: {config.database.connection_url}")
    print(f"Redis URL: {config.redis.connection_url}")
    print(f"Is Production: {config.is_production}")
    
    # Validate
    if config.is_production:
        errors = config.validate_production_config()
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
