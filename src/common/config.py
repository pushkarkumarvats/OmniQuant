"""
Production Configuration Management
Handles all system configuration with validation and environment-based settings
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseSettings, Field, validator, SecretStr
from pathlib import Path
import os
from enum import Enum


class Environment(str, Enum):
    """Deployment environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    host: str = Field("localhost", env="DB_HOST")
    port: int = Field(5432, env="DB_PORT")
    database: str = Field("omniquant", env="DB_NAME")
    username: str = Field("postgres", env="DB_USER")
    password: SecretStr = Field(..., env="DB_PASSWORD")
    pool_size: int = Field(10, env="DB_POOL_SIZE")
    max_overflow: int = Field(20, env="DB_MAX_OVERFLOW")
    
    @property
    def connection_url(self) -> str:
        """Get SQLAlchemy connection URL"""
        return f"postgresql://{self.username}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "DB_"


class RedisConfig(BaseSettings):
    """Redis configuration"""
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    password: Optional[SecretStr] = Field(None, env="REDIS_PASSWORD")
    db: int = Field(0, env="REDIS_DB")
    max_connections: int = Field(50, env="REDIS_MAX_CONNECTIONS")
    
    @property
    def connection_url(self) -> str:
        """Get Redis connection URL"""
        if self.password:
            return f"redis://:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_prefix = "REDIS_"


class BrokerConfig(BaseSettings):
    """Broker API configuration"""
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
    """Trading system configuration"""
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
    """Security configuration"""
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


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration"""
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
    """API server configuration"""
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
    """Main application configuration"""
    
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
    
    @validator("environment", pre=True)
    def validate_environment(cls, v):
        """Validate environment setting"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT
    
    def validate_production_config(self) -> List[str]:
        """
        Validate production configuration
        Returns list of validation errors
        """
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
    """Get global configuration instance"""
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
    """Reload configuration from environment"""
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
