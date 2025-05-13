"""
Service Settings Module

This module provides standardized service-specific settings classes for all services
in the forex trading platform. Each service has its own settings class that extends
the BaseAppSettings class with service-specific configuration.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import Field, SecretStr, field_validator, computed_field

from common_lib.config.standardized_config import BaseAppSettings


class AnalysisEngineSettings(BaseAppSettings):
    """
    Analysis Engine Service settings.
    
    This class extends the base application settings with Analysis Engine Service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("analysis-engine", description="Name of the service")
    
    # Analysis Engine specific settings
    ENABLE_PREDICTIVE_ANALYTICS: bool = Field(
        True,
        description="Enable predictive analytics features"
    )
    MAX_PARALLEL_ANALYSIS: int = Field(
        8,
        description="Maximum number of parallel analysis tasks"
    )
    DEFAULT_TIMEFRAME: str = Field(
        "1h",
        description="Default timeframe for analysis"
    )
    INDICATOR_CACHE_TTL: int = Field(
        3600,
        description="Indicator cache TTL in seconds"
    )
    PATTERN_DETECTION_SENSITIVITY: float = Field(
        0.7,
        description="Pattern detection sensitivity (0.0-1.0)"
    )
    ENABLE_GPU_ACCELERATION: bool = Field(
        False,
        description="Enable GPU acceleration for computations"
    )
    GPU_MEMORY_LIMIT_MB: int = Field(
        1024,
        description="GPU memory limit in MB"
    )
    MODEL_DIRECTORY: str = Field(
        "data/models",
        description="Directory for ML models"
    )
    DATA_DIRECTORY: str = Field(
        "data",
        description="Directory for data files"
    )
    
    @field_validator("PATTERN_DETECTION_SENSITIVITY")
    def validate_pattern_detection_sensitivity(cls, v: float) -> float:
        """Validate pattern detection sensitivity."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Pattern detection sensitivity must be between 0.0 and 1.0")
        return v


class DataPipelineSettings(BaseAppSettings):
    """
    Data Pipeline Service settings.
    
    This class extends the base application settings with Data Pipeline Service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("data-pipeline", description="Name of the service")
    
    # Data Pipeline specific settings
    DATA_SOURCES: List[str] = Field(
        ["alpha_vantage", "yahoo_finance", "oanda"],
        description="Enabled data sources"
    )
    POLLING_INTERVAL_SECONDS: int = Field(
        60,
        description="Data polling interval in seconds"
    )
    MAX_HISTORICAL_DAYS: int = Field(
        365,
        description="Maximum historical days to fetch"
    )
    BATCH_SIZE: int = Field(
        100,
        description="Batch size for data processing"
    )
    DATA_STORAGE_PATH: str = Field(
        "data/market_data",
        description="Path for storing market data"
    )
    ENABLE_DATA_VALIDATION: bool = Field(
        True,
        description="Enable data validation"
    )
    ENABLE_DATA_NORMALIZATION: bool = Field(
        True,
        description="Enable data normalization"
    )
    ALPHA_VANTAGE_API_KEY: SecretStr = Field(
        "",
        description="Alpha Vantage API key"
    )
    YAHOO_FINANCE_API_KEY: SecretStr = Field(
        "",
        description="Yahoo Finance API key"
    )
    OANDA_API_KEY: SecretStr = Field(
        "",
        description="OANDA API key"
    )


class FeatureStoreSettings(BaseAppSettings):
    """
    Feature Store Service settings.
    
    This class extends the base application settings with Feature Store Service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("feature-store", description="Name of the service")
    
    # Feature Store specific settings
    FEATURE_REGISTRY_PATH: str = Field(
        "data/feature_registry",
        description="Path for feature registry"
    )
    ENABLE_FEATURE_VERSIONING: bool = Field(
        True,
        description="Enable feature versioning"
    )
    MAX_FEATURE_VERSIONS: int = Field(
        10,
        description="Maximum number of feature versions to keep"
    )
    FEATURE_CACHE_TTL: int = Field(
        3600,
        description="Feature cache TTL in seconds"
    )
    MAX_PARALLEL_COMPUTATIONS: int = Field(
        4,
        description="Maximum number of parallel feature computations"
    )
    ENABLE_ONLINE_SERVING: bool = Field(
        True,
        description="Enable online feature serving"
    )
    ENABLE_OFFLINE_SERVING: bool = Field(
        True,
        description="Enable offline feature serving"
    )
    FEATURE_STORE_TYPE: str = Field(
        "local",
        description="Feature store type (local, redis, s3)"
    )
    
    @field_validator("MAX_PARALLEL_COMPUTATIONS")
    def validate_max_parallel_computations(cls, v: int) -> int:
        """Validate maximum number of parallel computations."""
        if v < 1:
            raise ValueError("Maximum parallel computations must be at least 1")
        return v


class MLIntegrationSettings(BaseAppSettings):
    """
    ML Integration Service settings.
    
    This class extends the base application settings with ML Integration Service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("ml-integration", description="Name of the service")
    
    # ML Integration specific settings
    MODEL_REGISTRY_PATH: str = Field(
        "data/model_registry",
        description="Path for model registry"
    )
    ENABLE_MODEL_VERSIONING: bool = Field(
        True,
        description="Enable model versioning"
    )
    MAX_MODEL_VERSIONS: int = Field(
        5,
        description="Maximum number of model versions to keep"
    )
    MODEL_SERVING_TIMEOUT: int = Field(
        30,
        description="Model serving timeout in seconds"
    )
    ENABLE_MODEL_MONITORING: bool = Field(
        True,
        description="Enable model monitoring"
    )
    MONITORING_INTERVAL_SECONDS: int = Field(
        300,
        description="Model monitoring interval in seconds"
    )
    DEFAULT_MODEL_FRAMEWORK: str = Field(
        "sklearn",
        description="Default model framework (sklearn, tensorflow, pytorch)"
    )
    ENABLE_BATCH_PREDICTIONS: bool = Field(
        True,
        description="Enable batch predictions"
    )
    BATCH_SIZE: int = Field(
        32,
        description="Batch size for predictions"
    )


class MLWorkbenchSettings(BaseAppSettings):
    """
    ML Workbench Service settings.
    
    This class extends the base application settings with ML Workbench Service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("ml-workbench", description="Name of the service")
    
    # ML Workbench specific settings
    NOTEBOOK_DIRECTORY: str = Field(
        "notebooks",
        description="Directory for Jupyter notebooks"
    )
    ENABLE_JUPYTER: bool = Field(
        True,
        description="Enable Jupyter notebook integration"
    )
    JUPYTER_PORT: int = Field(
        8888,
        description="Jupyter notebook port"
    )
    EXPERIMENT_TRACKING_BACKEND: str = Field(
        "mlflow",
        description="Experiment tracking backend (mlflow, tensorboard)"
    )
    MLFLOW_TRACKING_URI: str = Field(
        "http://localhost:5000",
        description="MLflow tracking URI"
    )
    ENABLE_EXPERIMENT_SHARING: bool = Field(
        True,
        description="Enable experiment sharing"
    )
    MAX_EXPERIMENT_HISTORY: int = Field(
        100,
        description="Maximum number of experiment runs to keep"
    )
    ENABLE_GPU_SUPPORT: bool = Field(
        False,
        description="Enable GPU support for ML workbench"
    )
    DEFAULT_PYTHON_ENVIRONMENT: str = Field(
        "python3",
        description="Default Python environment"
    )


class MonitoringAlertingSettings(BaseAppSettings):
    """
    Monitoring & Alerting Service settings.
    
    This class extends the base application settings with Monitoring & Alerting Service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("monitoring-alerting", description="Name of the service")
    
    # Monitoring & Alerting specific settings
    PROMETHEUS_ENDPOINT: str = Field(
        "http://localhost:9090",
        description="Prometheus endpoint"
    )
    GRAFANA_ENDPOINT: str = Field(
        "http://localhost:3000",
        description="Grafana endpoint"
    )
    ALERT_CHANNELS: List[str] = Field(
        ["email", "slack"],
        description="Enabled alert channels"
    )
    EMAIL_SMTP_SERVER: str = Field(
        "smtp.example.com",
        description="SMTP server for email alerts"
    )
    EMAIL_SMTP_PORT: int = Field(
        587,
        description="SMTP port for email alerts"
    )
    EMAIL_USERNAME: str = Field(
        "",
        description="SMTP username for email alerts"
    )
    EMAIL_PASSWORD: SecretStr = Field(
        "",
        description="SMTP password for email alerts"
    )
    SLACK_WEBHOOK_URL: SecretStr = Field(
        "",
        description="Slack webhook URL for alerts"
    )
    ALERT_THROTTLING_SECONDS: int = Field(
        300,
        description="Alert throttling period in seconds"
    )
    ENABLE_ANOMALY_DETECTION: bool = Field(
        True,
        description="Enable anomaly detection for metrics"
    )
    SCRAPE_INTERVAL_SECONDS: int = Field(
        15,
        description="Metrics scrape interval in seconds"
    )


class PortfolioManagementSettings(BaseAppSettings):
    """
    Portfolio Management Service settings.
    
    This class extends the base application settings with Portfolio Management Service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("portfolio-management", description="Name of the service")
    
    # Portfolio Management specific settings
    DEFAULT_RISK_PERCENTAGE: float = Field(
        2.0,
        description="Default risk percentage per trade"
    )
    MAX_PORTFOLIO_RISK_PERCENTAGE: float = Field(
        20.0,
        description="Maximum portfolio risk percentage"
    )
    POSITION_SIZING_METHOD: str = Field(
        "risk_based",
        description="Position sizing method (fixed, risk_based, kelly)"
    )
    ENABLE_PORTFOLIO_REBALANCING: bool = Field(
        True,
        description="Enable automatic portfolio rebalancing"
    )
    REBALANCING_THRESHOLD_PERCENTAGE: float = Field(
        5.0,
        description="Rebalancing threshold percentage"
    )
    REBALANCING_INTERVAL_DAYS: int = Field(
        30,
        description="Rebalancing interval in days"
    )
    ENABLE_RISK_MANAGEMENT: bool = Field(
        True,
        description="Enable risk management features"
    )
    MAX_DRAWDOWN_PERCENTAGE: float = Field(
        25.0,
        description="Maximum drawdown percentage"
    )
    ENABLE_PERFORMANCE_TRACKING: bool = Field(
        True,
        description="Enable performance tracking"
    )
    
    @field_validator("DEFAULT_RISK_PERCENTAGE")
    def validate_default_risk_percentage(cls, v: float) -> float:
        """Validate default risk percentage."""
        if v < 0.1 or v > 10.0:
            raise ValueError("Default risk percentage must be between 0.1 and 10.0")
        return v


class StrategyExecutionSettings(BaseAppSettings):
    """
    Strategy Execution Engine settings.
    
    This class extends the base application settings with Strategy Execution Engine-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("strategy-execution", description="Name of the service")
    
    # Strategy Execution specific settings
    STRATEGY_DIRECTORY: str = Field(
        "strategies",
        description="Directory for strategy definitions"
    )
    EXECUTION_MODE: str = Field(
        "live",
        description="Execution mode (live, paper, backtest)"
    )
    MAX_CONCURRENT_STRATEGIES: int = Field(
        10,
        description="Maximum number of concurrent strategies"
    )
    STRATEGY_EXECUTION_INTERVAL: int = Field(
        60,
        description="Strategy execution interval in seconds"
    )
    ENABLE_STRATEGY_OPTIMIZATION: bool = Field(
        True,
        description="Enable strategy optimization"
    )
    OPTIMIZATION_METHOD: str = Field(
        "grid",
        description="Optimization method (grid, random, bayesian)"
    )
    ENABLE_WALK_FORWARD_TESTING: bool = Field(
        True,
        description="Enable walk-forward testing"
    )
    BACKTEST_DATA_SOURCE: str = Field(
        "local",
        description="Backtest data source (local, api)"
    )
    ENABLE_STRATEGY_VALIDATION: bool = Field(
        True,
        description="Enable strategy validation before execution"
    )


class TradingGatewaySettings(BaseAppSettings):
    """
    Trading Gateway Service settings.
    
    This class extends the base application settings with Trading Gateway Service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("trading-gateway", description="Name of the service")
    
    # Trading Gateway specific settings
    ENABLED_BROKERS: List[str] = Field(
        ["oanda", "interactive_brokers", "paper"],
        description="Enabled brokers"
    )
    DEFAULT_BROKER: str = Field(
        "paper",
        description="Default broker"
    )
    ORDER_TIMEOUT_SECONDS: int = Field(
        30,
        description="Order execution timeout in seconds"
    )
    ENABLE_ORDER_VALIDATION: bool = Field(
        True,
        description="Enable order validation"
    )
    MAX_ORDER_RETRIES: int = Field(
        3,
        description="Maximum number of order retries"
    )
    OANDA_API_KEY: SecretStr = Field(
        "",
        description="OANDA API key"
    )
    OANDA_ACCOUNT_ID: str = Field(
        "",
        description="OANDA account ID"
    )
    IB_TWS_HOST: str = Field(
        "localhost",
        description="Interactive Brokers TWS host"
    )
    IB_TWS_PORT: int = Field(
        7496,
        description="Interactive Brokers TWS port"
    )
    IB_CLIENT_ID: int = Field(
        1,
        description="Interactive Brokers client ID"
    )
    ENABLE_TRADING_HOURS_CHECK: bool = Field(
        True,
        description="Enable trading hours check"
    )
    TRADING_HOURS_TIMEZONE: str = Field(
        "UTC",
        description="Trading hours timezone"
    )


class UIServiceSettings(BaseAppSettings):
    """
    UI Service settings.
    
    This class extends the base application settings with UI Service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("ui", description="Name of the service")
    
    # UI Service specific settings
    STATIC_FILES_DIRECTORY: str = Field(
        "static",
        description="Directory for static files"
    )
    ENABLE_WEBSOCKETS: bool = Field(
        True,
        description="Enable WebSocket support"
    )
    WEBSOCKET_PING_INTERVAL: int = Field(
        30,
        description="WebSocket ping interval in seconds"
    )
    DEFAULT_THEME: str = Field(
        "light",
        description="Default UI theme (light, dark)"
    )
    ENABLE_CHART_CACHING: bool = Field(
        True,
        description="Enable chart caching"
    )
    CHART_CACHE_TTL: int = Field(
        300,
        description="Chart cache TTL in seconds"
    )
    DEFAULT_CHART_TIMEFRAME: str = Field(
        "1h",
        description="Default chart timeframe"
    )
    DEFAULT_INDICATORS: List[str] = Field(
        ["sma", "ema", "rsi"],
        description="Default chart indicators"
    )
    ENABLE_NOTIFICATIONS: bool = Field(
        True,
        description="Enable in-app notifications"
    )
    MAX_NOTIFICATION_HISTORY: int = Field(
        100,
        description="Maximum notification history"
    )
    SESSION_TIMEOUT_MINUTES: int = Field(
        60,
        description="Session timeout in minutes"
    )
