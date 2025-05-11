"""
Configuration Module

This module provides configuration management for the Trading Gateway Service.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Type

from pydantic import BaseModel, Field, validator

from common_lib.config import (
    ServiceSpecificConfig,
    ConfigManager
)


# Define service-specific configuration
class TradingGatewayServiceConfig(ServiceSpecificConfig):
    """
    Service-specific configuration for the Trading Gateway Service.
    
    This class defines the service-specific configuration parameters.
    """
    
    # Service configuration
    api_prefix: str = Field("/api/v1", description="API prefix")
    cors_origins: List[str] = Field(["*"], description="CORS origins")
    max_workers: int = Field(4, description="Maximum number of worker threads")
    cache_size: int = Field(1000, description="Maximum number of items in the cache")
    max_requests_per_minute: int = Field(60, description="Maximum number of API requests per minute")
    max_retries: int = Field(3, description="Maximum number of retries for failed requests")
    retry_delay_seconds: int = Field(5, description="Delay between retries in seconds")
    timeout_seconds: int = Field(30, description="Timeout for API requests in seconds")
    
    # Kafka configuration
    kafka_bootstrap_servers: str = Field("localhost:9092", description="Comma-separated list of Kafka broker addresses")
    kafka_consumer_group_prefix: str = Field("trading-gateway", description="Prefix for Kafka consumer groups")
    kafka_auto_create_topics: bool = Field(True, description="Whether to automatically create Kafka topics")
    kafka_producer_acks: str = Field("all", description="Kafka producer acknowledgment setting")
    
    # Order management configuration
    order_max_orders_per_user: int = Field(100, description="Maximum number of orders per user")
    order_max_open_positions: int = Field(50, description="Maximum number of open positions")
    order_max_order_value: float = Field(100000.0, description="Maximum order value")
    order_min_order_value: float = Field(10.0, description="Minimum order value")
    order_default_order_type: str = Field("market", description="Default order type")
    order_default_time_in_force: str = Field("gtc", description="Default time in force")
    order_enable_stop_loss: bool = Field(True, description="Whether to enable stop loss")
    order_enable_take_profit: bool = Field(True, description="Whether to enable take profit")
    order_enable_trailing_stop: bool = Field(True, description="Whether to enable trailing stop")
    
    # Risk management configuration
    risk_max_leverage: float = Field(10.0, description="Maximum leverage")
    risk_margin_call_level: float = Field(0.8, description="Margin call level")
    risk_liquidation_level: float = Field(0.5, description="Liquidation level")
    risk_max_drawdown_percent: float = Field(20.0, description="Maximum drawdown percentage")
    risk_max_daily_loss_percent: float = Field(5.0, description="Maximum daily loss percentage")
    risk_max_position_size_percent: float = Field(10.0, description="Maximum position size percentage")
    
    # Broker configuration
    broker_default_broker: str = Field("interactive_brokers", description="Default broker")
    broker_interactive_brokers_host: str = Field("localhost", description="Interactive Brokers host")
    broker_interactive_brokers_port: int = Field(7496, description="Interactive Brokers port")
    broker_interactive_brokers_client_id: int = Field(1, description="Interactive Brokers client ID")
    broker_interactive_brokers_account_id: str = Field("", description="Interactive Brokers account ID")
    broker_interactive_brokers_paper_trading: bool = Field(True, description="Whether to use paper trading")
    broker_interactive_brokers_timeout_seconds: int = Field(30, description="Interactive Brokers timeout in seconds")
    broker_oanda_api_key: str = Field("", description="OANDA API key")
    broker_oanda_account_id: str = Field("", description="OANDA account ID")
    broker_oanda_environment: str = Field("practice", description="OANDA environment")
    broker_oanda_timeout_seconds: int = Field(30, description="OANDA timeout in seconds")
    broker_fxcm_api_key: str = Field("", description="FXCM API key")
    broker_fxcm_account_id: str = Field("", description="FXCM account ID")
    broker_fxcm_environment: str = Field("demo", description="FXCM environment")
    broker_fxcm_timeout_seconds: int = Field(30, description="FXCM timeout in seconds")
    
    # Simulation configuration
    simulation_enable_simulation: bool = Field(True, description="Whether to enable simulation")
    simulation_initial_balance: float = Field(10000.0, description="Initial balance for simulation")
    simulation_commission_rate: float = Field(0.001, description="Commission rate for simulation")
    simulation_slippage_model: str = Field("random", description="Slippage model for simulation")
    simulation_max_slippage_pips: int = Field(2, description="Maximum slippage in pips for simulation")
    
    # Authentication configuration
    auth_jwt_secret: str = Field("your-secret-key", description="JWT secret key")
    auth_jwt_expiration_minutes: int = Field(60, description="JWT expiration in minutes")
    auth_refresh_token_expiration_days: int = Field(7, description="Refresh token expiration in days")
    
    # Rate limiting configuration
    rate_max_requests_per_minute: int = Field(60, description="Maximum number of requests per minute")
    rate_max_orders_per_minute: int = Field(10, description="Maximum number of orders per minute")
    rate_max_cancellations_per_minute: int = Field(20, description="Maximum number of cancellations per minute")
    
    @validator("max_workers")
    def validate_max_workers(cls, v):
        """Validate maximum number of workers."""
        if v < 1:
            raise ValueError("Maximum workers must be at least 1")
        return v
    
    @validator("cache_size")
    def validate_cache_size(cls, v):
        """Validate cache size."""
        if v < 0:
            raise ValueError("Cache size must be non-negative")
        return v
    
    @validator("max_requests_per_minute")
    def validate_max_requests_per_minute(cls, v):
        """Validate maximum number of API requests per minute."""
        if v < 1:
            raise ValueError("Maximum requests per minute must be at least 1")
        return v
    
    @validator("max_retries")
    def validate_max_retries(cls, v):
        """Validate maximum number of retries."""
        if v < 0:
            raise ValueError("Maximum retries must be non-negative")
        return v
    
    @validator("retry_delay_seconds")
    def validate_retry_delay_seconds(cls, v):
        """Validate retry delay."""
        if v < 0:
            raise ValueError("Retry delay must be non-negative")
        return v
    
    @validator("timeout_seconds")
    def validate_timeout_seconds(cls, v):
        """Validate timeout."""
        if v < 0:
            raise ValueError("Timeout must be non-negative")
        return v
    
    @validator("order_max_orders_per_user")
    def validate_order_max_orders_per_user(cls, v):
        """Validate maximum number of orders per user."""
        if v < 1:
            raise ValueError("Maximum orders per user must be at least 1")
        return v
    
    @validator("order_max_open_positions")
    def validate_order_max_open_positions(cls, v):
        """Validate maximum number of open positions."""
        if v < 1:
            raise ValueError("Maximum open positions must be at least 1")
        return v
    
    @validator("order_max_order_value")
    def validate_order_max_order_value(cls, v):
        """Validate maximum order value."""
        if v <= 0:
            raise ValueError("Maximum order value must be positive")
        return v
    
    @validator("order_min_order_value")
    def validate_order_min_order_value(cls, v):
        """Validate minimum order value."""
        if v <= 0:
            raise ValueError("Minimum order value must be positive")
        return v
    
    @validator("risk_max_leverage")
    def validate_risk_max_leverage(cls, v):
        """Validate maximum leverage."""
        if v <= 0:
            raise ValueError("Maximum leverage must be positive")
        return v
    
    @validator("risk_margin_call_level")
    def validate_risk_margin_call_level(cls, v):
        """Validate margin call level."""
        if v <= 0 or v > 1:
            raise ValueError("Margin call level must be between 0 and 1")
        return v
    
    @validator("risk_liquidation_level")
    def validate_risk_liquidation_level(cls, v):
        """Validate liquidation level."""
        if v <= 0 or v > 1:
            raise ValueError("Liquidation level must be between 0 and 1")
        return v
    
    @validator("risk_max_drawdown_percent")
    def validate_risk_max_drawdown_percent(cls, v):
        """Validate maximum drawdown percentage."""
        if v <= 0 or v > 100:
            raise ValueError("Maximum drawdown percentage must be between 0 and 100")
        return v
    
    @validator("risk_max_daily_loss_percent")
    def validate_risk_max_daily_loss_percent(cls, v):
        """Validate maximum daily loss percentage."""
        if v <= 0 or v > 100:
            raise ValueError("Maximum daily loss percentage must be between 0 and 100")
        return v
    
    @validator("risk_max_position_size_percent")
    def validate_risk_max_position_size_percent(cls, v):
        """Validate maximum position size percentage."""
        if v <= 0 or v > 100:
            raise ValueError("Maximum position size percentage must be between 0 and 100")
        return v
    
    @validator("broker_interactive_brokers_port")
    def validate_broker_interactive_brokers_port(cls, v):
        """Validate Interactive Brokers port."""
        if v < 0 or v > 65535:
            raise ValueError("Interactive Brokers port must be between 0 and 65535")
        return v
    
    @validator("simulation_initial_balance")
    def validate_simulation_initial_balance(cls, v):
        """Validate initial balance for simulation."""
        if v <= 0:
            raise ValueError("Initial balance for simulation must be positive")
        return v
    
    @validator("simulation_commission_rate")
    def validate_simulation_commission_rate(cls, v):
        """Validate commission rate for simulation."""
        if v < 0 or v > 1:
            raise ValueError("Commission rate for simulation must be between 0 and 1")
        return v
    
    @validator("simulation_max_slippage_pips")
    def validate_simulation_max_slippage_pips(cls, v):
        """Validate maximum slippage in pips for simulation."""
        if v < 0:
            raise ValueError("Maximum slippage in pips for simulation must be non-negative")
        return v
    
    @validator("auth_jwt_expiration_minutes")
    def validate_auth_jwt_expiration_minutes(cls, v):
        """Validate JWT expiration in minutes."""
        if v <= 0:
            raise ValueError("JWT expiration in minutes must be positive")
        return v
    
    @validator("auth_refresh_token_expiration_days")
    def validate_auth_refresh_token_expiration_days(cls, v):
        """Validate refresh token expiration in days."""
        if v <= 0:
            raise ValueError("Refresh token expiration in days must be positive")
        return v
    
    @validator("rate_max_requests_per_minute")
    def validate_rate_max_requests_per_minute(cls, v):
        """Validate maximum number of requests per minute."""
        if v <= 0:
            raise ValueError("Maximum number of requests per minute must be positive")
        return v
    
    @validator("rate_max_orders_per_minute")
    def validate_rate_max_orders_per_minute(cls, v):
        """Validate maximum number of orders per minute."""
        if v <= 0:
            raise ValueError("Maximum number of orders per minute must be positive")
        return v
    
    @validator("rate_max_cancellations_per_minute")
    def validate_rate_max_cancellations_per_minute(cls, v):
        """Validate maximum number of cancellations per minute."""
        if v <= 0:
            raise ValueError("Maximum number of cancellations per minute must be positive")
        return v


# Create a singleton ConfigManager instance
config_manager = ConfigManager(
    config_path=os.environ.get("CONFIG_PATH", "config/config.yaml"),
    service_specific_model=TradingGatewayServiceConfig,
    env_prefix=os.environ.get("CONFIG_ENV_PREFIX", "TRADING_GATEWAY_"),
    default_config_path=os.environ.get("DEFAULT_CONFIG_PATH", "trading_gateway_service/config/default/config.yaml")
)


# Helper functions to access configuration
def get_service_config() -> TradingGatewayServiceConfig:
    """
    Get the service-specific configuration.
    
    Returns:
        Service-specific configuration
    """
    return config_manager.get_service_specific_config()


def get_database_config():
    """
    Get the database configuration.
    
    Returns:
        Database configuration
    """
    return config_manager.get_database_config()


def get_logging_config():
    """
    Get the logging configuration.
    
    Returns:
        Logging configuration
    """
    return config_manager.get_logging_config()


def get_service_client_config(service_name: str):
    """
    Get the configuration for a specific service client.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Service client configuration
    """
    return config_manager.get_service_client_config(service_name)


def get_trading_gateway_config():
    """
    Get the trading gateway configuration.
    
    Returns:
        Trading gateway configuration
    """
    service_config = get_service_config()
    return {
        "order_management": {
            "max_orders_per_user": service_config.order_max_orders_per_user,
            "max_open_positions": service_config.order_max_open_positions,
            "max_order_value": service_config.order_max_order_value,
            "min_order_value": service_config.order_min_order_value,
            "default_order_type": service_config.order_default_order_type,
            "default_time_in_force": service_config.order_default_time_in_force,
            "enable_stop_loss": service_config.order_enable_stop_loss,
            "enable_take_profit": service_config.order_enable_take_profit,
            "enable_trailing_stop": service_config.order_enable_trailing_stop
        },
        "risk_management": {
            "max_leverage": service_config.risk_max_leverage,
            "margin_call_level": service_config.risk_margin_call_level,
            "liquidation_level": service_config.risk_liquidation_level,
            "max_drawdown_percent": service_config.risk_max_drawdown_percent,
            "max_daily_loss_percent": service_config.risk_max_daily_loss_percent,
            "max_position_size_percent": service_config.risk_max_position_size_percent
        },
        "brokers": {
            "default_broker": service_config.broker_default_broker,
            "interactive_brokers": {
                "host": service_config.broker_interactive_brokers_host,
                "port": service_config.broker_interactive_brokers_port,
                "client_id": service_config.broker_interactive_brokers_client_id,
                "account_id": service_config.broker_interactive_brokers_account_id,
                "paper_trading": service_config.broker_interactive_brokers_paper_trading,
                "timeout_seconds": service_config.broker_interactive_brokers_timeout_seconds
            },
            "oanda": {
                "api_key": service_config.broker_oanda_api_key,
                "account_id": service_config.broker_oanda_account_id,
                "environment": service_config.broker_oanda_environment,
                "timeout_seconds": service_config.broker_oanda_timeout_seconds
            },
            "fxcm": {
                "api_key": service_config.broker_fxcm_api_key,
                "account_id": service_config.broker_fxcm_account_id,
                "environment": service_config.broker_fxcm_environment,
                "timeout_seconds": service_config.broker_fxcm_timeout_seconds
            }
        },
        "simulation": {
            "enable_simulation": service_config.simulation_enable_simulation,
            "initial_balance": service_config.simulation_initial_balance,
            "commission_rate": service_config.simulation_commission_rate,
            "slippage_model": service_config.simulation_slippage_model,
            "max_slippage_pips": service_config.simulation_max_slippage_pips
        },
        "authentication": {
            "jwt_secret": service_config.auth_jwt_secret,
            "jwt_expiration_minutes": service_config.auth_jwt_expiration_minutes,
            "refresh_token_expiration_days": service_config.auth_refresh_token_expiration_days
        },
        "rate_limiting": {
            "max_requests_per_minute": service_config.rate_max_requests_per_minute,
            "max_orders_per_minute": service_config.rate_max_orders_per_minute,
            "max_cancellations_per_minute": service_config.rate_max_cancellations_per_minute
        }
    }


def is_development() -> bool:
    """
    Check if the application is running in development mode.
    
    Returns:
        True if the application is running in development mode, False otherwise
    """
    return config_manager.is_development()


def is_testing() -> bool:
    """
    Check if the application is running in testing mode.
    
    Returns:
        True if the application is running in testing mode, False otherwise
    """
    return config_manager.is_testing()


def is_production() -> bool:
    """
    Check if the application is running in production mode.
    
    Returns:
        True if the application is running in production mode, False otherwise
    """
    return config_manager.is_production()
