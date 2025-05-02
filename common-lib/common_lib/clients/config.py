"""
Client Configuration Module

This module provides centralized configuration for service clients.
It defines standard configurations for different services and environments.
"""

import os
import logging
from typing import Dict, Any

from common_lib.clients.base_client import ClientConfig
from common_lib.clients.client_factory import register_client_config

logger = logging.getLogger(__name__)

# Standard service configurations
SERVICE_CONFIGS = {
    # Analysis Engine Service
    "analysis-engine-service": {
        "base_url": os.environ.get("ANALYSIS_ENGINE_API_URL", "http://analysis-engine-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("ANALYSIS_ENGINE_TIMEOUT", "30.0")),
        "max_retries": int(os.environ.get("ANALYSIS_ENGINE_MAX_RETRIES", "3")),
        "circuit_breaker_failure_threshold": int(os.environ.get("ANALYSIS_ENGINE_CB_THRESHOLD", "5")),
        "circuit_breaker_reset_timeout_seconds": int(os.environ.get("ANALYSIS_ENGINE_CB_RESET", "60")),
        "bulkhead_max_concurrent": int(os.environ.get("ANALYSIS_ENGINE_MAX_CONCURRENT", "20")),
        "service_name": "analysis-engine-service"
    },
    
    # Data Pipeline Service
    "data-pipeline-service": {
        "base_url": os.environ.get("DATA_PIPELINE_API_URL", "http://data-pipeline-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("DATA_PIPELINE_TIMEOUT", "30.0")),
        "max_retries": int(os.environ.get("DATA_PIPELINE_MAX_RETRIES", "3")),
        "circuit_breaker_failure_threshold": int(os.environ.get("DATA_PIPELINE_CB_THRESHOLD", "5")),
        "circuit_breaker_reset_timeout_seconds": int(os.environ.get("DATA_PIPELINE_CB_RESET", "60")),
        "bulkhead_max_concurrent": int(os.environ.get("DATA_PIPELINE_MAX_CONCURRENT", "20")),
        "service_name": "data-pipeline-service"
    },
    
    # Feature Store Service
    "feature-store-service": {
        "base_url": os.environ.get("FEATURE_STORE_API_URL", "http://feature-store-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("FEATURE_STORE_TIMEOUT", "20.0")),
        "max_retries": int(os.environ.get("FEATURE_STORE_MAX_RETRIES", "3")),
        "circuit_breaker_failure_threshold": int(os.environ.get("FEATURE_STORE_CB_THRESHOLD", "5")),
        "circuit_breaker_reset_timeout_seconds": int(os.environ.get("FEATURE_STORE_CB_RESET", "60")),
        "bulkhead_max_concurrent": int(os.environ.get("FEATURE_STORE_MAX_CONCURRENT", "20")),
        "service_name": "feature-store-service"
    },
    
    # ML Integration Service
    "ml-integration-service": {
        "base_url": os.environ.get("ML_INTEGRATION_API_URL", "http://ml-integration-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("ML_INTEGRATION_TIMEOUT", "30.0")),
        "max_retries": int(os.environ.get("ML_INTEGRATION_MAX_RETRIES", "3")),
        "circuit_breaker_failure_threshold": int(os.environ.get("ML_INTEGRATION_CB_THRESHOLD", "5")),
        "circuit_breaker_reset_timeout_seconds": int(os.environ.get("ML_INTEGRATION_CB_RESET", "60")),
        "bulkhead_max_concurrent": int(os.environ.get("ML_INTEGRATION_MAX_CONCURRENT", "20")),
        "service_name": "ml-integration-service"
    },
    
    # ML Workbench Service
    "ml-workbench-service": {
        "base_url": os.environ.get("ML_WORKBENCH_API_URL", "http://ml-workbench-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("ML_WORKBENCH_TIMEOUT", "30.0")),
        "max_retries": int(os.environ.get("ML_WORKBENCH_MAX_RETRIES", "3")),
        "circuit_breaker_failure_threshold": int(os.environ.get("ML_WORKBENCH_CB_THRESHOLD", "5")),
        "circuit_breaker_reset_timeout_seconds": int(os.environ.get("ML_WORKBENCH_CB_RESET", "60")),
        "bulkhead_max_concurrent": int(os.environ.get("ML_WORKBENCH_MAX_CONCURRENT", "20")),
        "service_name": "ml-workbench-service"
    },
    
    # Monitoring & Alerting Service
    "monitoring-alerting-service": {
        "base_url": os.environ.get("MONITORING_ALERTING_API_URL", "http://monitoring-alerting-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("MONITORING_ALERTING_TIMEOUT", "10.0")),
        "max_retries": int(os.environ.get("MONITORING_ALERTING_MAX_RETRIES", "3")),
        "circuit_breaker_failure_threshold": int(os.environ.get("MONITORING_ALERTING_CB_THRESHOLD", "5")),
        "circuit_breaker_reset_timeout_seconds": int(os.environ.get("MONITORING_ALERTING_CB_RESET", "60")),
        "bulkhead_max_concurrent": int(os.environ.get("MONITORING_ALERTING_MAX_CONCURRENT", "20")),
        "service_name": "monitoring-alerting-service"
    },
    
    # Portfolio Management Service
    "portfolio-management-service": {
        "base_url": os.environ.get("PORTFOLIO_MANAGEMENT_API_URL", "http://portfolio-management-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("PORTFOLIO_MANAGEMENT_TIMEOUT", "20.0")),
        "max_retries": int(os.environ.get("PORTFOLIO_MANAGEMENT_MAX_RETRIES", "3")),
        "circuit_breaker_failure_threshold": int(os.environ.get("PORTFOLIO_MANAGEMENT_CB_THRESHOLD", "5")),
        "circuit_breaker_reset_timeout_seconds": int(os.environ.get("PORTFOLIO_MANAGEMENT_CB_RESET", "60")),
        "bulkhead_max_concurrent": int(os.environ.get("PORTFOLIO_MANAGEMENT_MAX_CONCURRENT", "20")),
        "service_name": "portfolio-management-service"
    },
    
    # Risk Management Service
    "risk-management-service": {
        "base_url": os.environ.get("RISK_MANAGEMENT_API_URL", "http://risk-management-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("RISK_MANAGEMENT_TIMEOUT", "15.0")),
        "max_retries": int(os.environ.get("RISK_MANAGEMENT_MAX_RETRIES", "3")),
        "circuit_breaker_failure_threshold": int(os.environ.get("RISK_MANAGEMENT_CB_THRESHOLD", "5")),
        "circuit_breaker_reset_timeout_seconds": int(os.environ.get("RISK_MANAGEMENT_CB_RESET", "60")),
        "bulkhead_max_concurrent": int(os.environ.get("RISK_MANAGEMENT_MAX_CONCURRENT", "20")),
        "service_name": "risk-management-service"
    },
    
    # Strategy Execution Engine
    "strategy-execution-engine": {
        "base_url": os.environ.get("STRATEGY_EXECUTION_API_URL", "http://strategy-execution-engine:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("STRATEGY_EXECUTION_TIMEOUT", "20.0")),
        "max_retries": int(os.environ.get("STRATEGY_EXECUTION_MAX_RETRIES", "3")),
        "circuit_breaker_failure_threshold": int(os.environ.get("STRATEGY_EXECUTION_CB_THRESHOLD", "5")),
        "circuit_breaker_reset_timeout_seconds": int(os.environ.get("STRATEGY_EXECUTION_CB_RESET", "60")),
        "bulkhead_max_concurrent": int(os.environ.get("STRATEGY_EXECUTION_MAX_CONCURRENT", "20")),
        "service_name": "strategy-execution-engine"
    },
    
    # Trading Gateway Service
    "trading-gateway-service": {
        "base_url": os.environ.get("TRADING_GATEWAY_API_URL", "http://trading-gateway-service:8000/api/v1"),
        "timeout_seconds": float(os.environ.get("TRADING_GATEWAY_TIMEOUT", "15.0")),
        "max_retries": int(os.environ.get("TRADING_GATEWAY_MAX_RETRIES", "3")),
        "circuit_breaker_failure_threshold": int(os.environ.get("TRADING_GATEWAY_CB_THRESHOLD", "5")),
        "circuit_breaker_reset_timeout_seconds": int(os.environ.get("TRADING_GATEWAY_CB_RESET", "60")),
        "bulkhead_max_concurrent": int(os.environ.get("TRADING_GATEWAY_MAX_CONCURRENT", "20")),
        "service_name": "trading-gateway-service"
    },
}


def register_default_configs() -> None:
    """Register default configurations for all services."""
    for service_name, config_dict in SERVICE_CONFIGS.items():
        try:
            config = ClientConfig(**config_dict)
            register_client_config(service_name, config)
            logger.debug(f"Registered default configuration for {service_name}")
        except Exception as e:
            logger.error(f"Error registering configuration for {service_name}: {str(e)}")


# Register default configurations on module import
register_default_configs()
