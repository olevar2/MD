"""
Fixtures for service interaction tests.
"""

import os
import pytest
import asyncio
import logging
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock

from common_lib.config import (
    ConfigManager,
    DatabaseConfig,
    LoggingConfig,
    ServiceClientConfig,
    RetryConfig,
    CircuitBreakerConfig
)
from common_lib.service_client import ResilientServiceClient


@pytest.fixture
def config_manager():
    """Create a mock config manager."""
    mock_config_manager = MagicMock(spec=ConfigManager)
    
    # Mock database config
    mock_db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        username="postgres",
        password="postgres",
        database="test",
        min_connections=1,
        max_connections=10
    )
    mock_config_manager.get_database_config.return_value = mock_db_config
    
    # Mock logging config
    mock_logging_config = LoggingConfig(
        level="INFO",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        file=None
    )
    mock_config_manager.get_logging_config.return_value = mock_logging_config
    
    # Mock service client configs
    mock_service_client_configs = {
        "market_data_service": ServiceClientConfig(
            base_url="http://localhost:8001",
            timeout=30,
            retry=RetryConfig(max_retries=3, initial_backoff=1.0),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0)
        ),
        "data_pipeline_service": ServiceClientConfig(
            base_url="http://localhost:8002",
            timeout=30,
            retry=RetryConfig(max_retries=3, initial_backoff=1.0),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0)
        ),
        "feature_store_service": ServiceClientConfig(
            base_url="http://localhost:8003",
            timeout=30,
            retry=RetryConfig(max_retries=3, initial_backoff=1.0),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0)
        ),
        "analysis_engine_service": ServiceClientConfig(
            base_url="http://localhost:8004",
            timeout=30,
            retry=RetryConfig(max_retries=3, initial_backoff=1.0),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0)
        ),
        "trading_service": ServiceClientConfig(
            base_url="http://localhost:8005",
            timeout=30,
            retry=RetryConfig(max_retries=3, initial_backoff=1.0),
            circuit_breaker=CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0)
        )
    }
    
    def get_service_client_config(service_name):
        return mock_service_client_configs.get(service_name)
    
    mock_config_manager.get_service_client_config.side_effect = get_service_client_config
    
    return mock_config_manager


@pytest.fixture
def mock_service_client():
    """Create a mock service client."""
    mock_client = AsyncMock(spec=ResilientServiceClient)
    
    # Mock get method
    mock_client.get.return_value = {"status": "success"}
    
    # Mock post method
    mock_client.post.return_value = {"status": "success", "id": "123"}
    
    # Mock put method
    mock_client.put.return_value = {"status": "success"}
    
    # Mock delete method
    mock_client.delete.return_value = {"status": "success"}
    
    return mock_client


@pytest.fixture
def mock_service_clients(mock_service_client):
    """Create mock service clients."""
    return {
        "market_data_service": mock_service_client,
        "data_pipeline_service": mock_service_client,
        "feature_store_service": mock_service_client,
        "analysis_engine_service": mock_service_client,
        "trading_service": mock_service_client
    }


@pytest.fixture
def mock_database():
    """Create a mock database."""
    mock_db = AsyncMock()
    
    # Mock connect method
    mock_db.connect = AsyncMock()
    
    # Mock close method
    mock_db.close = AsyncMock()
    
    # Mock execute method
    mock_db.execute.return_value = "OK"
    
    # Mock fetch method
    mock_db.fetch.return_value = [{"id": 1, "name": "test"}]
    
    # Mock fetchrow method
    mock_db.fetchrow.return_value = {"id": 1, "name": "test"}
    
    # Mock fetchval method
    mock_db.fetchval.return_value = 1
    
    # Mock transaction method
    mock_transaction = AsyncMock()
    mock_db.transaction.return_value = mock_transaction
    
    return mock_db


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return MagicMock(spec=logging.Logger)
