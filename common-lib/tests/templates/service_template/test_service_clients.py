"""
Unit tests for the service template service clients module.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from common_lib.templates.service_template.service_clients import ServiceClients, ResilientServiceClientConfig, ResilientServiceClient


class TestServiceClients:
    """Tests for the ServiceClients class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def service_clients(self, mock_logger):
        """Create a service clients instance."""
        return ServiceClients(logger=mock_logger)

    def test_init(self, service_clients, mock_logger):
        """Test initialization."""
        assert service_clients.logger == mock_logger
        assert service_clients._clients == {}

    @patch("common_lib.templates.service_template.service_clients.get_service_client_config")
    @patch("common_lib.templates.service_template.service_clients.ResilientServiceClient")
    def test_get_client(self, mock_resilient_service_client, mock_get_service_client_config, service_clients):
        """Test get_client method."""
        # Mock the get_service_client_config function
        mock_service_client_config = MagicMock()
        mock_service_client_config.base_url = "http://localhost:8000"
        mock_service_client_config.timeout = 30
        mock_service_client_config.retry = MagicMock()
        mock_service_client_config.retry.max_retries = 3
        mock_service_client_config.retry.initial_backoff = 1.0
        mock_service_client_config.circuit_breaker = MagicMock()
        mock_service_client_config.circuit_breaker.failure_threshold = 5
        mock_service_client_config.circuit_breaker.recovery_timeout = 30.0
        mock_get_service_client_config.return_value = mock_service_client_config

        # Mock the ResilientServiceClient
        mock_client = MagicMock()
        mock_resilient_service_client.return_value = mock_client

        # Call the method
        client = service_clients.get_client("test-service")

        # Verify the result
        assert client == mock_client

        # Verify the mocks were called
        mock_get_service_client_config.assert_called_once_with("test-service")
        mock_resilient_service_client.assert_called_once()

        # Verify the client was cached
        assert service_clients._clients["test-service"] == mock_client

        # Call the method again
        client2 = service_clients.get_client("test-service")

        # Verify the result
        assert client2 == mock_client

        # Verify the mocks were not called again
        assert mock_get_service_client_config.call_count == 1
        assert mock_resilient_service_client.call_count == 1

    @patch("common_lib.templates.service_template.service_clients.get_service_client_config")
    def test_get_client_config_not_found(self, mock_get_service_client_config, service_clients):
        """Test get_client method with config not found."""
        # Mock the get_service_client_config function
        mock_get_service_client_config.return_value = None

        # Call the method
        with pytest.raises(ValueError) as excinfo:
            service_clients.get_client("test-service")

        # Verify the exception
        assert str(excinfo.value) == "Service client configuration not found for test-service"

        # Verify the mock was called
        mock_get_service_client_config.assert_called_once_with("test-service")

    @pytest.mark.asyncio
    async def test_connect_all(self, service_clients):
        """Test connect_all method."""
        # Create mock clients
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()

        # Add clients to the service clients
        service_clients._clients = {
            "service1": mock_client1,
            "service2": mock_client2
        }

        # Call the method
        await service_clients.connect_all()

        # Verify the mocks were called
        mock_client1.connect.assert_called_once()
        mock_client2.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_all(self, service_clients):
        """Test close_all method."""
        # Create mock clients
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()

        # Add clients to the service clients
        service_clients._clients = {
            "service1": mock_client1,
            "service2": mock_client2
        }

        # Call the method
        await service_clients.close_all()

        # Verify the mocks were called
        mock_client1.close.assert_called_once()
        mock_client2.close.assert_called_once()

    @patch("common_lib.templates.service_template.service_clients.ServiceClients.get_client")
    def test_get_market_data_client(self, mock_get_client, service_clients):
        """Test get_market_data_client method."""
        # Mock the get_client method
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Call the method
        client = service_clients.get_market_data_client()

        # Verify the result
        assert client == mock_client

        # Verify the mock was called
        mock_get_client.assert_called_once_with("market_data_service")

    @patch("common_lib.templates.service_template.service_clients.ServiceClients.get_client")
    def test_get_data_pipeline_client(self, mock_get_client, service_clients):
        """Test get_data_pipeline_client method."""
        # Mock the get_client method
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Call the method
        client = service_clients.get_data_pipeline_client()

        # Verify the result
        assert client == mock_client

        # Verify the mock was called
        mock_get_client.assert_called_once_with("data_pipeline_service")

    @patch("common_lib.templates.service_template.service_clients.ServiceClients.get_client")
    def test_get_feature_store_client(self, mock_get_client, service_clients):
        """Test get_feature_store_client method."""
        # Mock the get_client method
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Call the method
        client = service_clients.get_feature_store_client()

        # Verify the result
        assert client == mock_client

        # Verify the mock was called
        mock_get_client.assert_called_once_with("feature_store_service")

    @patch("common_lib.templates.service_template.service_clients.ServiceClients.get_client")
    def test_get_analysis_engine_client(self, mock_get_client, service_clients):
        """Test get_analysis_engine_client method."""
        # Mock the get_client method
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Call the method
        client = service_clients.get_analysis_engine_client()

        # Verify the result
        assert client == mock_client

        # Verify the mock was called
        mock_get_client.assert_called_once_with("analysis_engine_service")

    @patch("common_lib.templates.service_template.service_clients.ServiceClients.get_client")
    def test_get_trading_client(self, mock_get_client, service_clients):
        """Test get_trading_client method."""
        # Mock the get_client method
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Call the method
        client = service_clients.get_trading_client()

        # Verify the result
        assert client == mock_client

        # Verify the mock was called
        mock_get_client.assert_called_once_with("trading_service")
