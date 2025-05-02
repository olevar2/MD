"""Tests for the Portfolio Management Service client."""
import pytest
from unittest.mock import AsyncMock, patch

from risk_management_service.clients.portfolio_management_client import PortfolioManagementClient

@pytest.fixture
def mock_response():
    """Create a mock successful response."""
    mock = AsyncMock()
    mock.raise_for_status.return_value = None
    mock.__aenter__.return_value = mock
    return mock


@pytest.fixture
async def client():
    """Create a PortfolioManagementClient instance for testing."""
    client = PortfolioManagementClient()
    with patch.object(client, '_make_request', return_value={}):
        yield client


@pytest.mark.asyncio
async def test_get_portfolio_summary(client, mock_response):
    """Test getting a portfolio summary."""
    expected_summary = {
        'total_exposure': 100000.0,
        'positions': [
            {
                'symbol': 'EURUSD',
                'current_value': 100000.0,
            }
        ]
    }
    
    with patch.object(client, '_make_request', return_value=expected_summary):
        summary = await client.get_portfolio_summary('test-account')
        assert summary == expected_summary


@pytest.mark.asyncio
async def test_get_portfolio_risk(client, mock_response):
    """Test getting portfolio risk metrics."""
    expected_metrics = {
        'var': 5000.0,
        'max_drawdown': 10.0,
        'sharpe_ratio': 1.5
    }
    
    with patch.object(client, '_make_request', return_value=expected_metrics):
        metrics = await client.get_portfolio_risk('test-account')
        assert metrics == expected_metrics


@pytest.mark.asyncio
async def test_get_total_exposure(client, mock_response):
    """Test getting total exposure."""
    mock_summary = {
        'total_exposure': 100000.0
    }
    
    with patch.object(client, '_make_request', return_value=mock_summary):
        exposure = await client.get_total_exposure('test-account')
        assert exposure == 100000.0


@pytest.mark.asyncio
async def test_get_symbol_exposure(client, mock_response):
    """Test getting exposure for a specific symbol."""
    mock_summary = {
        'positions': [
            {
                'symbol': 'EURUSD',
                'current_value': 100000.0
            },
            {
                'symbol': 'GBPUSD',
                'current_value': 50000.0
            }
        ]
    }
    
    with patch.object(client, '_make_request', return_value=mock_summary):
        exposure = await client.get_symbol_exposure('test-account', 'EURUSD')
        assert exposure == 100000.0
