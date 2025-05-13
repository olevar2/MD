"""
Tests for the basic reconciliation module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from core.basic_reconciliation import BasicReconciliation


@pytest.fixture
def basic_reconciliation():
    """Create a BasicReconciliation instance with mocked dependencies."""
    return BasicReconciliation(
        account_repository=AsyncMock(),
        portfolio_repository=AsyncMock(),
        trading_gateway_client=AsyncMock(),
        event_publisher=AsyncMock(),
        reconciliation_repository=AsyncMock()
    )


@pytest.mark.asyncio
async def test_perform_reconciliation_no_discrepancies(basic_reconciliation):
    """Test perform_reconciliation with no discrepancies."""
    # Arrange
    internal_data = {
        "balance": 1000.0,
        "equity": 1100.0,
        "margin_used": 100.0,
        "free_margin": 900.0
    }
    
    broker_data = {
        "balance": 1000.0,
        "equity": 1100.0,
        "margin_used": 100.0,
        "free_margin": 900.0
    }
    
    tolerance = 0.01
    
    # Act
    result = await basic_reconciliation.perform_reconciliation(
        internal_data, broker_data, tolerance
    )
    
    # Assert
    assert result["discrepancies"] == []
    assert result["matched_fields"] == 4
    assert result["reconciliation_level"] == "basic"


@pytest.mark.asyncio
async def test_perform_reconciliation_with_discrepancies(basic_reconciliation):
    """Test perform_reconciliation with discrepancies."""
    # Arrange
    internal_data = {
        "balance": 1000.0,
        "equity": 1100.0,
        "margin_used": 100.0,
        "free_margin": 900.0
    }
    
    broker_data = {
        "balance": 1020.0,  # 2% difference
        "equity": 1100.0,
        "margin_used": 100.0,
        "free_margin": 900.0
    }
    
    tolerance = 0.01  # 1% tolerance
    
    # Act
    result = await basic_reconciliation.perform_reconciliation(
        internal_data, broker_data, tolerance
    )
    
    # Assert
    assert len(result["discrepancies"]) == 1
    assert result["discrepancies"][0]["field"] == "balance"
    assert result["discrepancies"][0]["internal_value"] == 1000.0
    assert result["discrepancies"][0]["broker_value"] == 1020.0
    assert result["discrepancies"][0]["absolute_difference"] == 20.0
    assert result["discrepancies"][0]["percentage_difference"] == 2.0
    assert result["discrepancies"][0]["severity"] == "medium"
    assert result["matched_fields"] == 3
    assert result["reconciliation_level"] == "basic"


@pytest.mark.asyncio
async def test_perform_reconciliation_with_zero_values(basic_reconciliation):
    """Test perform_reconciliation with zero values."""
    # Arrange
    internal_data = {
        "balance": 0.0,
        "equity": 0.0,
        "margin_used": 0.0,
        "free_margin": 0.0
    }
    
    broker_data = {
        "balance": 0.0,
        "equity": 0.0,
        "margin_used": 0.0,
        "free_margin": 0.0
    }
    
    tolerance = 0.01
    
    # Act
    result = await basic_reconciliation.perform_reconciliation(
        internal_data, broker_data, tolerance
    )
    
    # Assert
    assert result["discrepancies"] == []
    assert result["matched_fields"] == 4
    assert result["reconciliation_level"] == "basic"


@pytest.mark.asyncio
async def test_perform_reconciliation_with_missing_fields(basic_reconciliation):
    """Test perform_reconciliation with missing fields."""
    # Arrange
    internal_data = {
        "balance": 1000.0,
        "equity": 1100.0
    }
    
    broker_data = {
        "balance": 1000.0,
        "equity": 1100.0
    }
    
    tolerance = 0.01
    
    # Act
    result = await basic_reconciliation.perform_reconciliation(
        internal_data, broker_data, tolerance
    )
    
    # Assert
    assert result["discrepancies"] == []
    assert result["matched_fields"] == 4  # Still 4 because missing fields are treated as 0
    assert result["reconciliation_level"] == "basic"