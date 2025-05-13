"""
Tests for the account reconciliation facade.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from portfolio_management_service.services.account_reconciliation import AccountReconciliationService


@pytest.fixture
def reconciliation_service():
    """Create an AccountReconciliationService instance with mocked dependencies."""
    return AccountReconciliationService(
        account_repository=AsyncMock(),
        portfolio_repository=AsyncMock(),
        trading_gateway_client=AsyncMock(),
        event_publisher=AsyncMock(),
        reconciliation_repository=AsyncMock()
    )


@pytest.mark.asyncio
async def test_reconcile_account_basic(reconciliation_service):
    """Test reconcile_account with basic reconciliation level."""
    # Arrange
    account_id = "12345"
    reconciliation_level = "basic"
    tolerance = 0.01
    notification_threshold = 1.0
    auto_fix = False
    
    # Mock the internal methods
    reconciliation_service.base._get_internal_account_data = AsyncMock(return_value={
        "balance": 1000.0,
        "equity": 1100.0,
        "margin_used": 100.0,
        "free_margin": 900.0
    })
    
    reconciliation_service.base._get_broker_account_data = AsyncMock(return_value={
        "balance": 1000.0,
        "equity": 1100.0,
        "margin_used": 100.0,
        "free_margin": 900.0
    })
    
    reconciliation_service.basic_reconciliation.perform_reconciliation = AsyncMock(return_value={
        "discrepancies": [],
        "matched_fields": 4,
        "reconciliation_level": "basic"
    })
    
    reconciliation_service.base._create_reconciliation_report = AsyncMock(return_value={
        "reconciliation_id": "test-id",
        "account_id": account_id,
        "reconciliation_level": reconciliation_level,
        "start_time": datetime.utcnow(),
        "tolerance_percentage": tolerance * 100,
        "discrepancies": {
            "total_count": 0,
            "by_severity": {},
            "by_field_type": {},
            "total_monetary_difference": 0.0,
            "details": []
        },
        "matched_fields": 4,
        "status": "completed"
    })
    
    # Act
    result = await reconciliation_service.reconcile_account(
        account_id=account_id,
        reconciliation_level=reconciliation_level,
        tolerance=tolerance,
        notification_threshold=notification_threshold,
        auto_fix=auto_fix
    )
    
    # Assert
    assert result["account_id"] == account_id
    assert result["reconciliation_level"] == reconciliation_level
    assert result["discrepancies"]["total_count"] == 0
    assert result["status"] == "completed"
    
    # Verify method calls
    reconciliation_service.base._get_internal_account_data.assert_called_once_with(account_id, None)
    reconciliation_service.base._get_broker_account_data.assert_called_once_with(account_id, None)
    reconciliation_service.basic_reconciliation.perform_reconciliation.assert_called_once()
    reconciliation_service.base._create_reconciliation_report.assert_called_once()


@pytest.mark.asyncio
async def test_reconcile_account_positions(reconciliation_service):
    """Test reconcile_account with positions reconciliation level."""
    # Arrange
    account_id = "12345"
    reconciliation_level = "positions"
    tolerance = 0.01
    notification_threshold = 1.0
    auto_fix = False
    
    # Mock the internal methods
    reconciliation_service.base._get_internal_account_data = AsyncMock(return_value={
        "balance": 1000.0,
        "equity": 1100.0,
        "margin_used": 100.0,
        "free_margin": 900.0,
        "positions": [
            {
                "position_id": "pos1",
                "instrument": "EURUSD",
                "direction": "buy",
                "size": 1.0,
                "open_price": 1.1000,
                "current_price": 1.1050,
                "unrealized_pnl": 50.0
            }
        ]
    })
    
    reconciliation_service.base._get_broker_account_data = AsyncMock(return_value={
        "balance": 1000.0,
        "equity": 1100.0,
        "margin_used": 100.0,
        "free_margin": 900.0,
        "positions": [
            {
                "position_id": "pos1",
                "instrument": "EURUSD",
                "direction": "buy",
                "size": 1.0,
                "open_price": 1.1000,
                "current_price": 1.1050,
                "unrealized_pnl": 50.0
            }
        ]
    })
    
    reconciliation_service.position_reconciliation.perform_reconciliation = AsyncMock(return_value={
        "discrepancies": [],
        "matched_fields": 5,  # 4 account fields + 1 position
        "reconciliation_level": "positions"
    })
    
    reconciliation_service.base._create_reconciliation_report = AsyncMock(return_value={
        "reconciliation_id": "test-id",
        "account_id": account_id,
        "reconciliation_level": reconciliation_level,
        "start_time": datetime.utcnow(),
        "tolerance_percentage": tolerance * 100,
        "discrepancies": {
            "total_count": 0,
            "by_severity": {},
            "by_field_type": {},
            "total_monetary_difference": 0.0,
            "details": []
        },
        "matched_fields": 5,
        "status": "completed"
    })
    
    # Act
    result = await reconciliation_service.reconcile_account(
        account_id=account_id,
        reconciliation_level=reconciliation_level,
        tolerance=tolerance,
        notification_threshold=notification_threshold,
        auto_fix=auto_fix
    )
    
    # Assert
    assert result["account_id"] == account_id
    assert result["reconciliation_level"] == reconciliation_level
    assert result["discrepancies"]["total_count"] == 0
    assert result["status"] == "completed"
    
    # Verify method calls
    reconciliation_service.base._get_internal_account_data.assert_called_once_with(account_id, None)
    reconciliation_service.base._get_broker_account_data.assert_called_once_with(account_id, None)
    reconciliation_service.position_reconciliation.perform_reconciliation.assert_called_once()
    reconciliation_service.base._create_reconciliation_report.assert_called_once()