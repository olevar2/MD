import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC
import uuid

# Fixtures moved from analysis-coordinator-service/tests/unit/conftest.py

@pytest.fixture
def mock_settings():
    """
    Create a mock for the Settings object.
    """
    settings_mock = MagicMock()
    settings_mock.market_analysis_service_url = "http://mock-market-analysis:8000"
    settings_mock.causal_analysis_service_url = "http://mock-causal-analysis:8000"
    settings_mock.backtesting_service_url = "http://mock-backtesting:8000"
    settings_mock.database_connection_string = "sqlite:///:memory:"
    settings_mock.service_name = "test-analysis-coordinator-service"
    settings_mock.log_level = "DEBUG"
    settings_mock.api_prefix = "/test/api/v1"
    settings_mock.retry_count = 1
    settings_mock.retry_backoff_factor = 0.1
    settings_mock.circuit_breaker_failure_threshold = 1
    settings_mock.circuit_breaker_recovery_timeout = 1
    settings_mock.task_cleanup_interval_hours = 1
    settings_mock.task_max_age_days = 1
    return settings_mock

@pytest.fixture
def mock_coordinator_service(mock_settings):
    """
    Create a mock for CoordinatorService with mocked dependencies.
    """
    # Patch get_settings to return the mock_settings fixture
    with patch('analysis_coordinator_service.core.service_dependencies.get_settings', return_value=mock_settings):
        service = AsyncMock()

        # Generate a task ID to use consistently
        task_id = str(uuid.uuid4())

        # Mock run_integrated_analysis method
        service.run_integrated_analysis.return_value = MagicMock(
            task_id=task_id,
            status="PENDING", # Use string representation for mock
            created_at=datetime.now(UTC),
            estimated_completion_time=datetime.now(UTC)
        )

        # Mock create_analysis_task method
        service.create_analysis_task.return_value = MagicMock(
            task_id=task_id,
            service_type="MARKET_ANALYSIS", # Use string representation for mock
            status="PENDING", # Use string representation for mock
            created_at=datetime.now(UTC),
            estimated_completion_time=datetime.now(UTC)
        )

        # Mock get_task_result method
        service.get_task_result.return_value = MagicMock(
            task_id=task_id,
            service_type="MARKET_ANALYSIS", # Use string representation for mock
            status="COMPLETED", # Use string representation for mock
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
            error=None
        )

        # Mock get_task_status method
        service.get_task_status.return_value = MagicMock(
            task_id=task_id,
            service_type="MARKET_ANALYSIS", # Use string representation for mock
            status="RUNNING", # Use string representation for mock
            created_at=datetime.now(UTC),
            progress=0.5,
            message="Processing data"
        )

        # Mock delete_task method
        service.delete_task.return_value = True

        # Mock cancel_task method
        service.cancel_task.return_value = True

        # Mock list_tasks method
        service.list_tasks.return_value = [
            MagicMock(
                task_id=task_id,
                service_type="MARKET_ANALYSIS", # Use string representation for mock
                status="COMPLETED", # Use string representation for mock
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
                error=None
            )
        ]

        # Mock get_available_services method
        service.get_available_services.return_value = {
            "market_analysis": [
                "pattern_recognition",
                "support_resistance",
                "market_regime",
                "correlation_analysis"
            ],
            "causal_analysis": [
                "causal_graph",
                "intervention_effect",
                "counterfactual_scenario"
            ],
            "backtesting": [
                "strategy_backtest",
                "performance_analysis",
                "optimization"
            ]
        }

        return service