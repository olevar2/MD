import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timedelta, UTC
import uuid
import traceback

from fastapi.testclient import TestClient
from fastapi import FastAPI
from analysis_coordinator_service.api.v1.coordinator import router as coordinator_router
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskResponse,
    AnalysisTaskResult,
    AnalysisServiceType,
    IntegratedAnalysisResponse,
    AnalysisTaskStatus
)

# Create a test app
app = FastAPI()
app.include_router(coordinator_router)

# Create a test client
client = TestClient(app)

def create_mock_coordinator_service():
    """
    Create a mock for CoordinatorService.
    """
    service = AsyncMock()
    
    # Generate a task ID to use consistently
    task_id = str(uuid.uuid4())
    
    # Mock run_integrated_analysis method
    service.run_integrated_analysis.return_value = IntegratedAnalysisResponse(
        task_id=task_id,
        status=AnalysisTaskStatusEnum.PENDING,
        created_at=datetime.now(UTC),
        estimated_completion_time=datetime.now(UTC)
    )
    
    # Mock create_analysis_task method
    service.create_analysis_task.return_value = AnalysisTaskResponse(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.PENDING,
        created_at=datetime.now(UTC),
        estimated_completion_time=datetime.now(UTC)
    )
    
    # Mock get_task_result method
    service.get_task_result.return_value = AnalysisTaskResult(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.COMPLETED,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        result={"patterns": [{"type": "head_and_shoulders", "confidence": 0.85}]},
        error=None
    )
    
    # Mock get_task_status method
    service.get_task_status.return_value = AnalysisTaskStatus(
        task_id=task_id,
        service_type=AnalysisServiceType.MARKET_ANALYSIS,
        status=AnalysisTaskStatusEnum.RUNNING,
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
        AnalysisTaskResult(
            task_id=task_id,
            service_type=AnalysisServiceType.MARKET_ANALYSIS,
            status=AnalysisTaskStatusEnum.COMPLETED,
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

def debug_test_get_task_result():
    """
    Debug the get_task_result test.
    """
    try:
        # Create mock
        mock_coordinator_service = create_mock_coordinator_service()
        
        # Get task_id
        task_id = mock_coordinator_service.get_task_result.return_value.task_id
        
        # Patch the dependency
        with patch('analysis_coordinator_service.api.v1.coordinator.get_coordinator_service', return_value=mock_coordinator_service):
            # Make request
            response = client.get(f"/coordinator/tasks/{task_id}")
            
            # Print response
            print(f"Response status code: {response.status_code}")
            print(f"Response body: {response.text}")
            
            # Assert
            assert response.status_code == 200
            
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    debug_test_get_task_result()