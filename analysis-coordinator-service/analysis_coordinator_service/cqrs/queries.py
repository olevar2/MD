"""
Query models for the Analysis Coordinator Service.

This module provides the query models for the Analysis Coordinator Service.
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from common_lib.cqrs.queries import Query
from analysis_coordinator_service.models.coordinator_models import AnalysisServiceType


class GetAnalysisTaskQuery(Query):
    """Query to get an analysis task by ID."""
    
    task_id: str = Field(..., description="ID of the task to retrieve")


class ListAnalysisTasksQuery(Query):
    """Query to list analysis tasks."""
    
    service_type: Optional[AnalysisServiceType] = Field(None, description="Filter by service type")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    status: Optional[str] = Field(None, description="Filter by status")
    limit: int = Field(10, description="Maximum number of tasks to return")
    offset: int = Field(0, description="Offset for pagination")


class GetIntegratedAnalysisTaskQuery(Query):
    """Query to get an integrated analysis task by ID."""
    
    task_id: str = Field(..., description="ID of the integrated task to retrieve")


class ListIntegratedAnalysisTasksQuery(Query):
    """Query to list integrated analysis tasks."""
    
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    status: Optional[str] = Field(None, description="Filter by status")
    limit: int = Field(10, description="Maximum number of tasks to return")
    offset: int = Field(0, description="Offset for pagination")


class GetAnalysisTaskStatusQuery(Query):
    """Query to get the status of an analysis task."""
    
    task_id: str = Field(..., description="ID of the task to get status for")


class GetAvailableServicesQuery(Query):
    """Query to get available analysis services."""
    pass