"""
Command models for the Analysis Coordinator Service.

This module provides the command models for the Analysis Coordinator Service.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from common_lib.cqrs.commands import Command
from analysis_coordinator_service.models.coordinator_models import AnalysisServiceType


class RunIntegratedAnalysisCommand(Command):
    """Command to run an integrated analysis across multiple analysis services."""
    
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    services: List[AnalysisServiceType] = Field(..., description="Services to use for analysis")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the analysis")


class CreateAnalysisTaskCommand(Command):
    """Command to create a new analysis task."""
    
    service_type: AnalysisServiceType = Field(..., description="Type of analysis service")
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the analysis")


class CancelAnalysisTaskCommand(Command):
    """Command to cancel a running analysis task."""
    
    task_id: str = Field(..., description="ID of the task to cancel")


class DeleteAnalysisTaskCommand(Command):
    """Command to delete an analysis task."""
    
    task_id: str = Field(..., description="ID of the task to delete")