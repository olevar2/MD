"""
Test dependency injection module for the Analysis Coordinator Service.

This module provides the dependency injection for the Analysis Coordinator Service tests.
"""
from unittest.mock import AsyncMock, MagicMock
from typing import Optional

from common_lib.cqrs.commands import CommandBus
from common_lib.cqrs.queries import QueryBus
from analysis_coordinator_service.services.coordinator_service import CoordinatorService
from analysis_coordinator_service.adapters.market_analysis_adapter import MarketAnalysisAdapter
from analysis_coordinator_service.adapters.causal_analysis_adapter import CausalAnalysisAdapter
from analysis_coordinator_service.adapters.backtesting_adapter import BacktestingAdapter
from analysis_coordinator_service.repositories.task_repository import TaskRepository

# Mock instances
_mock_coordinator_service: Optional[AsyncMock] = None
_mock_task_repository: Optional[AsyncMock] = None
_mock_market_analysis_adapter: Optional[AsyncMock] = None
_mock_causal_analysis_adapter: Optional[AsyncMock] = None
_mock_backtesting_adapter: Optional[AsyncMock] = None
_command_bus: Optional[CommandBus] = None
_query_bus: Optional[QueryBus] = None


def set_mock_coordinator_service(mock_service: AsyncMock):
    """
    Set the mock coordinator service.
    """
    global _mock_coordinator_service
    _mock_coordinator_service = mock_service


def set_mock_task_repository(mock_repository: AsyncMock):
    """
    Set the mock task repository.
    """
    global _mock_task_repository
    _mock_task_repository = mock_repository


def get_market_analysis_adapter():
    """
    Get the mock market analysis adapter.
    """
    global _mock_market_analysis_adapter
    if _mock_market_analysis_adapter is None:
        _mock_market_analysis_adapter = AsyncMock(spec=MarketAnalysisAdapter)
    return _mock_market_analysis_adapter


def get_causal_analysis_adapter():
    """
    Get the mock causal analysis adapter.
    """
    global _mock_causal_analysis_adapter
    if _mock_causal_analysis_adapter is None:
        _mock_causal_analysis_adapter = AsyncMock(spec=CausalAnalysisAdapter)
    return _mock_causal_analysis_adapter


def get_backtesting_adapter():
    """
    Get the mock backtesting adapter.
    """
    global _mock_backtesting_adapter
    if _mock_backtesting_adapter is None:
        _mock_backtesting_adapter = AsyncMock(spec=BacktestingAdapter)
    return _mock_backtesting_adapter


def get_task_repository():
    """
    Get the mock task repository.
    """
    global _mock_task_repository
    if _mock_task_repository is None:
        _mock_task_repository = AsyncMock(spec=TaskRepository)
    return _mock_task_repository


def get_coordinator_service():
    """
    Get the mock coordinator service.
    """
    global _mock_coordinator_service
    if _mock_coordinator_service is None:
        _mock_coordinator_service = AsyncMock(spec=CoordinatorService)
    return _mock_coordinator_service


def get_command_bus() -> CommandBus:
    """
    Get the command bus.
    
    Returns:
        The command bus
    """
    global _command_bus
    
    if _command_bus is None:
        _command_bus = CommandBus()
    
    return _command_bus


def get_query_bus() -> QueryBus:
    """
    Get the query bus.
    
    Returns:
        The query bus
    """
    global _query_bus
    
    if _query_bus is None:
        _query_bus = QueryBus()
    
    return _query_bus