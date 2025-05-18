"""
Dependency injection module for the Analysis Coordinator Service.

This module provides the dependency injection for the Analysis Coordinator Service.
"""
import logging
from typing import Optional

from common_lib.cqrs.commands import CommandBus
from common_lib.cqrs.queries import QueryBus
from analysis_coordinator_service.cqrs.handlers.command_handlers import (
    RunIntegratedAnalysisCommandHandler,
    CreateAnalysisTaskCommandHandler,
    CancelAnalysisTaskCommandHandler,
    DeleteAnalysisTaskCommandHandler
)
from analysis_coordinator_service.cqrs.handlers.query_handlers import (
    GetAnalysisTaskQueryHandler,
    ListAnalysisTasksQueryHandler,
    GetIntegratedAnalysisTaskQueryHandler,
    ListIntegratedAnalysisTasksQueryHandler,
    GetAnalysisTaskStatusQueryHandler,
    GetAvailableServicesQueryHandler
)
from analysis_coordinator_service.cqrs.commands import (
    RunIntegratedAnalysisCommand,
    CreateAnalysisTaskCommand,
    CancelAnalysisTaskCommand,
    DeleteAnalysisTaskCommand
)
from analysis_coordinator_service.cqrs.queries import (
    GetAnalysisTaskQuery,
    ListAnalysisTasksQuery,
    GetIntegratedAnalysisTaskQuery,
    ListIntegratedAnalysisTasksQuery,
    GetAnalysisTaskStatusQuery,
    GetAvailableServicesQuery
)
from analysis_coordinator_service.repositories.read_repositories import TaskReadRepository
from analysis_coordinator_service.repositories.write_repositories import TaskWriteRepository
from analysis_coordinator_service.services.coordinator_service import CoordinatorService
from analysis_coordinator_service.adapters.market_analysis_adapter import MarketAnalysisAdapter
from analysis_coordinator_service.adapters.causal_analysis_adapter import CausalAnalysisAdapter
from analysis_coordinator_service.adapters.backtesting_adapter import BacktestingAdapter
from analysis_coordinator_service.config.settings import get_settings

logger = logging.getLogger(__name__)

# Singleton instances
_command_bus: Optional[CommandBus] = None
_query_bus: Optional[QueryBus] = None


def get_market_analysis_adapter(settings):
    """
    Get the market analysis adapter.
    """
    return MarketAnalysisAdapter(base_url=settings.market_analysis_service_url)


def get_causal_analysis_adapter(settings):
    """
    Get the causal analysis adapter.
    """
    return CausalAnalysisAdapter(base_url=settings.causal_analysis_service_url)


def get_backtesting_adapter(settings):
    """
    Get the backtesting adapter.
    """
    return BacktestingAdapter(base_url=settings.backtesting_service_url)


def get_task_read_repository(settings):
    """
    Get the task read repository.
    """
    return TaskReadRepository(connection_string=settings.database_connection_string)


def get_task_write_repository(settings):
    """
    Get the task write repository.
    """
    return TaskWriteRepository(connection_string=settings.database_connection_string)


def get_task_repository(settings):
    """
    Get the task repository (write repository for backward compatibility).
    """
    return get_task_write_repository(settings)


def get_coordinator_service(settings):
    """
    Get the coordinator service.
    """
    return CoordinatorService(
        market_analysis_adapter=get_market_analysis_adapter(settings),
        causal_analysis_adapter=get_causal_analysis_adapter(settings),
        backtesting_adapter=get_backtesting_adapter(settings),
        task_repository=get_task_write_repository(settings)
    )


def get_command_bus() -> CommandBus:
    """
    Get the command bus.
    
    Returns:
        The command bus
    """
    global _command_bus
    
    if _command_bus is None:
        _command_bus = CommandBus()
        settings = get_settings() # Get settings once
        
        # Create repositories
        task_write_repository = get_task_write_repository(settings)
        
        # Create services
        coordinator_service = get_coordinator_service(settings)
        
        # Create command handlers
        run_integrated_analysis_handler = RunIntegratedAnalysisCommandHandler(
            coordinator_service=coordinator_service,
            repository=task_write_repository
        )
        create_analysis_task_handler = CreateAnalysisTaskCommandHandler(
            coordinator_service=coordinator_service,
            repository=task_write_repository
        )
        cancel_analysis_task_handler = CancelAnalysisTaskCommandHandler(
            repository=task_write_repository
        )
        delete_analysis_task_handler = DeleteAnalysisTaskCommandHandler(
            repository=task_write_repository
        )
        
        # Register command handlers
        _command_bus.register_handler(RunIntegratedAnalysisCommand, run_integrated_analysis_handler)
        _command_bus.register_handler(CreateAnalysisTaskCommand, create_analysis_task_handler)
        _command_bus.register_handler(CancelAnalysisTaskCommand, cancel_analysis_task_handler)
        _command_bus.register_handler(DeleteAnalysisTaskCommand, delete_analysis_task_handler)
    
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
        settings = get_settings() # Get settings once
        
        # Create repositories
        task_read_repository = get_task_read_repository(settings)
        
        # Create services
        coordinator_service = get_coordinator_service(settings)
        
        # Create query handlers
        get_analysis_task_handler = GetAnalysisTaskQueryHandler(
            repository=task_read_repository
        )
        list_analysis_tasks_handler = ListAnalysisTasksQueryHandler(
            repository=task_read_repository
        )
        get_integrated_analysis_task_handler = GetIntegratedAnalysisTaskQueryHandler(
            repository=task_read_repository
        )
        list_integrated_analysis_tasks_handler = ListIntegratedAnalysisTasksQueryHandler(
            repository=task_read_repository
        )
        get_analysis_task_status_handler = GetAnalysisTaskStatusQueryHandler(
            repository=task_read_repository
        )
        get_available_services_handler = GetAvailableServicesQueryHandler(
            coordinator_service=coordinator_service
        )
        
        # Register query handlers
        _query_bus.register_handler(GetAnalysisTaskQuery, get_analysis_task_handler)
        _query_bus.register_handler(ListAnalysisTasksQuery, list_analysis_tasks_handler)
        _query_bus.register_handler(GetIntegratedAnalysisTaskQuery, get_integrated_analysis_task_handler)
        _query_bus.register_handler(ListIntegratedAnalysisTasksQuery, list_integrated_analysis_tasks_handler)
        _query_bus.register_handler(GetAnalysisTaskStatusQuery, get_analysis_task_status_handler)
        _query_bus.register_handler(GetAvailableServicesQuery, get_available_services_handler)
    
    return _query_bus