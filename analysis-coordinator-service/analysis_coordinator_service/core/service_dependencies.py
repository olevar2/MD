from typing import Dict, Any
from fastapi import Depends

from analysis_coordinator_service.services.coordinator_service import CoordinatorService
from analysis_coordinator_service.adapters.market_analysis_adapter import MarketAnalysisAdapter
from analysis_coordinator_service.adapters.causal_analysis_adapter import CausalAnalysisAdapter
from analysis_coordinator_service.adapters.backtesting_adapter import BacktestingAdapter
from analysis_coordinator_service.adapters.analysis_engine_grpc_client import AnalysisEngineGrpcClient
from analysis_coordinator_service.repositories.task_repository import TaskRepository
from analysis_coordinator_service.config.settings import get_settings

def get_market_analysis_adapter():
    """
    Get the market analysis adapter.
    """
    settings = get_settings()
    return MarketAnalysisAdapter(base_url=settings.market_analysis_service_url)

def get_analysis_engine_grpc_client():
    """
    Get the Analysis Engine gRPC client.
    """
    settings = get_settings()
    return AnalysisEngineGrpcClient(grpc_server_address=settings.analysis_engine_grpc_url)

def get_causal_analysis_adapter():
    """
    Get the causal analysis adapter.
    """
    settings = get_settings()
    return CausalAnalysisAdapter(base_url=settings.causal_analysis_service_url)

def get_backtesting_adapter():
    """
    Get the backtesting adapter.
    """
    settings = get_settings()
    return BacktestingAdapter(base_url=settings.backtesting_service_url)

def get_task_repository():
    """
    Get the task repository.
    """
    settings = get_settings()
    return TaskRepository(connection_string=settings.database_connection_string)

def get_coordinator_service(
    market_analysis_adapter: MarketAnalysisAdapter = Depends(get_market_analysis_adapter),
    causal_analysis_adapter: CausalAnalysisAdapter = Depends(get_causal_analysis_adapter),
    backtesting_adapter: BacktestingAdapter = Depends(get_backtesting_adapter),
    task_repository: TaskRepository = Depends(get_task_repository),
    # analysis_engine_client: AnalysisEngineGrpcClient = Depends(get_analysis_engine_grpc_client) # Add if needed by CoordinatorService
):
    """
    Get the coordinator service with all dependencies.
    """
    # TODO: Determine if AnalysisEngineGrpcClient needs to be passed to CoordinatorService directly
    # or used by a specific workflow/adapter that CoordinatorService uses.
    # For now, it's injectable but not directly passed to CoordinatorService constructor.
    return CoordinatorService(
        market_analysis_adapter=market_analysis_adapter,
        causal_analysis_adapter=causal_analysis_adapter,
        backtesting_adapter=backtesting_adapter,
        task_repository=task_repository
        # analysis_engine_client=analysis_engine_client, # Uncomment if CoordinatorService needs it
    )