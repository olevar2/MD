from typing import Optional
from functools import lru_cache

from backtesting_service.services.backtest_service import BacktestService
from backtesting_service.repositories.backtest_repository import BacktestRepository
from backtesting_service.adapters.data_pipeline_adapter import DataPipelineAdapter
from backtesting_service.adapters.analysis_coordinator_adapter import AnalysisCoordinatorAdapter
from backtesting_service.adapters.strategy_execution_adapter import StrategyExecutionAdapter

@lru_cache()
def get_data_pipeline_adapter() -> DataPipelineAdapter:
    """
    Get the data pipeline adapter with dependency injection.
    """
    return DataPipelineAdapter()

@lru_cache()
def get_analysis_coordinator_adapter() -> AnalysisCoordinatorAdapter:
    """
    Get the analysis coordinator adapter with dependency injection.
    """
    return AnalysisCoordinatorAdapter()

@lru_cache()
def get_strategy_execution_adapter() -> StrategyExecutionAdapter:
    """
    Get the strategy execution adapter with dependency injection.
    """
    return StrategyExecutionAdapter()

@lru_cache()
def get_backtest_repository() -> BacktestRepository:
    """
    Get the backtest repository with dependency injection.
    """
    return BacktestRepository()

@lru_cache()
def get_backtest_service() -> BacktestService:
    """
    Get the backtest service with dependency injection.
    """
    data_pipeline_adapter = get_data_pipeline_adapter()
    analysis_coordinator_adapter = get_analysis_coordinator_adapter()
    strategy_execution_adapter = get_strategy_execution_adapter()
    backtest_repository = get_backtest_repository()
    
    return BacktestService(
        data_pipeline_adapter=data_pipeline_adapter,
        analysis_coordinator_adapter=analysis_coordinator_adapter,
        strategy_execution_adapter=strategy_execution_adapter,
        backtest_repository=backtest_repository
    )