"""
Distributed Computing API Endpoints

This module provides API endpoints for distributed computing.
"""
import os
import sys
import time
import json
import logging
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, Body
from pydantic import BaseModel, Field
from analysis_engine.utils.distributed_computing import DistributedTaskManager, DistributedTask
from analysis_engine.utils.distributed_tracing import DistributedTracer
from analysis_engine.multi_asset.optimized_confluence_detector import OptimizedConfluenceDetector
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
from analysis_engine.ml.ml_confluence_detector import MLConfluenceDetector
from analysis_engine.ml.model_manager import ModelManager
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/distributed', tags=['Distributed Computing'])
tracer = DistributedTracer(service_name='analysis-engine')
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class TaskSubmitRequest(BaseModel):
    """Request model for submitting a task."""
    function_name: str = Field(..., description='Function name to execute')
    args: List[Any] = Field([], description='Function arguments')
    kwargs: Dict[str, Any] = Field({}, description='Function keyword arguments'
        )
    priority: int = Field(0, description=
        'Task priority (higher values have higher priority)')
    timeout: Optional[float] = Field(None, description=
        'Task timeout in seconds')
    worker_id: Optional[str] = Field(None, description=
        'Worker ID to submit the task to')


class TaskSubmitResponse(BaseModel):
    """Response model for submitting a task."""
    task_id: str = Field(..., description='Task ID')
    status: str = Field(..., description='Task status')
    worker_id: str = Field(..., description='Worker ID')


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    task_id: str = Field(..., description='Task ID')
    function_name: str = Field(..., description='Function name')
    status: str = Field(..., description='Task status')
    created_at: float = Field(..., description='Task creation timestamp')
    started_at: Optional[float] = Field(None, description=
        'Task start timestamp')
    completed_at: Optional[float] = Field(None, description=
        'Task completion timestamp')
    error: Optional[str] = Field(None, description='Task error message')
    worker_id: Optional[str] = Field(None, description='Worker ID')


class TaskResultResponse(BaseModel):
    """Response model for task result."""
    task_id: str = Field(..., description='Task ID')
    status: str = Field(..., description='Task status')
    result: Any = Field(None, description='Task result')
    error: Optional[str] = Field(None, description='Task error message')
    execution_time: Optional[float] = Field(None, description=
        'Task execution time in seconds')


class WorkerStatsResponse(BaseModel):
    """Response model for worker statistics."""
    worker_id: str = Field(..., description='Worker ID')
    status: str = Field(..., description='Worker status')
    max_concurrent_tasks: int = Field(..., description=
        'Maximum number of concurrent tasks')
    use_processes: bool = Field(..., description=
        'Whether the worker uses processes')
    queue_size: int = Field(..., description='Task queue size')
    tasks_pending: int = Field(..., description='Number of pending tasks')
    tasks_running: int = Field(..., description='Number of running tasks')
    tasks_completed: int = Field(..., description='Number of completed tasks')
    tasks_failed: int = Field(..., description='Number of failed tasks')
    tasks_timeout: int = Field(..., description='Number of timed out tasks')
    total_execution_time: float = Field(..., description=
        'Total execution time in seconds')
    avg_execution_time: float = Field(..., description=
        'Average execution time in seconds')


class ManagerStatsResponse(BaseModel):
    """Response model for task manager statistics."""
    status: str = Field(..., description='Task manager status')
    num_workers: int = Field(..., description='Number of workers')
    max_concurrent_tasks_per_worker: int = Field(..., description=
        'Maximum number of concurrent tasks per worker')
    use_processes: bool = Field(..., description=
        'Whether the workers use processes')
    tasks_pending: int = Field(..., description='Number of pending tasks')
    tasks_running: int = Field(..., description='Number of running tasks')
    tasks_completed: int = Field(..., description='Number of completed tasks')
    tasks_failed: int = Field(..., description='Number of failed tasks')
    tasks_timeout: int = Field(..., description='Number of timed out tasks')
    total_execution_time: float = Field(..., description=
        'Total execution time in seconds')
    avg_execution_time: float = Field(..., description=
        'Average execution time in seconds')
    workers: Dict[str, WorkerStatsResponse] = Field(..., description=
        'Worker statistics')


def get_task_manager():
    """Get the task manager."""
    functions = {'detect_confluence': detect_confluence,
        'detect_confluence_ml': detect_confluence_ml, 'analyze_divergence':
        analyze_divergence, 'analyze_divergence_ml': analyze_divergence_ml,
        'calculate_currency_strength': calculate_currency_strength}
    task_manager = DistributedTaskManager(functions=functions, num_workers=
        4, max_concurrent_tasks_per_worker=4, use_processes=False)
    if task_manager.status != 'running':
        asyncio.create_task(task_manager.start())
    return task_manager


async def detect_confluence(symbol: str, price_data: Dict[str, Any],
    signal_type: str, signal_direction: str, related_pairs: Optional[Dict[
    str, float]]=None, use_currency_strength: bool=True,
    min_confirmation_strength: float=0.3) ->Dict[str, Any]:
    """
    Detect confluence.

    Args:
        symbol: Currency pair
        price_data: Price data
        signal_type: Signal type
        signal_direction: Signal direction
        related_pairs: Related pairs
        use_currency_strength: Whether to use currency strength
        min_confirmation_strength: Minimum confirmation strength

    Returns:
        Confluence detection result
    """
    correlation_service = MockCorrelationService()
    currency_strength_analyzer = CurrencyStrengthAnalyzer()
    detector = OptimizedConfluenceDetector(correlation_service=
        correlation_service, currency_strength_analyzer=
        currency_strength_analyzer, correlation_threshold=0.7,
        lookback_periods=20, cache_ttl_minutes=60, max_workers=4)
    price_data_dfs = {}
    for pair, data in price_data.items():
        price_data_dfs[pair] = pd.DataFrame(data)
    result = detector.detect_confluence_optimized(symbol=symbol, price_data
        =price_data_dfs, signal_type=signal_type, signal_direction=
        signal_direction, related_pairs=related_pairs,
        use_currency_strength=use_currency_strength,
        min_confirmation_strength=min_confirmation_strength)
    return result


async def detect_confluence_ml(symbol: str, price_data: Dict[str, Any],
    signal_type: str, signal_direction: str, related_pairs: Optional[Dict[
    str, float]]=None, use_currency_strength: bool=True,
    min_confirmation_strength: float=0.3) ->Dict[str, Any]:
    """
    Detect confluence using ML.

    Args:
        symbol: Currency pair
        price_data: Price data
        signal_type: Signal type
        signal_direction: Signal direction
        related_pairs: Related pairs
        use_currency_strength: Whether to use currency strength
        min_confirmation_strength: Minimum confirmation strength

    Returns:
        Confluence detection result
    """
    correlation_service = MockCorrelationService()
    currency_strength_analyzer = CurrencyStrengthAnalyzer()
    model_manager = ModelManager(model_dir='models', use_gpu=False,
        correlation_service=correlation_service, currency_strength_analyzer
        =currency_strength_analyzer)
    detector = model_manager.load_ml_confluence_detector()
    price_data_dfs = {}
    for pair, data in price_data.items():
        price_data_dfs[pair] = pd.DataFrame(data)
    result = detector.detect_confluence_ml(symbol=symbol, price_data=
        price_data_dfs, signal_type=signal_type, signal_direction=
        signal_direction, related_pairs=related_pairs,
        use_currency_strength=use_currency_strength,
        min_confirmation_strength=min_confirmation_strength)
    return result


async def analyze_divergence(symbol: str, price_data: Dict[str, Any],
    related_pairs: Optional[Dict[str, float]]=None) ->Dict[str, Any]:
    """
    Analyze divergence.

    Args:
        symbol: Currency pair
        price_data: Price data
        related_pairs: Related pairs

    Returns:
        Divergence analysis result
    """
    correlation_service = MockCorrelationService()
    currency_strength_analyzer = CurrencyStrengthAnalyzer()
    detector = OptimizedConfluenceDetector(correlation_service=
        correlation_service, currency_strength_analyzer=
        currency_strength_analyzer, correlation_threshold=0.7,
        lookback_periods=20, cache_ttl_minutes=60, max_workers=4)
    price_data_dfs = {}
    for pair, data in price_data.items():
        price_data_dfs[pair] = pd.DataFrame(data)
    result = detector.analyze_divergence_optimized(symbol=symbol,
        price_data=price_data_dfs, related_pairs=related_pairs)
    return result


async def analyze_divergence_ml(symbol: str, price_data: Dict[str, Any],
    related_pairs: Optional[Dict[str, float]]=None) ->Dict[str, Any]:
    """
    Analyze divergence using ML.

    Args:
        symbol: Currency pair
        price_data: Price data
        related_pairs: Related pairs

    Returns:
        Divergence analysis result
    """
    correlation_service = MockCorrelationService()
    currency_strength_analyzer = CurrencyStrengthAnalyzer()
    model_manager = ModelManager(model_dir='models', use_gpu=False,
        correlation_service=correlation_service, currency_strength_analyzer
        =currency_strength_analyzer)
    detector = model_manager.load_ml_confluence_detector()
    price_data_dfs = {}
    for pair, data in price_data.items():
        price_data_dfs[pair] = pd.DataFrame(data)
    result = detector.analyze_divergence_ml(symbol=symbol, price_data=
        price_data_dfs, related_pairs=related_pairs)
    return result


async def calculate_currency_strength(price_data: Dict[str, Any], timeframe:
    str='H1', method: str='combined') ->Dict[str, Any]:
    """
    Calculate currency strength.

    Args:
        price_data: Price data
        timeframe: Timeframe
        method: Method

    Returns:
        Currency strength result
    """
    currency_strength_analyzer = CurrencyStrengthAnalyzer()
    price_data_dfs = {}
    for pair, data in price_data.items():
        price_data_dfs[pair] = pd.DataFrame(data)
    result = currency_strength_analyzer.calculate_currency_strength(price_data
        =price_data_dfs, method=method)
    return {'timeframe': timeframe, 'method': method, 'currencies': result,
        'strongest': max(result.items(), key=lambda x: x[1])[0], 'weakest':
        min(result.items(), key=lambda x: x[1])[0]}


class MockCorrelationService:
    """Mock correlation service for testing."""

    @with_resilience('get_all_correlations')
    async def get_all_correlations(self) ->Dict[str, Dict[str, float]]:
        """
        Get all correlations between pairs.

        Returns:
            Dictionary mapping pairs to dictionaries of correlations
        """
        return {'EURUSD': {'GBPUSD': 0.85, 'AUDUSD': 0.75, 'USDCAD': -0.65,
            'USDJPY': -0.55, 'EURGBP': 0.62, 'EURJPY': 0.78}, 'GBPUSD': {
            'EURUSD': 0.85, 'AUDUSD': 0.7, 'USDCAD': -0.6, 'USDJPY': -0.5,
            'EURGBP': -0.58, 'GBPJPY': 0.75}, 'USDJPY': {'EURUSD': -0.55,
            'GBPUSD': -0.5, 'AUDUSD': -0.45, 'USDCAD': 0.4, 'EURJPY': 0.65,
            'GBPJPY': 0.7}, 'AUDUSD': {'EURUSD': 0.75, 'GBPUSD': 0.7,
            'USDCAD': -0.55, 'USDJPY': -0.45}}


@router.post('/tasks', response_model=TaskSubmitResponse)
@async_with_exception_handling
async def submit_task(request: TaskSubmitRequest, task_manager:
    DistributedTaskManager=Depends(get_task_manager)):
    """
    Submit a task for distributed execution.

    This endpoint submits a task to the distributed task manager for execution.
    """
    with tracer.start_span('submit_task') as span:
        span.set_attribute('function_name', request.function_name)
        try:
            task_id = await task_manager.submit_task(function_name=request.
                function_name, args=request.args, kwargs=request.kwargs,
                priority=request.priority, timeout=request.timeout,
                worker_id=request.worker_id)
            task = task_manager.get_task(task_id)
            worker_id = request.worker_id or 'unknown'
            response = TaskSubmitResponse(task_id=task_id, status=task.
                status, worker_id=worker_id)
            return response
        except Exception as e:
            logger.error(f'Error submitting task: {e}')
            raise HTTPException(status_code=500, detail=str(e))


@router.get('/tasks/{task_id}', response_model=TaskStatusResponse)
@async_with_exception_handling
async def get_task_status(task_id: str=Path(..., description='Task ID'),
    task_manager: DistributedTaskManager=Depends(get_task_manager)):
    """
    Get the status of a task.

    This endpoint returns the status of a task.
    """
    with tracer.start_span('get_task_status') as span:
        span.set_attribute('task_id', task_id)
        try:
            task = task_manager.get_task(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail=
                    f'Task {task_id} not found')
            worker_id = None
            for worker_id, worker in task_manager.workers.items():
                if worker.get_task(task_id) is not None:
                    break
            response = TaskStatusResponse(task_id=task.task_id,
                function_name=task.function_name, status=task.status,
                created_at=task.created_at, started_at=task.started_at,
                completed_at=task.completed_at, error=task.error, worker_id
                =worker_id)
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f'Error getting task status: {e}')
            raise HTTPException(status_code=500, detail=str(e))


@router.get('/tasks/{task_id}/result', response_model=TaskResultResponse)
@async_with_exception_handling
async def get_task_result(task_id: str=Path(..., description='Task ID'),
    timeout: Optional[float]=Query(None, description='Timeout in seconds'),
    task_manager: DistributedTaskManager=Depends(get_task_manager)):
    """
    Get the result of a task.

    This endpoint returns the result of a task. If the task is not yet completed,
    it will wait for the task to complete up to the specified timeout.
    """
    with tracer.start_span('get_task_result') as span:
        span.set_attribute('task_id', task_id)
        try:
            task = task_manager.get_task(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail=
                    f'Task {task_id} not found')
            if task.status == 'pending' or task.status == 'running':
                try:
                    result = await task_manager.get_task_result(task_id,
                        timeout)
                    task = task_manager.get_task(task_id)
                except TimeoutError:
                    return TaskResultResponse(task_id=task_id, status=
                        'running', result=None, error=None, execution_time=None
                        )
                except Exception as e:
                    return TaskResultResponse(task_id=task_id, status=
                        'failed', result=None, error=str(e), execution_time
                        =None)
            execution_time = None
            if task.started_at and task.completed_at:
                execution_time = task.completed_at - task.started_at
            response = TaskResultResponse(task_id=task.task_id, status=task
                .status, result=task.result, error=task.error,
                execution_time=execution_time)
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f'Error getting task result: {e}')
            raise HTTPException(status_code=500, detail=str(e))


@router.get('/stats', response_model=ManagerStatsResponse)
@async_with_exception_handling
async def get_manager_stats(task_manager: DistributedTaskManager=Depends(
    get_task_manager)):
    """
    Get task manager statistics.

    This endpoint returns statistics about the task manager and its workers.
    """
    with tracer.start_span('get_manager_stats'):
        try:
            stats = task_manager.get_stats()
            return stats
        except Exception as e:
            logger.error(f'Error getting manager stats: {e}')
            raise HTTPException(status_code=500, detail=str(e))
