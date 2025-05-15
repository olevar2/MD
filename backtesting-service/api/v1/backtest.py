"""
Backtesting API routes.

This module provides the API routes for backtesting.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any, Optional
from datetime import datetime

from common_lib.cqrs.commands import CommandBus
from common_lib.cqrs.queries import QueryBus
from backtesting_service.utils.dependency_injection import get_command_bus, get_query_bus
from backtesting_service.cqrs.commands import (
    RunBacktestCommand,
    OptimizeStrategyCommand,
    RunWalkForwardTestCommand,
    CancelBacktestCommand,
    DeleteBacktestCommand
)
from backtesting_service.cqrs.queries import (
    GetBacktestQuery,
    ListBacktestsQuery,
    GetOptimizationQuery,
    ListOptimizationsQuery,
    GetWalkForwardTestQuery,
    ListWalkForwardTestsQuery,
    ListStrategiesQuery
)
from backtesting_service.models.backtest_models import (
    BacktestRequest,
    BacktestResponse,
    BacktestResult,
    BacktestStatus,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationResult,
    WalkForwardTestRequest,
    WalkForwardTestResponse,
    WalkForwardTestResult,
    StrategyListResponse
)

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Run a backtest with the specified configuration.
    """
    try:
        # Create command
        command = RunBacktestCommand(
            correlation_id=None,
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_balance=request.initial_balance,
            parameters=request.parameters
        )
        
        # Dispatch command
        backtest_id = await command_bus.dispatch(command)
        
        # Return response
        return BacktestResponse(
            backtest_id=backtest_id,
            status=BacktestStatus.PENDING,
            message="Backtest started successfully",
            estimated_completion_time=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run backtest: {str(e)}")


@router.get("/{backtest_id}", response_model=BacktestResult)
async def get_backtest(
    backtest_id: str,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get a backtest by ID.
    """
    try:
        # Create query
        query = GetBacktestQuery(
            correlation_id=None,
            backtest_id=backtest_id
        )
        
        # Dispatch query
        backtest = await query_bus.dispatch(query)
        
        if not backtest:
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")
        
        return backtest
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backtest: {str(e)}")


@router.get("/", response_model=List[BacktestResult])
async def list_backtests(
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    List backtests.
    """
    try:
        # Create query
        query = ListBacktestsQuery(
            correlation_id=None,
            strategy_id=strategy_id,
            symbol=symbol,
            limit=limit,
            offset=offset
        )
        
        # Dispatch query
        result = await query_bus.dispatch(query)
        
        return result.backtests
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backtests: {str(e)}")


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_strategy(
    request: OptimizationRequest,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Optimize a strategy with the specified configuration.
    """
    try:
        # Create command
        command = OptimizeStrategyCommand(
            correlation_id=None,
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_balance=request.initial_balance,
            parameters_to_optimize=request.parameters_to_optimize,
            optimization_metric=request.optimization_metric,
            num_iterations=request.num_iterations
        )
        
        # Dispatch command
        optimization_id = await command_bus.dispatch(command)
        
        # Return response
        return OptimizationResponse(
            optimization_id=optimization_id,
            status="pending",
            message="Optimization started successfully",
            estimated_completion_time=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize strategy: {str(e)}")


@router.get("/optimize/{optimization_id}", response_model=OptimizationResult)
async def get_optimization(
    optimization_id: str,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get an optimization by ID.
    """
    try:
        # Create query
        query = GetOptimizationQuery(
            correlation_id=None,
            optimization_id=optimization_id
        )
        
        # Dispatch query
        optimization = await query_bus.dispatch(query)
        
        if not optimization:
            raise HTTPException(status_code=404, detail=f"Optimization {optimization_id} not found")
        
        return optimization
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization: {str(e)}")


@router.post("/walk-forward", response_model=WalkForwardTestResponse)
async def run_walk_forward_test(
    request: WalkForwardTestRequest,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Run a walk-forward test with the specified configuration.
    """
    try:
        # Create command
        command = RunWalkForwardTestCommand(
            correlation_id=None,
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_balance=request.initial_balance,
            parameters=request.parameters,
            optimization_window=request.optimization_window,
            test_window=request.test_window,
            optimization_metric=request.optimization_metric,
            parameters_to_optimize=request.parameters_to_optimize
        )
        
        # Dispatch command
        test_id = await command_bus.dispatch(command)
        
        # Return response
        return WalkForwardTestResponse(
            test_id=test_id,
            status="pending",
            message="Walk-forward test started successfully",
            estimated_completion_time=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run walk-forward test: {str(e)}")


@router.get("/walk-forward/{test_id}", response_model=WalkForwardTestResult)
async def get_walk_forward_test(
    test_id: str,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get a walk-forward test by ID.
    """
    try:
        # Create query
        query = GetWalkForwardTestQuery(
            correlation_id=None,
            test_id=test_id
        )
        
        # Dispatch query
        test = await query_bus.dispatch(query)
        
        if not test:
            raise HTTPException(status_code=404, detail=f"Walk-forward test {test_id} not found")
        
        return test
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get walk-forward test: {str(e)}")


@router.get("/strategies", response_model=StrategyListResponse)
async def list_strategies(
    category: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    List available strategies.
    """
    try:
        # Create query
        query = ListStrategiesQuery(
            correlation_id=None,
            category=category,
            limit=limit,
            offset=offset
        )
        
        # Dispatch query
        strategies = await query_bus.dispatch(query)
        
        return strategies
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list strategies: {str(e)}")


@router.post("/{backtest_id}/cancel")
async def cancel_backtest(
    backtest_id: str,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Cancel a running backtest.
    """
    try:
        # Create command
        command = CancelBacktestCommand(
            correlation_id=None,
            backtest_id=backtest_id
        )
        
        # Dispatch command
        await command_bus.dispatch(command)
        
        return {"message": f"Backtest {backtest_id} cancelled successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel backtest: {str(e)}")


@router.delete("/{backtest_id}")
async def delete_backtest(
    backtest_id: str,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Delete a backtest.
    """
    try:
        # Create command
        command = DeleteBacktestCommand(
            correlation_id=None,
            backtest_id=backtest_id
        )
        
        # Dispatch command
        await command_bus.dispatch(command)
        
        return {"message": f"Backtest {backtest_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete backtest: {str(e)}")