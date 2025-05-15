"""
Backtesting API Routes

This module defines the API routes for backtesting.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional, Any

from app.services.backtest_service import BacktestService
from app.core.service_dependencies import get_backtest_service
from app.models.backtest_models import (
    BacktestRequest,
    BacktestResponse,
    BacktestResult,
    BacktestListResponse,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationResult,
    WalkForwardTestRequest,
    WalkForwardTestResponse,
    WalkForwardTestResult,
    StrategyListResponse,
    StrategyMetadata
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["backtest"])

@router.post("/backtests", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """
    Run a backtest for a trading strategy.

    This endpoint runs a backtest for a trading strategy against historical market data.

    Parameters:
    - strategy_id: ID of the strategy to backtest
    - symbol: Symbol to backtest
    - timeframe: Timeframe for the backtest
    - start_date: Start date for the backtest
    - end_date: End date for the backtest
    - initial_balance: Initial balance for the backtest
    - parameters: Parameters for the backtest

    Returns:
    - backtest_id: ID of the backtest
    - status: Status of the backtest
    - message: Message about the backtest
    - estimated_completion_time: Estimated completion time
    """
    try:
        logger.info(f"Running backtest for strategy {request.strategy_id} on {request.symbol} from {request.start_date} to {request.end_date}")
        return await backtest_service.run_backtest(request)
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtests/{backtest_id}", response_model=BacktestResult)
async def get_backtest_result(
    backtest_id: str,
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """
    Get the result of a backtest.

    This endpoint retrieves the result of a backtest by its ID.

    Parameters:
    - backtest_id: ID of the backtest

    Returns:
    - backtest_id: ID of the backtest
    - strategy_id: ID of the strategy
    - start_date: Start date of the backtest
    - end_date: End date of the backtest
    - initial_balance: Initial balance for the backtest
    - final_balance: Final balance after the backtest
    - total_trades: Total number of trades
    - winning_trades: Number of winning trades
    - losing_trades: Number of losing trades
    - performance_metrics: Performance metrics for the backtest
    - trades: List of trades
    - equity_curve: Equity curve for the backtest
    - parameters: Parameters used for the backtest
    """
    try:
        logger.info(f"Getting backtest result for {backtest_id}")
        result = await backtest_service.get_backtest_result(backtest_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtests", response_model=List[Dict[str, Any]])
async def list_backtests(
    strategy_id: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """
    List backtests with optional filtering.

    This endpoint lists backtests with optional filtering by strategy ID and symbol.

    Parameters:
    - strategy_id: Optional strategy ID to filter by
    - symbol: Optional symbol to filter by
    - limit: Maximum number of results to return
    - offset: Offset for pagination

    Returns:
    - backtests: List of backtests
    - count: Number of backtests
    """
    try:
        logger.info(f"Listing backtests with strategy_id={strategy_id}, symbol={symbol}, limit={limit}, offset={offset}")
        return await backtest_service.list_backtests(
            strategy_id=strategy_id,
            symbol=symbol,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing backtests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_strategy(
    request: OptimizationRequest,
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """
    Optimize a trading strategy.

    This endpoint optimizes a trading strategy by finding the best parameters.

    Parameters:
    - strategy_id: ID of the strategy to optimize
    - symbol: Symbol to optimize for
    - timeframe: Timeframe for the optimization
    - start_date: Start date for the optimization
    - end_date: End date for the optimization
    - initial_balance: Initial balance for the optimization
    - parameters_to_optimize: Parameters to optimize with ranges
    - optimization_metric: Metric to optimize for
    - optimization_method: Optimization method to use
    - max_evaluations: Maximum number of evaluations

    Returns:
    - optimization_id: ID of the optimization
    - status: Status of the optimization
    - message: Message about the optimization
    - estimated_completion_time: Estimated completion time
    """
    try:
        logger.info(f"Optimizing strategy {request.strategy_id} on {request.symbol} from {request.start_date} to {request.end_date}")
        return await backtest_service.optimize_strategy(request)
    except Exception as e:
        logger.error(f"Error optimizing strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimize/{optimization_id}", response_model=OptimizationResult)
async def get_optimization_result(
    optimization_id: str,
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """
    Get the result of an optimization.

    This endpoint retrieves the result of an optimization by its ID.

    Parameters:
    - optimization_id: ID of the optimization

    Returns:
    - optimization_id: ID of the optimization
    - strategy_id: ID of the strategy
    - symbol: Symbol optimized for
    - timeframe: Timeframe for the optimization
    - start_date: Start date of the optimization
    - end_date: End date of the optimization
    - optimization_metric: Metric optimized for
    - optimization_method: Optimization method used
    - best_parameters: Best parameters found
    - best_metric_value: Best metric value found
    - evaluations: Number of evaluations performed
    - all_results: All optimization results
    """
    try:
        logger.info(f"Getting optimization result for {optimization_id}")
        result = await backtest_service.get_optimization_result(optimization_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Optimization {optimization_id} not found")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting optimization result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/walk-forward", response_model=WalkForwardTestResponse)
async def run_walk_forward_test(
    request: WalkForwardTestRequest,
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """
    Run a walk-forward test for a trading strategy.

    This endpoint runs a walk-forward test for a trading strategy.

    Parameters:
    - strategy_id: ID of the strategy to test
    - symbol: Symbol to test
    - timeframe: Timeframe for the test
    - start_date: Start date for the test
    - end_date: End date for the test
    - initial_balance: Initial balance for the test
    - parameters: Base parameters for the strategy
    - optimization_window: Number of bars in the optimization window
    - test_window: Number of bars in the test window
    - optimization_metric: Metric to optimize for
    - parameters_to_optimize: Parameters to optimize with ranges

    Returns:
    - test_id: ID of the walk-forward test
    - status: Status of the test
    - message: Message about the test
    - estimated_completion_time: Estimated completion time
    """
    try:
        logger.info(f"Running walk-forward test for strategy {request.strategy_id} on {request.symbol} from {request.start_date} to {request.end_date}")
        return await backtest_service.run_walk_forward_test(request)
    except Exception as e:
        logger.error(f"Error running walk-forward test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/walk-forward/{test_id}", response_model=WalkForwardTestResult)
async def get_walk_forward_test_result(
    test_id: str,
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """
    Get the result of a walk-forward test.

    This endpoint retrieves the result of a walk-forward test by its ID.

    Parameters:
    - test_id: ID of the walk-forward test

    Returns:
    - test_id: ID of the walk-forward test
    - strategy_id: ID of the strategy
    - symbol: Symbol tested
    - timeframe: Timeframe for the test
    - start_date: Start date of the test
    - end_date: End date of the test
    - initial_balance: Initial balance for the test
    - final_balance: Final balance after the test
    - total_trades: Total number of trades
    - winning_trades: Number of winning trades
    - losing_trades: Number of losing trades
    - performance_metrics: Performance metrics for the test
    - trades: List of trades
    - equity_curve: Equity curve for the test
    - windows: List of optimization and test windows
    - parameters_by_window: Parameters used for each window
    """
    try:
        logger.info(f"Getting walk-forward test result for {test_id}")
        result = await backtest_service.get_walk_forward_test_result(test_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Walk-forward test {test_id} not found")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting walk-forward test result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies", response_model=List[StrategyMetadata])
async def list_strategies(
    backtest_service: BacktestService = Depends(get_backtest_service)
):
    """
    List available strategies.

    This endpoint lists all available strategies.

    Returns:
    - strategies: List of available strategies
    - count: Number of strategies
    """
    try:
        logger.info("Listing available strategies")
        result = await backtest_service.list_strategies()
        return result.strategies
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))
