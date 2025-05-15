"""
API routes for the Backtesting Service.

This module defines the API routes for the Backtesting Service, including
backtesting, walk-forward optimization, Monte Carlo simulation, and stress testing.
"""
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from backtesting_service.services.backtesting_service import BacktestingService
from backtesting_service.utils.correlation_id import get_correlation_id
from backtesting_service.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["backtesting"])


from backtesting_service.models.backtest_models import (
    StrategyConfig as ModelStrategyConfig,  # Alias to avoid name clash if needed later
    BacktestRequest as ModelBacktestRequest,
    TradeResult as ModelTradeResult, # Assuming TradeMetrics in routes.py maps to TradeResult
    PerformanceMetrics as ModelPerformanceMetrics,
    BacktestResponse as ModelBacktestResponse
)

# Use the imported models, potentially aliasing if direct names are preferred in routes
# For now, let's assume we want to use the direct names from backtest_models for clarity
# and will adjust endpoint signatures accordingly.

class WalkForwardRequest(BaseModel):
    """Request model for walk-forward optimization"""
    strategy_config: ModelStrategyConfig = Field(..., description='Strategy configuration')
    parameter_ranges: Dict[str, List[Any]] = Field(..., description='Parameter ranges for optimization')
    start_date: datetime = Field(..., description='Start date for backtesting')
    end_date: datetime = Field(..., description='End date for backtesting')
    instruments: List[str] = Field(..., description='List of instruments to backtest')
    initial_capital: float = Field(10000.0, description='Initial capital for backtesting')
    window_size_days: int = Field(90, description='Size of each window in days')
    anchor_size_days: int = Field(30, description='Size of each anchor (out-of-sample) period in days')
    optimization_metric: str = Field('sharpe_ratio', description='Metric to optimize for')
    parallel_jobs: int = Field(1, description='Number of parallel jobs for optimization')


class MonteCarloRequest(BaseModel):
    """Request model for Monte Carlo simulation"""
    backtest_id: str = Field(..., description='ID of the backtest to simulate')
    num_simulations: int = Field(1000, description='Number of Monte Carlo simulations to run')
    confidence_level: float = Field(0.95, description='Confidence level for results (0-1)')
    simulation_method: str = Field('bootstrap', description='Simulation method (bootstrap, parametric)')
    simulation_parameters: Optional[Dict[str, Any]] = Field(None, description='Additional simulation parameters')


class StressTestRequest(BaseModel):
    """Request model for stress testing"""
    backtest_id: str = Field(..., description='ID of the backtest to stress test')
    stress_scenarios: List[Dict[str, Any]] = Field(..., description='Stress scenarios to test')
    apply_to_parameters: Optional[List[str]] = Field(None, description='Parameters to apply stress to')


class TradeMetrics(BaseModel):
    """Model for trade metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_holding_period: float


# PerformanceMetrics and BacktestResponse are now imported from backtesting_service.models.backtest_models
# The local definitions are removed.

class WalkForwardResponse(BaseModel):
    """Response model for walk-forward optimization results"""
    optimization_id: str = Field(..., description='Unique identifier for the optimization')
    strategy_id: str = Field(..., description='Strategy identifier')
    windows: List[Dict[str, Any]] = Field(..., description='Results for each window')
    optimal_parameters: Dict[str, Any] = Field(..., description='Optimal parameters')
    in_sample_metrics: PerformanceMetrics = Field(..., description='In-sample performance metrics')
    out_of_sample_metrics: PerformanceMetrics = Field(..., description='Out-of-sample performance metrics')
    robustness_score: float = Field(..., description='Robustness score (0-1)')


class MonteCarloResponse(BaseModel):
    """Response model for Monte Carlo simulation results"""
    simulation_id: str = Field(..., description='Unique identifier for the simulation')
    backtest_id: str = Field(..., description='Original backtest ID')
    num_simulations: int = Field(..., description='Number of simulations run')
    confidence_level: float = Field(..., description='Confidence level used')
    expected_return: float = Field(..., description='Expected return')
    return_range: List[float] = Field(..., description='Return range at confidence level')
    max_drawdown_range: List[float] = Field(..., description='Max drawdown range at confidence level')
    probability_of_profit: float = Field(..., description='Probability of profit')
    probability_of_target: Optional[float] = Field(None, description='Probability of reaching target return')
    simulation_percentiles: Dict[str, List[float]] = Field(..., description='Percentiles for key metrics')


class StressTestResponse(BaseModel):
    """Response model for stress test results"""
    stress_test_id: str = Field(..., description='Unique identifier for the stress test')
    backtest_id: str = Field(..., description='Original backtest ID')
    baseline_metrics: PerformanceMetrics = Field(..., description='Baseline performance metrics')
    scenario_results: List[Dict[str, Any]] = Field(..., description='Results for each stress scenario')
    impact_summary: Dict[str, Any] = Field(..., description='Summary of stress test impacts')
    most_vulnerable_to: str = Field(..., description='Scenario with highest negative impact')


def get_backtesting_service():
    """Dependency for getting the BacktestingService instance."""
    return BacktestingService()


@router.post("/run", response_model=ModelBacktestResponse) # Use imported ModelBacktestResponse
async def run_backtest(
    request: ModelBacktestRequest, # Use imported ModelBacktestRequest
    req: Request,
    backtesting_service: BacktestingService = Depends(get_backtesting_service)
):
    """
    Run a backtest for a trading strategy with the specified configuration.
    
    This endpoint executes a backtest for the specified strategy over the given time period
    and returns detailed performance metrics and trade history.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to run backtest for strategy {request.strategy_config.strategy_id}", 
                extra={"correlation_id": correlation_id})
    
    try:
        result = await backtesting_service.run_backtest(
            strategy_id=request.strategy_config.strategy_id,
            strategy_parameters=request.strategy_config.parameters,
            risk_settings=request.strategy_config.risk_settings or {},
            position_sizing=request.strategy_config.position_sizing or {},
            start_date=request.start_date,
            end_date=request.end_date,
            instruments=request.instruments,
            initial_capital=request.initial_capital,
            commission_model=request.commission_model,
            commission_settings=request.commission_settings or {},
            slippage_model=request.slippage_model,
            slippage_settings=request.slippage_settings or {},
            data_source=request.data_source,
            data_parameters=request.data_parameters or {},
            correlation_id=correlation_id
        )
        return result
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")


@router.get("/{backtest_id}", response_model=BacktestResponse)
async def get_backtest_results(
    backtest_id: str,
    req: Request,
    include_trades: bool = Query(True, description='Include trade details in response'),
    include_equity_curve: bool = Query(True, description='Include equity curve in response'),
    backtesting_service: BacktestingService = Depends(get_backtesting_service)
):
    """
    Get the results of a previously run backtest.
    
    This endpoint retrieves the detailed results of a backtest that was previously executed,
    including performance metrics, trade history, and equity curve.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to get backtest results for {backtest_id}", 
                extra={"correlation_id": correlation_id})
    
    try:
        result = await backtesting_service.get_backtest_results(
            backtest_id=backtest_id,
            include_trades=include_trades,
            include_equity_curve=include_equity_curve,
            correlation_id=correlation_id
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Backtest not found: {backtest_id}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving backtest results: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error retrieving backtest results: {str(e)}")


@router.post("/walk-forward", response_model=WalkForwardResponse)
async def run_walk_forward_optimization(
    request: WalkForwardRequest,
    req: Request,
    backtesting_service: BacktestingService = Depends(get_backtesting_service)
):
    """
    Run walk-forward optimization for a trading strategy.
    
    This endpoint performs walk-forward optimization, which helps prevent overfitting
    by testing the strategy on out-of-sample data after optimizing on in-sample data.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to run walk-forward optimization for strategy {request.strategy_config.strategy_id}", 
                extra={"correlation_id": correlation_id})
    
    try:
        result = await backtesting_service.run_walk_forward_optimization(
            strategy_id=request.strategy_config.strategy_id,
            strategy_parameters=request.strategy_config.parameters,
            parameter_ranges=request.parameter_ranges,
            start_date=request.start_date,
            end_date=request.end_date,
            instruments=request.instruments,
            initial_capital=request.initial_capital,
            window_size_days=request.window_size_days,
            anchor_size_days=request.anchor_size_days,
            optimization_metric=request.optimization_metric,
            parallel_jobs=request.parallel_jobs,
            correlation_id=correlation_id
        )
        return result
    except Exception as e:
        logger.error(f"Error running walk-forward optimization: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error running walk-forward optimization: {str(e)}")


@router.post("/monte-carlo", response_model=MonteCarloResponse)
async def run_monte_carlo_simulation(
    request: MonteCarloRequest,
    req: Request,
    backtesting_service: BacktestingService = Depends(get_backtesting_service)
):
    """
    Run Monte Carlo simulation for a previously executed backtest.
    
    This endpoint performs Monte Carlo simulation to estimate the range of possible
    outcomes and provide statistical confidence intervals for backtest results.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to run Monte Carlo simulation for backtest {request.backtest_id}", 
                extra={"correlation_id": correlation_id})
    
    try:
        result = await backtesting_service.run_monte_carlo_simulation(
            backtest_id=request.backtest_id,
            num_simulations=request.num_simulations,
            confidence_level=request.confidence_level,
            simulation_method=request.simulation_method,
            simulation_parameters=request.simulation_parameters or {},
            correlation_id=correlation_id
        )
        return result
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error running Monte Carlo simulation: {str(e)}")


@router.post("/stress-test", response_model=StressTestResponse)
async def run_stress_test(
    request: StressTestRequest,
    req: Request,
    backtesting_service: BacktestingService = Depends(get_backtesting_service)
):
    """
    Run stress test for a previously executed backtest.
    
    This endpoint performs stress testing to evaluate how a strategy would perform
    under extreme market conditions or specific adverse scenarios.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to run stress test for backtest {request.backtest_id}", 
                extra={"correlation_id": correlation_id})
    
    try:
        result = await backtesting_service.run_stress_test(
            backtest_id=request.backtest_id,
            stress_scenarios=request.stress_scenarios,
            apply_to_parameters=request.apply_to_parameters,
            correlation_id=correlation_id
        )
        return result
    except Exception as e:
        logger.error(f"Error running stress test: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error running stress test: {str(e)}")


@router.get("/strategies", response_model=List[Dict[str, Any]])
async def get_available_strategies(
    req: Request,
    backtesting_service: BacktestingService = Depends(get_backtesting_service)
):
    """
    Get a list of available strategies for backtesting.
    
    This endpoint returns a list of strategies that can be used for backtesting,
    including their IDs, names, descriptions, and parameter specifications.
    """
    correlation_id = get_correlation_id(req)
    logger.info("Received request to get available strategies", extra={"correlation_id": correlation_id})
    
    try:
        strategies = await backtesting_service.get_available_strategies(correlation_id=correlation_id)
        return strategies
    except Exception as e:
        logger.error(f"Error retrieving available strategies: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error retrieving available strategies: {str(e)}")


@router.get("/data-sources", response_model=List[Dict[str, Any]])
async def get_available_data_sources(
    req: Request,
    backtesting_service: BacktestingService = Depends(get_backtesting_service)
):
    """
    Get a list of available data sources for backtesting.
    
    This endpoint returns a list of data sources that can be used for backtesting,
    including their IDs, names, descriptions, and available instruments.
    """
    correlation_id = get_correlation_id(req)
    logger.info("Received request to get available data sources", extra={"correlation_id": correlation_id})
    
    try:
        data_sources = await backtesting_service.get_available_data_sources(correlation_id=correlation_id)
        return data_sources
    except Exception as e:
        logger.error(f"Error retrieving available data sources: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error retrieving available data sources: {str(e)}")


@router.post("/{backtest_id}/cancel", response_model=Dict[str, Any])
async def cancel_backtest(
    backtest_id: str,
    req: Request,
    backtesting_service: BacktestingService = Depends(get_backtesting_service)
):
    """
    Cancel a running backtest.
    
    This endpoint cancels a backtest that is currently running.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to cancel backtest {backtest_id}", extra={"correlation_id": correlation_id})
    
    try:
        result = await backtesting_service.cancel_backtest(
            backtest_id=backtest_id,
            correlation_id=correlation_id
        )
        return result
    except Exception as e:
        logger.error(f"Error canceling backtest: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error canceling backtest: {str(e)}")


@router.get("/list", response_model=Dict[str, Any])
async def list_backtests(
    req: Request,
    status: Optional[str] = None,
    strategy_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
    backtesting_service: BacktestingService = Depends(get_backtesting_service)
):
    """
    List backtests with optional filtering.
    
    This endpoint returns a list of backtests that match the specified filters.
    """
    correlation_id = get_correlation_id(req)
    logger.info("Received request to list backtests", extra={"correlation_id": correlation_id})
    
    try:
        result = await backtesting_service.list_backtests(
            status=status,
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
            correlation_id=correlation_id
        )
        return result
    except Exception as e:
        logger.error(f"Error listing backtests: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error listing backtests: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint.
    
    This endpoint returns the health status of the service.
    """
    return {
        "status": "healthy",
        "service": "backtesting-service",
        "timestamp": datetime.utcnow().isoformat()
    }