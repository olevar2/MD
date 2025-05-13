"""
Standardized Backtesting API for Analysis Engine Service.

This module provides standardized API endpoints for backtesting capabilities,
including strategy backtesting, walk-forward optimization, Monte Carlo simulation,
and stress testing.

All endpoints follow the platform's standardized API design patterns.
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException, Body, Request
from pydantic import BaseModel, Field
from analysis_engine.analysis.backtesting.core import BacktestConfiguration, BacktestResult, DataSplit
from analysis_engine.analysis.backtesting.walk_forward import WalkForwardOptimizationConfig, WalkForwardResult
from analysis_engine.analysis.backtesting.monte_carlo import MonteCarloSimulationConfig, MonteCarloResult
from analysis_engine.analysis.backtesting.stress_testing import StressTestConfig, StressTestResult
from analysis_engine.services.backtesting_service import BacktestingService
from analysis_engine.api.dependencies import get_backtesting_service
from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError, AnalysisError, BacktestingError, get_correlation_id_from_request
from analysis_engine.monitoring.structured_logging import get_structured_logger


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class StrategyConfig(BaseModel):
    """Model for strategy configuration"""
    strategy_id: str = Field(..., description='Identifier for the strategy')
    parameters: Dict[str, Any] = Field(default_factory=dict, description=
        'Strategy parameters')
    risk_settings: Optional[Dict[str, Any]] = Field(None, description=
        'Risk management settings')
    position_sizing: Optional[Dict[str, Any]] = Field(None, description=
        'Position sizing settings')


class BacktestRequest(BaseModel):
    """Request model for backtesting"""
    strategy_config: StrategyConfig = Field(..., description=
        'Strategy configuration')
    start_date: datetime = Field(..., description='Start date for backtesting')
    end_date: datetime = Field(..., description='End date for backtesting')
    instruments: List[str] = Field(..., description=
        'List of instruments to backtest')
    initial_capital: float = Field(10000.0, description=
        'Initial capital for backtesting')
    commission_model: Optional[str] = Field('fixed', description=
        'Commission model (fixed, percentage)')
    commission_settings: Optional[Dict[str, Any]] = Field(None, description
        ='Commission settings')
    slippage_model: Optional[str] = Field('fixed', description=
        'Slippage model (fixed, percentage, variable)')
    slippage_settings: Optional[Dict[str, Any]] = Field(None, description=
        'Slippage settings')
    data_source: Optional[str] = Field('historical', description=
        'Data source (historical, generated)')
    data_parameters: Optional[Dict[str, Any]] = Field(None, description=
        'Data source parameters')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'strategy_config': {'strategy_id':
            'moving_average_crossover', 'parameters': {'fast_period': 10,
            'slow_period': 30}, 'risk_settings': {'max_drawdown_pct': 20,
            'max_risk_per_trade_pct': 2}, 'position_sizing': {'method':
            'fixed_risk', 'risk_pct': 1}}, 'start_date':
            '2023-01-01T00:00:00Z', 'end_date': '2023-12-31T23:59:59Z',
            'instruments': ['EUR_USD', 'GBP_USD'], 'initial_capital': 
            10000.0, 'commission_model': 'fixed', 'commission_settings': {
            'fixed_commission': 5.0}, 'slippage_model': 'fixed',
            'slippage_settings': {'fixed_pips': 1}, 'data_source':
            'historical', 'data_parameters': {'timeframe': 'H1'}}}


class WalkForwardRequest(BaseModel):
    """Request model for walk-forward optimization"""
    strategy_config: StrategyConfig = Field(..., description=
        'Strategy configuration')
    parameter_ranges: Dict[str, List[Any]] = Field(..., description=
        'Parameter ranges for optimization')
    start_date: datetime = Field(..., description='Start date for backtesting')
    end_date: datetime = Field(..., description='End date for backtesting')
    instruments: List[str] = Field(..., description=
        'List of instruments to backtest')
    initial_capital: float = Field(10000.0, description=
        'Initial capital for backtesting')
    window_size_days: int = Field(90, description='Size of each window in days'
        )
    anchor_size_days: int = Field(30, description=
        'Size of each anchor (out-of-sample) period in days')
    optimization_metric: str = Field('sharpe_ratio', description=
        'Metric to optimize for')
    parallel_jobs: int = Field(1, description=
        'Number of parallel jobs for optimization')


class MonteCarloRequest(BaseModel):
    """Request model for Monte Carlo simulation"""
    backtest_id: str = Field(..., description='ID of the backtest to simulate')
    num_simulations: int = Field(1000, description=
        'Number of Monte Carlo simulations to run')
    confidence_level: float = Field(0.95, description=
        'Confidence level for results (0-1)')
    simulation_method: str = Field('bootstrap', description=
        'Simulation method (bootstrap, parametric)')
    simulation_parameters: Optional[Dict[str, Any]] = Field(None,
        description='Additional simulation parameters')


class StressTestRequest(BaseModel):
    """Request model for stress testing"""
    backtest_id: str = Field(..., description=
        'ID of the backtest to stress test')
    stress_scenarios: List[Dict[str, Any]] = Field(..., description=
        'Stress scenarios to test')
    apply_to_parameters: Optional[List[str]] = Field(None, description=
        'Parameters to apply stress to')


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


class PerformanceMetrics(BaseModel):
    """Model for performance metrics"""
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    value_at_risk: float
    expected_shortfall: float


class BacktestResponse(BaseModel):
    """Response model for backtest results"""
    backtest_id: str = Field(..., description=
        'Unique identifier for the backtest')
    strategy_id: str = Field(..., description='Strategy identifier')
    start_date: datetime = Field(..., description='Start date of the backtest')
    end_date: datetime = Field(..., description='End date of the backtest')
    instruments: List[str] = Field(..., description=
        'Instruments used in the backtest')
    initial_capital: float = Field(..., description='Initial capital')
    final_capital: float = Field(..., description='Final capital')
    trade_metrics: TradeMetrics = Field(..., description='Trade metrics')
    performance_metrics: PerformanceMetrics = Field(..., description=
        'Performance metrics')
    equity_curve: List[Dict[str, Any]] = Field(..., description=
        'Equity curve data points')
    trades: List[Dict[str, Any]] = Field(..., description='List of trades')
    drawdowns: List[Dict[str, Any]] = Field(..., description=
        'List of drawdowns')


class WalkForwardResponse(BaseModel):
    """Response model for walk-forward optimization results"""
    optimization_id: str = Field(..., description=
        'Unique identifier for the optimization')
    strategy_id: str = Field(..., description='Strategy identifier')
    windows: List[Dict[str, Any]] = Field(..., description=
        'Results for each window')
    optimal_parameters: Dict[str, Any] = Field(..., description=
        'Optimal parameters')
    in_sample_metrics: PerformanceMetrics = Field(..., description=
        'In-sample performance metrics')
    out_of_sample_metrics: PerformanceMetrics = Field(..., description=
        'Out-of-sample performance metrics')
    robustness_score: float = Field(..., description='Robustness score (0-1)')


class MonteCarloResponse(BaseModel):
    """Response model for Monte Carlo simulation results"""
    simulation_id: str = Field(..., description=
        'Unique identifier for the simulation')
    backtest_id: str = Field(..., description='Original backtest ID')
    num_simulations: int = Field(..., description='Number of simulations run')
    confidence_level: float = Field(..., description='Confidence level used')
    expected_return: float = Field(..., description='Expected return')
    return_range: List[float] = Field(..., description=
        'Return range at confidence level')
    max_drawdown_range: List[float] = Field(..., description=
        'Max drawdown range at confidence level')
    probability_of_profit: float = Field(..., description=
        'Probability of profit')
    probability_of_target: Optional[float] = Field(None, description=
        'Probability of reaching target return')
    simulation_percentiles: Dict[str, List[float]] = Field(..., description
        ='Percentiles for key metrics')


class StressTestResponse(BaseModel):
    """Response model for stress test results"""
    stress_test_id: str = Field(..., description=
        'Unique identifier for the stress test')
    backtest_id: str = Field(..., description='Original backtest ID')
    baseline_metrics: PerformanceMetrics = Field(..., description=
        'Baseline performance metrics')
    scenario_results: List[Dict[str, Any]] = Field(..., description=
        'Results for each stress scenario')
    impact_summary: Dict[str, Any] = Field(..., description=
        'Summary of stress test impacts')
    most_vulnerable_to: str = Field(..., description=
        'Scenario with highest negative impact')


router = APIRouter(prefix='/v1/analysis/backtesting', tags=['Backtesting'])
logger = get_structured_logger(__name__)


@router.post('/run', response_model=BacktestResponse, summary=
    'Run a backtest', description=
    'Run a backtest for a trading strategy with the specified configuration.')
@async_with_exception_handling
async def run_backtest(request: BacktestRequest, request_obj: Request,
    backtesting_service: BacktestingService=Depends(get_backtesting_service)):
    """
    Run a backtest for a trading strategy with the specified configuration.

    This endpoint executes a backtest for the specified strategy over the given time period
    and returns detailed performance metrics and trade history.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        config = BacktestConfiguration(strategy_id=request.strategy_config.
            strategy_id, strategy_parameters=request.strategy_config.
            parameters, risk_settings=request.strategy_config.risk_settings or
            {}, position_sizing=request.strategy_config.position_sizing or
            {}, start_date=request.start_date, end_date=request.end_date,
            instruments=request.instruments, initial_capital=request.
            initial_capital, commission_model=request.commission_model,
            commission_settings=request.commission_settings or {},
            slippage_model=request.slippage_model, slippage_settings=
            request.slippage_settings or {}, data_source=request.
            data_source, data_parameters=request.data_parameters or {})
        result = await backtesting_service.run_backtest(config,
            correlation_id=correlation_id)
        logger.info(
            f'Backtest completed for strategy {request.strategy_config.strategy_id}'
            , extra={'correlation_id': correlation_id, 'strategy_id':
            request.strategy_config.strategy_id, 'backtest_id': result.
            backtest_id, 'instruments': request.instruments, 'start_date':
            request.start_date.isoformat(), 'end_date': request.end_date.
            isoformat(), 'total_return_pct': result.performance_metrics.
            total_return_pct, 'trade_count': result.trade_metrics.total_trades}
            )
        return result
    except Exception as e:
        logger.error(
            f'Error running backtest for strategy {request.strategy_config.strategy_id}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        raise BacktestingError(message=f'Error running backtest: {str(e)}',
            correlation_id=correlation_id)


@router.get('/{backtest_id}', response_model=BacktestResponse, summary=
    'Get backtest results', description=
    'Get the results of a previously run backtest.')
@async_with_exception_handling
async def get_backtest_results(backtest_id: str, request_obj: Request,
    include_trades: bool=Query(True, description=
    'Include trade details in response'), include_equity_curve: bool=Query(
    True, description='Include equity curve in response'),
    backtesting_service: BacktestingService=Depends(get_backtesting_service)):
    """
    Get the results of a previously run backtest.

    This endpoint retrieves the detailed results of a backtest that was previously executed,
    including performance metrics, trade history, and equity curve.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        result = await backtesting_service.get_backtest_results(backtest_id,
            include_trades=include_trades, include_equity_curve=
            include_equity_curve, correlation_id=correlation_id)
        if not result:
            logger.warning(f'Backtest not found: {backtest_id}', extra={
                'correlation_id': correlation_id})
            raise HTTPException(status_code=404, detail=
                f'Backtest not found: {backtest_id}')
        logger.info(f'Retrieved backtest results for {backtest_id}', extra=
            {'correlation_id': correlation_id, 'backtest_id': backtest_id,
            'strategy_id': result.strategy_id})
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f'Error retrieving backtest results for {backtest_id}: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise BacktestingError(message=
            f'Error retrieving backtest results: {str(e)}', correlation_id=
            correlation_id)


@router.post('/walk-forward', response_model=WalkForwardResponse, summary=
    'Run walk-forward optimization', description=
    'Run walk-forward optimization for a trading strategy.')
@async_with_exception_handling
async def run_walk_forward_optimization(request: WalkForwardRequest,
    request_obj: Request, backtesting_service: BacktestingService=Depends(
    get_backtesting_service)):
    """
    Run walk-forward optimization for a trading strategy.

    This endpoint performs walk-forward optimization, which helps prevent overfitting
    by testing the strategy on out-of-sample data after optimizing on in-sample data.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        config = WalkForwardOptimizationConfig(strategy_id=request.
            strategy_config.strategy_id, strategy_parameters=request.
            strategy_config.parameters, parameter_ranges=request.
            parameter_ranges, start_date=request.start_date, end_date=
            request.end_date, instruments=request.instruments,
            initial_capital=request.initial_capital, window_size_days=
            request.window_size_days, anchor_size_days=request.
            anchor_size_days, optimization_metric=request.
            optimization_metric, parallel_jobs=request.parallel_jobs)
        result = await backtesting_service.run_walk_forward_optimization(config
            , correlation_id=correlation_id)
        logger.info(
            f'Walk-forward optimization completed for strategy {request.strategy_config.strategy_id}'
            , extra={'correlation_id': correlation_id, 'strategy_id':
            request.strategy_config.strategy_id, 'optimization_id': result.
            optimization_id, 'instruments': request.instruments,
            'start_date': request.start_date.isoformat(), 'end_date':
            request.end_date.isoformat(), 'window_count': len(result.
            windows), 'robustness_score': result.robustness_score})
        return result
    except Exception as e:
        logger.error(
            f'Error running walk-forward optimization for strategy {request.strategy_config.strategy_id}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        raise BacktestingError(message=
            f'Error running walk-forward optimization: {str(e)}',
            correlation_id=correlation_id)


@router.post('/monte-carlo', response_model=MonteCarloResponse, summary=
    'Run Monte Carlo simulation', description=
    'Run Monte Carlo simulation for a previously executed backtest.')
@async_with_exception_handling
async def run_monte_carlo_simulation(request: MonteCarloRequest,
    request_obj: Request, backtesting_service: BacktestingService=Depends(
    get_backtesting_service)):
    """
    Run Monte Carlo simulation for a previously executed backtest.

    This endpoint performs Monte Carlo simulation to estimate the range of possible
    outcomes and provide statistical confidence intervals for backtest results.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        config = MonteCarloSimulationConfig(backtest_id=request.backtest_id,
            num_simulations=request.num_simulations, confidence_level=
            request.confidence_level, simulation_method=request.
            simulation_method, simulation_parameters=request.
            simulation_parameters or {})
        result = await backtesting_service.run_monte_carlo_simulation(config,
            correlation_id=correlation_id)
        logger.info(
            f'Monte Carlo simulation completed for backtest {request.backtest_id}'
            , extra={'correlation_id': correlation_id, 'backtest_id':
            request.backtest_id, 'simulation_id': result.simulation_id,
            'num_simulations': request.num_simulations, 'confidence_level':
            request.confidence_level, 'expected_return': result.
            expected_return, 'probability_of_profit': result.
            probability_of_profit})
        return result
    except Exception as e:
        logger.error(
            f'Error running Monte Carlo simulation for backtest {request.backtest_id}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        raise BacktestingError(message=
            f'Error running Monte Carlo simulation: {str(e)}',
            correlation_id=correlation_id)


@router.post('/stress-test', response_model=StressTestResponse, summary=
    'Run stress test', description=
    'Run stress test for a previously executed backtest.')
@async_with_exception_handling
async def run_stress_test(request: StressTestRequest, request_obj: Request,
    backtesting_service: BacktestingService=Depends(get_backtesting_service)):
    """
    Run stress test for a previously executed backtest.

    This endpoint performs stress testing to evaluate how a strategy would perform
    under extreme market conditions or specific adverse scenarios.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        config = StressTestConfig(backtest_id=request.backtest_id,
            stress_scenarios=request.stress_scenarios, apply_to_parameters=
            request.apply_to_parameters)
        result = await backtesting_service.run_stress_test(config,
            correlation_id=correlation_id)
        logger.info(f'Stress test completed for backtest {request.backtest_id}'
            , extra={'correlation_id': correlation_id, 'backtest_id':
            request.backtest_id, 'stress_test_id': result.stress_test_id,
            'scenario_count': len(request.stress_scenarios),
            'most_vulnerable_to': result.most_vulnerable_to})
        return result
    except Exception as e:
        logger.error(
            f'Error running stress test for backtest {request.backtest_id}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        raise BacktestingError(message=
            f'Error running stress test: {str(e)}', correlation_id=
            correlation_id)


@router.get('/strategies', response_model=List[Dict[str, Any]], summary=
    'Get available strategies', description=
    'Get a list of available strategies for backtesting.')
@async_with_exception_handling
async def get_available_strategies(request_obj: Request,
    backtesting_service: BacktestingService=Depends(get_backtesting_service)):
    """
    Get a list of available strategies for backtesting.

    This endpoint returns a list of strategies that can be used for backtesting,
    including their IDs, names, descriptions, and parameter specifications.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        strategies = await backtesting_service.get_available_strategies(
            correlation_id=correlation_id)
        logger.info(f'Retrieved {len(strategies)} available strategies',
            extra={'correlation_id': correlation_id, 'strategy_count': len(
            strategies)})
        return strategies
    except Exception as e:
        logger.error(f'Error retrieving available strategies: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise BacktestingError(message=
            f'Error retrieving available strategies: {str(e)}',
            correlation_id=correlation_id)


@router.get('/data-sources', response_model=List[Dict[str, Any]], summary=
    'Get available data sources', description=
    'Get a list of available data sources for backtesting.')
@async_with_exception_handling
async def get_available_data_sources(request_obj: Request,
    backtesting_service: BacktestingService=Depends(get_backtesting_service)):
    """
    Get a list of available data sources for backtesting.

    This endpoint returns a list of data sources that can be used for backtesting,
    including their IDs, names, descriptions, and available instruments.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        data_sources = await backtesting_service.get_available_data_sources(
            correlation_id=correlation_id)
        logger.info(f'Retrieved {len(data_sources)} available data sources',
            extra={'correlation_id': correlation_id, 'data_source_count':
            len(data_sources)})
        return data_sources
    except Exception as e:
        logger.error(f'Error retrieving available data sources: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise BacktestingError(message=
            f'Error retrieving available data sources: {str(e)}',
            correlation_id=correlation_id)


legacy_router = APIRouter(prefix='/api/v1/backtesting', tags=[
    'Backtesting (Legacy)'])


@legacy_router.post('/run')
async def legacy_run_backtest(request: BacktestRequest, request_obj:
    Request=None, backtesting_service: BacktestingService=Depends(
    get_backtesting_service)):
    """
    Legacy endpoint for running a backtest.
    Consider migrating to /api/v1/analysis/backtesting/run
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/backtesting/run'
        )
    return await run_backtest(request, request_obj, backtesting_service)


@legacy_router.get('/{backtest_id}')
async def legacy_get_backtest_results(backtest_id: str, request_obj:
    Request=None, include_trades: bool=Query(True, description=
    'Include trade details in response'), include_equity_curve: bool=Query(
    True, description='Include equity curve in response'),
    backtesting_service: BacktestingService=Depends(get_backtesting_service)):
    """
    Legacy endpoint for getting backtest results.
    Consider migrating to /api/v1/analysis/backtesting/{backtest_id}
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/backtesting/{backtest_id}'
        )
    return await get_backtest_results(backtest_id, request_obj,
        include_trades, include_equity_curve, backtesting_service)


@legacy_router.post('/walk-forward')
async def legacy_run_walk_forward_optimization(request: WalkForwardRequest,
    request_obj: Request=None, backtesting_service: BacktestingService=
    Depends(get_backtesting_service)):
    """
    Legacy endpoint for running walk-forward optimization.
    Consider migrating to /api/v1/analysis/backtesting/walk-forward
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/backtesting/walk-forward'
        )
    return await run_walk_forward_optimization(request, request_obj,
        backtesting_service)


@legacy_router.post('/monte-carlo')
async def legacy_run_monte_carlo_simulation(request: MonteCarloRequest,
    request_obj: Request=None, backtesting_service: BacktestingService=
    Depends(get_backtesting_service)):
    """
    Legacy endpoint for running Monte Carlo simulation.
    Consider migrating to /api/v1/analysis/backtesting/monte-carlo
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/backtesting/monte-carlo'
        )
    return await run_monte_carlo_simulation(request, request_obj,
        backtesting_service)


@legacy_router.post('/stress-test')
async def legacy_run_stress_test(request: StressTestRequest, request_obj:
    Request=None, backtesting_service: BacktestingService=Depends(
    get_backtesting_service)):
    """
    Legacy endpoint for running stress test.
    Consider migrating to /api/v1/analysis/backtesting/stress-test
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/backtesting/stress-test'
        )
    return await run_stress_test(request, request_obj, backtesting_service)


@legacy_router.get('/strategies')
async def legacy_get_available_strategies(request_obj: Request=None,
    backtesting_service: BacktestingService=Depends(get_backtesting_service)):
    """
    Legacy endpoint for getting available strategies.
    Consider migrating to /api/v1/analysis/backtesting/strategies
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/backtesting/strategies'
        )
    return await get_available_strategies(request_obj, backtesting_service)


@legacy_router.get('/data-sources')
async def legacy_get_available_data_sources(request_obj: Request=None,
    backtesting_service: BacktestingService=Depends(get_backtesting_service)):
    """
    Legacy endpoint for getting available data sources.
    Consider migrating to /api/v1/analysis/backtesting/data-sources
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/backtesting/data-sources'
        )
    return await get_available_data_sources(request_obj, backtesting_service)


def setup_backtesting_routes(app: FastAPI) ->None:
    """
    Set up backtesting routes.

    Args:
        app: FastAPI application
    """
    app.include_router(router, prefix='/api')
    app.include_router(legacy_router)
