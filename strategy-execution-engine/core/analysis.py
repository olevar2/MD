"""
Analysis API for Strategy Execution Engine

This module provides API endpoints for strategy analysis.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from core.performance_analyzer import performance_analyzer
from core.error import AnalysisError, DataFetchError
logger = logging.getLogger(__name__)
analysis_router = APIRouter(prefix='/api/v1/analysis', tags=['analysis'])


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class PerformanceAnalysisResponse(BaseModel):
    """Performance analysis response model."""
    backtest_id: str
    strategy_id: str
    start_date: str
    end_date: str
    metrics: Dict[str, Any]
    trade_stats: Dict[str, Any]
    drawdown_stats: Dict[str, Any]
    monthly_returns: Dict[str, float]


class StrategyComparisonResponse(BaseModel):
    """Strategy comparison response model."""
    strategies: List[str]
    metrics_comparison: Dict[str, Dict[str, Any]]
    trade_stats_comparison: Dict[str, Dict[str, Any]]
    drawdown_comparison: Dict[str, Dict[str, Any]]
    timestamp: str


@analysis_router.get('/performance/{backtest_id}', response_model=
    PerformanceAnalysisResponse)
@async_with_exception_handling
async def analyze_backtest_performance(backtest_id: str):
    """
    Analyze backtest performance.
    
    Args:
        backtest_id: Backtest ID
        
    Returns:
        Dict: Performance analysis
    """
    try:
        analysis = performance_analyzer.analyze_backtest(backtest_id)
        return analysis
    except DataFetchError as e:
        logger.warning(f'Data fetch error: {e}')
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=
            str(e))
    except AnalysisError as e:
        logger.error(f'Analysis error: {e}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {e}', exc_info=True)
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=
            f'Unexpected error: {str(e)}')


@analysis_router.get('/compare', response_model=StrategyComparisonResponse)
@async_with_exception_handling
async def compare_strategies(strategy_ids: List[str]=Query(..., description
    ='List of strategy IDs to compare')):
    """
    Compare multiple strategies.
    
    Args:
        strategy_ids: List of strategy IDs to compare
        
    Returns:
        Dict: Strategy comparison
    """
    if not strategy_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail
            ='No strategy IDs provided')
    try:
        comparison = performance_analyzer.compare_strategies(strategy_ids)
        return comparison
    except AnalysisError as e:
        logger.error(f'Analysis error: {e}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {e}', exc_info=True)
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=
            f'Unexpected error: {str(e)}')


@analysis_router.get('/backtests')
@async_with_exception_handling
async def list_backtests(strategy_id: Optional[str]=None):
    """
    List all backtests, optionally filtered by strategy ID.
    
    Args:
        strategy_id: Optional strategy ID to filter by
        
    Returns:
        List: Backtest results
    """
    try:
        backtests = performance_analyzer.get_all_backtest_results(strategy_id)
        summary = []
        for backtest in backtests:
            summary.append({'backtest_id': backtest.get('backtest_id'),
                'strategy_id': backtest.get('strategy_id'), 'start_date':
                backtest.get('start_date'), 'end_date': backtest.get(
                'end_date'), 'initial_capital': backtest.get(
                'initial_capital'), 'metrics': backtest.get('metrics', {})})
        return {'backtests': summary}
    except Exception as e:
        logger.error(f'Unexpected error: {e}', exc_info=True)
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=
            f'Unexpected error: {str(e)}')


def setup_analysis_routes(app):
    """
    Set up analysis routes for the application.
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(analysis_router)
    logger.info('Analysis routes configured')
