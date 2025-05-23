"""
Consolidated Effectiveness Analysis API

This module provides unified API endpoints for analyzing trading tool effectiveness,
combining functionality previously spread across multiple modules.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from core_foundations.models.auth import User
from analysis_engine.db.connection import get_db_session
from analysis_engine.api.auth import get_current_user
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.market_regime_analysis import MarketRegimeAnalysisService
from analysis_engine.services.dashboard_generator import ToolEffectivenessDashboardGenerator
from analysis_engine.tools.effectiveness.enhanced_effectiveness_framework import EnhancedEffectivenessAnalyzer
from analysis_engine.api.v1.models import MarketRegimeEnum, TimeframeEnum
logger = logging.getLogger(__name__)
router = APIRouter()


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class BaseAnalysisRequest(BaseModel):
    """
    BaseAnalysisRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    tool_id: str
    timeframe: Optional[TimeframeEnum] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


class RegimeAnalysisRequest(BaseAnalysisRequest):
    """
    RegimeAnalysisRequest class that inherits from BaseAnalysisRequest.
    
    Attributes:
        Add attributes here
    """

    analysis_level: str = Field('basic', description=
        "Level of analysis: 'basic' or 'advanced' (includes statistical significance)"
        )
    min_sample_size: int = Field(10, description=
        'Minimum sample size for advanced analysis')
    significance_threshold: float = Field(0.05, description=
        'P-value threshold for statistical significance in advanced analysis')


class TemporalAnalysisRequest(BaseAnalysisRequest):
    """
    TemporalAnalysisRequest class that inherits from BaseAnalysisRequest.
    
    Attributes:
        Add attributes here
    """

    analysis_type: str = Field('trend', description=
        "Type of temporal analysis: 'trend' or 'decay'")
    trend_period_days: int = Field(90, description=
        'Total period in days for trend analysis')
    trend_interval_days: int = Field(7, description=
        'Interval in days for trend calculation')
    decay_window_days: int = Field(30, description=
        'Rolling window size in days for decay detection')
    decay_significance_threshold: float = Field(0.05, description=
        'P-value threshold for decay significance')


class OptimalConditionsRequest(BaseAnalysisRequest):
    """
    OptimalConditionsRequest class that inherits from BaseAnalysisRequest.
    
    Attributes:
        Add attributes here
    """

    min_sample_size: int = Field(10, description=
        'Minimum sample size per condition')


class ToolComplementarityRequest(BaseModel):
    """
    ToolComplementarityRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    tool_ids: List[str]
    timeframe: Optional[TimeframeEnum] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


class DashboardRequest(BaseModel):
    """
    DashboardRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    timeframe: Optional[TimeframeEnum] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


class ComprehensiveAnalysisRequest(BaseModel):
    """
    ComprehensiveAnalysisRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    tool_id: str
    tool_results: List[Dict[str, Any]]
    market_regimes: Optional[List[Dict[str, Any]]] = None
    timeframe_results: Optional[Dict[str, List[Dict[str, Any]]]] = None
    baseline_results: Optional[List[Dict[str, Any]]] = None
    min_sample_size: Optional[int] = Query(30, description=
        'Minimum sample size for statistical analysis')
    significance_threshold: Optional[float] = Query(0.05, description=
        'P-value threshold for statistical significance')


class CrossTimeframeAnalysisRequest(BaseModel):
    """
    CrossTimeframeAnalysisRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    tool_id: str
    timeframe_results: Dict[str, List[Dict[str, Any]]]


@router.post('/regime', response_model=Dict[str, Any])
@async_with_exception_handling
async def analyze_regime_performance(request: RegimeAnalysisRequest,
    current_user: User=Depends(get_current_user), db: Session=Depends(
    get_db_session)):
    """Analyze tool performance across different market regimes."""
    try:
        repo = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repo)
        analyzer_config = {'significance_threshold': request.
            significance_threshold, 'min_sample_size': request.min_sample_size}
        analyzer = EnhancedEffectivenessAnalyzer(analyzer_config)
        if request.analysis_level == 'basic':
            result = service.get_regime_performance_matrix(tool_id=request.
                tool_id, timeframe=request.timeframe.value if request.
                timeframe else None, instrument=request.instrument,
                from_date=request.from_date, to_date=request.to_date)
        elif request.analysis_level == 'advanced':
            logger.warning(
                'Advanced regime analysis requires fetching raw data - implementation pending.'
                )
            result = service.get_regime_performance_matrix(tool_id=request.
                tool_id, timeframe=request.timeframe.value if request.
                timeframe else None, instrument=request.instrument,
                from_date=request.from_date, to_date=request.to_date)
            result['analysis_level'] = 'advanced (placeholder - using basic)'
        else:
            raise HTTPException(status_code=400, detail=
                "Invalid analysis_level. Use 'basic' or 'advanced'.")
        return {'status': 'success', 'tool_id': request.tool_id, 'results':
            result}
    except Exception as e:
        logger.error(
            f'Error in regime analysis for tool {request.tool_id}: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=
            'Failed to perform regime analysis. Please check logs for details.'
            )


@router.post('/temporal', response_model=Dict[str, Any])
@async_with_exception_handling
async def analyze_temporal_performance(request: TemporalAnalysisRequest,
    current_user: User=Depends(get_current_user), db: Session=Depends(
    get_db_session)):
    """Analyze tool performance trends or decay over time."""
    try:
        repo = ToolEffectivenessRepository(db)
        dashboard_gen = ToolEffectivenessDashboardGenerator(repo)
        analyzer_config = {'significance_threshold': request.
            decay_significance_threshold, 'performance_decay_window_days':
            request.decay_window_days}
        analyzer = EnhancedEffectivenessAnalyzer(analyzer_config)
        if request.analysis_type == 'trend':
            result = dashboard_gen.generate_performance_trends(tool_id=
                request.tool_id, period_days=request.trend_period_days,
                interval_days=request.trend_interval_days)
        elif request.analysis_type == 'decay':
            logger.warning(
                'Performance decay analysis requires fetching raw data - implementation pending.'
                )
            result = {'message': 'Decay analysis implementation pending'}
        else:
            raise HTTPException(status_code=400, detail=
                "Invalid analysis_type. Use 'trend' or 'decay'.")
        return {'status': 'success', 'tool_id': request.tool_id,
            'analysis_type': request.analysis_type, 'results': result}
    except Exception as e:
        logger.error(
            f'Error in temporal analysis for tool {request.tool_id}: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=
            'Failed to perform temporal analysis. Please check logs for details.'
            )


@router.post('/optimal-conditions', response_model=Dict[str, Any])
@async_with_exception_handling
async def find_optimal_conditions(request: OptimalConditionsRequest,
    current_user: User=Depends(get_current_user), db: Session=Depends(
    get_db_session)):
    """Find the optimal market conditions for the tool."""
    try:
        repo = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repo)
        result = service.find_optimal_market_conditions(tool_id=request.
            tool_id, min_sample_size=request.min_sample_size, timeframe=
            request.timeframe.value if request.timeframe else None,
            instrument=request.instrument, from_date=request.from_date,
            to_date=request.to_date)
        return {'status': 'success', 'tool_id': request.tool_id, 'results':
            result}
    except Exception as e:
        logger.error(
            f'Error finding optimal conditions for tool {request.tool_id}: {str(e)}'
            )
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=
            'Failed to find optimal conditions. Please check logs for details.'
            )


@router.post('/complementarity', response_model=Dict[str, Any])
@async_with_exception_handling
async def analyze_complementarity(request: ToolComplementarityRequest,
    current_user: User=Depends(get_current_user), db: Session=Depends(
    get_db_session)):
    """Analyze how well different tools complement each other."""
    try:
        repo = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repo)
        result = service.compute_tool_complementarity(tool_ids=request.
            tool_ids, timeframe=request.timeframe.value if request.
            timeframe else None, instrument=request.instrument, from_date=
            request.from_date, to_date=request.to_date)
        return {'status': 'success', 'tool_ids': request.tool_ids,
            'results': result}
    except Exception as e:
        logger.error(
            f'Error analyzing complementarity for tools {request.tool_ids}: {str(e)}'
            )
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=
            'Failed to perform complementarity analysis. Please check logs for details.'
            )


@router.post('/dashboard', response_model=Dict[str, Any])
@async_with_exception_handling
async def generate_dashboard(request: DashboardRequest, current_user: User=
    Depends(get_current_user), db: Session=Depends(get_db_session)):
    """Generate data for the effectiveness dashboard."""
    try:
        repo = ToolEffectivenessRepository(db)
        generator = ToolEffectivenessDashboardGenerator(repo)
        result = generator.generate_dashboard_data(timeframe=request.
            timeframe.value if request.timeframe else None, instrument=
            request.instrument, from_date=request.from_date, to_date=
            request.to_date)
        return {'status': 'success', 'filters': request.dict(), 'results':
            result}
    except Exception as e:
        logger.error(f'Error generating dashboard data: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=
            'Failed to generate dashboard data. Please check logs for details.'
            )


@router.post('/cross-timeframe', response_model=Dict[str, Any])
@async_with_exception_handling
async def analyze_cross_timeframe(request: CrossTimeframeAnalysisRequest,
    current_user: User=Depends(get_current_user)):
    """Analyze tool consistency across different timeframes."""
    try:
        if not request.timeframe_results or len(request.timeframe_results) < 2:
            raise HTTPException(status_code=400, detail=
                'Results for at least two timeframes required.')
        analyzer = EnhancedEffectivenessAnalyzer()
        result = analyzer.analyze_cross_timeframe_consistency(
            tool_results_by_timeframe=request.timeframe_results)
        return {'status': 'success', 'tool_id': request.tool_id, 'results':
            result}
    except Exception as e:
        logger.error(
            f'Error in cross-timeframe analysis for tool {request.tool_id}: {str(e)}'
            )
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=
            'Failed to perform cross-timeframe analysis. Please check logs for details.'
            )


@router.get('/metrics', response_model=Dict[str, Any])
def get_basic_metrics(tool_id: Optional[str]=Query(None, description=
    'Filter by tool ID'), market_regime: Optional[MarketRegimeEnum]=Query(
    None, description='Filter by market regime'), timeframe: Optional[
    TimeframeEnum]=Query(None, description='Filter by timeframe'),
    instrument: Optional[str]=Query(None, description=
    'Filter by trading instrument'), from_date: Optional[datetime]=Query(
    None, description='Filter by start date'), to_date: Optional[datetime]=
    Query(None, description='Filter by end date'), current_user: User=
    Depends(get_current_user), db: Session=Depends(get_db_session)):
    """Get basic effectiveness metrics based on registered outcomes."""
    logger.warning('Basic metrics endpoint implementation pending.')
    return {'status': 'pending', 'message':
        'Basic metrics endpoint implementation pending'}


@router.post('/comprehensive', response_model=Dict[str, Any])
@async_with_exception_handling
async def analyze_comprehensive(request: ComprehensiveAnalysisRequest,
    current_user: User=Depends(get_current_user)):
    """Generate a comprehensive effectiveness report."""
    try:
        config = {'min_sample_size': request.min_sample_size,
            'significance_threshold': request.significance_threshold}
        analyzer = EnhancedEffectivenessAnalyzer(config)
        report = analyzer.generate_comprehensive_report(tool_id=request.
            tool_id, tool_results=request.tool_results, market_regimes=
            request.market_regimes, timeframe_results=request.
            timeframe_results, baseline_results=request.baseline_results)
        if 'error' in report and not isinstance(report['error'], dict):
            raise HTTPException(status_code=422, detail=report['error'])
        return {'status': 'success', 'tool_id': request.tool_id, 'results':
            report}
    except Exception as e:
        logger.error(
            f'Error in comprehensive analysis for tool {request.tool_id}: {str(e)}'
            )
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail=
            'Failed to perform comprehensive analysis. Please check logs for details.'
            )
