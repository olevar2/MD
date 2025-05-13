"""
Analysis Results API Module

This module defines FastAPI endpoints for accessing analysis results from
various technical analysis components.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from fastapi.responses import JSONResponse
from datetime import datetime
from analysis_engine.models.analysis_result import AnalysisResult
from analysis_engine.models.market_data import MarketData
from analysis_engine.services.analysis_service import AnalysisService
from analysis_engine.api.auth.auth_handler import get_current_user, User
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/v1/analysis', tags=['analysis'], responses=
    {(404): {'description': 'Analysis not found'}})
analysis_service = AnalysisService()


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@router.on_startup
async def initialize_service():
    """Initialize the analysis service when the API starts"""
    await analysis_service.initialize()


@router.get('/', summary='List available analyzers')
@async_with_exception_handling
async def list_analyzers(current_user: User=Depends(get_current_user)):
    """
    List all available technical analysis components
    """
    try:
        analyzers = await analysis_service.list_available_analyzers()
        return {'analyzers': analyzers, 'count': len(analyzers)}
    except Exception as e:
        logger.error(f'Error listing analyzers: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to list analyzers: {str(e)}')


@router.get('/{analyzer_name}', summary='Get analyzer details')
@async_with_exception_handling
async def get_analyzer_details(analyzer_name: str=Path(..., description=
    'Name of the analyzer'), current_user: User=Depends(get_current_user)):
    """
    Get details about a specific analyzer
    """
    try:
        analyzer = await analysis_service.get_analyzer_details(analyzer_name)
        if not analyzer:
            raise HTTPException(status_code=404, detail=
                f'Analyzer {analyzer_name} not found')
        return analyzer
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting analyzer details for {analyzer_name}: {e}'
            )
        raise HTTPException(status_code=500, detail=
            f'Failed to get analyzer details: {str(e)}')


@router.post('/{analyzer_name}/analyze', summary='Run analysis')
@async_with_exception_handling
async def run_analysis(data: Dict[str, Any], analyzer_name: str=Path(...,
    description='Name of the analyzer'), current_user: User=Depends(
    get_current_user)):
    """
    Run analysis using the specified analyzer

    The data format depends on the analyzer type:
    - For regular analyzers: {"market_data": {...}}
    - For multi-timeframe analyzers: {"market_data": {"1h": {...}, "4h": {...}, ...}}
    """
    try:
        if 'market_data' not in data:
            raise HTTPException(status_code=400, detail=
                'Market data is required')
        market_data = data['market_data']
        if isinstance(market_data, dict) and all(isinstance(v, dict) for v in
            market_data.values()):
            converted_data = {tf: MarketData.from_dict(md) for tf, md in
                market_data.items()}
        else:
            converted_data = MarketData.from_dict(market_data)
        result = await analysis_service.run_analysis(analyzer_name,
            converted_data)
        if isinstance(result, AnalysisResult):
            return {'analyzer_name': result.analyzer_name, 'result_data':
                result.result_data, 'is_valid': result.is_valid, 'metadata':
                result.metadata}
        else:
            return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error running analysis with {analyzer_name}: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to run analysis: {str(e)}')


@router.get('/{analyzer_name}/performance', summary=
    'Get analyzer performance metrics')
@async_with_exception_handling
async def get_performance_metrics(analyzer_name: str=Path(..., description=
    'Name of the analyzer'), current_user: User=Depends(get_current_user)):
    """
    Get performance metrics for a specific analyzer
    """
    try:
        metrics = await analysis_service.get_analyzer_performance(analyzer_name
            )
        if not metrics:
            raise HTTPException(status_code=404, detail=
                f'Performance metrics for analyzer {analyzer_name} not found')
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f'Error getting performance metrics for {analyzer_name}: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get performance metrics: {str(e)}')


@router.get('/{analyzer_name}/effectiveness', summary=
    'Get analyzer effectiveness metrics')
@async_with_exception_handling
async def get_effectiveness_metrics(analyzer_name: str=Path(...,
    description='Name of the analyzer'), timeframe: Optional[str]=Query(
    None, description='Filter by timeframe'), instrument: Optional[str]=
    Query(None, description='Filter by instrument'), current_user: User=
    Depends(get_current_user)):
    """
    Get effectiveness metrics for a specific analyzer
    """
    try:
        metrics = await analysis_service.get_analyzer_effectiveness(
            analyzer_name, timeframe=timeframe, instrument=instrument)
        if not metrics:
            raise HTTPException(status_code=404, detail=
                f'Effectiveness metrics for analyzer {analyzer_name} not found'
                )
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f'Error getting effectiveness metrics for {analyzer_name}: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get effectiveness metrics: {str(e)}')


@router.post('/{analyzer_name}/effectiveness/record', summary=
    'Record effectiveness')
@async_with_exception_handling
async def record_effectiveness(data: Dict[str, Any], analyzer_name: str=
    Path(..., description='Name of the analyzer'), current_user: User=
    Depends(get_current_user)):
    """
    Record effectiveness data for a specific analyzer prediction

    Required data:
    - analysis_result: The original analysis result
    - actual_outcome: The actual market outcome
    - Optional: timeframe, instrument
    """
    try:
        required_fields = ['analysis_result', 'actual_outcome']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=
                    f'Missing required field: {field}')
        timeframe = data.get('timeframe')
        instrument = data.get('instrument')
        log_id = await analysis_service.record_effectiveness(analyzer_name=
            analyzer_name, analysis_result=data['analysis_result'],
            actual_outcome=data['actual_outcome'], timeframe=timeframe,
            instrument=instrument)
        return {'log_id': log_id, 'status': 'recorded', 'analyzer_name':
            analyzer_name, 'timestamp': datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error recording effectiveness for {analyzer_name}: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to record effectiveness: {str(e)}')


@router.post('/multi_timeframe/analyze', summary='Run multi-timeframe analysis'
    )
@async_with_exception_handling
async def run_multi_timeframe_analysis(data: Dict[str, Any], current_user:
    User=Depends(get_current_user)):
    """
    Run multi-timeframe analysis with all required timeframes

    Required data format:
    {
        "market_data": {
            "1m": {...},
            "5m": {...},
            "1h": {...},
            ...
        },
        "parameters": {
            "timeframes": ["1m", "5m", "1h"],
            "primary_timeframe": "1h",
            ...
        }
    }
    """
    try:
        if 'market_data' not in data:
            raise HTTPException(status_code=400, detail=
                'Market data is required')
        market_data = {tf: MarketData.from_dict(md) for tf, md in data[
            'market_data'].items()}
        parameters = data.get('parameters', {})
        result = await analysis_service.run_multi_timeframe_analysis(
            market_data, parameters)
        if isinstance(result, AnalysisResult):
            return {'analyzer_name': result.analyzer_name, 'result_data':
                result.result_data, 'is_valid': result.is_valid, 'metadata':
                result.metadata}
        else:
            return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error running multi-timeframe analysis: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to run multi-timeframe analysis: {str(e)}')


@router.post('/confluence/analyze', summary='Run confluence analysis')
@async_with_exception_handling
async def run_confluence_analysis(data: Dict[str, Any], current_user: User=
    Depends(get_current_user)):
    """
    Run confluence analysis to detect when multiple signals align

    Required data format:
    {
        "symbol": str,
        "timeframe": str,
        "market_data": {
            "open": List[float],
            "high": List[float],
            "low": List[float],
            "close": List[float],
            "volume": List[float],
            "timestamp": List[str]
        },
        "parameters": {
            "min_tools_for_confluence": int,
            "effectiveness_threshold": float,
            ...
        }
    }
    """
    try:
        required_fields = ['symbol', 'timeframe', 'market_data']
        for field in required_fields:
            if field not in data:
                raise HTTPException(status_code=400, detail=
                    f'Missing required field: {field}')
        parameters = data.get('parameters', {})
        result = await analysis_service.run_confluence_analysis(data,
            parameters)
        if isinstance(result, AnalysisResult):
            return {'analyzer_name': result.analyzer_name, 'result': result
                .result, 'is_valid': result.is_valid, 'metadata': result.
                metadata}
        else:
            return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error running confluence analysis: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to run confluence analysis: {str(e)}')
