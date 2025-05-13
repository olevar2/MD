"""
Adapter API Module

This module provides API endpoints that use the adapter pattern.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel
from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer
from common_lib.errors.base_exceptions import BaseError, ValidationError, DataError, ServiceError
from analysis_engine.api.dependencies import get_analysis_provider, get_indicator_provider, get_pattern_recognizer
from analysis_engine.core.logging import get_logger
logger = get_logger('analysis-engine-service.adapter-api')
adapter_router = APIRouter(prefix='/api/v1/adapter', tags=['adapter'],
    responses={(404): {'description': 'Not found'}})


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AnalysisResponse(BaseModel):
    """Response model for analysis."""
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: Optional[datetime] = None
    indicators: Dict[str, Any]
    patterns: Dict[str, Any]


class IndicatorResponse(BaseModel):
    """Response model for indicator."""
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str


class PatternResponse(BaseModel):
    """Response model for pattern."""
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str


@adapter_router.get('/analysis/{symbol}/{timeframe}', response_model=
    AnalysisResponse)
@async_with_exception_handling
async def get_analysis(symbol: str=Path(..., description=
    "The trading symbol (e.g., 'EURUSD')"), timeframe: str=Path(...,
    description="The timeframe (e.g., '1m', '5m', '1h', '1d')"), start_time:
    datetime=Query(..., description='Start time for the data'), end_time:
    Optional[datetime]=Query(None, description='End time for the data'),
    indicators: Optional[List[str]]=Query(None, description=
    'List of indicators to include'), patterns: Optional[List[str]]=Query(
    None, description='List of patterns to detect'), analysis_provider:
    IAnalysisProvider=Depends(get_analysis_provider)):
    """
    Get analysis for a symbol.
    
    Args:
        symbol: The trading symbol (e.g., "EURUSD")
        timeframe: The timeframe (e.g., "1m", "5m", "1h", "1d")
        start_time: Start time for the data
        end_time: Optional end time for the data
        indicators: Optional list of indicators to include
        patterns: Optional list of patterns to detect
        analysis_provider: The analysis provider adapter
        
    Returns:
        AnalysisResponse containing the analysis results
    """
    try:
        result = await analysis_provider.analyze_market_data(symbol=symbol,
            timeframe=timeframe, start_time=start_time, end_time=end_time,
            indicators=indicators, patterns=patterns)
        return AnalysisResponse(symbol=symbol, timeframe=timeframe,
            start_time=start_time, end_time=end_time, indicators=result.get
            ('indicators', {}), patterns=result.get('patterns', {}))
    except ValidationError as e:
        logger.warning(f'Validation error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except DataError as e:
        logger.warning(f'Data error: {str(e)}')
        raise HTTPException(status_code=404, detail=str(e))
    except ServiceError as e:
        logger.error(f'Service error: {str(e)}')
        raise HTTPException(status_code=503, detail=str(e))
    except BaseError as e:
        logger.error(f'Base error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Unexpected error: {str(e)}')


@adapter_router.get('/indicators', response_model=List[IndicatorResponse])
@async_with_exception_handling
async def get_indicators(indicator_provider: IIndicatorProvider=Depends(
    get_indicator_provider)):
    """
    Get available indicators.
    
    Args:
        indicator_provider: The indicator provider adapter
        
    Returns:
        List of IndicatorResponse containing information about available indicators
    """
    try:
        result = await indicator_provider.get_all_indicators_info()
        return [IndicatorResponse(name=indicator.get('name', ''),
            description=indicator.get('description', ''), parameters=
            indicator.get('parameters', {}), category=indicator.get(
            'category', '')) for indicator in result]
    except ServiceError as e:
        logger.error(f'Service error: {str(e)}')
        raise HTTPException(status_code=503, detail=str(e))
    except BaseError as e:
        logger.error(f'Base error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Unexpected error: {str(e)}')


@adapter_router.get('/patterns', response_model=List[PatternResponse])
@async_with_exception_handling
async def get_patterns(pattern_recognizer: IPatternRecognizer=Depends(
    get_pattern_recognizer)):
    """
    Get available patterns.
    
    Args:
        pattern_recognizer: The pattern recognizer adapter
        
    Returns:
        List of PatternResponse containing information about available patterns
    """
    try:
        result = await pattern_recognizer.get_all_patterns_info()
        return [PatternResponse(name=pattern.get('name', ''), description=
            pattern.get('description', ''), parameters=pattern.get(
            'parameters', {}), category=pattern.get('category', '')) for
            pattern in result]
    except ServiceError as e:
        logger.error(f'Service error: {str(e)}')
        raise HTTPException(status_code=503, detail=str(e))
    except BaseError as e:
        logger.error(f'Base error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Unexpected error: {str(e)}')
