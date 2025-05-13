"""
Standardized Market Manipulation Detection API for Analysis Engine Service.

This module provides standardized API endpoints for detecting potential market manipulation
patterns in forex data, including stop hunting, fake breakouts, and unusual
price-volume relationships.

All endpoints follow the platform's standardized API design patterns.
"""
from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Body, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime
from core_foundations.models.auth import User
from analysis_engine.analysis.manipulation.detector import MarketManipulationAnalyzer
from analysis_engine.db.connection import get_db_session
from analysis_engine.api.auth import get_current_user
from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError, AnalysisError, ManipulationDetectionError, InsufficientDataError, get_correlation_id_from_request
from analysis_engine.monitoring.structured_logging import get_structured_logger


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class OHLCVData(BaseModel):
    """Model for OHLCV data point"""
    timestamp: str = Field(..., description='ISO datetime string')
    open: float = Field(..., description='Open price')
    high: float = Field(..., description='High price')
    low: float = Field(..., description='Low price')
    close: float = Field(..., description='Close price')
    volume: float = Field(..., description='Volume')


class Metadata(BaseModel):
    """Model for market data metadata"""
    symbol: str = Field(..., description='Symbol/currency pair')
    timeframe: str = Field(..., description=
        "Timeframe (e.g., '1h', '4h', '1d')")


class ManipulationDetectionRequest(BaseModel):
    """Request model for manipulation detection"""
    ohlcv: List[OHLCVData] = Field(..., description='OHLCV data points')
    metadata: Metadata = Field(..., description='Market data metadata')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'ohlcv': [{'timestamp':
            '2025-04-01T12:00:00', 'open': 1.2345, 'high': 1.236, 'low': 
            1.234, 'close': 1.2355, 'volume': 1000}, {'timestamp':
            '2025-04-01T13:00:00', 'open': 1.2355, 'high': 1.237, 'low': 
            1.235, 'close': 1.2365, 'volume': 1100}], 'metadata': {'symbol':
            'EUR/USD', 'timeframe': '1h'}}}


class AnalysisResponse(BaseModel):
    """Response model for manipulation detection analysis"""
    status: str = Field(..., description='Status of the analysis')
    symbol: str = Field(..., description='Symbol/currency pair')
    timeframe: str = Field(..., description='Timeframe')
    results: Dict[str, Any] = Field(..., description='Analysis results')
    timestamp: str = Field(..., description=
        'ISO datetime string of when the analysis was performed')


router = APIRouter(prefix='/v1/analysis/manipulation-detection', tags=[
    'Manipulation Detection'])
logger = get_structured_logger(__name__)


@router.post('/detect', response_model=AnalysisResponse, summary=
    'Detect manipulation patterns', description=
    'Analyze market data for potential manipulation patterns.')
@async_with_exception_handling
async def detect_manipulation_patterns(request:
    ManipulationDetectionRequest, request_obj: Request, sensitivity:
    Optional[float]=Query(1.0, description=
    'Detection sensitivity multiplier (0.5-2.0)'), include_protection:
    Optional[bool]=Query(True, description=
    'Include protection recommendations'), current_user: User=Depends(
    get_current_user)):
    """
    Analyze market data for potential manipulation patterns.

    Returns detected patterns, clusters, and optional protection recommendations.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        if len(request.ohlcv) < 100:
            raise InsufficientDataError(message=
                'OHLCV data must contain at least 100 data points',
                correlation_id=correlation_id)
        detector_params = {}
        if sensitivity != 1.0:
            if sensitivity < 0.5 or sensitivity > 2.0:
                raise HTTPException(status_code=400, detail=
                    'Sensitivity must be between 0.5 and 2.0')
            detector_params['volume_z_threshold'] = 2.0 / sensitivity
            detector_params['price_reversal_threshold'] = 0.5 / sensitivity
            detector_params['confidence_high_threshold'] = 0.8 - (sensitivity -
                1.0) * 0.1
            detector_params['confidence_medium_threshold'] = 0.6 - (sensitivity
                 - 1.0) * 0.1
        detector = MarketManipulationAnalyzer(detector_params)
        data = {'ohlcv': [item.dict() for item in request.ohlcv],
            'metadata': request.metadata.dict()}
        result = detector.analyze(data)
        if not result.is_valid:
            raise ManipulationDetectionError(message=result.result_data.get
                ('error', 'Analysis failed'), correlation_id=correlation_id)
        if (not include_protection and 'protection_recommendations' in
            result.result_data):
            result.result_data['protection_recommendations'] = []
        logger.info(
            f'Detected manipulation patterns for {request.metadata.symbol} ({request.metadata.timeframe})'
            , extra={'correlation_id': correlation_id, 'symbol': request.
            metadata.symbol, 'timeframe': request.metadata.timeframe,
            'sensitivity': sensitivity, 'data_points': len(request.ohlcv)})
        response = AnalysisResponse(status='success', symbol=request.
            metadata.symbol, timeframe=request.metadata.timeframe, results=
            result.result_data, timestamp=datetime.now().isoformat())
        return response
    except InsufficientDataError:
        raise
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error detecting manipulation patterns: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise ManipulationDetectionError(message=
            'Failed to detect manipulation patterns', correlation_id=
            correlation_id)


@router.post('/stop-hunting', response_model=AnalysisResponse, summary=
    'Detect stop hunting', description=
    'Specifically analyze for stop hunting patterns.')
@async_with_exception_handling
async def detect_stop_hunting(request: ManipulationDetectionRequest,
    request_obj: Request, lookback: Optional[int]=Query(30, description=
    'Lookback period for stop hunting detection'), recovery_threshold:
    Optional[float]=Query(0.5, description='Recovery percentage threshold'),
    current_user: User=Depends(get_current_user)):
    """
    Specifically analyze for stop hunting patterns.

    Returns detected stop hunting patterns and related support/resistance levels.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        detector_params = {'stop_hunting_lookback': lookback,
            'stop_hunting_recovery': recovery_threshold}
        detector = MarketManipulationAnalyzer(detector_params)
        data = {'ohlcv': [item.dict() for item in request.ohlcv],
            'metadata': request.metadata.dict()}
        result = detector.analyze(data)
        if not result.is_valid:
            raise ManipulationDetectionError(message=result.result_data.get
                ('error', 'Analysis failed'), correlation_id=correlation_id)
        filtered_results = {'stop_hunting_patterns': result.result_data.get
            ('detected_patterns', {}).get('stop_hunting', []),
            'pattern_count': result.result_data.get('pattern_count', {}).
            get('stop_hunting', 0), 'support_resistance': result.
            result_data.get('support_resistance', {}),
            'manipulation_likelihood': result.result_data.get(
            'manipulation_likelihood', {}), 'protection_recommendations': [
            rec for rec in result.result_data.get(
            'protection_recommendations', []) if rec.get('trigger') ==
            'stop_hunting' or rec.get('trigger') == 'manipulation_cluster']}
        logger.info(
            f'Detected stop hunting patterns for {request.metadata.symbol} ({request.metadata.timeframe})'
            , extra={'correlation_id': correlation_id, 'symbol': request.
            metadata.symbol, 'timeframe': request.metadata.timeframe,
            'lookback': lookback, 'recovery_threshold': recovery_threshold,
            'data_points': len(request.ohlcv)})
        response = AnalysisResponse(status='success', symbol=request.
            metadata.symbol, timeframe=request.metadata.timeframe, results=
            filtered_results, timestamp=datetime.now().isoformat())
        return response
    except InsufficientDataError:
        raise
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error detecting stop hunting patterns: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise ManipulationDetectionError(message=
            'Failed to detect stop hunting patterns', correlation_id=
            correlation_id)


@router.post('/fake-breakouts', response_model=AnalysisResponse, summary=
    'Detect fake breakouts', description=
    'Specifically analyze for fake breakout patterns.')
@async_with_exception_handling
async def detect_fake_breakouts(request: ManipulationDetectionRequest,
    request_obj: Request, threshold: Optional[float]=Query(0.7, description
    ='Fake breakout detection threshold'), current_user: User=Depends(
    get_current_user)):
    """
    Specifically analyze for fake breakout patterns.

    Returns detected fake breakout patterns and related support/resistance levels.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        detector_params = {'fake_breakout_threshold': threshold}
        detector = MarketManipulationAnalyzer(detector_params)
        data = {'ohlcv': [item.dict() for item in request.ohlcv],
            'metadata': request.metadata.dict()}
        result = detector.analyze(data)
        if not result.is_valid:
            raise ManipulationDetectionError(message=result.result_data.get
                ('error', 'Analysis failed'), correlation_id=correlation_id)
        filtered_results = {'fake_breakout_patterns': result.result_data.
            get('detected_patterns', {}).get('fake_breakouts', []),
            'pattern_count': result.result_data.get('pattern_count', {}).
            get('fake_breakouts', 0), 'support_resistance': result.
            result_data.get('support_resistance', {}),
            'manipulation_likelihood': result.result_data.get(
            'manipulation_likelihood', {}), 'protection_recommendations': [
            rec for rec in result.result_data.get(
            'protection_recommendations', []) if rec.get('trigger') ==
            'fake_breakout' or rec.get('trigger') == 'manipulation_cluster']}
        logger.info(
            f'Detected fake breakout patterns for {request.metadata.symbol} ({request.metadata.timeframe})'
            , extra={'correlation_id': correlation_id, 'symbol': request.
            metadata.symbol, 'timeframe': request.metadata.timeframe,
            'threshold': threshold, 'data_points': len(request.ohlcv)})
        response = AnalysisResponse(status='success', symbol=request.
            metadata.symbol, timeframe=request.metadata.timeframe, results=
            filtered_results, timestamp=datetime.now().isoformat())
        return response
    except InsufficientDataError:
        raise
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error detecting fake breakout patterns: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise ManipulationDetectionError(message=
            'Failed to detect fake breakout patterns', correlation_id=
            correlation_id)


@router.post('/volume-anomalies', response_model=AnalysisResponse, summary=
    'Detect volume anomalies', description=
    'Specifically analyze for volume anomalies.')
@async_with_exception_handling
async def detect_volume_anomalies(request: ManipulationDetectionRequest,
    request_obj: Request, z_threshold: Optional[float]=Query(2.0,
    description='Z-score threshold for volume anomaly detection'),
    current_user: User=Depends(get_current_user)):
    """
    Specifically analyze for volume anomalies.

    Returns detected volume anomalies and potential manipulation patterns.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        detector_params = {'volume_z_threshold': z_threshold}
        detector = MarketManipulationAnalyzer(detector_params)
        data = {'ohlcv': [item.dict() for item in request.ohlcv],
            'metadata': request.metadata.dict()}
        result = detector.analyze(data)
        if not result.is_valid:
            raise ManipulationDetectionError(message=result.result_data.get
                ('error', 'Analysis failed'), correlation_id=correlation_id)
        filtered_results = {'volume_anomalies': result.result_data.get(
            'detected_patterns', {}).get('volume_anomalies', []),
            'pattern_count': result.result_data.get('pattern_count', {}).
            get('volume_anomalies', 0), 'manipulation_likelihood': result.
            result_data.get('manipulation_likelihood', {}),
            'protection_recommendations': [rec for rec in result.
            result_data.get('protection_recommendations', []) if rec.get(
            'trigger') == 'volume_anomaly' or rec.get('trigger') ==
            'manipulation_cluster']}
        logger.info(
            f'Detected volume anomalies for {request.metadata.symbol} ({request.metadata.timeframe})'
            , extra={'correlation_id': correlation_id, 'symbol': request.
            metadata.symbol, 'timeframe': request.metadata.timeframe,
            'z_threshold': z_threshold, 'data_points': len(request.ohlcv)})
        response = AnalysisResponse(status='success', symbol=request.
            metadata.symbol, timeframe=request.metadata.timeframe, results=
            filtered_results, timestamp=datetime.now().isoformat())
        return response
    except InsufficientDataError:
        raise
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error detecting volume anomalies: {str(e)}', extra={
            'correlation_id': correlation_id}, exc_info=True)
        raise ManipulationDetectionError(message=
            'Failed to detect volume anomalies', correlation_id=correlation_id)


legacy_router = APIRouter(prefix='/api/v1/manipulation', tags=[
    'Manipulation Detection (Legacy)'])


@legacy_router.post('/detect', response_model=AnalysisResponse)
async def legacy_detect_manipulation_patterns(data: Dict[str, Any],
    sensitivity: Optional[float]=Query(1.0, description=
    'Detection sensitivity multiplier (0.5-2.0)'), include_protection:
    Optional[bool]=Query(True, description=
    'Include protection recommendations'), request_obj: Request=None,
    current_user: User=Depends(get_current_user)):
    """
    Legacy endpoint for detecting manipulation patterns.
    Consider migrating to /api/v1/analysis/manipulation-detection/detect
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/manipulation-detection/detect'
        )
    if not data or not isinstance(data, dict) or 'ohlcv' not in data:
        raise HTTPException(status_code=400, detail=
            'Request must contain OHLCV data')
    ohlcv_data = data.get('ohlcv', [])
    metadata = data.get('metadata', {})
    ohlcv_models = [OHLCVData(**item) for item in ohlcv_data]
    metadata_model = Metadata(**metadata)
    request = ManipulationDetectionRequest(ohlcv=ohlcv_models, metadata=
        metadata_model)
    return await detect_manipulation_patterns(request, request_obj,
        sensitivity, include_protection, current_user)


@legacy_router.post('/stop-hunting', response_model=AnalysisResponse)
async def legacy_detect_stop_hunting(data: Dict[str, Any], lookback:
    Optional[int]=Query(30, description=
    'Lookback period for stop hunting detection'), recovery_threshold:
    Optional[float]=Query(0.5, description='Recovery percentage threshold'),
    request_obj: Request=None, current_user: User=Depends(get_current_user)):
    """
    Legacy endpoint for detecting stop hunting patterns.
    Consider migrating to /api/v1/analysis/manipulation-detection/stop-hunting
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/manipulation-detection/stop-hunting'
        )
    if not data or not isinstance(data, dict) or 'ohlcv' not in data:
        raise HTTPException(status_code=400, detail=
            'Request must contain OHLCV data')
    ohlcv_data = data.get('ohlcv', [])
    metadata = data.get('metadata', {})
    ohlcv_models = [OHLCVData(**item) for item in ohlcv_data]
    metadata_model = Metadata(**metadata)
    request = ManipulationDetectionRequest(ohlcv=ohlcv_models, metadata=
        metadata_model)
    return await detect_stop_hunting(request, request_obj, lookback,
        recovery_threshold, current_user)


@legacy_router.post('/fake-breakouts', response_model=AnalysisResponse)
async def legacy_detect_fake_breakouts(data: Dict[str, Any], threshold:
    Optional[float]=Query(0.7, description=
    'Fake breakout detection threshold'), request_obj: Request=None,
    current_user: User=Depends(get_current_user)):
    """
    Legacy endpoint for detecting fake breakout patterns.
    Consider migrating to /api/v1/analysis/manipulation-detection/fake-breakouts
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/manipulation-detection/fake-breakouts'
        )
    if not data or not isinstance(data, dict) or 'ohlcv' not in data:
        raise HTTPException(status_code=400, detail=
            'Request must contain OHLCV data')
    ohlcv_data = data.get('ohlcv', [])
    metadata = data.get('metadata', {})
    ohlcv_models = [OHLCVData(**item) for item in ohlcv_data]
    metadata_model = Metadata(**metadata)
    request = ManipulationDetectionRequest(ohlcv=ohlcv_models, metadata=
        metadata_model)
    return await detect_fake_breakouts(request, request_obj, threshold,
        current_user)


@legacy_router.post('/volume-anomalies', response_model=AnalysisResponse)
async def legacy_detect_volume_anomalies(data: Dict[str, Any], z_threshold:
    Optional[float]=Query(2.0, description=
    'Z-score threshold for volume anomaly detection'), request_obj: Request
    =None, current_user: User=Depends(get_current_user)):
    """
    Legacy endpoint for detecting volume anomalies.
    Consider migrating to /api/v1/analysis/manipulation-detection/volume-anomalies
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/manipulation-detection/volume-anomalies'
        )
    if not data or not isinstance(data, dict) or 'ohlcv' not in data:
        raise HTTPException(status_code=400, detail=
            'Request must contain OHLCV data')
    ohlcv_data = data.get('ohlcv', [])
    metadata = data.get('metadata', {})
    ohlcv_models = [OHLCVData(**item) for item in ohlcv_data]
    metadata_model = Metadata(**metadata)
    request = ManipulationDetectionRequest(ohlcv=ohlcv_models, metadata=
        metadata_model)
    return await detect_volume_anomalies(request, request_obj, z_threshold,
        current_user)


def setup_manipulation_detection_routes(app: FastAPI) ->None:
    """
    Set up manipulation detection routes.

    Args:
        app: FastAPI application
    """
    app.include_router(router, prefix='/api')
    app.include_router(legacy_router)
