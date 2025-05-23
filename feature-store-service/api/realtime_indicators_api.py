"""
Real-time Indicators API

This module provides API endpoints for accessing real-time incrementally calculated indicators.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime
from core_foundations.utils.logger import get_logger
from ..services.incremental_processor import RealTimeFeatureProcessor
from ..models.schemas import IndicatorConfig, TickData, IndicatorResponse
realtime_indicators_router = APIRouter(prefix='/api/realtime-indicators',
    tags=['realtime-indicators'], responses={(404): {'description':
    'Not found'}})
realtime_processor = RealTimeFeatureProcessor()
logger = get_logger('realtime-indicators-api')


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@realtime_indicators_router.post('/configure')
@async_with_exception_handling
async def configure_indicators(configs: List[IndicatorConfig]):
    """
    Configure the real-time indicator processor with specific indicators
    
    Args:
        configs: List of indicator configurations
        
    Returns:
        Dict with status and configured indicators
    """
    try:
        config_dicts = [config.dict() for config in configs]
        realtime_processor._configure_indicators(config_dicts)
        return {'status': 'success', 'configured_indicators': len(
            config_dicts), 'indicator_names': [conf.name for conf in configs]}
    except Exception as e:
        logger.error(f'Error configuring indicators: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to configure indicators: {str(e)}')


@realtime_indicators_router.post('/process-tick/{symbol}')
@async_with_exception_handling
async def process_tick(symbol: str, tick_data: TickData):
    """
    Process a new tick and get updated indicator values
    
    Args:
        symbol: Trading symbol (e.g., 'EUR_USD')
        tick_data: Latest tick data
        
    Returns:
        Updated indicator values
    """
    try:
        tick_dict = tick_data.dict()
        result = realtime_processor.process_tick(symbol, tick_dict)
        return {'symbol': symbol, 'timestamp': result.get('timestamp',
            datetime.now()), 'indicators': result.get('indicators', {})}
    except Exception as e:
        logger.error(f'Error processing tick for {symbol}: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to process tick: {str(e)}')


@realtime_indicators_router.get('/latest/{symbol}')
@async_with_exception_handling
async def get_latest_indicators(symbol: str):
    """
    Get the latest indicator values for a symbol
    
    Args:
        symbol: Trading symbol (e.g., 'EUR_USD')
        
    Returns:
        Latest indicator values
    """
    try:
        result = realtime_processor.get_latest_indicators(symbol)
        if not result or not result.get('indicators'):
            raise HTTPException(status_code=404, detail=
                f'No indicators found for symbol: {symbol}')
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting latest indicators for {symbol}: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get latest indicators: {str(e)}')


@realtime_indicators_router.get('/available-symbols')
async def get_available_symbols():
    """
    Get list of symbols with available indicator data
    
    Returns:
        List of symbols
    """
    result = realtime_processor.get_latest_indicators()
    return {'symbols': result.get('symbols', [])}


@realtime_indicators_router.delete('/reset/{symbol}')
@async_with_exception_handling
async def reset_indicators(symbol: str):
    """
    Reset indicator state for a symbol
    
    Args:
        symbol: Trading symbol to reset
        
    Returns:
        Status message
    """
    try:
        realtime_processor.reset(symbol)
        return {'status': 'success', 'message':
            f'Reset indicators for symbol: {symbol}'}
    except Exception as e:
        logger.error(f'Error resetting indicators for {symbol}: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to reset indicators: {str(e)}')


@realtime_indicators_router.delete('/reset-all')
@async_with_exception_handling
async def reset_all_indicators():
    """
    Reset indicator state for all symbols
    
    Returns:
        Status message
    """
    try:
        realtime_processor.reset()
        return {'status': 'success', 'message': 'Reset all indicators'}
    except Exception as e:
        logger.error(f'Error resetting all indicators: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to reset all indicators: {str(e)}')


@realtime_indicators_router.websocket('/ws/{symbol}')
@async_with_exception_handling
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time indicator updates
    
    Args:
        websocket: WebSocket connection
        symbol: Trading symbol to subscribe to
    """
    await websocket.accept()

    @async_with_exception_handling
    async def send_indicators(data: Dict[str, Any]):
    """
    Send indicators.
    
    Args:
        data: Description of data
        Any]: Description of Any]
    
    """

        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f'Error sending indicator update: {str(e)}')
    realtime_processor.register_listener(symbol, send_indicators)
    try:
        initial_data = realtime_processor.get_latest_indicators(symbol)
        if initial_data and 'indicators' in initial_data and initial_data[
            'indicators']:
            await websocket.send_json(initial_data)
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                if 'action' in message and message['action'] == 'ping':
                    await websocket.send_json({'action': 'pong'})
            except json.JSONDecodeError:
                logger.warning(f'Received invalid JSON: {data}')
            except Exception as e:
                logger.error(f'Error processing WebSocket message: {str(e)}')
    except WebSocketDisconnect:
        logger.info(f'WebSocket disconnected for {symbol}')
    except Exception as e:
        logger.error(f'WebSocket error: {str(e)}')
    finally:
        realtime_processor.unregister_listener(symbol, send_indicators)
        logger.info(f'WebSocket connection closed for {symbol}')
