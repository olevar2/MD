"""
Indicator API module.

This module provides API endpoints for retrieving indicator metadata and documentation.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
from core_foundations.models.schemas import TimeFrame
from core_foundations.utils.logger import get_logger
from core.indicator_registry import IndicatorRegistry
from repositories.feature_storage import FeatureStorage
from core.main_1 import feature_storage
from services.enhanced_indicator_service import EnhancedIndicatorService
logger = get_logger('feature-store-service.indicator-api')
indicator_router = APIRouter(prefix='/api/v1/indicators', tags=[
    'indicators'], responses={(404): {'description': 'Not found'}})
indicator_registry: Optional[IndicatorRegistry] = None
enhanced_indicator_service: Optional[EnhancedIndicatorService] = None


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ParameterInfo(BaseModel):
    """Information about a parameter for an indicator."""
    name: str
    description: str
    type: str
    default: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    options: Optional[List[str]] = None


class IndicatorInfo(BaseModel):
    """Information about a technical indicator."""
    id: str
    name: str
    description: str
    category: str
    parameters: List[ParameterInfo]


class CacheStatistics(BaseModel):
    """Statistics about the indicator caching system."""
    memory_cache: Dict[str, Any]
    disk_cache: Optional[Dict[str, Any]]
    performance: Dict[str, Any]


@indicator_router.get('/', response_model=List[IndicatorInfo])
@async_with_exception_handling
async def get_all_indicators(category: Optional[str]=Query(None,
    description='Filter by category')):
    """
    Get a list of all available technical indicators.

    If category is provided, only indicators in that category are returned.
    """
    if indicator_registry is None:
        raise HTTPException(status_code=503, detail=
            'Indicator registry is not initialized')
    try:
        if category:
            indicators = indicator_registry.get_indicators_by_category(category
                )
        else:
            indicators = indicator_registry.get_all_indicators()
        result = []
        for indicator_id, indicator_class in indicators.items():
            metadata = indicator_class.get_metadata()
            parameters = []
            for param_name, param_info in metadata['parameters'].items():
                parameters.append(ParameterInfo(name=param_name,
                    description=param_info.get('description', ''), type=
                    param_info.get('type', 'string'), default=str(
                    param_info.get('default')) if param_info.get('default')
                     is not None else None, min=param_info.get('min'), max=
                    param_info.get('max'), options=param_info.get('options')))
            result.append(IndicatorInfo(id=metadata['id'], name=metadata[
                'name'], description=metadata['description'], category=
                metadata['category'], parameters=parameters))
        result.sort(key=lambda x: (x.category, x.name))
        return result
    except Exception as e:
        logger.error(f'Error getting indicators: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@indicator_router.get('/{indicator_id}', response_model=IndicatorInfo)
@async_with_exception_handling
async def get_indicator(indicator_id: str):
    """
    Get detailed information about a specific technical indicator.
    """
    if indicator_registry is None:
        raise HTTPException(status_code=503, detail=
            'Indicator registry is not initialized')
    try:
        indicator_class = indicator_registry.get_indicator(indicator_id)
        if not indicator_class:
            raise HTTPException(status_code=404, detail=
                f"Indicator '{indicator_id}' not found")
        metadata = indicator_class.get_metadata()
        parameters = []
        for param_name, param_info in metadata['parameters'].items():
            parameters.append(ParameterInfo(name=param_name, description=
                param_info.get('description', ''), type=param_info.get(
                'type', 'string'), default=str(param_info.get('default')) if
                param_info.get('default') is not None else None, min=
                param_info.get('min'), max=param_info.get('max'), options=
                param_info.get('options')))
        result = IndicatorInfo(id=metadata['id'], name=metadata['name'],
            description=metadata['description'], category=metadata[
            'category'], parameters=parameters)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting indicator '{indicator_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@indicator_router.get('/categories', response_model=List[str])
@async_with_exception_handling
async def get_categories():
    """
    Get a list of all indicator categories.
    """
    if indicator_registry is None:
        raise HTTPException(status_code=503, detail=
            'Indicator registry is not initialized')
    try:
        return indicator_registry.get_categories()
    except Exception as e:
        logger.error(f'Error getting categories: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@indicator_router.get('/{indicator_id}/values')
@async_with_exception_handling
async def get_indicator_values(indicator_id: str, symbol: str, timeframe:
    str, start_date: Optional[datetime]=None, end_date: Optional[datetime]=
    None, columns: Optional[List[str]]=Query(None)):
    """
    Get computed values for a specific indicator.
    """
    try:
        indicator_class = indicator_registry.get_indicator(indicator_id)
        if not indicator_class:
            raise HTTPException(status_code=404, detail=
                f'Indicator {indicator_id} not found')
        data = await feature_storage.get_indicator_data(indicator_id=
            indicator_id, symbol=symbol, timeframe=timeframe, start_date=
            start_date, end_date=end_date, columns=columns)
        if data.empty:
            return {'data': [], 'columns': []}
        result = []
        for timestamp, row in data.iterrows():
            record = {'timestamp': timestamp.isoformat()}
            for col in data.columns:
                record[col] = float(row[col]) if pd.notna(row[col]) else None
            result.append(record)
        return {'data': result, 'columns': list(data.columns)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error retrieving indicator values: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@indicator_router.get('/presets')
async def get_indicator_presets():
    """
    Get predefined indicator presets for common analysis scenarios.
    """
    presets = {'trend_following': {'name': 'Trend Following', 'description':
        'Indicators useful for trend following strategies', 'indicators': [
        'sma', 'ema', 'macd', 'adx']}, 'swing_trading': {'name':
        'Swing Trading', 'description':
        'Indicators useful for swing trading strategies', 'indicators': [
        'rsi', 'stochastic', 'bollinger_bands', 'fibonacci_retracements']},
        'volatility_based': {'name': 'Volatility Based', 'description':
        'Indicators focusing on market volatility', 'indicators': ['atr',
        'bollinger_bands', 'keltner_channels', 'vix']}, 'mean_reversion': {
        'name': 'Mean Reversion', 'description':
        'Indicators useful for mean reversion strategies', 'indicators': [
        'rsi', 'bollinger_bands', 'stochastic', 'macd']}, 'volume_analysis':
        {'name': 'Volume Analysis', 'description':
        'Indicators focused on volume analysis', 'indicators': ['obv',
        'mfi', 'vwap', 'volume']}}
    return {'presets': presets}


@indicator_router.get('/cache/stats', response_model=CacheStatistics)
@async_with_exception_handling
async def get_cache_statistics():
    """
    Get statistics about the indicator caching system.
    """
    if enhanced_indicator_service is None:
        raise HTTPException(status_code=503, detail=
            'Enhanced indicator service with caching is not enabled')
    try:
        stats = enhanced_indicator_service.get_cache_stats()
        return stats
    except Exception as e:
        logger.error(f'Error retrieving cache statistics: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@indicator_router.post('/cache/clear/symbol/{symbol}')
@async_with_exception_handling
async def clear_cache_for_symbol(symbol: str):
    """
    Clear all cached indicator data for a specific symbol.
    """
    if enhanced_indicator_service is None:
        raise HTTPException(status_code=503, detail=
            'Enhanced indicator service with caching is not enabled')
    try:
        cleared_count = enhanced_indicator_service.clear_cache_for_symbol(
            symbol)
        return {'message':
            f'Cleared {cleared_count} cache entries for symbol {symbol}'}
    except Exception as e:
        logger.error(f'Error clearing cache for symbol {symbol}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@indicator_router.post('/cache/clear/indicator/{indicator_type}')
@async_with_exception_handling
async def clear_cache_for_indicator(indicator_type: str):
    """
    Clear all cached data for a specific indicator type.
    """
    if enhanced_indicator_service is None:
        raise HTTPException(status_code=503, detail=
            'Enhanced indicator service with caching is not enabled')
    try:
        cleared_count = enhanced_indicator_service.clear_cache_for_indicator(
            indicator_type)
        return {'message':
            f'Cleared {cleared_count} cache entries for indicator {indicator_type}'
            }
    except Exception as e:
        logger.error(
            f'Error clearing cache for indicator {indicator_type}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


def set_indicator_registry(registry: IndicatorRegistry, indicator_service:
    Optional[EnhancedIndicatorService]=None):
    """
    Set the indicator registry instance and enhanced indicator service if available.

    Args:
        registry: The indicator registry instance
        indicator_service: Optional enhanced indicator service with caching support
    """
    global indicator_registry, enhanced_indicator_service
    indicator_registry = registry
    if indicator_service:
        enhanced_indicator_service = indicator_service
        logger.info(
            'Enhanced indicator service with caching support is enabled')
