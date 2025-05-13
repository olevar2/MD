"""
Correlation API Module

This module provides API endpoints for accessing cross-asset correlation data.
"""
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from analysis_engine.multi_asset.correlation_tracking_service import CorrelationTrackingService
from analysis_engine.services.multi_asset_service import MultiAssetService
from analysis_engine.multi_asset.asset_registry import AssetClass
from analysis_engine.api.dependencies import get_current_user
from analysis_engine.models.user import User
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/v1/correlations', tags=['correlations'],
    responses={(404): {'description': 'Not found'}})


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def get_correlation_service(multi_asset_service: MultiAssetService=Depends()
    ) ->CorrelationTrackingService:
    """Dependency for getting the correlation tracking service"""
    return CorrelationTrackingService(asset_registry=multi_asset_service.
        asset_registry)


@router.get('/matrix', summary='Get correlation matrix')
@async_with_exception_handling
async def get_correlation_matrix(symbols: List[str]=Query(..., description=
    'List of symbols to include in the correlation matrix'),
    correlation_service: CorrelationTrackingService=Depends(
    get_correlation_service), current_user: User=Depends(get_current_user)):
    """
    Get the correlation matrix for a list of symbols.
    
    Returns a matrix of correlation values between all specified symbols.
    """
    try:
        matrix = await correlation_service.get_correlation_matrix(symbols)
        return {'correlation_matrix': matrix}
    except Exception as e:
        logger.error(f'Error getting correlation matrix: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get correlation matrix: {str(e)}')


@router.get('/symbol/{symbol}', summary='Get symbol correlations')
@async_with_exception_handling
async def get_symbol_correlations(symbol: str=Path(..., description=
    'Symbol to find correlations for'), threshold: float=Query(0.7,
    description='Minimum correlation threshold', ge=0.0, le=1.0), limit:
    int=Query(10, description='Maximum number of results to return', ge=1,
    le=50), correlation_service: CorrelationTrackingService=Depends(
    get_correlation_service), multi_asset_service: MultiAssetService=
    Depends(), current_user: User=Depends(get_current_user)):
    """
    Get the assets most correlated with the specified symbol.
    
    Returns a list of correlated assets with correlation values.
    """
    try:
        asset_info = multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            raise HTTPException(status_code=404, detail=
                f'Asset not found: {symbol}')
        correlations = await correlation_service.get_highest_correlations(
            symbol, threshold, limit)
        return {'symbol': symbol, 'asset_class': asset_info.get(
            'asset_class'), 'correlations': correlations}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting symbol correlations: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get symbol correlations: {str(e)}')


@router.get('/cross-asset/{symbol}', summary='Get cross-asset correlations')
@async_with_exception_handling
async def get_cross_asset_correlations(symbol: str=Path(..., description=
    'Symbol to find correlations for'), asset_classes: Optional[List[str]]=
    Query(None, description='Asset classes to include'),
    correlation_service: CorrelationTrackingService=Depends(
    get_correlation_service), multi_asset_service: MultiAssetService=
    Depends(), current_user: User=Depends(get_current_user)):
    """
    Get correlations between a symbol and symbols from other asset classes.
    
    Returns a dictionary mapping asset classes to lists of correlated symbols.
    """
    try:
        asset_info = multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            raise HTTPException(status_code=404, detail=
                f'Asset not found: {symbol}')
        asset_class_enums = None
        if asset_classes:
            try:
                asset_class_enums = [AssetClass(ac.lower()) for ac in
                    asset_classes]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=
                    f'Invalid asset class: {str(e)}')
        results = await correlation_service.get_cross_asset_correlations(symbol
            , asset_class_enums)
        string_results = {(ac.value if isinstance(ac, AssetClass) else ac):
            vals for ac, vals in results.items()}
        return {'symbol': symbol, 'asset_class': asset_info.get(
            'asset_class'), 'cross_asset_correlations': string_results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting cross-asset correlations: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get cross-asset correlations: {str(e)}')


@router.get('/changes/{symbol1}/{symbol2}', summary='Get correlation changes')
@async_with_exception_handling
async def get_correlation_changes(symbol1: str=Path(..., description=
    'First symbol'), symbol2: str=Path(..., description='Second symbol'),
    correlation_service: CorrelationTrackingService=Depends(
    get_correlation_service), multi_asset_service: MultiAssetService=
    Depends(), current_user: User=Depends(get_current_user)):
    """
    Get correlation changes over different time periods.
    
    Returns correlation values for different lookback periods.
    """
    try:
        if not multi_asset_service.get_asset_info(symbol1):
            raise HTTPException(status_code=404, detail=
                f'Asset not found: {symbol1}')
        if not multi_asset_service.get_asset_info(symbol2):
            raise HTTPException(status_code=404, detail=
                f'Asset not found: {symbol2}')
        changes = await correlation_service.get_correlation_changes(symbol1,
            symbol2)
        regime_change = (await correlation_service.
            find_correlation_regime_change(symbol1, symbol2))
        return {'symbol1': symbol1, 'symbol2': symbol2,
            'correlation_changes': changes, 'regime_change': regime_change}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting correlation changes: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get correlation changes: {str(e)}')


@router.get('/visualization', summary='Get correlation visualization data')
@async_with_exception_handling
async def get_visualization_data(symbols: List[str]=Query(..., description=
    'List of symbols to include'), correlation_service:
    CorrelationTrackingService=Depends(get_correlation_service),
    current_user: User=Depends(get_current_user)):
    """
    Get data formatted for correlation visualization.
    
    Returns nodes and links for a correlation network visualization.
    """
    try:
        data = correlation_service.get_correlation_visualization_data(symbols)
        return data
    except Exception as e:
        logger.error(f'Error getting visualization data: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get visualization data: {str(e)}')


@router.post('/update', summary='Update correlations')
@async_with_exception_handling
async def update_correlations(correlation_service:
    CorrelationTrackingService=Depends(get_correlation_service),
    current_user: User=Depends(get_current_user)):
    """
    Update correlation data for all assets.
    
    Returns the number of correlation records updated.
    """
    try:
        updated_count = await correlation_service.update_asset_correlations()
        return {'status': 'success', 'updated_count': updated_count}
    except Exception as e:
        logger.error(f'Error updating correlations: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to update correlations: {str(e)}')
