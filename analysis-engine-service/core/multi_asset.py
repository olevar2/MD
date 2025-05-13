"""
Multi-Asset API Module

This module provides API endpoints for working with multiple asset 
classes and markets.
"""
import logging
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from analysis_engine.services.multi_asset_service import MultiAssetService
from analysis_engine.multi_asset.asset_registry import AssetClass
from analysis_engine.api.dependencies import get_current_user
from analysis_engine.models.user import User
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/multi-asset', tags=['multi-asset'], responses={
    (404): {'description': 'Not found'}})


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def get_multi_asset_service():
    """Dependency for getting the MultiAssetService instance"""
    return MultiAssetService()


@router.get('/assets', summary='List all available assets')
@async_with_exception_handling
async def list_assets(asset_class: Optional[str]=Query(None, description=
    'Filter by asset class'), multi_asset_service: MultiAssetService=
    Depends(get_multi_asset_service), current_user: User=Depends(
    get_current_user)):
    """
    List all available assets with optional filtering by asset class.
    
    Returns a list of asset symbols and display names.
    """
    try:
        asset_class_enum = None
        if asset_class:
            try:
                asset_class_enum = AssetClass(asset_class.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=
                    f"Invalid asset class: {asset_class}. Must be one of: {', '.join([a.value for a in AssetClass])}"
                    )
        if asset_class_enum:
            symbols = multi_asset_service.list_assets_by_class(asset_class_enum
                )
        else:
            symbols = []
            for cls in AssetClass:
                symbols.extend(multi_asset_service.list_assets_by_class(cls))
        assets = []
        for symbol in symbols:
            asset_info = multi_asset_service.get_asset_info(symbol)
            if asset_info:
                assets.append({'symbol': symbol, 'display_name': asset_info
                    .get('display_name', symbol), 'asset_class': asset_info
                    .get('asset_class'), 'market_type': asset_info.get(
                    'market_type'), 'base_currency': asset_info.get(
                    'base_currency'), 'quote_currency': asset_info.get(
                    'quote_currency')})
        return {'assets': assets}
    except Exception as e:
        logger.error(f'Error listing assets: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to list assets: {str(e)}')


@router.get('/assets/{symbol}', summary='Get asset details')
@async_with_exception_handling
async def get_asset_details(symbol: str=Path(..., description=
    'Asset symbol'), multi_asset_service: MultiAssetService=Depends(
    get_multi_asset_service), current_user: User=Depends(get_current_user)):
    """
    Get detailed information about a specific asset.
    """
    try:
        asset_info = multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            raise HTTPException(status_code=404, detail=
                f'Asset not found: {symbol}')
        return asset_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting asset details: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get asset details: {str(e)}')


@router.get('/correlations/{symbol}', summary='Get correlated assets')
@async_with_exception_handling
async def get_correlated_assets(symbol: str=Path(..., description=
    'Asset symbol'), threshold: float=Query(0.6, description=
    'Minimum correlation threshold', ge=0.0, le=1.0), multi_asset_service:
    MultiAssetService=Depends(get_multi_asset_service), current_user: User=
    Depends(get_current_user)):
    """
    Get assets correlated with the specified symbol above the given threshold.
    """
    try:
        asset_info = multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            raise HTTPException(status_code=404, detail=
                f'Asset not found: {symbol}')
        correlated_assets = (multi_asset_service.asset_registry.
            get_correlated_assets(symbol, threshold))
        return {'correlations': correlated_assets}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting correlated assets: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get correlated assets: {str(e)}')


@router.get('/groups', summary='List asset groups')
@async_with_exception_handling
async def list_asset_groups(multi_asset_service: MultiAssetService=Depends(
    get_multi_asset_service), current_user: User=Depends(get_current_user)):
    """
    List all available asset groups.
    """
    try:
        groups = {}
        for group_name in multi_asset_service.asset_registry._asset_groups.keys(
            ):
            groups[group_name] = multi_asset_service.get_asset_group(group_name
                )
        return {'groups': groups}
    except Exception as e:
        logger.error(f'Error listing asset groups: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to list asset groups: {str(e)}')


@router.get('/groups/{group_name}', summary='Get assets in group')
@async_with_exception_handling
async def get_asset_group(group_name: str=Path(..., description=
    'Group name'), multi_asset_service: MultiAssetService=Depends(
    get_multi_asset_service), current_user: User=Depends(get_current_user)):
    """
    Get all assets in a named group.
    """
    try:
        symbols = multi_asset_service.get_asset_group(group_name)
        if not symbols:
            raise HTTPException(status_code=404, detail=
                f'Group not found or empty: {group_name}')
        assets = []
        for symbol in symbols:
            asset_info = multi_asset_service.get_asset_info(symbol)
            if asset_info:
                assets.append({'symbol': symbol, 'display_name': asset_info
                    .get('display_name', symbol), 'asset_class': asset_info
                    .get('asset_class'), 'market_type': asset_info.get(
                    'market_type')})
        return {'group_name': group_name, 'assets': assets}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting asset group: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get asset group: {str(e)}')


@router.get('/analysis-parameters/{symbol}', summary='Get analysis parameters')
@async_with_exception_handling
async def get_analysis_parameters(symbol: str=Path(..., description=
    'Asset symbol'), multi_asset_service: MultiAssetService=Depends(
    get_multi_asset_service), current_user: User=Depends(get_current_user)):
    """
    Get asset-specific analysis parameters.
    """
    try:
        asset_info = multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            raise HTTPException(status_code=404, detail=
                f'Asset not found: {symbol}')
        parameters = multi_asset_service.get_analysis_parameters(symbol)
        return {'symbol': symbol, 'parameters': parameters}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting analysis parameters: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get analysis parameters: {str(e)}')
