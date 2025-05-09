"""
Cache API Module.

This module provides API endpoints for monitoring and managing the caching system.
"""

import logging
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends

from ml_integration_service.caching.model_inference_cache import (
    get_cache_stats,
    clear_model_cache
)
from ml_integration_service.caching.feature_vector_cache import (
    get_feature_cache_stats,
    clear_feature_cache
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/cache",
    tags=["cache"],
    responses={404: {"description": "Not found"}},
)


@router.get("/stats", response_model=Dict[str, Any])
async def get_cache_statistics() -> Dict[str, Any]:
    """
    Get statistics about the caching system.
    
    Returns:
        Dictionary with cache statistics
    """
    try:
        # Get statistics from different cache systems
        model_stats = get_cache_stats()
        feature_stats = get_feature_cache_stats()
        
        # Combine statistics
        return {
            "model_cache": model_stats,
            "feature_cache": feature_stats,
            "total_entries": model_stats["total_entries"] + feature_stats["total_entries"],
            "total_active_entries": model_stats["active_entries"] + feature_stats["active_entries"],
            "total_expired_entries": model_stats["expired_entries"] + feature_stats["expired_entries"]
        }
    except Exception as e:
        logger.error(f"Error getting cache statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting cache statistics: {str(e)}")


@router.post("/clear", response_model=Dict[str, Any])
async def clear_cache(
    model_name: str = None,
    symbol: str = None,
    clear_all: bool = False
) -> Dict[str, Any]:
    """
    Clear the cache.
    
    Args:
        model_name: Optional model name to clear only entries for that model
        symbol: Optional symbol to clear only entries for that symbol
        clear_all: Whether to clear all caches
        
    Returns:
        Dictionary with result
    """
    try:
        if clear_all:
            # Clear all caches
            clear_model_cache()
            clear_feature_cache()
            return {"status": "success", "message": "All caches cleared"}
        elif model_name or symbol:
            # Clear specific entries
            clear_model_cache(model_name, symbol)
            clear_feature_cache(model_name, symbol)
            
            message = "Cleared cache"
            if model_name:
                message += f" for model {model_name}"
            if symbol:
                message += f" for symbol {symbol}"
                
            return {"status": "success", "message": message}
        else:
            return {"status": "error", "message": "No cache clearing parameters provided"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")
