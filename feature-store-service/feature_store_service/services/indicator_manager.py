"""
Indicator Manager Service.

This service provides dependency injection for the incremental indicator service
and manages its lifecycle.
"""

import os
from typing import Optional
from fastapi import Depends

from core_foundations.utils.logger import get_logger
from core_foundations.config.config_loader import ConfigLoader
from feature_store_service.computation.incremental.indicator_service import IncrementalIndicatorService
from feature_store_service.storage.feature_storage import FeatureStorage, get_feature_storage
from data_pipeline_service.services.ohlcv_service import OHLCVService, get_ohlcv_service

logger = get_logger("feature-store-service.indicator-manager")

# Singleton instance
_indicator_service_instance: Optional[IncrementalIndicatorService] = None


async def initialize_indicator_service(
    feature_storage: FeatureStorage,
    ohlcv_service: OHLCVService,
) -> IncrementalIndicatorService:
    """
    Initialize the incremental indicator service.
    
    Args:
        feature_storage: Storage for computed indicators
        ohlcv_service: Service for retrieving OHLCV data
        
    Returns:
        Initialized IncrementalIndicatorService
    """
    config = ConfigLoader().get_config()
    
    # Get state persistence path from config
    state_persistence_path = None
    if hasattr(config, "feature_store_service"):
        if hasattr(config.feature_store_service, "indicator_state_path"):
            state_persistence_path = config.feature_store_service.indicator_state_path
            
    if not state_persistence_path:
        # Use a default path in the project directory
        state_persistence_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "indicator_states"
        )
        
    # Create the service instance
    service = IncrementalIndicatorService(
        feature_storage=feature_storage,
        ohlcv_service=ohlcv_service,
        state_persistence_path=state_persistence_path
    )
    
    # Load saved states if available
    try:
        count = await service.load_all_saved_states()
        logger.info(f"Loaded {count} indicator states from {state_persistence_path}")
    except Exception as e:
        logger.warning(f"Failed to load indicator states: {str(e)}")
        
    return service


async def get_indicator_manager(
    feature_storage: FeatureStorage = Depends(get_feature_storage),
    ohlcv_service: OHLCVService = Depends(get_ohlcv_service),
) -> IncrementalIndicatorService:
    """
    Get the incremental indicator service instance.
    
    This function is used as a dependency for FastAPI endpoints.
    
    Args:
        feature_storage: Storage for computed indicators
        ohlcv_service: Service for retrieving OHLCV data
        
    Returns:
        IncrementalIndicatorService instance
    """
    global _indicator_service_instance
    
    if _indicator_service_instance is None:
        _indicator_service_instance = await initialize_indicator_service(
            feature_storage=feature_storage,
            ohlcv_service=ohlcv_service
        )
        
    return _indicator_service_instance