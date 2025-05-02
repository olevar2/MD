"""
Dependency Injection Setup for Feature Store Service.

This module provides functions to initialize and inject service dependencies
using FastAPI's Depends mechanism.
"""

import os
from functools import lru_cache
from typing import Dict, Any

from core_foundations.utils.logger import get_logger

# Import necessary classes (adjust paths as needed)
from .caching.config import CacheConfig
from .caching.enhanced_cache_manager import EnhancedCacheManager
from .indicators.indicator_registry import IndicatorRegistry
from .storage.feature_storage import FeatureStorage
from .services.enhanced_indicator_service import EnhancedIndicatorService
from .services.incremental_processor import RealTimeFeatureProcessor
from .computation.incremental.indicator_service import IncrementalIndicatorService
from .computation.feature_computation_engine import FeatureComputationEngine
from data_pipeline_service.services.ohlcv_service import OHLCVService, get_ohlcv_service # Assuming this is the correct way to get OHLCV service

logger = get_logger("feature-store-service.dependencies")

# --- Configuration Loading ---

@lru_cache()
def get_cache_config() -> Dict[str, Any]:
    """Loads cache configuration from file."""
    try:
        cache_config_path = os.path.join(os.path.dirname(__file__), "..", "config", "cache_config.json")
        config_data = CacheConfig.load_config(cache_config_path)
        logger.info(f"Loaded cache configuration from {cache_config_path}")
        return config_data
    except Exception as e:
        logger.error(f"Failed to load cache configuration: {e}", exc_info=True)
        # Return default or raise error? Returning default for now.
        return {}

# --- Singleton Service Instances ---
# Use @lru_cache(maxsize=None) for simple singletons if state allows

@lru_cache(maxsize=None)
def get_indicator_registry() -> IndicatorRegistry:
    """Provides a singleton instance of IndicatorRegistry."""
    logger.debug("Initializing IndicatorRegistry singleton")
    registry = IndicatorRegistry()
    # Initialization like register_all_indicators happens in main.startup_event
    return registry

@lru_cache(maxsize=None)
def get_feature_storage() -> FeatureStorage:
    """Provides a singleton instance of FeatureStorage."""
    logger.debug("Initializing FeatureStorage singleton")
    storage = FeatureStorage()
    # Initialization like .initialize() happens in main.startup_event
    return storage

@lru_cache(maxsize=None)
def get_enhanced_cache_manager() -> EnhancedCacheManager:
    """Provides a singleton instance of EnhancedCacheManager."""
    logger.debug("Initializing EnhancedCacheManager singleton")
    config = get_cache_config()
    # Assuming EnhancedCacheManager takes config dict directly
    cache_manager = EnhancedCacheManager(config=config.get('enhanced_cache', {}))
    logger.info(f"Initialized EnhancedCacheManager with config: {config.get('enhanced_cache', {})}")
    return cache_manager

@lru_cache(maxsize=None)
def get_enhanced_indicator_service() -> EnhancedIndicatorService:
    """Provides a singleton instance of EnhancedIndicatorService."""
    logger.debug("Initializing EnhancedIndicatorService singleton")
    registry = get_indicator_registry()
    cache_manager = get_enhanced_cache_manager()
    storage = get_feature_storage()
    config = get_cache_config() # Or specific config section if needed

    # EnhancedIndicatorService now takes cache_manager directly
    service = EnhancedIndicatorService(
        registry=registry,
        cache_manager=cache_manager,
        storage=storage,
        config=config # Pass the loaded config
    )
    logger.info("Initialized EnhancedIndicatorService singleton")
    return service

@lru_cache(maxsize=None)
def get_feature_computation_engine() -> FeatureComputationEngine:
    """Provides a singleton instance of FeatureComputationEngine."""
    logger.debug("Initializing FeatureComputationEngine singleton")
    engine = FeatureComputationEngine(
        indicator_registry=get_indicator_registry(),
        feature_storage=get_feature_storage(),
        enhanced_indicator_service=get_enhanced_indicator_service()
    )
    logger.info("Initialized FeatureComputationEngine singleton")
    return engine

@lru_cache(maxsize=None)
def get_realtime_feature_processor() -> RealTimeFeatureProcessor:
    """Provides a singleton instance of RealTimeFeatureProcessor."""
    logger.debug("Initializing RealTimeFeatureProcessor singleton")
    # Assuming RealTimeFeatureProcessor has default constructor or gets config elsewhere
    processor = RealTimeFeatureProcessor()
    logger.info("Initialized RealTimeFeatureProcessor singleton")
    return processor

# Note: IncrementalIndicatorService needs OHLCVService, which might be async to get
# FastAPI handles async dependencies correctly.
async def get_incremental_indicator_service() -> IncrementalIndicatorService:
    """Provides an instance of IncrementalIndicatorService."""
    # This might not be a strict singleton if state needs careful management,
    # but for now, let's assume it can be treated as one per request or cached.
    # Using a simple cache for demonstration. A more robust approach might be needed.

    # Check if already initialized (simple in-memory flag)
    # A more robust approach might involve checking state in storage or using FastAPI's app state
    global _incremental_service_instance
    if _incremental_service_instance is None:
        logger.debug("Initializing IncrementalIndicatorService instance")
        storage = get_feature_storage()
        # Assuming get_ohlcv_service is an async dependency provider
        ohlcv_service: OHLCVService = await get_ohlcv_service()
        _incremental_service_instance = IncrementalIndicatorService(storage, ohlcv_service)
        # Initialization like load_all_saved_states happens in main.startup_event
        logger.info("Initialized IncrementalIndicatorService instance")

    return _incremental_service_instance

# Global variable to hold the singleton instance for the async dependency
_incremental_service_instance: IncrementalIndicatorService | None = None

# Function to reset the async singleton (e.g., for testing)
def reset_incremental_service_instance():
    global _incremental_service_instance
    _incremental_service_instance = None

