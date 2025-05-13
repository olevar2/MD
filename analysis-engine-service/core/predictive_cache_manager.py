"""
This file has been replaced by the standardized cache management implementation.

The original implementation has been backed up to:
analysis-engine-service\analysis_engine\utils/predictive_cache_manager.py.bak.20250512230259

Please use the standardized cache management from common-lib instead:

from common_lib.caching import (
    AdaptiveCacheManager,
    PredictiveCacheManager,
    cached,
    get_cache_manager,
    get_predictive_cache_manager
)
"""

from common_lib.caching import (
    AdaptiveCacheManager,
    PredictiveCacheManager,
    cached,
    get_cache_manager,
    get_predictive_cache_manager
)

# For backward compatibility
CacheManager = AdaptiveCacheManager
EnhancedCacheManager = AdaptiveCacheManager
cache = get_cache_manager()
cache_manager = get_cache_manager()
