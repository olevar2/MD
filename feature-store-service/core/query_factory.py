"""
Query Factory Module.

This module provides factory functions for creating optimized query utilities.
"""

from typing import Optional, Dict
import asyncpg

from core_foundations.utils.logger import get_logger
from core.timeseries_optimized_queries import TimeSeriesQueryOptimizer

logger = get_logger("feature-store-service.query-factory")

# Singleton instance
_query_optimizer_instance: Optional[TimeSeriesQueryOptimizer] = None


async def get_query_optimizer(
    db_pool: asyncpg.Pool,
    config: Optional[Dict] = None
) -> TimeSeriesQueryOptimizer:
    """
    Get or create an instance of the time-series query optimizer.
    
    Args:
        db_pool: Database connection pool
        config: Configuration options for the query optimizer
        
    Returns:
        TimeSeriesQueryOptimizer instance
    """
    global _query_optimizer_instance
    
    if _query_optimizer_instance is None:
        # Set default config values
        cache_enabled = True
        cache_ttl_seconds = 300  # 5 minutes
        max_cache_items = 1000
        
        # Override with provided config if available
        if config:
            cache_enabled = config_manager.get('cache_enabled', cache_enabled)
            cache_ttl_seconds = config_manager.get('cache_ttl_seconds', cache_ttl_seconds)
            max_cache_items = config_manager.get('max_cache_items', max_cache_items)
            
        # Create the query optimizer
        _query_optimizer_instance = TimeSeriesQueryOptimizer(
            db_pool=db_pool,
            cache_enabled=cache_enabled,
            cache_ttl_seconds=cache_ttl_seconds,
            max_cache_items=max_cache_items
        )
        
        # Start periodic cache cleanup in the background
        import asyncio
        asyncio.create_task(
            _query_optimizer_instance.periodic_cache_cleanup(interval_seconds=60)
        )
        
        logger.info("Time series query optimizer initialized")
        
    return _query_optimizer_instance