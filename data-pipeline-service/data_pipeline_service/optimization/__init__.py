"""
Optimization Module.

This module provides utilities for optimizing database queries and other operations.
"""

from data_pipeline_service.optimization.query_optimizer import (
    QueryOptimizer,
    optimize_query,
    query_optimizer
)

from data_pipeline_service.optimization.index_manager import (
    IndexManager,
    get_index_manager
)

from data_pipeline_service.optimization.connection_pool import (
    OptimizedConnectionPool,
    optimized_pool,
    get_optimized_sa_session,
    get_optimized_asyncpg_connection,
    initialize_optimized_pool,
    close_optimized_pool
)

__all__ = [
    'QueryOptimizer',
    'optimize_query',
    'query_optimizer',
    'IndexManager',
    'get_index_manager',
    'OptimizedConnectionPool',
    'optimized_pool',
    'get_optimized_sa_session',
    'get_optimized_asyncpg_connection',
    'initialize_optimized_pool',
    'close_optimized_pool'
]
