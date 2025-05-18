"""
Benchmark for database connection pool.
"""
import os
import sys
import time
import logging
import asyncio
import argparse
import json
import random
from typing import Dict, List, Any, Optional, Tuple
import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from common_lib.database import (
    get_connection_pool,
    get_sync_db_session,
    get_async_db_session,
    get_asyncpg_connection,
)
from benchmarks.database.benchmark_framework import DatabaseBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Database configuration
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
DB_NAME = os.environ.get("DB_NAME", "forex_platform")
DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


async def setup_database():
    """Set up the database for benchmarking."""
    logger.info("Setting up database...")
    
    # Create a direct connection to the database
    conn = await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )
    
    try:
        # Create benchmark schema
        await conn.execute("CREATE SCHEMA IF NOT EXISTS benchmark")
        
        # Create benchmark table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark.connection_pool_test (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                value INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        
        # Create benchmark time series table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark.connection_pool_time_series (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                value DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (time, symbol)
            )
        """)
        
        # Try to create hypertable (will fail if TimescaleDB is not installed)
        try:
            await conn.execute("""
                SELECT create_hypertable('benchmark.connection_pool_time_series', 'time', if_not_exists => TRUE)
            """)
        except Exception as e:
            logger.warning(f"Failed to create hypertable: {e}")
        
        logger.info("Database setup complete")
    finally:
        # Close the connection
        await conn.close()


async def cleanup_database():
    """Clean up the database after benchmarking."""
    logger.info("Cleaning up database...")
    
    # Create a direct connection to the database
    conn = await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )
    
    try:
        # Truncate benchmark tables
        await conn.execute("TRUNCATE TABLE benchmark.connection_pool_test")
        await conn.execute("TRUNCATE TABLE benchmark.connection_pool_time_series")
        
        logger.info("Database cleanup complete")
    finally:
        # Close the connection
        await conn.close()


async def benchmark_connection_acquisition(pool_size: int, max_overflow: int, concurrency: int):
    """
    Benchmark connection acquisition.
    
    Args:
        pool_size: Size of the connection pool
        max_overflow: Maximum number of overflow connections
        concurrency: Number of concurrent connections to acquire
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking connection acquisition with pool_size={pool_size}, max_overflow={max_overflow}, concurrency={concurrency}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("ConnectionPoolBenchmark")
    
    # Create connection pool
    pool = get_connection_pool(
        "benchmark_service",
        database_url=DB_URL,
        async_database_url=DB_URL,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
        prepared_statement_cache_size=100,
    )
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Define a simple query function
    async def execute_query(i: int):
        async with pool.get_async_session() as session:
            result = await session.execute("SELECT 1")
            return result.scalar_one()
    
    # Benchmark connection acquisition
    await benchmark.benchmark_async_throughput(
        name=f"connection_acquisition_pool_{pool_size}_overflow_{max_overflow}_concurrency_{concurrency}",
        category="connection_acquisition",
        func=execute_query,
        args_list=[(i,) for i in range(concurrency)],
        concurrency=concurrency,
        metadata={
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "concurrency": concurrency,
        },
    )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def benchmark_query_execution(pool_size: int, query_complexity: str, concurrency: int):
    """
    Benchmark query execution.
    
    Args:
        pool_size: Size of the connection pool
        query_complexity: Complexity of the query (simple, medium, complex)
        concurrency: Number of concurrent queries to execute
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking query execution with pool_size={pool_size}, query_complexity={query_complexity}, concurrency={concurrency}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("QueryExecutionBenchmark")
    
    # Create connection pool
    pool = get_connection_pool(
        "benchmark_service",
        database_url=DB_URL,
        async_database_url=DB_URL,
        pool_size=pool_size,
        max_overflow=pool_size,  # Same as pool_size
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
        prepared_statement_cache_size=100,
    )
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Define queries based on complexity
    if query_complexity == "simple":
        query = "SELECT 1"
        params = {}
    elif query_complexity == "medium":
        query = """
            SELECT * FROM benchmark.connection_pool_test
            WHERE value > :value
            ORDER BY value
            LIMIT 100
        """
        params = {"value": 0}
    elif query_complexity == "complex":
        query = """
            SELECT
                t1.name,
                t1.value,
                t2.time,
                t2.value as time_value
            FROM
                benchmark.connection_pool_test t1
            JOIN
                benchmark.connection_pool_time_series t2
            ON
                t1.name = t2.symbol
            WHERE
                t1.value > :value
                AND t2.time > :time
            ORDER BY
                t2.time DESC
            LIMIT 100
        """
        from datetime import datetime, timedelta
        params = {
            "value": 0,
            "time": datetime.now() - timedelta(days=1),
        }
    else:
        raise ValueError(f"Invalid query complexity: {query_complexity}")
    
    # Define a query function
    async def execute_query(i: int):
        async with pool.get_async_session() as session:
            result = await session.execute(text(query), params)
            return result.fetchall()
    
    # Benchmark query execution
    await benchmark.benchmark_async_throughput(
        name=f"query_execution_pool_{pool_size}_complexity_{query_complexity}_concurrency_{concurrency}",
        category="query_execution",
        func=execute_query,
        args_list=[(i,) for i in range(concurrency)],
        concurrency=concurrency,
        metadata={
            "pool_size": pool_size,
            "query_complexity": query_complexity,
            "concurrency": concurrency,
        },
    )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def benchmark_connection_reuse(pool_size: int, operations_per_connection: int, concurrency: int):
    """
    Benchmark connection reuse.
    
    Args:
        pool_size: Size of the connection pool
        operations_per_connection: Number of operations to perform with each connection
        concurrency: Number of concurrent connections to acquire
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking connection reuse with pool_size={pool_size}, operations_per_connection={operations_per_connection}, concurrency={concurrency}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("ConnectionReuseBenchmark")
    
    # Create connection pool
    pool = get_connection_pool(
        "benchmark_service",
        database_url=DB_URL,
        async_database_url=DB_URL,
        pool_size=pool_size,
        max_overflow=pool_size,  # Same as pool_size
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
        prepared_statement_cache_size=100,
    )
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Define a function that performs multiple operations with a single connection
    async def execute_operations(i: int):
        async with pool.get_async_session() as session:
            for j in range(operations_per_connection):
                result = await session.execute(
                    "INSERT INTO benchmark.connection_pool_test (name, value) VALUES (:name, :value) RETURNING id",
                    {"name": f"test_{i}_{j}", "value": i * 100 + j},
                )
                id = result.scalar_one()
                
                result = await session.execute(
                    "SELECT * FROM benchmark.connection_pool_test WHERE id = :id",
                    {"id": id},
                )
                row = result.fetchone()
                
                result = await session.execute(
                    "UPDATE benchmark.connection_pool_test SET value = :value WHERE id = :id",
                    {"id": id, "value": row.value * 2},
                )
            
            await session.commit()
    
    # Benchmark connection reuse
    await benchmark.benchmark_async_throughput(
        name=f"connection_reuse_pool_{pool_size}_operations_{operations_per_connection}_concurrency_{concurrency}",
        category="connection_reuse",
        func=execute_operations,
        args_list=[(i,) for i in range(concurrency)],
        concurrency=concurrency,
        metadata={
            "pool_size": pool_size,
            "operations_per_connection": operations_per_connection,
            "concurrency": concurrency,
        },
    )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def benchmark_direct_asyncpg(concurrency: int):
    """
    Benchmark direct asyncpg connection.
    
    Args:
        concurrency: Number of concurrent connections to acquire
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking direct asyncpg connection with concurrency={concurrency}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("DirectAsyncpgBenchmark")
    
    # Create connection pool
    pool = get_connection_pool(
        "benchmark_service",
        database_url=DB_URL,
        async_database_url=DB_URL,
        pool_size=10,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
        prepared_statement_cache_size=100,
    )
    
    # Initialize the pool
    await pool.initialize_asyncpg()
    
    # Define a function that uses direct asyncpg connection
    async def execute_query(i: int):
        async with pool.get_asyncpg_connection() as conn:
            result = await conn.fetchval("SELECT 1")
            return result
    
    # Benchmark direct asyncpg connection
    await benchmark.benchmark_async_throughput(
        name=f"direct_asyncpg_concurrency_{concurrency}",
        category="direct_asyncpg",
        func=execute_query,
        args_list=[(i,) for i in range(concurrency)],
        concurrency=concurrency,
        metadata={
            "concurrency": concurrency,
        },
    )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def run_benchmarks():
    """Run all benchmarks."""
    logger.info("Running connection pool benchmarks...")
    
    # Set up database
    await setup_database()
    
    try:
        # Create combined benchmark
        benchmark = DatabaseBenchmark("ConnectionPoolBenchmarks")
        
        # Benchmark connection acquisition with different pool sizes
        for pool_size in [5, 10, 20]:
            for max_overflow in [0, pool_size, pool_size * 2]:
                for concurrency in [10, 50, 100]:
                    result = await benchmark_connection_acquisition(pool_size, max_overflow, concurrency)
                    benchmark.results.update(result.results)
        
        # Benchmark query execution with different complexities
        for pool_size in [5, 10, 20]:
            for query_complexity in ["simple", "medium", "complex"]:
                for concurrency in [10, 50, 100]:
                    result = await benchmark_query_execution(pool_size, query_complexity, concurrency)
                    benchmark.results.update(result.results)
        
        # Benchmark connection reuse
        for pool_size in [5, 10, 20]:
            for operations_per_connection in [1, 5, 10]:
                for concurrency in [10, 50, 100]:
                    result = await benchmark_connection_reuse(pool_size, operations_per_connection, concurrency)
                    benchmark.results.update(result.results)
        
        # Benchmark direct asyncpg
        for concurrency in [10, 50, 100]:
            result = await benchmark_direct_asyncpg(concurrency)
            benchmark.results.update(result.results)
        
        # Save results
        benchmark.save_results()
        benchmark.save_csv()
        
        # Print results
        benchmark.print_results()
        
        # Plot results
        benchmark.plot_results(categories=["connection_acquisition"], save_path="benchmarks/database/results/connection_acquisition.png")
        benchmark.plot_results(categories=["query_execution"], save_path="benchmarks/database/results/query_execution.png")
        benchmark.plot_results(categories=["connection_reuse"], save_path="benchmarks/database/results/connection_reuse.png")
        benchmark.plot_results(categories=["direct_asyncpg"], save_path="benchmarks/database/results/direct_asyncpg.png")
        
        logger.info("Connection pool benchmarks complete")
    finally:
        # Clean up database
        await cleanup_database()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Connection pool benchmark")
    parser.add_argument("--db-host", default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, default=5432, help="Database port")
    parser.add_argument("--db-user", default="postgres", help="Database user")
    parser.add_argument("--db-password", default="postgres", help="Database password")
    parser.add_argument("--db-name", default="forex_platform", help="Database name")
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["DB_HOST"] = args.db_host
    os.environ["DB_PORT"] = str(args.db_port)
    os.environ["DB_USER"] = args.db_user
    os.environ["DB_PASSWORD"] = args.db_password
    os.environ["DB_NAME"] = args.db_name
    
    # Run benchmarks
    asyncio.run(run_benchmarks())


if __name__ == "__main__":
    main()