"""
Benchmark for prepared statements.
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
    with_prepared_statement,
    async_with_prepared_statement,
    execute_prepared_statement,
    execute_prepared_statement_async,
    execute_prepared_statement_asyncpg,
    fetch_prepared_statement_asyncpg,
    get_prepared_statement_cache,
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
            CREATE TABLE IF NOT EXISTS benchmark.prepared_statements_test (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                value INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        
        # Insert test data
        await conn.execute("TRUNCATE TABLE benchmark.prepared_statements_test")
        
        for i in range(1000):
            await conn.execute(
                "INSERT INTO benchmark.prepared_statements_test (name, value) VALUES ($1, $2)",
                f"test_{i}",
                i,
            )
        
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
        await conn.execute("TRUNCATE TABLE benchmark.prepared_statements_test")
        
        logger.info("Database cleanup complete")
    finally:
        # Close the connection
        await conn.close()


async def benchmark_prepared_vs_unprepared(query_complexity: str, executions: int):
    """
    Benchmark prepared vs. unprepared statements.
    
    Args:
        query_complexity: Complexity of the query (simple, medium, complex)
        executions: Number of executions
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking prepared vs. unprepared statements with query_complexity={query_complexity}, executions={executions}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("PreparedVsUnpreparedBenchmark")
    
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
    await pool.initialize_async()
    
    # Define queries based on complexity
    if query_complexity == "simple":
        query = "SELECT * FROM benchmark.prepared_statements_test WHERE id = :id"
        params = {"id": 1}
    elif query_complexity == "medium":
        query = """
            SELECT * FROM benchmark.prepared_statements_test
            WHERE value > :value
            ORDER BY value
            LIMIT 100
        """
        params = {"value": 500}
    elif query_complexity == "complex":
        query = """
            SELECT
                t1.id,
                t1.name,
                t1.value,
                t1.created_at,
                (SELECT COUNT(*) FROM benchmark.prepared_statements_test t2 WHERE t2.value < t1.value) as count_less,
                (SELECT AVG(value) FROM benchmark.prepared_statements_test t3 WHERE t3.value > :min_value) as avg_value
            FROM
                benchmark.prepared_statements_test t1
            WHERE
                t1.value BETWEEN :min_value AND :max_value
            ORDER BY
                t1.value DESC
            LIMIT 100
        """
        params = {
            "min_value": 200,
            "max_value": 800,
        }
    else:
        raise ValueError(f"Invalid query complexity: {query_complexity}")
    
    # Define a function that uses unprepared statements
    async def execute_unprepared(session: AsyncSession, params: Dict[str, Any]):
        result = await session.execute(text(query), params)
        return result.fetchall()
    
    # Define a function that uses prepared statements
    @async_with_prepared_statement("benchmark_service", f"prepared_{query_complexity}")
    async def execute_prepared(session: AsyncSession, params: Dict[str, Any]):
        result = await execute_prepared_statement_async(
            session,
            query,
            params,
            "benchmark_service",
            f"prepared_{query_complexity}",
        )
        return result.fetchall()
    
    # Benchmark unprepared statements
    async with pool.get_async_session() as session:
        # Warm up
        for _ in range(5):
            await execute_unprepared(session, params)
        
        # Benchmark
        await benchmark.benchmark_async(
            name=f"unprepared_{query_complexity}",
            category="unprepared",
            func=execute_unprepared,
            args=(session, params),
            repeat=executions,
            metadata={
                "query_complexity": query_complexity,
                "executions": executions,
            },
        )
    
    # Benchmark prepared statements
    async with pool.get_async_session() as session:
        # Warm up
        for _ in range(5):
            await execute_prepared(session, params)
        
        # Benchmark
        await benchmark.benchmark_async(
            name=f"prepared_{query_complexity}",
            category="prepared",
            func=execute_prepared,
            args=(session, params),
            repeat=executions,
            metadata={
                "query_complexity": query_complexity,
                "executions": executions,
            },
        )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def benchmark_prepared_statement_cache(cache_size: int, unique_queries: int, executions: int):
    """
    Benchmark prepared statement cache.
    
    Args:
        cache_size: Size of the prepared statement cache
        unique_queries: Number of unique queries
        executions: Number of executions per query
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking prepared statement cache with cache_size={cache_size}, unique_queries={unique_queries}, executions={executions}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("PreparedStatementCacheBenchmark")
    
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
        prepared_statement_cache_size=cache_size,
    )
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Get the prepared statement cache
    cache = get_prepared_statement_cache("benchmark_service")
    
    # Define a function that executes a prepared statement
    async def execute_prepared_statement_with_id(session: AsyncSession, id: int):
        # Create a unique query for each ID
        query = f"SELECT * FROM benchmark.prepared_statements_test WHERE id = :id AND value > {id % unique_queries}"
        
        # Execute the prepared statement
        result = await execute_prepared_statement_async(
            session,
            query,
            {"id": id},
            "benchmark_service",
            f"prepared_statement_{id % unique_queries}",
        )
        
        return result.fetchall()
    
    # Benchmark prepared statement cache
    async with pool.get_async_session() as session:
        # Warm up
        for i in range(min(unique_queries, 10)):
            await execute_prepared_statement_with_id(session, i)
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(executions):
            for i in range(unique_queries):
                await execute_prepared_statement_with_id(session, i)
        
        end_time = time.time()
        
        # Calculate throughput
        total_executions = executions * unique_queries
        throughput = total_executions / (end_time - start_time)
        
        # Get cache statistics
        cache_size_actual = len(cache._cache)
        cache_hits = cache.hits
        cache_misses = cache.misses
        cache_hit_ratio = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        # Create result
        result = benchmark.results.get(f"prepared_statement_cache_{cache_size}_{unique_queries}", None)
        if result is None:
            result = benchmark.BenchmarkResult(
                name=f"prepared_statement_cache_{cache_size}_{unique_queries}",
                category="prepared_statement_cache",
            )
            benchmark.results[result.name] = result
        
        result.add_execution_time(end_time - start_time)
        result.add_throughput(throughput)
        result.add_metadata("cache_size", cache_size)
        result.add_metadata("unique_queries", unique_queries)
        result.add_metadata("executions", executions)
        result.add_metadata("cache_size_actual", cache_size_actual)
        result.add_metadata("cache_hits", cache_hits)
        result.add_metadata("cache_misses", cache_misses)
        result.add_metadata("cache_hit_ratio", cache_hit_ratio)
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def benchmark_asyncpg_prepared_statements(executions: int):
    """
    Benchmark asyncpg prepared statements.
    
    Args:
        executions: Number of executions
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking asyncpg prepared statements with executions={executions}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("AsyncpgPreparedStatementsBenchmark")
    
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
    
    # Define a function that uses unprepared asyncpg statements
    async def execute_unprepared_asyncpg(conn: asyncpg.Connection, id: int):
        result = await conn.fetch(
            "SELECT * FROM benchmark.prepared_statements_test WHERE id = $1",
            id,
        )
        return result
    
    # Define a function that uses prepared asyncpg statements
    async def execute_prepared_asyncpg(conn: asyncpg.Connection, id: int):
        stmt_name = await prepare_asyncpg_statement(
            conn,
            "SELECT * FROM benchmark.prepared_statements_test WHERE id = $1",
            "benchmark_service",
            "asyncpg_prepared",
        )
        
        result = await fetch_prepared_statement_asyncpg(
            conn,
            stmt_name,
            [id],
            "benchmark_service",
            "asyncpg_prepared",
        )
        
        return result
    
    # Benchmark unprepared asyncpg statements
    async with pool.get_asyncpg_connection() as conn:
        # Warm up
        for _ in range(5):
            await execute_unprepared_asyncpg(conn, 1)
        
        # Benchmark
        await benchmark.benchmark_async(
            name="unprepared_asyncpg",
            category="unprepared_asyncpg",
            func=execute_unprepared_asyncpg,
            args=(conn, 1),
            repeat=executions,
            metadata={
                "executions": executions,
            },
        )
    
    # Benchmark prepared asyncpg statements
    async with pool.get_asyncpg_connection() as conn:
        # Warm up
        for _ in range(5):
            await execute_prepared_asyncpg(conn, 1)
        
        # Benchmark
        await benchmark.benchmark_async(
            name="prepared_asyncpg",
            category="prepared_asyncpg",
            func=execute_prepared_asyncpg,
            args=(conn, 1),
            repeat=executions,
            metadata={
                "executions": executions,
            },
        )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def run_benchmarks():
    """Run all benchmarks."""
    logger.info("Running prepared statements benchmarks...")
    
    # Set up database
    await setup_database()
    
    try:
        # Create combined benchmark
        benchmark = DatabaseBenchmark("PreparedStatementsBenchmarks")
        
        # Benchmark prepared vs. unprepared statements
        for query_complexity in ["simple", "medium", "complex"]:
            for executions in [100, 1000]:
                result = await benchmark_prepared_vs_unprepared(query_complexity, executions)
                benchmark.results.update(result.results)
        
        # Benchmark prepared statement cache
        for cache_size in [10, 50, 100, 200]:
            for unique_queries in [5, 20, 50, 100]:
                for executions in [10, 100]:
                    result = await benchmark_prepared_statement_cache(cache_size, unique_queries, executions)
                    benchmark.results.update(result.results)
        
        # Benchmark asyncpg prepared statements
        for executions in [100, 1000, 10000]:
            result = await benchmark_asyncpg_prepared_statements(executions)
            benchmark.results.update(result.results)
        
        # Save results
        benchmark.save_results()
        benchmark.save_csv()
        
        # Print results
        benchmark.print_results()
        
        # Plot results
        benchmark.plot_results(categories=["prepared", "unprepared"], save_path="benchmarks/database/results/prepared_vs_unprepared.png")
        benchmark.plot_results(categories=["prepared_statement_cache"], save_path="benchmarks/database/results/prepared_statement_cache.png")
        benchmark.plot_results(categories=["prepared_asyncpg", "unprepared_asyncpg"], save_path="benchmarks/database/results/asyncpg_prepared.png")
        
        logger.info("Prepared statements benchmarks complete")
    finally:
        # Clean up database
        await cleanup_database()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepared statements benchmark")
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