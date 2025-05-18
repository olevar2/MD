"""
Benchmark for bulk operations.
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
from sqlalchemy import Table, Column, Integer, String, MetaData, TIMESTAMP, text, Float
from sqlalchemy.ext.asyncio import AsyncSession

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from common_lib.database import (
    get_connection_pool,
    get_sync_db_session,
    get_async_db_session,
    get_asyncpg_connection,
    bulk_insert,
    bulk_insert_async,
    bulk_update,
    bulk_update_async,
    bulk_delete,
    bulk_delete_async,
    bulk_insert_asyncpg,
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

# Define SQLAlchemy tables for testing
metadata = MetaData()

bulk_test_table = Table(
    "bulk_test",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("value", Integer),
    Column("created_at", TIMESTAMP),
    schema="benchmark",
)

bulk_time_series = Table(
    "bulk_time_series",
    metadata,
    Column("time", TIMESTAMP, primary_key=True),
    Column("symbol", String, primary_key=True),
    Column("value", Float),
    schema="benchmark",
)


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
            CREATE TABLE IF NOT EXISTS benchmark.bulk_test (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                value INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        
        # Create benchmark time series table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark.bulk_time_series (
                time TIMESTAMPTZ NOT NULL,
                symbol TEXT NOT NULL,
                value DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (time, symbol)
            )
        """)
        
        # Try to create hypertable (will fail if TimescaleDB is not installed)
        try:
            await conn.execute("""
                SELECT create_hypertable('benchmark.bulk_time_series', 'time', if_not_exists => TRUE)
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
        await conn.execute("TRUNCATE TABLE benchmark.bulk_test")
        await conn.execute("TRUNCATE TABLE benchmark.bulk_time_series")
        
        logger.info("Database cleanup complete")
    finally:
        # Close the connection
        await conn.close()


def generate_test_data(count: int) -> List[Dict[str, Any]]:
    """
    Generate test data for bulk operations.
    
    Args:
        count: Number of records to generate
        
    Returns:
        List of dictionaries with test data
    """
    from datetime import datetime
    
    return [
        {
            "name": f"test_{i}",
            "value": i,
            "created_at": datetime.now(),
        }
        for i in range(count)
    ]


def generate_time_series_data(count: int) -> List[Dict[str, Any]]:
    """
    Generate time series data for bulk operations.
    
    Args:
        count: Number of records to generate
        
    Returns:
        List of dictionaries with time series data
    """
    from datetime import datetime, timedelta
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]
    start_time = datetime.now() - timedelta(days=30)
    
    data = []
    for i in range(count):
        symbol = symbols[i % len(symbols)]
        time = start_time + timedelta(minutes=i)
        value = 1.0 + (i % 1000) / 10000.0
        
        data.append({
            "time": time,
            "symbol": symbol,
            "value": value,
        })
    
    return data


async def benchmark_bulk_insert(row_count: int, chunk_size: int):
    """
    Benchmark bulk insert.
    
    Args:
        row_count: Number of rows to insert
        chunk_size: Size of each chunk
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking bulk insert with row_count={row_count}, chunk_size={chunk_size}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("BulkInsertBenchmark")
    
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
    
    # Generate test data
    test_data = generate_test_data(row_count)
    
    # Define a function that performs a regular insert
    async def regular_insert(session: AsyncSession, data: List[Dict[str, Any]]):
        for item in data:
            await session.execute(
                text("INSERT INTO benchmark.bulk_test (name, value, created_at) VALUES (:name, :value, :created_at)"),
                item,
            )
        await session.commit()
    
    # Define a function that performs a bulk insert
    async def bulk_insert_operation(session: AsyncSession, data: List[Dict[str, Any]], chunk_size: int):
        await bulk_insert_async(
            session,
            bulk_test_table,
            data,
            "benchmark_service",
            return_defaults=False,
            chunk_size=chunk_size,
        )
        await session.commit()
    
    # Benchmark regular insert
    async with pool.get_async_session() as session:
        # Clean up before benchmark
        await session.execute(text("TRUNCATE TABLE benchmark.bulk_test"))
        await session.commit()
        
        # Benchmark
        await benchmark.benchmark_async(
            name=f"regular_insert_{row_count}",
            category="regular_insert",
            func=regular_insert,
            args=(session, test_data),
            repeat=1,
            metadata={
                "row_count": row_count,
            },
        )
    
    # Benchmark bulk insert
    async with pool.get_async_session() as session:
        # Clean up before benchmark
        await session.execute(text("TRUNCATE TABLE benchmark.bulk_test"))
        await session.commit()
        
        # Benchmark
        await benchmark.benchmark_async(
            name=f"bulk_insert_{row_count}_chunk_{chunk_size}",
            category="bulk_insert",
            func=bulk_insert_operation,
            args=(session, test_data, chunk_size),
            repeat=1,
            metadata={
                "row_count": row_count,
                "chunk_size": chunk_size,
            },
        )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def benchmark_bulk_update(row_count: int, chunk_size: int):
    """
    Benchmark bulk update.
    
    Args:
        row_count: Number of rows to update
        chunk_size: Size of each chunk
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking bulk update with row_count={row_count}, chunk_size={chunk_size}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("BulkUpdateBenchmark")
    
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
    
    # Generate test data
    test_data = generate_test_data(row_count)
    
    # Insert test data
    async with pool.get_async_session() as session:
        # Clean up before benchmark
        await session.execute(text("TRUNCATE TABLE benchmark.bulk_test"))
        await session.commit()
        
        # Insert test data
        await bulk_insert_async(
            session,
            bulk_test_table,
            test_data,
            "benchmark_service",
            return_defaults=True,
            chunk_size=chunk_size,
        )
        await session.commit()
    
    # Prepare update data
    async with pool.get_async_session() as session:
        result = await session.execute(text("SELECT * FROM benchmark.bulk_test"))
        rows = result.fetchall()
        
        update_data = []
        for row in rows:
            update_data.append({
                "id": row.id,
                "name": f"updated_{row.name}",
                "value": row.value * 2,
            })
    
    # Define a function that performs a regular update
    async def regular_update(session: AsyncSession, data: List[Dict[str, Any]]):
        for item in data:
            await session.execute(
                text("UPDATE benchmark.bulk_test SET name = :name, value = :value WHERE id = :id"),
                item,
            )
        await session.commit()
    
    # Define a function that performs a bulk update
    async def bulk_update_operation(session: AsyncSession, data: List[Dict[str, Any]], chunk_size: int):
        await bulk_update_async(
            session,
            bulk_test_table,
            data,
            "id",
            "benchmark_service",
            chunk_size=chunk_size,
        )
        await session.commit()
    
    # Benchmark regular update
    async with pool.get_async_session() as session:
        # Benchmark
        await benchmark.benchmark_async(
            name=f"regular_update_{row_count}",
            category="regular_update",
            func=regular_update,
            args=(session, update_data),
            repeat=1,
            metadata={
                "row_count": row_count,
            },
        )
    
    # Benchmark bulk update
    async with pool.get_async_session() as session:
        # Benchmark
        await benchmark.benchmark_async(
            name=f"bulk_update_{row_count}_chunk_{chunk_size}",
            category="bulk_update",
            func=bulk_update_operation,
            args=(session, update_data, chunk_size),
            repeat=1,
            metadata={
                "row_count": row_count,
                "chunk_size": chunk_size,
            },
        )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def benchmark_bulk_delete(row_count: int, chunk_size: int):
    """
    Benchmark bulk delete.
    
    Args:
        row_count: Number of rows to delete
        chunk_size: Size of each chunk
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking bulk delete with row_count={row_count}, chunk_size={chunk_size}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("BulkDeleteBenchmark")
    
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
    
    # Generate test data
    test_data = generate_test_data(row_count)
    
    # Insert test data
    async with pool.get_async_session() as session:
        # Clean up before benchmark
        await session.execute(text("TRUNCATE TABLE benchmark.bulk_test"))
        await session.commit()
        
        # Insert test data
        await bulk_insert_async(
            session,
            bulk_test_table,
            test_data,
            "benchmark_service",
            return_defaults=True,
            chunk_size=chunk_size,
        )
        await session.commit()
    
    # Prepare delete data
    async with pool.get_async_session() as session:
        result = await session.execute(text("SELECT id FROM benchmark.bulk_test"))
        rows = result.fetchall()
        
        delete_ids = [row.id for row in rows]
    
    # Define a function that performs a regular delete
    async def regular_delete(session: AsyncSession, ids: List[int]):
        for id in ids:
            await session.execute(
                text("DELETE FROM benchmark.bulk_test WHERE id = :id"),
                {"id": id},
            )
        await session.commit()
    
    # Define a function that performs a bulk delete
    async def bulk_delete_operation(session: AsyncSession, ids: List[int], chunk_size: int):
        await bulk_delete_async(
            session,
            bulk_test_table,
            ids,
            "id",
            "benchmark_service",
            chunk_size=chunk_size,
        )
        await session.commit()
    
    # Benchmark regular delete
    async with pool.get_async_session() as session:
        # Benchmark
        await benchmark.benchmark_async(
            name=f"regular_delete_{row_count}",
            category="regular_delete",
            func=regular_delete,
            args=(session, delete_ids),
            repeat=1,
            metadata={
                "row_count": row_count,
            },
        )
    
    # Re-insert test data
    async with pool.get_async_session() as session:
        # Clean up before benchmark
        await session.execute(text("TRUNCATE TABLE benchmark.bulk_test"))
        await session.commit()
        
        # Insert test data
        await bulk_insert_async(
            session,
            bulk_test_table,
            test_data,
            "benchmark_service",
            return_defaults=True,
            chunk_size=chunk_size,
        )
        await session.commit()
        
        # Get IDs again
        result = await session.execute(text("SELECT id FROM benchmark.bulk_test"))
        rows = result.fetchall()
        
        delete_ids = [row.id for row in rows]
    
    # Benchmark bulk delete
    async with pool.get_async_session() as session:
        # Benchmark
        await benchmark.benchmark_async(
            name=f"bulk_delete_{row_count}_chunk_{chunk_size}",
            category="bulk_delete",
            func=bulk_delete_operation,
            args=(session, delete_ids, chunk_size),
            repeat=1,
            metadata={
                "row_count": row_count,
                "chunk_size": chunk_size,
            },
        )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def benchmark_bulk_insert_asyncpg(row_count: int, chunk_size: int):
    """
    Benchmark bulk insert with asyncpg.
    
    Args:
        row_count: Number of rows to insert
        chunk_size: Size of each chunk
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking bulk insert with asyncpg with row_count={row_count}, chunk_size={chunk_size}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("BulkInsertAsyncpgBenchmark")
    
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
    
    # Generate test data
    test_data = generate_test_data(row_count)
    
    # Prepare data for asyncpg
    columns = ["name", "value", "created_at"]
    values = [[item[col] for col in columns] for item in test_data]
    
    # Define a function that performs a regular insert with asyncpg
    async def regular_insert_asyncpg(conn: asyncpg.Connection, data: List[Dict[str, Any]]):
        for item in data:
            await conn.execute(
                "INSERT INTO benchmark.bulk_test (name, value, created_at) VALUES ($1, $2, $3)",
                item["name"],
                item["value"],
                item["created_at"],
            )
    
    # Define a function that performs a bulk insert with asyncpg
    async def bulk_insert_asyncpg_operation(conn: asyncpg.Connection, columns: List[str], values: List[List[Any]], chunk_size: int):
        await bulk_insert_asyncpg(
            conn,
            "benchmark.bulk_test",
            columns,
            values,
            "benchmark_service",
            chunk_size=chunk_size,
            return_ids=False,
        )
    
    # Benchmark regular insert with asyncpg
    async with pool.get_asyncpg_connection() as conn:
        # Clean up before benchmark
        await conn.execute("TRUNCATE TABLE benchmark.bulk_test")
        
        # Benchmark
        await benchmark.benchmark_async(
            name=f"regular_insert_asyncpg_{row_count}",
            category="regular_insert_asyncpg",
            func=regular_insert_asyncpg,
            args=(conn, test_data),
            repeat=1,
            metadata={
                "row_count": row_count,
            },
        )
    
    # Benchmark bulk insert with asyncpg
    async with pool.get_asyncpg_connection() as conn:
        # Clean up before benchmark
        await conn.execute("TRUNCATE TABLE benchmark.bulk_test")
        
        # Benchmark
        await benchmark.benchmark_async(
            name=f"bulk_insert_asyncpg_{row_count}_chunk_{chunk_size}",
            category="bulk_insert_asyncpg",
            func=bulk_insert_asyncpg_operation,
            args=(conn, columns, values, chunk_size),
            repeat=1,
            metadata={
                "row_count": row_count,
                "chunk_size": chunk_size,
            },
        )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def benchmark_time_series_bulk_insert(row_count: int, chunk_size: int):
    """
    Benchmark time series bulk insert.
    
    Args:
        row_count: Number of rows to insert
        chunk_size: Size of each chunk
        
    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking time series bulk insert with row_count={row_count}, chunk_size={chunk_size}")
    
    # Create benchmark
    benchmark = DatabaseBenchmark("TimeSeriesBulkInsertBenchmark")
    
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
    
    # Generate time series data
    time_series_data = generate_time_series_data(row_count)
    
    # Define a function that performs a bulk insert for time series
    async def bulk_insert_time_series(session: AsyncSession, data: List[Dict[str, Any]], chunk_size: int):
        await bulk_insert_async(
            session,
            bulk_time_series,
            data,
            "benchmark_service",
            return_defaults=False,
            chunk_size=chunk_size,
        )
        await session.commit()
    
    # Benchmark bulk insert for time series
    async with pool.get_async_session() as session:
        # Clean up before benchmark
        await session.execute(text("TRUNCATE TABLE benchmark.bulk_time_series"))
        await session.commit()
        
        # Benchmark
        await benchmark.benchmark_async(
            name=f"time_series_bulk_insert_{row_count}_chunk_{chunk_size}",
            category="time_series_bulk_insert",
            func=bulk_insert_time_series,
            args=(session, time_series_data, chunk_size),
            repeat=1,
            metadata={
                "row_count": row_count,
                "chunk_size": chunk_size,
            },
        )
    
    # Close the pool
    await pool.close_async()
    
    return benchmark


async def run_benchmarks():
    """Run all benchmarks."""
    logger.info("Running bulk operations benchmarks...")
    
    # Set up database
    await setup_database()
    
    try:
        # Create combined benchmark
        benchmark = DatabaseBenchmark("BulkOperationsBenchmarks")
        
        # Benchmark bulk insert
        for row_count in [100, 1000, 10000]:
            for chunk_size in [100, 500, 1000]:
                result = await benchmark_bulk_insert(row_count, chunk_size)
                benchmark.results.update(result.results)
        
        # Benchmark bulk update
        for row_count in [100, 1000, 10000]:
            for chunk_size in [100, 500, 1000]:
                result = await benchmark_bulk_update(row_count, chunk_size)
                benchmark.results.update(result.results)
        
        # Benchmark bulk delete
        for row_count in [100, 1000, 10000]:
            for chunk_size in [100, 500, 1000]:
                result = await benchmark_bulk_delete(row_count, chunk_size)
                benchmark.results.update(result.results)
        
        # Benchmark bulk insert with asyncpg
        for row_count in [100, 1000, 10000]:
            for chunk_size in [100, 500, 1000]:
                result = await benchmark_bulk_insert_asyncpg(row_count, chunk_size)
                benchmark.results.update(result.results)
        
        # Benchmark time series bulk insert
        for row_count in [1000, 10000, 100000]:
            for chunk_size in [100, 1000, 5000]:
                result = await benchmark_time_series_bulk_insert(row_count, chunk_size)
                benchmark.results.update(result.results)
        
        # Save results
        benchmark.save_results()
        benchmark.save_csv()
        
        # Print results
        benchmark.print_results()
        
        # Plot results
        benchmark.plot_results(categories=["regular_insert", "bulk_insert"], save_path="benchmarks/database/results/bulk_insert.png")
        benchmark.plot_results(categories=["regular_update", "bulk_update"], save_path="benchmarks/database/results/bulk_update.png")
        benchmark.plot_results(categories=["regular_delete", "bulk_delete"], save_path="benchmarks/database/results/bulk_delete.png")
        benchmark.plot_results(categories=["regular_insert_asyncpg", "bulk_insert_asyncpg"], save_path="benchmarks/database/results/bulk_insert_asyncpg.png")
        benchmark.plot_results(categories=["time_series_bulk_insert"], save_path="benchmarks/database/results/time_series_bulk_insert.png")
        
        logger.info("Bulk operations benchmarks complete")
    finally:
        # Clean up database
        await cleanup_database()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Bulk operations benchmark")
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