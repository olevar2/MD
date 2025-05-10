#!/usr/bin/env python
"""
Initialize the database schema.

This script creates the necessary database schema for the Historical Data Management service.
"""

import asyncio
import logging
import os
from typing import Optional

import asyncpg
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def init_db(db_url: Optional[str] = None) -> None:
    """
    Initialize the database schema.
    
    Args:
        db_url: Database connection URL
    """
    # Get database URL from environment variable or use default
    if db_url is None:
        db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/forex_platform"
        )
    
    logger.info(f"Initializing database: {db_url}")
    
    # Create engine
    engine = create_async_engine(db_url)
    
    # Create schema
    async with engine.begin() as conn:
        # Create schema if it doesn't exist
        await conn.execute(sa.text("CREATE SCHEMA IF NOT EXISTS historical_data"))
        
        # Check if TimescaleDB extension is available
        result = await conn.execute(sa.text(
            "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'"
        ))
        has_timescale = result.scalar() > 0
        
        if has_timescale:
            logger.info("TimescaleDB extension is available")
            
            # Create TimescaleDB extension if not already created
            await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
        else:
            logger.warning("TimescaleDB extension is not available, using regular tables")
    
    # Create metadata
    metadata = sa.MetaData(schema="historical_data")
    
    # Define tables
    ohlcv_table = sa.Table(
        "ohlcv_data",
        metadata,
        sa.Column("record_id", sa.String, primary_key=True),
        sa.Column("symbol", sa.String, nullable=False, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("timeframe", sa.String, nullable=False, index=True),
        sa.Column("source_type", sa.String, nullable=False),
        sa.Column("source_id", sa.String, nullable=False),
        sa.Column("data", sa.JSON, nullable=False),
        sa.Column("version", sa.Integer, nullable=False, default=1),
        sa.Column("is_correction", sa.Boolean, nullable=False, default=False),
        sa.Column("correction_of", sa.String, nullable=True),
        sa.Column("correction_type", sa.String, nullable=True),
        sa.Column("correction_reason", sa.String, nullable=True),
        sa.Column("metadata", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String, nullable=True),
    )
    
    tick_table = sa.Table(
        "tick_data",
        metadata,
        sa.Column("record_id", sa.String, primary_key=True),
        sa.Column("symbol", sa.String, nullable=False, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("source_type", sa.String, nullable=False),
        sa.Column("source_id", sa.String, nullable=False),
        sa.Column("data", sa.JSON, nullable=False),
        sa.Column("version", sa.Integer, nullable=False, default=1),
        sa.Column("is_correction", sa.Boolean, nullable=False, default=False),
        sa.Column("correction_of", sa.String, nullable=True),
        sa.Column("correction_type", sa.String, nullable=True),
        sa.Column("correction_reason", sa.String, nullable=True),
        sa.Column("metadata", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String, nullable=True),
    )
    
    alternative_table = sa.Table(
        "alternative_data",
        metadata,
        sa.Column("record_id", sa.String, primary_key=True),
        sa.Column("symbol", sa.String, nullable=False, index=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("source_type", sa.String, nullable=False),
        sa.Column("source_id", sa.String, nullable=False),
        sa.Column("data_type", sa.String, nullable=False, index=True),
        sa.Column("data", sa.JSON, nullable=False),
        sa.Column("version", sa.Integer, nullable=False, default=1),
        sa.Column("is_correction", sa.Boolean, nullable=False, default=False),
        sa.Column("correction_of", sa.String, nullable=True),
        sa.Column("correction_type", sa.String, nullable=True),
        sa.Column("correction_reason", sa.String, nullable=True),
        sa.Column("metadata", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String, nullable=True),
    )
    
    corrections_table = sa.Table(
        "data_corrections",
        metadata,
        sa.Column("correction_id", sa.String, primary_key=True),
        sa.Column("original_record_id", sa.String, nullable=False, index=True),
        sa.Column("corrected_record_id", sa.String, nullable=False, index=True),
        sa.Column("correction_type", sa.String, nullable=False),
        sa.Column("correction_reason", sa.String, nullable=False),
        sa.Column("correction_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("corrected_by", sa.String, nullable=False),
        sa.Column("approved_by", sa.String, nullable=True),
        sa.Column("approval_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON, nullable=True),
    )
    
    quality_reports_table = sa.Table(
        "data_quality_reports",
        metadata,
        sa.Column("report_id", sa.String, primary_key=True),
        sa.Column("symbol", sa.String, nullable=False, index=True),
        sa.Column("source_type", sa.String, nullable=False),
        sa.Column("timeframe", sa.String, nullable=True),
        sa.Column("start_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_records", sa.Integer, nullable=False),
        sa.Column("missing_records", sa.Integer, nullable=False),
        sa.Column("corrected_records", sa.Integer, nullable=False),
        sa.Column("quality_issues", sa.JSON, nullable=False),
        sa.Column("report_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", sa.JSON, nullable=True),
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        
        # Convert tables to TimescaleDB hypertables if TimescaleDB is available
        if has_timescale:
            for table_name in ["ohlcv_data", "tick_data", "alternative_data"]:
                await conn.execute(sa.text(
                    f"SELECT create_hypertable('historical_data.{table_name}', 'timestamp', "
                    f"if_not_exists => TRUE)"
                ))
            logger.info("Tables converted to TimescaleDB hypertables")
    
    logger.info("Database initialization complete")


if __name__ == "__main__":
    asyncio.run(init_db())
