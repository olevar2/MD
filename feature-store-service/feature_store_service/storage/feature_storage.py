"""
Feature Storage Module.

This module provides functionality for storing and retrieving computed features
from the TimeScaleDB database.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import sqlalchemy as sa
import asyncpg
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import declarative_base

from core_foundations.models.schemas import TimeFrame
from core_foundations.utils.logger import get_logger
from feature_store_service.db.db_core import (
    get_engine, get_db_session, create_asyncpg_pool, check_connection, Base
)
from feature_store_service.storage.query_factory import get_query_optimizer
from feature_store_service.storage.timeseries_optimized_queries import TimeSeriesQueryOptimizer

# Initialize logger
logger = get_logger("feature-store-service.feature-storage")

# Global instance for dependency injection
_feature_storage_instance: Optional['FeatureStorage'] = None


class FeatureStorage:
    """
    Storage for computed features and technical indicators.
    
    This class provides functionality for storing and retrieving computed features
    from the TimeScaleDB database.
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        ohlcv_table: str = "ohlcv",
        feature_table_prefix: str = "feature_",
        data_access_url: Optional[str] = None,
    ):
        """
        Initialize the feature storage.
        
        Args:
            db_url: Database connection URL (if None, uses centralized config)
            ohlcv_table: Name of the table storing OHLCV data
            feature_table_prefix: Prefix for feature tables
            data_access_url: URL for the data access API
        """
        self.db_url = db_url  # Only stored for reference, actual connection managed by db_core
        self.ohlcv_table = ohlcv_table
        self.feature_table_prefix = feature_table_prefix
        self.data_access_url = data_access_url or "http://localhost:8001"
        
        # Later we'll initialize the data access client
        self.data_access_client = None
        
        # For asyncpg connection pool
        self.pool: Optional[asyncpg.Pool] = None
        
        # Time series query optimizer
        self.query_optimizer: Optional[TimeSeriesQueryOptimizer] = None
    
    async def initialize(self) -> None:
        """
        Initialize the storage components.
        
        Creates database engine and ensures required tables exist.
        """
        # Get engine from centralized db_core (should be already initialized)
        self.engine = get_engine()
        
        # Create asyncpg pool for optimized queries
        try:
            self.pool = await create_asyncpg_pool()
            
            # Initialize query optimizer
            self.query_optimizer = await get_query_optimizer(
                db_pool=self.pool,
                config={
                    "cache_enabled": True,
                    "cache_ttl_seconds": 300,
                    "max_cache_items": 1000
                }
            )
            logger.info("Query optimizer initialized")
        except Exception as e:
            logger.error(f"Failed to create asyncpg pool: {str(e)}")
            raise
        
        # Verify the connection works
        try:
            async with get_db_session() as session:
                await session.execute(sa.text("SELECT 1"))
            logger.info("Database connection verified successfully")
        except Exception as e:
            logger.error(f"Failed to verify database connection: {str(e)}")
            raise
            
        # Initialize data access client
        from feature_store_service.computation.feature_computation_engine import DataAccessClient
        self.data_access_client = DataAccessClient(self.data_access_url)
        
        # Ensure feature schema exists
        await self._ensure_schema_exists()
        
        # Set the global instance
        global _feature_storage_instance
        _feature_storage_instance = self
    
    async def _ensure_schema_exists(self) -> None:
        """
        Ensure that the necessary schema and tables exist.
        """
        # Create schema and hypertables if they don't exist
        # This is a simplified version - we'd normally use migrations
        create_schema_sql = """
        CREATE SCHEMA IF NOT EXISTS features;
        
        -- Create the features registry table if it doesn't exist
        CREATE TABLE IF NOT EXISTS features.indicator_registry (
            indicator_id VARCHAR(100) PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            description TEXT,
            category VARCHAR(100),
            parameters JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            async with get_db_session() as session:
                await session.execute(sa.text(create_schema_sql))
                await session.commit()
            logger.info("Feature schema initialized")
        except Exception as e:
            logger.error(f"Failed to create feature schema: {str(e)}")
            raise
    
    async def close(self) -> None:
        """
        Close the database connection.
        """
        # Individual connections are managed by db_core now
        # We just need to close our asyncpg pool if it exists
        if self.pool:
            await self.pool.close()
            logger.info("Feature storage asyncpg pool closed")
    
    def check_connection(self) -> bool:
        """
        Check if the database connection is working.
        
        Returns:
            True if the connection is working, False otherwise
        """
        # Use centralized check_connection function
        return check_connection()
    
    async def register_indicator(
        self,
        indicator_id: str,
        name: str,
        description: str,
        category: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register an indicator in the registry.
        
        Args:
            indicator_id: Unique identifier for the indicator
            name: Human-readable name of the indicator
            description: Description of the indicator
            category: Category of the indicator
            parameters: Optional parameters for the indicator
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            async with get_db_session() as session:
                # Upsert the indicator registration
                upsert_sql = """
                INSERT INTO features.indicator_registry 
                    (indicator_id, name, description, category, parameters)
                VALUES 
                    (:indicator_id, :name, :description, :category, :parameters::jsonb)
                ON CONFLICT (indicator_id) 
                DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    category = EXCLUDED.category,
                    parameters = EXCLUDED.parameters
                """
                
                await session.execute(
                    sa.text(upsert_sql),
                    {
                        "indicator_id": indicator_id,
                        "name": name,
                        "description": description,
                        "category": category,
                        "parameters": parameters or {}
                    }
                )
                await session.commit()
                
            return True
        except Exception as e:
            logger.error(f"Failed to register indicator {indicator_id}: {str(e)}")
            return False
    
    async def _ensure_feature_table_exists(self, indicator_id: str) -> bool:
        """
        Ensure that a table exists for storing indicator data.
        
        Args:
            indicator_id: The ID of the indicator
            
        Returns:
            True if the table exists or was created, False on error
        """
        # Create a hypertable for the indicator if it doesn't exist
        table_name = f"{self.feature_table_prefix}{indicator_id}"
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS features.{table_name} (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            value FLOAT NOT NULL,
            column_name VARCHAR(100) NOT NULL,
            PRIMARY KEY (timestamp, symbol, timeframe, column_name)
        );
        
        -- Make it a TimescaleDB hypertable
        SELECT create_hypertable(
            'features.{table_name}', 
            'timestamp', 
            if_not_exists => TRUE
        );
        
        -- Create index for faster queries
        CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_timeframe 
        ON features.{table_name} (symbol, timeframe);
        
        -- Create index optimized for time-range queries
        CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp_range 
        ON features.{table_name} (timestamp DESC, symbol, timeframe);
        
        -- Create index for column name queries
        CREATE INDEX IF NOT EXISTS idx_{table_name}_column_name 
        ON features.{table_name} (column_name, symbol, timeframe);
        """
        
        try:
            async with get_db_session() as session:
                await session.execute(sa.text(create_table_sql))
                await session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to create feature table for {indicator_id}: {str(e)}")
            return False
    
    async def store_indicator_data(
        self,
        indicator_id: str,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        data: pd.DataFrame
    ) -> bool:
        """
        Store calculated indicator data in the database.
        
        Args:
            indicator_id: ID of the indicator
            symbol: Symbol the data is for
            timeframe: Timeframe of the data
            data: DataFrame with indicator values
            
        Returns:
            True if storage was successful, False otherwise
        """
        if data.empty:
            logger.warning(f"No data to store for indicator {indicator_id}")
            return False
            
        # Ensure the timeframe is a string
        if isinstance(timeframe, TimeFrame):
            timeframe_str = timeframe.value
        else:
            timeframe_str = str(timeframe)
            
        # Make sure the table exists
        if not await self._ensure_feature_table_exists(indicator_id):
            return False
            
        table_name = f"{self.feature_table_prefix}{indicator_id}"
        
        try:
            # Convert the DataFrame to long format for storage
            # This is more flexible for storing multi-column indicators
            records = []
            for column in data.columns:
                for timestamp, value in data[column].dropna().items():
                    if pd.notna(value):  # Only store non-NaN values
                        records.append({
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "timeframe": timeframe_str,
                            "value": float(value),
                            "column_name": column
                        })
            
            if not records:
                logger.warning(f"No valid records to store for indicator {indicator_id}")
                return False
                
            # Use COPY for efficient bulk insert
            copy_sql = f"""
            COPY features.{table_name} (timestamp, symbol, timeframe, value, column_name)
            FROM STDIN
            """
            
            async with self.engine.raw_connection() as conn:
                # Generate CSV data for COPY
                import io
                csv_data = io.StringIO()
                for record in records:
                    ts = record["timestamp"].isoformat() if isinstance(record["timestamp"], datetime) else record["timestamp"]
                    csv_data.write(f"{ts}\t{record['symbol']}\t{record['timeframe']}\t{record['value']}\t{record['column_name']}\n")
                
                csv_data.seek(0)
                
                # Execute COPY
                cursor = await conn.cursor()
                await cursor.copy_expert(copy_sql, csv_data)
                await conn.commit()
                
            # Invalidate cache for this indicator
            if self.query_optimizer:
                self.query_optimizer.invalidate_table_cache(f"features.{table_name}")
                
            logger.info(f"Stored {len(records)} records for indicator {indicator_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store indicator data: {str(e)}")
            return False
    
    async def get_indicator_data(
        self,
        indicator_id: str,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        columns: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve indicator data from the database using optimized queries.
        
        Args:
            indicator_id: ID of the indicator
            symbol: Symbol to retrieve data for
            timeframe: Timeframe to retrieve data for
            start_date: Optional start date
            end_date: Optional end date
            columns: Optional specific columns to retrieve
            use_cache: Whether to use query cache
            
        Returns:
            DataFrame with indicator values
        """
        # Ensure the timeframe is a string
        if isinstance(timeframe, TimeFrame):
            timeframe_str = timeframe.value
        else:
            timeframe_str = str(timeframe)
            
        table_name = f"features.{self.feature_table_prefix}{indicator_id}"
        
        # Default date range if not specified
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()
            
        try:
            # Use the optimized query utility if available
            if self.query_optimizer:
                # Build filters
                filters = {
                    "symbol": symbol,
                    "timeframe": timeframe_str
                }
                
                if columns:
                    filters["column_name"] = columns
                
                # Use the optimized query
                result = await self.query_optimizer.get_time_range_data(
                    table_name=table_name,
                    time_column="timestamp",
                    start_time=start_date,
                    end_time=end_date,
                    filters=filters,
                    use_cache=use_cache
                )
                
                if result.empty:
                    logger.warning(f"No data found for indicator {indicator_id}, symbol {symbol}, timeframe {timeframe_str}")
                    return pd.DataFrame()
                
                # Pivot the result to get the wide format
                pivoted = result.pivot(columns="column_name", values="value")
                return pivoted
            else:
                # Fallback to the original implementation
                logger.warning("Query optimizer not available, using standard query")
                return await self._get_indicator_data_fallback(
                    indicator_id, symbol, timeframe_str, start_date, end_date, columns
                )
        except Exception as e:
            logger.error(f"Failed to retrieve indicator data: {str(e)}")
            # Try the fallback method
            return await self._get_indicator_data_fallback(
                indicator_id, symbol, timeframe_str, start_date, end_date, columns
            )
    
    async def _get_indicator_data_fallback(
        self,
        indicator_id: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fallback method to retrieve indicator data using standard queries.
        
        Args:
            indicator_id: ID of the indicator
            symbol: Symbol to retrieve data for
            timeframe: Timeframe to retrieve data for
            start_date: Start date
            end_date: End date
            columns: Optional specific columns to retrieve
            
        Returns:
            DataFrame with indicator values
        """
        table_name = f"{self.feature_table_prefix}{indicator_id}"
        
        try:
            # Build the query
            query = f"""
            SELECT timestamp, column_name, value
            FROM features.{table_name}
            WHERE symbol = :symbol
            AND timeframe = :timeframe
            AND timestamp >= :start_date
            AND timestamp <= :end_date
            """
            
            if columns:
                query += " AND column_name IN :columns"
                
            query += " ORDER BY timestamp ASC"
            
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date
            }
            
            if columns:
                params["columns"] = tuple(columns)
                
            async with get_db_session() as session:
                result = await session.execute(sa.text(query), params)
                rows = result.fetchall()
                
            if not rows:
                logger.warning(f"No data found for indicator {indicator_id}, symbol {symbol}, timeframe {timeframe}")
                return pd.DataFrame()
                
            # Convert to DataFrame and pivot to wide format
            df = pd.DataFrame(rows, columns=["timestamp", "column_name", "value"])
            pivoted = df.pivot(index="timestamp", columns="column_name", values="value")
            
            return pivoted
        except Exception as e:
            logger.error(f"Failed to retrieve indicator data (fallback): {str(e)}")
            return pd.DataFrame()
    
    async def get_latest_indicator_values(
        self,
        indicator_id: str,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        columns: Optional[List[str]] = None,
        lookback_minutes: int = 60,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get the latest values for an indicator using optimized queries.
        
        Args:
            indicator_id: ID of the indicator
            symbol: Symbol to retrieve data for
            timeframe: Timeframe to retrieve data for
            columns: Optional specific columns to retrieve
            lookback_minutes: How far back to look for latest values
            use_cache: Whether to use query cache
            
        Returns:
            DataFrame with the latest indicator values
        """
        # Ensure the timeframe is a string
        if isinstance(timeframe, TimeFrame):
            timeframe_str = timeframe.value
        else:
            timeframe_str = str(timeframe)
            
        table_name = f"features.{self.feature_table_prefix}{indicator_id}"
        
        try:
            # Use the optimized query utility if available
            if self.query_optimizer:
                # Build filters
                filters = {
                    "symbol": symbol,
                    "timeframe": timeframe_str
                }
                
                if columns:
                    filters["column_name"] = columns
                
                # For latest values, use the optimized latest values query
                result = await self.query_optimizer.get_latest_values(
                    table_name=table_name,
                    time_column="timestamp",
                    group_by_columns=["column_name"],
                    value_columns=["value"],
                    filters=filters,
                    lookback_minutes=lookback_minutes,
                    use_cache=use_cache
                )
                
                if result.empty:
                    logger.warning(f"No latest data found for indicator {indicator_id}")
                    return pd.DataFrame()
                
                # Create a single row DataFrame with the latest values
                latest_values = {}
                for _, row in result.iterrows():
                    column_name = row["column_name"]
                    value = row["value"]
                    latest_values[column_name] = value
                
                # Convert to DataFrame
                return pd.DataFrame([latest_values])
            else:
                # Fallback to standard query
                logger.warning("Query optimizer not available, using standard query for latest values")
                return await self._get_latest_indicator_values_fallback(
                    indicator_id, symbol, timeframe_str, columns, lookback_minutes
                )
        except Exception as e:
            logger.error(f"Failed to retrieve latest indicator values: {str(e)}")
            # Try the fallback method
            return await self._get_latest_indicator_values_fallback(
                indicator_id, symbol, timeframe_str, columns, lookback_minutes
            )
    
    async def _get_latest_indicator_values_fallback(
        self,
        indicator_id: str,
        symbol: str,
        timeframe: str,
        columns: Optional[List[str]] = None,
        lookback_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Fallback method to retrieve latest indicator values using standard queries.
        
        Args:
            indicator_id: ID of the indicator
            symbol: Symbol to retrieve data for
            timeframe: Timeframe to retrieve data for
            columns: Optional specific columns to retrieve
            lookback_minutes: How far back to look for latest values
            
        Returns:
            DataFrame with latest indicator values
        """
        table_name = f"{self.feature_table_prefix}{indicator_id}"
        
        try:
            # Build the query to get latest values for each column_name
            query = f"""
            WITH ranked AS (
                SELECT 
                    timestamp, column_name, value,
                    ROW_NUMBER() OVER (PARTITION BY column_name ORDER BY timestamp DESC) as rn
                FROM 
                    features.{table_name}
                WHERE 
                    symbol = :symbol
                    AND timeframe = :timeframe
                    AND timestamp >= :lookback_time
            )
            SELECT timestamp, column_name, value
            FROM ranked
            WHERE rn = 1
            """
            
            if columns:
                query += " AND column_name IN :columns"
                
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback_time": datetime.utcnow() - timedelta(minutes=lookback_minutes)
            }
            
            if columns:
                params["columns"] = tuple(columns)
                
            async with get_db_session() as session:
                result = await session.execute(sa.text(query), params)
                rows = result.fetchall()
                
            if not rows:
                logger.warning(f"No latest data found for indicator {indicator_id}")
                return pd.DataFrame()
                
            # Convert to a single row DataFrame with the latest values
            latest_values = {}
            for timestamp, column_name, value in rows:
                latest_values[column_name] = value
                
            return pd.DataFrame([latest_values])
        except Exception as e:
            logger.error(f"Failed to retrieve latest indicator values (fallback): {str(e)}")
            return pd.DataFrame()
    
    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for the given symbol and timeframe.
        
        This method gets the data from the data pipeline service, which is the
        authoritative source for OHLCV data.
        
        Args:
            symbol: Symbol to fetch data for
            timeframe: Timeframe to fetch data for
            start_date: Optional start date
            end_date: Optional end date
            use_cache: Whether to use query cache
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.data_access_client is None:
            logger.error("Data access client not initialized")
            return pd.DataFrame()
            
        # Ensure we have datetime objects
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=365)  # Default to 1 year of data
        if end_date is None:
            end_date = datetime.utcnow()
            
        # Ensure the timeframe is a string
        if isinstance(timeframe, TimeFrame):
            timeframe_str = timeframe.value
        else:
            timeframe_str = str(timeframe)
            
        try:
            # Call the data access client
            result = await self.data_access_client.get_ohlcv_data(
                symbols=[symbol],
                timeframes=[timeframe_str],
                from_time=start_date,
                to_time=end_date,
                limit=10000  # Reasonable limit for computation
            )
            
            if symbol in result and timeframe_str in result[symbol]:
                return result[symbol][timeframe_str]
            else:
                logger.warning(f"No data found for {symbol} {timeframe_str}")
                # Try local database as fallback
                return await self._fetch_historical_data_from_db(symbol, timeframe_str, start_date, end_date, use_cache)
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {str(e)}")
            # As a fallback, try to get data directly from the database
            return await self._fetch_historical_data_from_db(symbol, timeframe_str, start_date, end_date, use_cache)
    
    async def _fetch_historical_data_from_db(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fallback method to fetch historical data directly from the database.
        
        Uses the optimized query patterns if available.
        
        Args:
            symbol: Symbol to fetch data for
            timeframe: Timeframe to fetch data for
            start_date: Start date
            end_date: End date
            use_cache: Whether to use query cache
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Use the optimized query utility if available
            if self.query_optimizer:
                # Build filters
                filters = {
                    "symbol": symbol,
                    "timeframe": timeframe
                }
                
                # Use the optimized query
                result = await self.query_optimizer.get_time_range_data(
                    table_name=self.ohlcv_table,
                    time_column="timestamp",
                    start_time=start_date,
                    end_time=end_date,
                    filters=filters,
                    use_cache=use_cache
                )
                
                if not result.empty:
                    return result
                else:
                    logger.warning(f"No historical data found for {symbol} {timeframe}")
                    return pd.DataFrame()
            else:
                # Fallback to standard query
                query = f"""
                SELECT 
                    timestamp, open, high, low, close, volume
                FROM 
                    {self.ohlcv_table}
                WHERE 
                    symbol = :symbol
                    AND timeframe = :timeframe
                    AND timestamp >= :start_date
                    AND timestamp <= :end_date
                ORDER BY 
                    timestamp ASC
                """
                
                async with get_db_session() as session:
                    result = await session.execute(
                        sa.text(query),
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "start_date": start_date,
                            "end_date": end_date
                        }
                    )
                    rows = result.fetchall()
                    
                if not rows:
                    logger.warning(f"No historical data found in database for {symbol} {timeframe}")
                    return pd.DataFrame()
                    
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df.set_index("timestamp", inplace=True)
                
                return df
        except Exception as e:
            logger.error(f"Failed to fetch historical data from database: {str(e)}")
            return pd.DataFrame()


async def get_feature_storage() -> FeatureStorage:
    """
    Get the feature storage instance for dependency injection.
    
    Returns:
        FeatureStorage instance
    """
    global _feature_storage_instance
    
    if _feature_storage_instance is None:
        # Create and initialize a new instance
        storage = FeatureStorage()
        await storage.initialize()
        _feature_storage_instance = storage
        
    return _feature_storage_instance