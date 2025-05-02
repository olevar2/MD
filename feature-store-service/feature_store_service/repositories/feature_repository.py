"""
Feature Repository Module.

Manages storage and retrieval of computed features in TimeScaleDB.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json

import pandas as pd
import numpy as np
from sqlalchemy import Column, String, Float, Table, MetaData, and_, select, func
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB, TIMESTAMP
from sqlalchemy.ext.asyncio import AsyncSession

from core_foundations.utils.logger import get_logger
from feature_store_service.db.db_core import get_db_session, get_engine

logger = get_logger("feature-repository")

# Define metadata
metadata = MetaData()

# Create feature values table
feature_values_table = Table(
    'feature_values',
    metadata,
    Column('symbol', String, nullable=False),
    Column('timeframe', String, nullable=False),
    Column('timestamp', TIMESTAMP(timezone=True), nullable=False),
    Column('feature_id', String, nullable=False),
    Column('value', DOUBLE_PRECISION, nullable=False),
    Column('metadata', JSONB, nullable=True),
)

# Create feature metadata table
feature_metadata_table = Table(
    'feature_metadata',
    metadata,
    Column('feature_id', String, primary_key=True),
    Column('name', String, nullable=False),
    Column('description', String, nullable=True),
    Column('created_at', TIMESTAMP(timezone=True), nullable=False),
    Column('updated_at', TIMESTAMP(timezone=True), nullable=False),
    Column('parameters', JSONB, nullable=True),
    Column('tags', JSONB, nullable=True),
)


class FeatureRepository:
    """
    Repository for storing and retrieving computed features from TimeScaleDB.
    
    This class provides methods for:
    - Storing feature values (outputs from indicators)
    - Retrieving stored features for use in model training or analysis
    - Managing feature metadata
    - Optimized batch operations
    """
    
    def __init__(self):
        """
        Initialize the feature repository using centralized database configuration.
        """
        # No need to create engine or session factory here
        # These are managed by db_core module
        pass
        
    async def initialize(self) -> None:
        """
        Initialize the database schema if it doesn't exist.
        """
        engine = get_engine()
        async with engine.begin() as conn:
            try:
                # Create tables if they don't exist
                await conn.run_sync(metadata.create_all)
                
                # Convert to TimescaleDB hypertable if needed
                await conn.execute(
                    """
                    SELECT create_hypertable('feature_values', 'timestamp', 
                                           if_not_exists => TRUE)
                    """
                )
                
                # Create indexes
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_feature_values_symbol_timeframe 
                    ON feature_values(symbol, timeframe);
                    
                    CREATE INDEX IF NOT EXISTS idx_feature_values_feature_id 
                    ON feature_values(feature_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_feature_values_timestamp 
                    ON feature_values(timestamp);
                    """
                )
                
                logger.info("Feature repository tables and indexes initialized")
            except Exception as e:
                logger.error(f"Error initializing feature repository: {e}")
                raise
    
    async def store_feature_values(
        self,
        feature_id: str,
        symbol: str,
        timeframe: str,
        values: List[Dict[str, Any]]
    ) -> int:
        """
        Store feature values in the database.
        
        Args:
            feature_id: Identifier for the feature
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe string (e.g., '1h')
            values: List of dictionaries with 'timestamp', 'value', and optional 'metadata'
            
        Returns:
            Number of values stored
        """
        if not values:
            return 0
            
        # Prepare values for insertion
        rows = [
            {
                'feature_id': feature_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': v['timestamp'],
                'value': v['value'],
                'metadata': v.get('metadata', None)
            }
            for v in values
        ]
        
        async with get_db_session() as session:
            try:
                # Use parameterized query for safety and efficiency
                await session.execute(
                    feature_values_table.insert().values(rows)
                )
                await session.commit()
                logger.debug(f"Stored {len(rows)} values for feature {feature_id}")
                return len(rows)
            except Exception as e:
                await session.rollback()
                logger.error(f"Error storing feature values: {e}")
                raise
    
    async def get_feature_values(
        self,
        feature_id: str,
        symbol: str,
        timeframe: str,
        from_time: datetime,
        to_time: datetime
    ) -> pd.DataFrame:
        """
        Retrieve stored feature values.
        
        Args:
            feature_id: Feature identifier
            symbol: Trading symbol
            timeframe: Data timeframe
            from_time: Start time for retrieval
            to_time: End time for retrieval
            
        Returns:
            DataFrame with feature values indexed by timestamp
        """
        try:
            async with get_db_session() as session:
                # Query feature values
                query = (
                    select(
                        feature_values_table.c.timestamp,
                        feature_values_table.c.feature_id,
                        feature_values_table.c.value
                    )
                    .where(
                        and_(
                            feature_values_table.c.symbol == symbol,
                            feature_values_table.c.timeframe == timeframe,
                            feature_values_table.c.feature_id.like(f"{feature_id}%"),
                            feature_values_table.c.timestamp >= from_time,
                            feature_values_table.c.timestamp <= to_time
                        )
                    )
                    .order_by(feature_values_table.c.timestamp)
                )
                
                result = await session.execute(query)
                rows = result.fetchall()
                
                if not rows:
                    logger.warning(f"No feature values found for {feature_id}:{symbol}:{timeframe}")
                    return pd.DataFrame()
                    
                # Convert to DataFrame
                data = []
                for timestamp, feature_id, value in rows:
                    # Extract the column name from the feature_id
                    if "_" in feature_id:
                        column = feature_id.split("_", 1)[1]
                    else:
                        column = feature_id
                        
                    data.append({
                        'timestamp': timestamp,
                        'column': column,
                        'value': value
                    })
                    
                df = pd.DataFrame(data)
                
                # Pivot to get each column as a separate column in the DataFrame
                pivot_df = df.pivot(index='timestamp', columns='column', values='value')
                
                return pivot_df
                
        except Exception as e:
            logger.error(f"Error retrieving feature values: {str(e)}")
            return pd.DataFrame()
            
    async def store_feature_metadata(
        self,
        feature_id: str,
        name: str,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Store metadata about a feature.
        
        Args:
            feature_id: Feature identifier
            name: Feature name
            description: Feature description
            parameters: Feature parameters
            tags: Tags for categorizing features
            
        Returns:
            True if successful, False otherwise
        """
        now = datetime.utcnow()
        
        try:
            async with get_db_session() as session:
                # Check if the feature metadata already exists
                query = select(feature_metadata_table).where(
                    feature_metadata_table.c.feature_id == feature_id
                )
                result = await session.execute(query)
                existing = result.fetchone()
                
                if existing:
                    # Update existing metadata
                    stmt = (
                        feature_metadata_table.update()
                        .where(feature_metadata_table.c.feature_id == feature_id)
                        .values(
                            name=name,
                            description=description,
                            updated_at=now,
                            parameters=parameters,
                            tags=tags
                        )
                    )
                else:
                    # Insert new metadata
                    stmt = feature_metadata_table.insert().values(
                        feature_id=feature_id,
                        name=name,
                        description=description,
                        created_at=now,
                        updated_at=now,
                        parameters=parameters,
                        tags=tags
                    )
                    
                await session.execute(stmt)
                await session.commit()
                
                logger.info(f"Stored metadata for feature {feature_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing feature metadata: {str(e)}")
            return False
            
    async def get_feature_metadata(
        self,
        feature_id: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get metadata about features.
        
        Args:
            feature_id: Specific feature ID to retrieve, or None for all
            tags: Filter features by tags
            
        Returns:
            List of feature metadata dictionaries
        """
        try:
            async with get_db_session() as session:
                query = select(feature_metadata_table)
                
                # Apply filters if specified
                if feature_id:
                    query = query.where(feature_metadata_table.c.feature_id == feature_id)
                    
                if tags:
                    # Filter by tags using PostgreSQL JSONB containment
                    for tag in tags:
                        query = query.where(
                            feature_metadata_table.c.tags.contains([tag])
                        )
                        
                result = await session.execute(query)
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                metadata_list = []
                for row in rows:
                    metadata_dict = {
                        'feature_id': row.feature_id,
                        'name': row.name,
                        'description': row.description,
                        'created_at': row.created_at,
                        'updated_at': row.updated_at,
                        'parameters': row.parameters,
                        'tags': row.tags
                    }
                    metadata_list.append(metadata_dict)
                    
                return metadata_list
                
        except Exception as e:
            logger.error(f"Error retrieving feature metadata: {str(e)}")
            return []
            
    async def delete_feature_values(
        self,
        feature_id: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        older_than: Optional[datetime] = None
    ) -> int:
        """
        Delete feature values based on criteria.
        
        Args:
            feature_id: Feature identifier
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            older_than: Optional timestamp to delete values older than
            
        Returns:
            Number of deleted rows
        """
        try:
            async with get_db_session() as session:
                # Build conditions
                conditions = [feature_values_table.c.feature_id.like(f"{feature_id}%")]
                
                if symbol:
                    conditions.append(feature_values_table.c.symbol == symbol)
                    
                if timeframe:
                    conditions.append(feature_values_table.c.timeframe == timeframe)
                    
                if older_than:
                    conditions.append(feature_values_table.c.timestamp < older_than)
                    
                # Execute delete
                stmt = feature_values_table.delete().where(and_(*conditions))
                result = await session.execute(stmt)
                await session.commit()
                
                row_count = result.rowcount
                logger.info(f"Deleted {row_count} feature values for {feature_id}")
                return row_count
                
        except Exception as e:
            logger.error(f"Error deleting feature values: {str(e)}")
            return 0

    async def close(self) -> None:
        """Close the database engine."""
        engine = get_engine()
        await engine.dispose()