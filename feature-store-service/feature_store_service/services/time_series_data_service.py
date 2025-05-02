"""
Time Series Data Service

This module provides high-performance data retrieval services for time series data,
leveraging the TimeSeriesQueryOptimizer for efficient database access.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
from functools import lru_cache

from feature_store_service.storage.time_series_query_optimizer import (
    TimeSeriesQueryOptimizer, TimeSeriesQueryContext
)
from feature_store_service.db.connection import get_db_connection
from feature_store_service.models.time_series import TimeSeriesData

class TimeSeriesDataService:
    """
    Service for efficient retrieval and management of time series data,
    with optimized query handling and caching.
    """
    
    def __init__(self, max_cache_size: int = 200, default_ttl: int = 600):
        """
        Initialize the time series data service
        
        Args:
            max_cache_size: Maximum number of query results to cache
            default_ttl: Default time-to-live for cached results in seconds
        """
        self.logger = logging.getLogger(__name__)
        self.optimizer = TimeSeriesQueryOptimizer(
            max_cache_size=max_cache_size,
            default_ttl=default_ttl
        )
        
    async def get_price_data(
        self, 
        symbol: str,
        timeframe: str,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str],
        include_current_candle: bool = False
    ) -> pd.DataFrame:
        """
        Get historical price data with optimization for repeated queries
        
        Args:
            symbol: Trading instrument symbol
            timeframe: Chart timeframe (e.g., '1m', '5m', '1h', '1d')
            start_time: Start of the time range
            end_time: End of the time range
            include_current_candle: Whether to include the current (incomplete) candle
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert string dates to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        # Prepare query parameters
        query_params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_time": start_time,
            "end_time": end_time,
            "include_current_candle": include_current_candle
        }
        
        # Use the query context for optimization
        with TimeSeriesQueryContext(self.optimizer, query_params) as (cache_hit, result, optimized_params):
            if cache_hit:
                self.logger.debug(f"Cache hit for {symbol} {timeframe} data")
                return result
            
            # No cache hit, perform database query with optimized parameters
            db = get_db_connection()
            
            # Convert optimized parameters for database query
            db_query_params = self._prepare_db_query_params(optimized_params)
            
            # Execute the database query
            data = await self._execute_db_query(db, db_query_params)
            
            # Cache the result
            self.optimizer.cache_result(optimized_params, data)
            
            return data
            
    async def get_indicator_data(
        self,
        symbol: str,
        indicator_name: str,
        parameters: Dict[str, Any],
        timeframe: str,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str]
    ) -> pd.DataFrame:
        """
        Get technical indicator data with optimization for repeated queries
        
        Args:
            symbol: Trading instrument symbol
            indicator_name: Name of the technical indicator
            parameters: Dictionary of indicator parameters
            timeframe: Chart timeframe
            start_time: Start of the time range
            end_time: End of the time range
            
        Returns:
            DataFrame with indicator data
        """
        # Convert string dates to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        # Prepare query parameters
        query_params = {
            "symbol": symbol,
            "indicator_name": indicator_name,
            "indicator_params": parameters,
            "timeframe": timeframe,
            "start_time": start_time,
            "end_time": end_time
        }
        
        # Use the query context for optimization
        with TimeSeriesQueryContext(self.optimizer, query_params) as (cache_hit, result, optimized_params):
            if cache_hit:
                self.logger.debug(f"Cache hit for {indicator_name} on {symbol} {timeframe}")
                return result
            
            # No cache hit, perform database query with optimized parameters
            db = get_db_connection()
            
            # Convert optimized parameters for database query
            db_query_params = self._prepare_db_query_params(optimized_params)
            
            # Execute the database query for indicator data
            data = await self._execute_indicator_query(db, db_query_params)
            
            # If the indicator doesn't exist in database, calculate it
            if data.empty:
                # Get price data for calculation
                price_data = await self.get_price_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=optimized_params["start_time"],
                    end_time=optimized_params["end_time"]
                )
                
                # Calculate indicator
                data = await self._calculate_indicator(
                    price_data=price_data,
                    indicator_name=indicator_name,
                    parameters=parameters
                )
                
                # Store in database for future use (background task)
                self._schedule_indicator_storage(
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator_name=indicator_name,
                    parameters=parameters,
                    data=data
                )
            
            # Cache the result
            self.optimizer.cache_result(optimized_params, data)
            
            return data
            
    async def get_multi_symbol_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_time: Union[datetime, str],
        end_time: Union[datetime, str],
        align_timestamps: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get price data for multiple symbols efficiently
        
        Args:
            symbols: List of trading instrument symbols
            timeframe: Chart timeframe
            start_time: Start of the time range
            end_time: End of the time range
            align_timestamps: Whether to align timestamps across all symbols
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        results = {}
        
        # If we have a lot of symbols, use parallel partitioning
        if len(symbols) > 5:
            # Create partitioning for efficient parallel retrieval
            symbol_partitions = [symbols[i:i+5] for i in range(0, len(symbols), 5)]
            
            for partition in symbol_partitions:
                # Process each partition
                for symbol in partition:
                    data = await self.get_price_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=start_time,
                        end_time=end_time
                    )
                    results[symbol] = data
        else:
            # For a small number of symbols, process sequentially
            for symbol in symbols:
                data = await self.get_price_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
                results[symbol] = data
        
        # Align timestamps if requested
        if align_timestamps and results:
            results = self._align_multi_symbol_data(results)
            
        return results
    
    def invalidate_cache(
        self, 
        symbol: Optional[str] = None,
        indicator_type: Optional[str] = None,
        timeframe: Optional[str] = None
    ):
        """
        Invalidate cache entries for specific symbols, indicators or timeframes
        
        Args:
            symbol: Symbol to invalidate cache for
            indicator_type: Indicator type to invalidate cache for
            timeframe: Timeframe to invalidate cache for
        """
        self.optimizer.invalidate_cache(
            symbol=symbol,
            indicator_type=indicator_type
        )
        self.logger.info(f"Invalidated cache for {symbol or 'all symbols'}, "
                        f"{indicator_type or 'all indicators'}, "
                        f"{timeframe or 'all timeframes'}")
    
    def get_optimizer_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the query optimizer performance
        
        Returns:
            Dictionary with optimizer statistics
        """
        return self.optimizer.get_query_statistics()
    
    async def _execute_db_query(self, db, query_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute a database query for price data
        
        Args:
            db: Database connection
            query_params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        try:
            # Build the SQL query
            query = """
                SELECT timestamp, open, high, low, close, volume 
                FROM price_data 
                WHERE symbol = :symbol 
                AND timeframe = :timeframe 
                AND timestamp >= :start_time 
                AND timestamp <= :end_time
                ORDER BY timestamp
            """
            
            # Add index hints if available
            if query_params.get("time_index"):
                query = query.replace("FROM price_data", "FROM price_data USING INDEX time_idx")
                
            if query_params.get("symbol_index"):
                query = query.replace("FROM price_data", "FROM price_data USING INDEX symbol_idx")
            
            # Execute the query
            result = await db.fetch_all(query, query_params)
            
            # Convert to DataFrame
            if result:
                data = pd.DataFrame(result)
                data.set_index('timestamp', inplace=True)
                return data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Database query error: {str(e)}")
            return pd.DataFrame()
    
    async def _execute_indicator_query(self, db, query_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute a database query for indicator data
        
        Args:
            db: Database connection
            query_params: Query parameters
            
        Returns:
            DataFrame with indicator data
        """
        try:
            # Build the SQL query
            query = """
                SELECT timestamp, value, additional_values
                FROM indicator_data 
                WHERE symbol = :symbol 
                AND indicator_name = :indicator_name
                AND parameters = :indicator_params_json
                AND timeframe = :timeframe 
                AND timestamp >= :start_time 
                AND timestamp <= :end_time
                ORDER BY timestamp
            """
            
            # Serialize parameters to JSON for storage comparison
            import json
            query_params["indicator_params_json"] = json.dumps(
                query_params["indicator_params"], 
                sort_keys=True
            )
            
            # Execute the query
            result = await db.fetch_all(query, query_params)
            
            # Convert to DataFrame
            if result:
                data = pd.DataFrame(result)
                
                # Parse additional_values JSON column if it exists
                if 'additional_values' in data.columns:
                    # Expand additional values columns
                    additional_df = pd.json_normalize(
                        data['additional_values'].apply(json.loads)
                    )
                    
                    # Merge with main dataframe
                    data = pd.concat([data.drop('additional_values', axis=1), additional_df], axis=1)
                    
                data.set_index('timestamp', inplace=True)
                return data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Indicator query error: {str(e)}")
            return pd.DataFrame()
    
    async def _calculate_indicator(
        self,
        price_data: pd.DataFrame,
        indicator_name: str,
        parameters: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calculate a technical indicator from price data
        
        Args:
            price_data: OHLCV price data
            indicator_name: Name of the indicator to calculate
            parameters: Dictionary of indicator parameters
            
        Returns:
            DataFrame with indicator values
        """
        try:
            # Import the indicator calculation module dynamically
            from feature_store_service.indicators.factory import create_indicator
            
            # Create and calculate the indicator
            indicator = create_indicator(indicator_name, parameters)
            indicator_data = indicator.calculate(price_data)
            
            return indicator_data
            
        except Exception as e:
            self.logger.error(f"Error calculating {indicator_name}: {str(e)}")
            return pd.DataFrame()
    
    def _prepare_db_query_params(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare query parameters for database queries
        
        Args:
            query_params: Original query parameters
            
        Returns:
            Prepared parameters for database query
        """
        # Make a copy to avoid modifying the original
        db_params = query_params.copy()
        
        # Ensure datetime objects are correctly formatted for DB
        if isinstance(db_params.get('start_time'), datetime):
            db_params['start_time'] = db_params['start_time'].isoformat()
            
        if isinstance(db_params.get('end_time'), datetime):
            db_params['end_time'] = db_params['end_time'].isoformat()
            
        return db_params
    
    def _schedule_indicator_storage(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        parameters: Dict[str, Any],
        data: pd.DataFrame
    ):
        """
        Schedule background task to store calculated indicator values in database
        
        Args:
            symbol: Trading instrument symbol
            timeframe: Chart timeframe
            indicator_name: Name of the indicator
            parameters: Dictionary of indicator parameters
            data: DataFrame with indicator values
        """
        try:
            from feature_store_service.tasks.storage_tasks import store_indicator_values
            
            # Convert DataFrame to format suitable for storage task
            records = data.reset_index().to_dict('records')
            
            # Create a background task
            store_indicator_values.delay(
                symbol=symbol,
                timeframe=timeframe,
                indicator_name=indicator_name,
                parameters=parameters,
                values=records
            )
            
        except Exception as e:
            self.logger.error(f"Error scheduling indicator storage: {str(e)}")
    
    def _align_multi_symbol_data(
        self, 
        symbol_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Align timestamp indexes across multiple symbol DataFrames
        
        Args:
            symbol_data: Dictionary mapping symbols to their DataFrames
            
        Returns:
            Dictionary with aligned DataFrames
        """
        if not symbol_data:
            return {}
            
        # Get unique timestamps across all DataFrames
        all_timestamps = pd.Index([])
        for df in symbol_data.values():
            if not df.empty:
                all_timestamps = all_timestamps.union(df.index)
        
        # Reindex all DataFrames to the same timestamp index
        aligned_data = {}
        for symbol, df in symbol_data.items():
            if df.empty:
                aligned_data[symbol] = df
            else:
                # Reindex and forward fill missing values
                aligned_df = df.reindex(all_timestamps).ffill()
                aligned_data[symbol] = aligned_df
        
        return aligned_data
