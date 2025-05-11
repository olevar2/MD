"""
Enhanced Indicator Service Module with caching support.

This is an enhanced version of the indicator service that integrates with
the multi-tiered caching system for improved performance.
"""
from typing import Dict, List, Optional, Union, Any
import logging
import asyncio
from datetime import datetime, timezone
import os

import pandas as pd

from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.indicators.indicator_registry import indicator_registry
from feature_store_service.computation.feature_computation_engine import FeatureComputationEngine
from feature_store_service.models.feature_models import (
    TimeSeriesTransformRequest,
    TimeSeriesTransformResponse,
    FeatureVectorRequest,
    FeatureResponse,
    FeatureMetadataResponse,
    FeatureQuery,
    FeatureVector
)
from feature_store_service.repositories.feature_repository import FeatureRepository
from feature_store_service.caching.cache_manager import CacheManager
from feature_store_service.caching.cache_key import CacheKey
import time # Added for timing cache misses

class EnhancedIndicatorService:
    \
\
\

    Enhanced service for managing technical indicator calculations with caching support.::

    This service integrates directly with the multi-tiered caching system
    to improve performance for repeated indicator calculations.
    \"\"\"

    def __init__(self, config: Dict[str, Any] = None):
        \"\"\"
        Initialize the enhanced indicator service with caching.

        Args:
            config: Optional configuration dictionary for the caching system
        \"\"\"
        self.logger = logging.getLogger(__name__)
        self._registry = indicator_registry # Assumes indicator_registry is globally available or injected
        self._repository = FeatureRepository()
        self._computation_engine = FeatureComputationEngine() # Consider if this is still needed here or just in main.py

        # Default cache configuration
        default_config = {
            'memory_cache_size': 1_000_000_000,  # 1GB
            'memory_cache_ttl': 300,  # 5 minutes
            'use_disk_cache': True,
            'disk_cache_path': os.path.join(os.getcwd(), 'feature_store_cache'),
            'disk_cache_size': 50_000_000_000,  # 50GB
            'disk_cache_ttl': 86400,  # 24 hours
        }

        # Merge with provided config
        if config:
            cache_config = {**default_config, **config}
        else:
            cache_config = default_config

        # Initialize the cache manager directly
        self.cache_manager = CacheManager(cache_config)

        # Remove initialization of CacheAwareIndicatorService
        # self.cache_aware_service = CacheAwareIndicatorService(
        #     cache_manager=self.cache_manager,
        #     indicator_factory=self._registry
        # )

        self.logger.info("Enhanced indicator service initialized with direct caching support")

    async def calculate_indicator_async(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        symbol: str = "generic",
        timeframe: str = "default",
        **indicator_params
    ) -> pd.DataFrame:
        \"\""
        Calculate a single indicator with caching support.

        Args:
            data: DataFrame containing financial data
            indicator_name: Name of the indicator to calculate
            symbol: Symbol for the data
            timeframe: Timeframe for the data
            **indicator_params: Parameters to pass to the indicator constructor

        Returns:
            DataFrame with the indicator values added as new columns
        \"\""
        # --- Merged logic from CacheAwareIndicatorService.calculate_indicator ---
        cache_key = CacheKey(
            indicator_type=indicator_name,
            params=indicator_params,
            symbol=symbol,
            timeframe=timeframe,
            data_hash=data # CacheKey will handle hashing
        )

        # Try to get from cache
        cached_result = await self.cache_manager.get(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for {indicator_name} ({symbol}/{timeframe})")
            return cached_result

        # Cache miss - calculate the indicator
        self.logger.debug(f"Cache miss for {indicator_name} ({symbol}/{timeframe}). Calculating...")
        start_time = time.time()
        try:
            # Use the indicator factory (_registry) to get and calculate
            indicator_instance = self._registry.get_indicator(indicator_name, **indicator_params)
            if not indicator_instance:
                 raise KeyError(f"Indicator '{indicator_name}' not found in registry")

            # Assuming calculate method exists and returns the DataFrame with results
            # This might need adjustment based on BaseIndicator interface
            result_df = indicator_instance.calculate(data.copy()) # Use copy to avoid modifying original data

            calc_time = time.time() - start_time
            self.logger.info(f"Calculated {indicator_name} ({symbol}/{timeframe}) in {calc_time:.4f} seconds")

            # Store the result in cache
            await self.cache_manager.put(cache_key, result_df)

            return result_df

        except KeyError as e:
            self.logger.error(f"Indicator '{indicator_name}' not found in registry: {str(e)}")
            raise
        except Exception as e:
            calc_time = time.time() - start_time
            self.logger.error(f"Error calculating {indicator_name} ({symbol}/{timeframe}) after {calc_time:.4f}s: {str(e)}", exc_info=True)
            # Optionally: Store None or an error marker in cache to prevent retries?
            # await self.cache_manager.put(cache_key, None, ttl=60) # Cache failure for 1 min
            raise
        # --- End of merged logic ---

    def calculate_indicator(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        symbol: str = "generic",
        timeframe: str = "default",
        **indicator_params
    ) -> pd.DataFrame:
        \"\""
        Calculate a single indicator with caching support (synchronous wrapper).

        Args:
            data: DataFrame containing financial data
            indicator_name: Name of the indicator to calculate
            symbol: Symbol for the data
            timeframe: Timeframe for the data
            **indicator_params: Parameters to pass to the indicator constructor

        Returns:
            DataFrame with the indicator values added as new columns
        \"\""
        # Run the async method in the event loop
        # Ensure an event loop is running or create one if necessary
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.calculate_indicator_async(
                data=data,
                indicator_name=indicator_name,
                symbol=symbol,
                timeframe=timeframe,
                **indicator_params
            ))
        else:
             # If a loop is already running (e.g., within FastAPI), use it
             # This might require making the endpoint itself async
             # For simplicity in a sync wrapper, we might block, but it's not ideal in async context
             # Consider if this sync wrapper is truly needed or if callers should be async
             return asyncio.run(self.calculate_indicator_async(
                 data=data,
                 indicator_name=indicator_name,
                 symbol=symbol,
                 timeframe=timeframe,
                 **indicator_params
             ))


    async def calculate_multiple_indicators_async(
        self,
        data: pd.DataFrame,
        indicators: List[Dict[str, Any]],
        symbol: str = "generic",
        timeframe: str = "default"
    ) -> pd.DataFrame:
        \"\""
        Calculate multiple indicators with caching support.

        Args:
            data: DataFrame containing financial data
            indicators: List of indicator configurations, each with:
                - 'name': Name of the indicator
                - 'params': Dictionary of parameters for the indicator
            symbol: Symbol for the data
            timeframe: Timeframe for the data

        Returns:
            DataFrame with all indicator values added as new columns
        \"\""
        # --- Merged logic from CacheAwareIndicatorService.calculate_batch ---
        self.logger.info(f"Calculating batch of {len(indicators)} indicators ({symbol}/{timeframe}) with cache support")
        results_df = data.copy() # Start with original data
        tasks = []
        indicator_configs_to_calculate = [] # Track indicators needing calculation

        # First pass: Check cache for all indicators
        for indicator_config in indicators:
            indicator_type = indicator_config['name']
            params = indicator_config.get('params', {})
            cache_key = CacheKey(
                indicator_type=indicator_type,
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                data_hash=data # Use original data hash for consistency
            )
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for batch indicator {indicator_type} ({symbol}/{timeframe})")
                # Merge cached result - be careful about column overlaps
                # Assuming cached_result contains ONLY the indicator columns
                results_df = pd.merge(results_df, cached_result, left_index=True, right_index=True, how='left', suffixes=('', '_cached'))
                # Handle potential duplicate columns if necessary
                results_df.drop([col for col in results_df.columns if '_cached' in col], axis=1, inplace=True, errors='ignore')
            else:
                self.logger.debug(f"Cache miss for batch indicator {indicator_type} ({symbol}/{timeframe})")
                indicator_configs_to_calculate.append(indicator_config)

        # Second pass: Calculate cache misses (potentially in parallel)
        if not indicator_configs_to_calculate:
            self.logger.info(f"All batch indicators ({symbol}/{timeframe}) served from cache.")
            return results_df

        self.logger.info(f"Calculating {len(indicator_configs_to_calculate)} indicators ({symbol}/{timeframe}) after cache check.")

        # Simple sequential calculation for now, parallelization can be added
        calculation_results = {}
        for indicator_config in indicator_configs_to_calculate:
             indicator_type = indicator_config['name']
             params = indicator_config.get('params', {})
             try:
                 # Reuse single calculation logic (which includes caching the result)
                 # Pass the *original* data to ensure cache key consistency if hashing is used
                 indicator_df = await self.calculate_indicator_async(
                     data=data, # Pass original data
                     indicator_name=indicator_type,
                     symbol=symbol,
                     timeframe=timeframe,
                     **params
                 )
                 # Extract only the newly calculated columns to merge
                 new_cols = [col for col in indicator_df.columns if col not in results_df.columns]
                 calculation_results[indicator_type] = indicator_df[new_cols]

             except Exception as e:
                 self.logger.error(f"Error calculating batch indicator {indicator_type} ({symbol}/{timeframe}): {str(e)}", exc_info=True)
                 # Decide how to handle partial failures - skip, return partial, raise?
                 # For now, we skip failed ones but log the error.

        # Merge calculation results
        for indicator_type, result_df in calculation_results.items():
             results_df = pd.merge(results_df, result_df, left_index=True, right_index=True, how='left')

        return results_df
        # --- End of merged logic ---


    def calculate_multiple_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[Dict[str, Any]],
        symbol: str = "generic",
        timeframe: str = "default"
    ) -> pd.DataFrame:
        \"\""
        Calculate multiple indicators with caching support (synchronous wrapper).

        Args:
            data: DataFrame containing financial data
            indicators: List of indicator configurations
            symbol: Symbol for the data
            timeframe: Timeframe for the data

        Returns:
            DataFrame with all indicator values added as new columns
        \"\""
        # Run the async method in the event loop
        # Ensure an event loop is running or create one if necessary
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.calculate_multiple_indicators_async(
                data=data,
                indicators=indicators,
                symbol=symbol,
                timeframe=timeframe
            ))
        else:
             # See comment in calculate_indicator sync wrapper
             return asyncio.run(self.calculate_multiple_indicators_async(
                 data=data,
                 indicators=indicators,
                 symbol=symbol,
                 timeframe=timeframe
             ))

    def get_available_indicators(self) -> List[str]:
        \"\""
        Get a list of all available indicators.

        Returns:
            List of indicator names registered in the system
        \"\""
        return self._registry.get_available_indicators()

    def get_indicator_metadata(self, indicator_name: str) -> Dict[str, Any]:
        \"\""
        Get metadata for a specific indicator.

        Args:
            indicator_name: Name of the indicator

        Returns:
            Dictionary with indicator metadata
        \"\""
        return self._registry.get_indicator_metadata(indicator_name)

    # --- Cache Management Methods ---
    # These methods now directly call the cache_manager

    async def get_cache_stats_async(self) -> Dict[str, Any]:
         \"\"\"Get statistics about the caching system (async).\"\""
         return await self.cache_manager.get_metrics()

    def get_cache_stats(self) -> Dict[str, Any]:
        \"\""
        Get statistics about the caching system.

        Returns:
            Dictionary with comprehensive cache statistics
        \"\""
        # Run the async method in the event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.get_cache_stats_async())
        else:
            return asyncio.run(self.get_cache_stats_async())


    async def clear_cache_for_symbol_async(self, symbol: str) -> Dict[str, int]:
        \"\"\"Clear all cached data for a specific symbol (async).\"\""
        return await self.cache_manager.invalidate_by_pattern(symbol=symbol)

    def clear_cache_for_symbol(self, symbol: str) -> Dict[str, int]:
        \"\""
        Clear all cached data for a specific symbol.

        Args:
            symbol: Symbol to clear cache for

        Returns:
            Number of cache entries cleared
        \"\""
        # Run the async method in the event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.clear_cache_for_symbol_async(symbol))
        else:
            return asyncio.run(self.clear_cache_for_symbol_async(symbol))


    async def clear_cache_for_indicator_async(self, indicator_type: str) -> Dict[str, int]:
        \"\"\"Clear all cached data for a specific indicator type (async).\"\""
        return await self.cache_manager.invalidate_by_pattern(indicator_type=indicator_type)

    def clear_cache_for_indicator(self, indicator_type: str) -> Dict[str, int]:
        \"\""
        Clear all cached data for a specific indicator type.

        Args:
            indicator_type: Indicator type to clear cache for

        Returns:
            Number of cache entries cleared
        \"\""
        # Run the async method in the event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.clear_cache_for_indicator_async(indicator_type))
        else:
            return asyncio.run(self.clear_cache_for_indicator_async(indicator_type))

    async def clear_all_cache_async(self) -> Dict[str, int]:
         \"\"\"Clear all cache entries (async).\"\""
         return await self.cache_manager.clear()

    def clear_all_cache(self) -> Dict[str, int]:
        \"\"\"Clear all cache entries.\"\""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.clear_all_cache_async())
        else:
            return asyncio.run(self.clear_all_cache_async())


    # --- Other Methods (Keep as is or refactor if needed) ---

    def create_common_indicator_set(
        self,
        data: pd.DataFrame,
        timeframe: str = 'default',
        symbol: str = 'generic'
    ) -> pd.DataFrame:
        \"\""
        Calculate a standard set of common indicators with caching support.

        Args:
            data: DataFrame containing OHLCV data
            timeframe: The timeframe to determine which indicators to use
                       ('short', 'medium', 'long', or 'default')
            symbol: Symbol for the data

        Returns:
            DataFrame with all standard indicators added
        \"\""
        # Define commonly used indicator sets by timeframe
        indicators = []

        # Add standard indicators for all timeframes
        indicators.extend([
            {"name": "SMA", "params": {"window": 20}},
            {"name": "EMA", "params": {"window": 14}},
            {"name": "RSI", "params": {"window": 14}},
            {"name": "MACD", "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}}
        ])

        # Add timeframe-specific indicators
        if timeframe == 'short':
            # Short-term indicators (intraday and day trading)
            indicators.extend([
                {"name": "SMA", "params": {"window": 5}},
                {"name": "SMA", "params": {"window": 10}},
                {"name": "Bollinger", "params": {"window": 10, "num_std": 2}},
                {"name": "ATR", "params": {"window": 7}},
                {"name": "Stochastic", "params": {"k_period": 5, "d_period": 3, "slowing": 3}},
                {"name": "HeikinAshiPatternRecognizer", "params": {"lookback_period": 20, "sensitivity": 0.8}}
            ])
        elif timeframe == 'medium':
            # Medium-term indicators (swing trading)
            indicators.extend([
                {"name": "SMA", "params": {"window": 50}},
                {"name": "Bollinger", "params": {"window": 20, "num_std": 2}},
                {"name": "ATR", "params": {"window": 14}},
                {"name": "ADX", "params": {"window": 14}},
                {"name": "Ichimoku", "params": {}},
                {"name": "IchimokuPatternRecognizer", "params": {"lookback_period": 50, "sensitivity": 0.75}},
                {"name": "RenkoPatternRecognizer", "params": {"lookback_period": 50, "sensitivity": 0.75}}
            ])
        elif timeframe == 'long':
            # Long-term indicators (position trading)
            indicators.extend([
                {"name": "SMA", "params": {"window": 100}},
                {"name": "SMA", "params": {"window": 200}},
                {"name": "Bollinger", "params": {"window": 50, "num_std": 2}},
                {"name": "ROC", "params": {"window": 100}},
                {"name": "ADX", "params": {"window": 25}},
                {"name": "WyckoffPatternRecognizer", "params": {"lookback_period": 100, "sensitivity": 0.7}},
                {"name": "VSAPatternRecognizer", "params": {"lookback_period": 100, "sensitivity": 0.7}}
            ])
        else:  # default
            # Default indicator set (balanced)
            indicators.extend([
                {"name": "SMA", "params": {"window": 50}},
                {"name": "Bollinger", "params": {"window": 20, "num_std": 2}},
                {"name": "ATR", "params": {"window": 14}},
                {"name": "ADX", "params": {"window": 14}},
                {"name": "AdvancedPatternFacade", "params": {"lookback_period": 50, "sensitivity": 0.75}}
            ])

        # Calculate all the indicators using the consolidated method
        # Use the synchronous wrapper for simplicity here, assuming this method is called synchronously
        result = self.calculate_multiple_indicators(
            data=data,
            indicators=indicators,
            symbol=symbol,
            timeframe=timeframe
        )

        self.logger.info(f"Created common indicator set for {timeframe} timeframe with {len(indicators)} indicators")
        return result


    def transform_time_series(self, request: TimeSeriesTransformRequest) -> TimeSeriesTransformResponse:
        \"\""
        Apply transformations to time series data.

        Args:
            request: The time series transformation request

        Returns:
            Response containing the transformed features
        \"\""
        self.logger.info(f"Processing time series transformation: {request.transformation_type}")

        # Delegate to the computation engine
        # Ensure computation engine is initialized correctly if needed here
        # Or remove this method if it belongs solely in the computation engine
        return self._computation_engine.transform_time_series(request)

    def get_feature_vectors(self, request: FeatureVectorRequest) -> FeatureResponse:
        \"\""
        Retrieve feature vectors based on the request parameters.

        Args:
            request: The feature vector request

        Returns:
            FeatureResponse with the requested feature vectors
        \"\""
        try:
            self.logger.info(f"Processing feature vector request for {len(request.feature_ids)} features")

            # Create a feature query from the request
            query = FeatureQuery(
                feature_ids=request.feature_ids,
                start_time=request.start_time,
                end_time=request.end_time,
                limit=request.limit,
                offset=request.offset
            )

            # Retrieve the features from the repository
            # Ensure repository is initialized correctly
            feature_data = self._repository.get_features(query)

            # Organize the features by timestamp
            feature_vectors = {}

            for feature in feature_data:
                timestamp = feature.timestamp
                feature_id = feature.feature_id
                value = feature.value

                if timestamp not in feature_vectors:
                    feature_vectors[timestamp] = {
                        "timestamp": timestamp,
                        "features": {}
                    }

                feature_vectors[timestamp]["features"][feature_id] = value

            # Convert to a sorted list
            result_vectors = list(feature_vectors.values())
            result_vectors.sort(key=lambda x: x["timestamp"])

            return FeatureResponse(
                status="success",
                message=f"Retrieved {len(result_vectors)} feature vectors",
                feature_vectors=result_vectors
            )

        except Exception as e:
            self.logger.error(f"Error retrieving feature vectors: {str(e)}")
            return FeatureResponse(
                status="error",
                message=f"Retrieval error: {str(e)}",
                feature_vectors=[]
            )

    def list_available_features(self) -> List[FeatureMetadataResponse]:
        \"\""
        List all available features in the feature store.

        Returns:
            List of FeatureMetadataResponse objects
        \"\""
        return self._repository.get_all_feature_metadata()

    def get_feature_metadata(self, feature_ids: List[str]) -> List[FeatureMetadataResponse]:
        \"\""
        Retrieve metadata for the specified features.

        Args:
            feature_ids: List of feature IDs to retrieve metadata for

        Returns:
            List of feature metadata objects
        \"\""
        return self._repository.get_feature_metadata_batch(feature_ids)

    def register_custom_indicator(self, name: str, indicator_class: type) -> None:
        \"\""
        Register a custom indicator.

        Args:
            name: Name for the indicator
            indicator_class: Class implementing the indicator
        \"\""
        if not issubclass(indicator_class, BaseIndicator):
            raise TypeError("Custom indicator must inherit from BaseIndicator")

        self._registry.register_indicator(name, indicator_class)
        self.logger.info(f"Registered custom indicator: {name}")
