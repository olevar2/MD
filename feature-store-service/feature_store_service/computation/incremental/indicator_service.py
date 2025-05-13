"""
Incremental Indicator Service.

This service manages incremental indicator calculations, providing efficient
real-time updates and state management for low-latency applications.
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
from core_foundations.utils.logger import get_logger
from feature_store_service.computation.incremental.base_incremental import IncrementalIndicator, IncrementalIndicatorFactory
from feature_store_service.storage.feature_storage import FeatureStorage
from data_pipeline_service.services.ohlcv_service import OHLCVService
logger = get_logger('feature-store-service.incremental-indicator-service')


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class IncrementalIndicatorService:
    """
    Service for managing incremental indicator calculations.
    
    This service maintains stateful indicators that can be efficiently updated
    with new data points, optimized for low-latency applications like real-time
    trading systems.
    """

    def __init__(self, feature_storage: FeatureStorage, ohlcv_service:
        OHLCVService, state_persistence_path: Optional[str]=None):
        """
        Initialize the incremental indicator service.
        
        Args:
            feature_storage: Storage for computed indicators
            ohlcv_service: Service for retrieving OHLCV data
            state_persistence_path: Path for persisting indicator states (optional)
        """
        self.feature_storage = feature_storage
        self.ohlcv_service = ohlcv_service
        self.state_persistence_path = state_persistence_path
        self.active_indicators: Dict[str, IncrementalIndicator] = {}

    @async_with_exception_handling
    async def get_or_initialize_indicator(self, symbol: str, timeframe: str,
        indicator_type: str, params: Dict[str, Any]=None, lookback_days: int=60
        ) ->Optional[IncrementalIndicator]:
        """
        Get an existing indicator instance or initialize a new one.
        
        Args:
            symbol: Symbol for the indicator
            timeframe: Timeframe for the indicator
            indicator_type: Type of indicator (e.g., "SMA", "EMA")
            params: Parameters for the indicator
            lookback_days: Number of days of history to use for initialization
            
        Returns:
            IncrementalIndicator instance if successful, None otherwise
        """
        indicator_key = self._generate_indicator_key(symbol, timeframe,
            indicator_type, params)
        if indicator_key in self.active_indicators:
            return self.active_indicators[indicator_key]
        indicator = IncrementalIndicatorFactory.create_indicator(indicator_type
            , **params or {})
        if not indicator:
            logger.error(f'Failed to create indicator of type {indicator_type}'
                )
            return None
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        try:
            data = await self.ohlcv_service.get_ohlcv_data(symbol=symbol,
                timeframe=timeframe, start_date=start_date, end_date=end_date)
            if data.empty:
                logger.warning(
                    f'No historical data available for {symbol} {timeframe}')
                return None
            indicator.initialize(data)
            if indicator.is_initialized:
                self.active_indicators[indicator_key] = indicator
                logger.info(
                    f'Initialized {indicator_type} for {symbol} {timeframe}')
                return indicator
            else:
                logger.error(
                    f'Failed to initialize {indicator_type} for {symbol} {timeframe}'
                    )
                return None
        except Exception as e:
            logger.error(f'Error initializing indicator: {str(e)}')
            return None

    @async_with_exception_handling
    async def update_indicator(self, symbol: str, timeframe: str,
        indicator_type: str, new_data_point: Dict[str, Union[float,
        datetime]], params: Dict[str, Any]=None) ->Dict[str, float]:
        """
        Update an indicator with a new data point.
        
        Args:
            symbol: Symbol for the indicator
            timeframe: Timeframe for the indicator
            indicator_type: Type of indicator (e.g., "SMA", "EMA")
            new_data_point: New data point for updating the indicator
            params: Parameters for the indicator
            
        Returns:
            Dictionary with the updated indicator values
        """
        indicator_key = self._generate_indicator_key(symbol, timeframe,
            indicator_type, params)
        indicator = self.active_indicators.get(indicator_key)
        if not indicator:
            indicator = await self.get_or_initialize_indicator(symbol,
                timeframe, indicator_type, params)
        if not indicator:
            logger.error(
                f'Failed to get/initialize indicator for update: {indicator_key}'
                )
            return {}
        try:
            result = indicator.update(new_data_point)
            if self.state_persistence_path:
                self._save_indicator_state(indicator_key, indicator)
            return result
        except Exception as e:
            logger.error(f'Error updating indicator {indicator_key}: {str(e)}')
            return {}

    @async_with_exception_handling
    async def update_all_indicators_for_symbol(self, symbol: str, timeframe:
        str, new_data_point: Dict[str, Union[float, datetime]]) ->Dict[str,
        Dict[str, float]]:
        """
        Update all active indicators for a specific symbol and timeframe.
        
        Args:
            symbol: Symbol to update indicators for
            timeframe: Timeframe to update indicators for
            new_data_point: New data point for updating the indicators
            
        Returns:
            Dictionary mapping indicator keys to their updated values
        """
        results = {}
        prefix = f'{symbol}_{timeframe}_'
        relevant_keys = [k for k in self.active_indicators.keys() if k.
            startswith(prefix)]
        for key in relevant_keys:
            indicator = self.active_indicators[key]
            try:
                updated_values = indicator.update(new_data_point)
                parts = key.split('_')
                if len(parts) > 2:
                    indicator_type = parts[2]
                    results[indicator_type] = updated_values
            except Exception as e:
                logger.error(f'Error updating indicator {key}: {str(e)}')
        return results

    def _generate_indicator_key(self, symbol: str, timeframe: str,
        indicator_type: str, params: Optional[Dict[str, Any]]) ->str:
        """
        Generate a unique key for an indicator configuration.
        
        Args:
            symbol: Symbol for the indicator
            timeframe: Timeframe for the indicator
            indicator_type: Type of indicator
            params: Parameters for the indicator
            
        Returns:
            Unique key string
        """
        params_str = ''
        if params:
            sorted_params = {k: params[k] for k in sorted(params.keys())}
            params_str = json.dumps(sorted_params, sort_keys=True)
            import hashlib
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        else:
            params_hash = 'default'
        return f'{symbol}_{timeframe}_{indicator_type}_{params_hash}'

    @with_exception_handling
    def _save_indicator_state(self, indicator_key: str, indicator:
        IncrementalIndicator) ->bool:
        """
        Save the indicator state to persistent storage.
        
        Args:
            indicator_key: Unique key for the indicator
            indicator: Indicator instance to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self.state_persistence_path:
            return False
        try:
            import os
            import pickle
            os.makedirs(self.state_persistence_path, exist_ok=True)
            file_path = os.path.join(self.state_persistence_path,
                f'{indicator_key}.pickle')
            with open(file_path, 'wb') as f:
                pickle.dump(indicator.get_state(), f)
            return True
        except Exception as e:
            logger.error(f'Error saving indicator state: {str(e)}')
            return False

    @with_exception_handling
    def _load_indicator_state(self, indicator_key: str) ->Optional[Dict[str,
        Any]]:
        """
        Load an indicator state from persistent storage.
        
        Args:
            indicator_key: Unique key for the indicator
            
        Returns:
            Dictionary containing the indicator state if successful, None otherwise
        """
        if not self.state_persistence_path:
            return None
        try:
            import os
            import pickle
            file_path = os.path.join(self.state_persistence_path,
                f'{indicator_key}.pickle')
            if not os.path.exists(file_path):
                return None
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
            return state
        except Exception as e:
            logger.error(f'Error loading indicator state: {str(e)}')
            return None

    @async_with_exception_handling
    async def load_all_saved_states(self) ->int:
        """
        Load all saved indicator states from persistent storage.
        
        Returns:
            Number of successfully loaded indicators
        """
        if not self.state_persistence_path:
            logger.warning('No state persistence path configured')
            return 0
        try:
            import os
            import pickle
            if not os.path.exists(self.state_persistence_path):
                logger.warning(
                    f'State persistence directory does not exist: {self.state_persistence_path}'
                    )
                return 0
            count = 0
            for filename in os.listdir(self.state_persistence_path):
                if not filename.endswith('.pickle'):
                    continue
                indicator_key = filename[:-7]
                state = self._load_indicator_state(indicator_key)
                if not state:
                    continue
                indicator_type = state.get('name')
                if not indicator_type:
                    logger.warning(
                        f'Invalid state file, missing indicator type: {filename}'
                        )
                    continue
                indicator = IncrementalIndicatorFactory.create_indicator(
                    indicator_type, **state.get('params', {}))
                if not indicator:
                    logger.error(
                        f'Failed to create indicator from saved state: {indicator_type}'
                        )
                    continue
                indicator.set_state(state)
                self.active_indicators[indicator_key] = indicator
                count += 1
            logger.info(f'Loaded {count} indicator states from persistence')
            return count
        except Exception as e:
            logger.error(f'Error loading saved indicator states: {str(e)}')
            return 0

    def get_all_active_indicators(self) ->Dict[str, Dict[str, Any]]:
        """
        Get information about all active indicators.
        
        Returns:
            Dictionary mapping indicator keys to their information
        """
        result = {}
        for key, indicator in self.active_indicators.items():
            parts = key.split('_')
            if len(parts) >= 4:
                symbol = parts[0]
                timeframe = parts[1]
                indicator_type = parts[2]
                result[key] = {'symbol': symbol, 'timeframe': timeframe,
                    'type': indicator_type, 'params': indicator.params,
                    'is_initialized': indicator.is_initialized,
                    'last_updated': indicator.last_timestamp}
        return result
