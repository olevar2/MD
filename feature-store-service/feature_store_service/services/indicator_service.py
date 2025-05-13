"""
Indicator Service Module.

Provides services for calculating and managing technical indicators.
"""
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime, timezone
import pandas as pd
from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.indicators.indicator_registry import indicator_registry
from feature_store_service.computation.feature_computation_engine import FeatureComputationEngine
from feature_store_service.models.feature_models import TimeSeriesTransformRequest, TimeSeriesTransformResponse, FeatureVectorRequest, FeatureResponse, FeatureMetadataResponse, FeatureQuery, FeatureVector
from feature_store_service.repositories.feature_repository import FeatureRepository


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class IndicatorService:
    """
    Service for managing technical indicator calculations.
    
    This service provides high-level methods for calculating
    technical indicators on financial data.
    """

    def __init__(self):
        """Initialize the indicator service."""
        self.logger = logging.getLogger(__name__)
        self._registry = indicator_registry
        self._calculation_cache = {}
        self._repository = FeatureRepository()
        self._computation_engine = FeatureComputationEngine()

    @with_exception_handling
    def calculate_indicator(self, data: pd.DataFrame, indicator_name: str,
        **indicator_params) ->pd.DataFrame:
        """
        Calculate a single indicator for the provided data.
        
        Args:
            data: DataFrame containing financial data
            indicator_name: Name of the indicator to calculate
            **indicator_params: Parameters to pass to the indicator constructor
            
        Returns:
            DataFrame with the indicator values added as new columns
            
        Raises:
            KeyError: If the indicator name is not registered
            ValueError: If the data doesn't contain the required columns
        """
        try:
            indicator = self._registry.create_indicator(indicator_name, **
                indicator_params)
            self.logger.info(
                f'Calculating {indicator_name} with parameters {indicator_params}'
                )
            result = indicator.calculate(data)
            return result
        except KeyError:
            self.logger.error(
                f"Indicator '{indicator_name}' not found in registry")
            raise
        except ValueError as e:
            self.logger.error(f'Error calculating {indicator_name}: {str(e)}')
            raise
        except Exception as e:
            self.logger.error(
                f'Unexpected error calculating {indicator_name}: {str(e)}')
            raise

    @with_exception_handling
    def calculate_multiple_indicators(self, data: pd.DataFrame, indicators:
        List[Dict[str, Any]]) ->pd.DataFrame:
        """
        Calculate multiple indicators for the provided data.
        
        Args:
            data: DataFrame containing financial data
            indicators: List of indicator configurations, each with:
                - 'name': Name of the indicator
                - 'params': Dictionary of parameters for the indicator
        Returns:
            DataFrame with all indicator values added as new columns
            
        Example:
            indicators = [
                {"name": "SMA", "params": {"window": 20}},
                {"name": "RSI", "params": {"window": 14}}
            ]
        """
        result = data.copy()
        for indicator_config in indicators:
            name = indicator_config['name']
            params = indicator_config_manager.get('params', {})
            try:
                result = self.calculate_indicator(result, name, **params)
            except Exception as e:
                self.logger.error(f'Error calculating {name}: {str(e)}')
                continue
        return result

    def get_available_indicators(self) ->List[str]:
        """
        Get a list of all available indicators.
        
        Returns:
            List of indicator names registered in the system
        """
        return self._registry.get_available_indicators()

    def get_indicator_metadata(self, indicator_name: str) ->Dict[str, Any]:
        """
        Get metadata for a specific indicator.
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Dictionary with indicator metadata
            
        Raises:
            KeyError: If the indicator name is not registered
        """
        return self._registry.get_indicator_metadata(indicator_name)

    @with_exception_handling
    def transform_time_series(self, request: TimeSeriesTransformRequest
        ) ->TimeSeriesTransformResponse:
        """
        Apply transformations to time series data.
        
        This method handles various transformations like normalization,
        standardization, differencing, etc.
        
        Args:
            request: The time series transformation request
            
        Returns:
            Response containing the transformed features
        """
        self.logger.info(
            f'Processing time series transformation: {request.transformation_type}'
            )
        try:
            features = []
            for instrument in request.instruments:
                for timeframe in request.timeframes:
                    transformed_data = (self._computation_engine.
                        transform_time_series(instrument=instrument,
                        timeframe=timeframe, transformation_type=request.
                        transformation_type, parameters=request.parameters,
                        start_time=request.start_time, end_time=request.
                        end_time))
                    feature_id = (
                        f'{request.transformation_type}_{instrument}_{timeframe}'
                        )
                    self._repository.store_feature_data(feature_id=
                        feature_id, instrument=instrument, timeframe=
                        timeframe, data=transformed_data)
                    feature_metadata = FeatureMetadataResponse(feature_id=
                        feature_id, name=
                        f'{request.transformation_type} of {instrument}',
                        instrument=instrument, timeframe=timeframe,
                        value_type='float', description=
                        f'{request.transformation_type} transformation of {instrument} with parameters {request.parameters}'
                        , parameters=request.parameters)
                    features.append(feature_metadata)
            return TimeSeriesTransformResponse(status='success', message=
                f'Successfully transformed time series data for {len(features)} features'
                , features=features)
        except Exception as e:
            error_msg = f'Error transforming time series data: {str(e)}'
            self.logger.error(error_msg)
            return TimeSeriesTransformResponse(status='error', message=
                error_msg, features=[])

    @with_exception_handling
    def get_feature_vectors(self, request: FeatureVectorRequest
        ) ->FeatureResponse:
        """
        Retrieve feature vectors for the requested feature IDs and time range.
        
        Args:
            request: The feature vector request
            
        Returns:
            Response containing the requested feature vectors
        """
        self.logger.info(
            f'Retrieving feature vectors for {len(request.feature_ids)} features'
            )
        try:
            query = FeatureQuery(feature_ids=request.feature_ids,
                start_time=request.start_time, end_time=request.end_time,
                limit=request.limit, offset=request.offset)
            feature_data = self._repository.query_features(query)
            feature_vectors = []
            for timestamp, features in feature_data.items():
                vector = {'timestamp': timestamp, 'features': features}
                feature_vectors.append(vector)
            return FeatureResponse(status='success', message=
                f'Successfully retrieved {len(feature_vectors)} feature vectors'
                , feature_vectors=feature_vectors)
        except Exception as e:
            error_msg = f'Error retrieving feature vectors: {str(e)}'
            self.logger.error(error_msg)
            return FeatureResponse(status='error', message=error_msg,
                feature_vectors=[])

    @with_exception_handling
    def get_feature_metadata(self, feature_ids: List[str]) ->List[
        FeatureMetadataResponse]:
        """
        Retrieve metadata for the specified features.
        
        Args:
            feature_ids: List of feature IDs to retrieve metadata for
            
        Returns:
            List of feature metadata objects
        """
        self.logger.info(f'Retrieving metadata for {len(feature_ids)} features'
            )
        try:
            metadata = []
            for feature_id in feature_ids:
                feature_metadata = self._repository.get_feature_metadata(
                    feature_id)
                if feature_metadata:
                    metadata_response = FeatureMetadataResponse(feature_id=
                        feature_id, name=feature_metadata.get('name',
                        feature_id), instrument=feature_metadata.get(
                        'instrument', ''), timeframe=feature_metadata.get(
                        'timeframe', ''), value_type=feature_metadata.get(
                        'value_type', 'float'), description=
                        feature_metadata.get('description', ''), parameters
                        =feature_metadata.get('parameters', {}))
                    metadata.append(metadata_response)
            return metadata
        except Exception as e:
            self.logger.error(f'Error retrieving feature metadata: {str(e)}')
            return []

    def register_custom_indicator(self, name: str, indicator_class: type
        ) ->None:
        """
        Register a custom indicator.
        
        Args:
            name: Name for the indicator
            indicator_class: Class implementing the indicator
            
        Raises:
            TypeError: If the class doesn't inherit from BaseIndicator
        """
        if not issubclass(indicator_class, BaseIndicator):
            raise TypeError('Custom indicator must inherit from BaseIndicator')
        self._registry.register_indicator(name, indicator_class)
        self.logger.info(f'Registered custom indicator: {name}')

    def create_common_indicator_set(self, data: pd.DataFrame, timeframe:
        str='default') ->pd.DataFrame:
        """
        Calculate a standard set of common indicators appropriate for the timeframe.
        
        Args:
            data: DataFrame containing OHLCV data
            timeframe: The timeframe to determine which indicators to use
                       ('short', 'medium', 'long', or 'default')
                       
        Returns:
            DataFrame with all standard indicators added
        """
        indicators = []
        indicators.extend([{'name': 'SMA', 'params': {'window': 20}}, {
            'name': 'EMA', 'params': {'window': 14}}, {'name': 'RSI',
            'params': {'window': 14}}, {'name': 'MACD', 'params': {
            'fast_period': 12, 'slow_period': 26, 'signal_period': 9}}])
        if timeframe == 'short':
            indicators.extend([{'name': 'SMA', 'params': {'window': 5}}, {
                'name': 'SMA', 'params': {'window': 10}}, {'name':
                'Bollinger', 'params': {'window': 10, 'num_std': 2}}, {
                'name': 'ATR', 'params': {'window': 7}}, {'name':
                'Stochastic', 'params': {'k_period': 5, 'd_period': 3,
                'slowing': 3}}])
        elif timeframe == 'medium':
            indicators.extend([{'name': 'SMA', 'params': {'window': 50}}, {
                'name': 'Bollinger', 'params': {'window': 20, 'num_std': 2}
                }, {'name': 'ATR', 'params': {'window': 14}}, {'name':
                'ADX', 'params': {'window': 14}}, {'name': 'Ichimoku',
                'params': {}}])
        elif timeframe == 'long':
            indicators.extend([{'name': 'SMA', 'params': {'window': 100}},
                {'name': 'SMA', 'params': {'window': 200}}, {'name':
                'Bollinger', 'params': {'window': 50, 'num_std': 2}}, {
                'name': 'ROC', 'params': {'window': 100}}, {'name': 'ADX',
                'params': {'window': 25}}])
        else:
            indicators.extend([{'name': 'SMA', 'params': {'window': 50}}, {
                'name': 'Bollinger', 'params': {'window': 20, 'num_std': 2}
                }, {'name': 'ATR', 'params': {'window': 14}}, {'name':
                'ADX', 'params': {'window': 14}}])
        result = self.calculate_multiple_indicators(data, indicators)
        self.logger.info(
            f'Created common indicator set for {timeframe} timeframe with {len(indicators)} indicators'
            )
        return result

    @with_exception_handling
    def calculate_time_series_transform(self, request:
        TimeSeriesTransformRequest) ->TimeSeriesTransformResponse:
        """
        Calculate time series transformations based on the request parameters.
        
        Args:
            request: The time series transformation request
            
        Returns:
            TimeSeriesTransformResponse with metadata about the calculated features
        """
        try:
            self.logger.info(
                f'Processing time series transform request for {request.transformation_type}'
                )
            result_features = self._computation_engine.transform_time_series(
                instruments=request.instruments, timeframes=request.
                timeframes, transform_type=request.transformation_type,
                parameters=request.parameters, start_time=request.
                start_time, end_time=request.end_time)
            feature_metadata_list = []
            for feature in result_features:
                feature_metadata = FeatureMetadataResponse(feature_id=
                    feature['feature_id'], name=feature['name'], instrument
                    =feature['instrument'], timeframe=feature['timeframe'],
                    value_type=feature['value_type'], description=feature[
                    'description'], parameters=feature['parameters'])
                feature_metadata_list.append(feature_metadata)
            return TimeSeriesTransformResponse(status='success', message=
                f'Successfully calculated {request.transformation_type} transform'
                , features=feature_metadata_list)
        except ValueError as e:
            self.logger.error(
                f'Invalid parameters for transformation: {str(e)}')
            return TimeSeriesTransformResponse(status='error', message=
                f'Invalid parameters: {str(e)}', features=[])
        except Exception as e:
            self.logger.error(
                f'Error calculating time series transform: {str(e)}')
            return TimeSeriesTransformResponse(status='error', message=
                f'Calculation error: {str(e)}', features=[])

    @with_exception_handling
    def get_feature_vectors(self, request: FeatureVectorRequest
        ) ->FeatureResponse:
        """
        Retrieve feature vectors based on the request parameters.
        
        Args:
            request: The feature vector request
            
        Returns:
            FeatureResponse with the requested feature vectors
        """
        try:
            self.logger.info(
                f'Processing feature vector request for {len(request.feature_ids)} features'
                )
            query = FeatureQuery(feature_ids=request.feature_ids,
                start_time=request.start_time, end_time=request.end_time,
                limit=request.limit, offset=request.offset)
            feature_data = self._repository.get_features(query)
            feature_vectors = {}
            for feature in feature_data:
                timestamp = feature.timestamp
                feature_id = feature.feature_id
                value = feature.value
                if timestamp not in feature_vectors:
                    feature_vectors[timestamp] = {'timestamp': timestamp,
                        'features': {}}
                feature_vectors[timestamp]['features'][feature_id] = value
            result_vectors = list(feature_vectors.values())
            result_vectors.sort(key=lambda x: x['timestamp'])
            return FeatureResponse(status='success', message=
                f'Retrieved {len(result_vectors)} feature vectors',
                feature_vectors=result_vectors)
        except Exception as e:
            self.logger.error(f'Error retrieving feature vectors: {str(e)}')
            return FeatureResponse(status='error', message=
                f'Retrieval error: {str(e)}', feature_vectors=[])

    @with_exception_handling
    def list_available_features(self) ->List[FeatureMetadataResponse]:
        """
        List all available features in the feature store.
        
        Returns:
            List of FeatureMetadataResponse objects
        """
        try:
            feature_metadata = self._repository.get_all_feature_metadata()
            return [FeatureMetadataResponse(feature_id=meta.feature_id,
                name=meta.name, instrument=meta.symbol, timeframe=meta.
                timeframe, value_type=meta.value_type, description=meta.
                description, parameters=meta.parameters) for meta in
                feature_metadata]
        except Exception as e:
            self.logger.error(f'Error listing available features: {str(e)}')
            raise

    @with_exception_handling
    def get_feature_metadata(self, feature_ids: List[str]) ->List[
        FeatureMetadataResponse]:
        """
        Retrieve metadata for the specified features.
        
        Args:
            feature_ids: List of feature IDs to retrieve metadata for
            
        Returns:
            List of feature metadata objects
        """
        self.logger.info(f'Retrieving metadata for {len(feature_ids)} features'
            )
        try:
            metadata = []
            for feature_id in feature_ids:
                feature_metadata = self._repository.get_feature_metadata(
                    feature_id)
                if feature_metadata:
                    metadata_response = FeatureMetadataResponse(feature_id=
                        feature_id, name=feature_metadata.get('name',
                        feature_id), instrument=feature_metadata.get(
                        'instrument', ''), timeframe=feature_metadata.get(
                        'timeframe', ''), value_type=feature_metadata.get(
                        'value_type', 'float'), description=
                        feature_metadata.get('description', ''), parameters
                        =feature_metadata.get('parameters', {}))
                    metadata.append(metadata_response)
            return metadata
        except Exception as e:
            self.logger.error(f'Error retrieving feature metadata: {str(e)}')
            return []
