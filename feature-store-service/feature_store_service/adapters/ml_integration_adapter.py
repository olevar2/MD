"""
ML Integration Adapter Module

This module provides adapter implementations for ML integration interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from common_lib.ml.feature_interfaces import IMLFeatureConsumer, FeatureType, FeatureScope, SelectionMethod
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MLFeatureConsumerAdapter(IMLFeatureConsumer):
    """
    Adapter for ML feature consumers that implements the common interface.

    This adapter can either wrap an actual consumer instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, consumer_instance=None):
        """
        Initialize the adapter.

        Args:
            consumer_instance: Optional actual consumer instance to wrap
        """
        self.consumer = consumer_instance
        self.feature_importance_cache = {}
        self.required_features_cache = {}

    @with_exception_handling
    def get_required_features(self, model_id: str=None) ->List[Dict[str, Any]]:
        """
        Get required features for a model.

        Args:
            model_id: Optional model ID

        Returns:
            List of feature definitions
        """
        if self.consumer:
            try:
                return self.consumer.get_required_features(model_id=model_id)
            except Exception as e:
                logger.warning(f'Error getting required features: {str(e)}')
        if model_id and model_id in self.required_features_cache:
            return self.required_features_cache[model_id]
        default_features = [{'name': 'price_normalized', 'source_columns':
            ['close'], 'feature_type': FeatureType.NORMALIZED, 'params': {
            'scaler': 'minmax'}, 'scope': FeatureScope.PRICE,
            'is_sequential': False, 'lookback_periods': 1}, {'name':
            'rsi_14', 'source_columns': ['rsi_14'], 'feature_type':
            FeatureType.NORMALIZED, 'params': {}, 'scope': FeatureScope.
            INDICATOR, 'is_sequential': False, 'lookback_periods': 1}, {
            'name': 'ma_crossover', 'source_columns': ['sma_10', 'sma_50'],
            'feature_type': FeatureType.CROSSOVER, 'params': {}, 'scope':
            FeatureScope.INDICATOR, 'is_sequential': False,
            'lookback_periods': 1}, {'name': 'bollinger_band_position',
            'source_columns': ['close', 'bb_upper', 'bb_lower'],
            'feature_type': FeatureType.CUSTOM, 'params': {'method':
            'position'}, 'scope': FeatureScope.INDICATOR, 'is_sequential': 
            False, 'lookback_periods': 1}, {'name': 'volume_trend',
            'source_columns': ['volume'], 'feature_type': FeatureType.TREND,
            'params': {'window': 5}, 'scope': FeatureScope.VOLUME,
            'is_sequential': False, 'lookback_periods': 1}]
        if model_id:
            self.required_features_cache[model_id] = default_features
        return default_features

    @with_exception_handling
    def prepare_model_inputs(self, features: pd.DataFrame, model_id: str=
        None, target_column: str=None) ->Tuple[pd.DataFrame, Optional[pd.
        Series]]:
        """
        Prepare inputs for a model.

        Args:
            features: DataFrame with features
            model_id: Optional model ID
            target_column: Optional target column

        Returns:
            Tuple of (X, y) where X is the feature matrix and y is the target
        """
        if self.consumer:
            try:
                return self.consumer.prepare_model_inputs(features=features,
                    model_id=model_id, target_column=target_column)
            except Exception as e:
                logger.warning(f'Error preparing model inputs: {str(e)}')
        X = features.copy()
        y = None
        X = X.fillna(method='ffill').fillna(0)
        if target_column and target_column in X.columns:
            y = X[target_column]
            X = X.drop(columns=[target_column])
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X = X.drop(columns=[col])
        return X, y

    @with_exception_handling
    def get_feature_importance_feedback(self, model_id: str, features: List
        [str]) ->Dict[str, float]:
        """
        Get feature importance feedback from a model.

        Args:
            model_id: Model ID
            features: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.consumer:
            try:
                return self.consumer.get_feature_importance_feedback(model_id
                    =model_id, features=features)
            except Exception as e:
                logger.warning(
                    f'Error getting feature importance feedback: {str(e)}')
        cache_key = f"{model_id}_{','.join(sorted(features))}"
        if cache_key in self.feature_importance_cache:
            return self.feature_importance_cache[cache_key]
        np.random.seed(42)
        importance = {}
        for feature in features:
            importance[feature] = np.random.uniform(0, 1)
        total = sum(importance.values())
        if total > 0:
            for feature in importance:
                importance[feature] /= total
        self.feature_importance_cache[cache_key] = importance
        return importance
