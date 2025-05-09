"""
Feature Store Adapter Module

This module provides adapter implementations for feature store interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from common_lib.ml.feature_interfaces import (
    IFeatureProvider, IFeatureTransformer, IFeatureSelector,
    FeatureType, FeatureScope, SelectionMethod
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureProviderAdapter(IFeatureProvider):
    """
    Adapter for feature providers that implements the common interface.
    
    This adapter can either wrap an actual provider instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, provider_instance=None):
        """
        Initialize the adapter.
        
        Args:
            provider_instance: Optional actual provider instance to wrap
        """
        self.provider = provider_instance
        self.feature_cache = {}
        self.metadata_cache = {}
        
        # Default available features
        self.default_features = [
            "close", "open", "high", "low", "volume",
            "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_10", "ema_20", "ema_50", "ema_200",
            "rsi_14", "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower",
            "atr_14", "adx_14", "stoch_k", "stoch_d"
        ]
    
    def get_available_features(self, symbol: str = None) -> List[str]:
        """
        Get a list of available features.
        
        Args:
            symbol: Optional symbol to get features for
            
        Returns:
            List of feature names
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return self.provider.get_available_features(symbol=symbol)
            except Exception as e:
                logger.warning(f"Error getting available features: {str(e)}")
        
        # Fallback to default features if no provider available
        return self.default_features
    
    def get_feature_data(
        self,
        feature_names: List[str],
        symbol: str,
        timeframe: str,
        start_date: str = None,
        end_date: str = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Get feature data for a symbol.
        
        Args:
            feature_names: List of feature names to get
            symbol: The trading symbol
            timeframe: The timeframe to get data for
            start_date: Optional start date
            end_date: Optional end date
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame with feature data
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return self.provider.get_feature_data(
                    feature_names=feature_names,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit
                )
            except Exception as e:
                logger.warning(f"Error getting feature data: {str(e)}")
        
        # Check if we have cached data for this request
        cache_key = f"{symbol}_{timeframe}_{','.join(sorted(feature_names))}"
        if cache_key in self.feature_cache:
            data = self.feature_cache[cache_key].copy()
            
            # Apply date filtering if needed
            if start_date:
                data = data[data.index >= pd.to_datetime(start_date)]
            if end_date:
                data = data[data.index <= pd.to_datetime(end_date)]
            if limit:
                data = data.tail(limit)
            
            return data
        
        # Fallback to generating synthetic data if no provider available
        end_date_dt = pd.to_datetime(end_date) if end_date else datetime.now()
        if limit:
            start_date_dt = end_date_dt - timedelta(days=limit)
        else:
            start_date_dt = pd.to_datetime(start_date) if start_date else end_date_dt - timedelta(days=100)
        
        # Create date range
        date_range = pd.date_range(start=start_date_dt, end=end_date_dt, freq='D')
        
        # Create DataFrame with random data
        np.random.seed(42)  # For reproducibility
        data = pd.DataFrame(index=date_range)
        
        # Generate OHLCV data
        if "close" in feature_names or any(f for f in feature_names if f.startswith("sma_") or f.startswith("ema_")):
            close = 100 + np.cumsum(np.random.normal(0, 1, len(date_range)))
            data["close"] = close
        
        if "open" in feature_names:
            data["open"] = data["close"].shift(1) * (1 + np.random.normal(0, 0.01, len(date_range)))
            data["open"].iloc[0] = data["close"].iloc[0] * 0.99
        
        if "high" in feature_names:
            data["high"] = data[["open", "close"]].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, len(date_range))))
        
        if "low" in feature_names:
            data["low"] = data[["open", "close"]].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, len(date_range))))
        
        if "volume" in feature_names:
            data["volume"] = np.random.randint(1000, 10000, len(date_range))
        
        # Generate indicator data
        for feature in feature_names:
            if feature in data.columns:
                continue
            
            if feature.startswith("sma_"):
                window = int(feature.split("_")[1])
                if "close" in data.columns:
                    data[feature] = data["close"].rolling(window=window).mean()
            
            elif feature.startswith("ema_"):
                window = int(feature.split("_")[1])
                if "close" in data.columns:
                    data[feature] = data["close"].ewm(span=window).mean()
            
            elif feature == "rsi_14":
                if "close" in data.columns:
                    delta = data["close"].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss
                    data[feature] = 100 - (100 / (1 + rs))
            
            elif feature in ["macd", "macd_signal", "macd_histogram"]:
                if "close" in data.columns:
                    ema12 = data["close"].ewm(span=12).mean()
                    ema26 = data["close"].ewm(span=26).mean()
                    macd = ema12 - ema26
                    macd_signal = macd.ewm(span=9).mean()
                    
                    if feature == "macd":
                        data[feature] = macd
                    elif feature == "macd_signal":
                        data[feature] = macd_signal
                    elif feature == "macd_histogram":
                        data[feature] = macd - macd_signal
            
            elif feature in ["bb_upper", "bb_middle", "bb_lower"]:
                if "close" in data.columns:
                    window = 20
                    std = data["close"].rolling(window=window).std()
                    middle = data["close"].rolling(window=window).mean()
                    
                    if feature == "bb_middle":
                        data[feature] = middle
                    elif feature == "bb_upper":
                        data[feature] = middle + 2 * std
                    elif feature == "bb_lower":
                        data[feature] = middle - 2 * std
            
            elif feature == "atr_14":
                if all(col in data.columns for col in ["high", "low", "close"]):
                    high_low = data["high"] - data["low"]
                    high_close = (data["high"] - data["close"].shift()).abs()
                    low_close = (data["low"] - data["close"].shift()).abs()
                    ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    data[feature] = ranges.rolling(window=14).mean()
            
            elif feature == "adx_14":
                # Simplified ADX calculation
                data[feature] = 25 + 10 * np.sin(np.linspace(0, 10, len(date_range)))
            
            elif feature in ["stoch_k", "stoch_d"]:
                if all(col in data.columns for col in ["high", "low", "close"]):
                    window = 14
                    lowest_low = data["low"].rolling(window=window).min()
                    highest_high = data["high"].rolling(window=window).max()
                    k = 100 * ((data["close"] - lowest_low) / (highest_high - lowest_low))
                    
                    if feature == "stoch_k":
                        data[feature] = k
                    elif feature == "stoch_d":
                        data[feature] = k.rolling(window=3).mean()
            
            else:
                # For any other feature, generate random data
                data[feature] = np.random.normal(0, 1, len(date_range))
        
        # Filter to only requested features
        data = data[feature_names]
        
        # Cache the data
        self.feature_cache[cache_key] = data.copy()
        
        return data
    
    def compute_features(
        self,
        feature_definitions: List[Dict[str, Any]],
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute features from raw data.
        
        Args:
            feature_definitions: List of feature definitions
            data: DataFrame with raw data
            
        Returns:
            DataFrame with computed features
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return self.provider.compute_features(
                    feature_definitions=feature_definitions,
                    data=data
                )
            except Exception as e:
                logger.warning(f"Error computing features: {str(e)}")
        
        # Fallback to simple feature computation if no provider available
        result = pd.DataFrame(index=data.index)
        
        for feature_def in feature_definitions:
            name = feature_def.get("name", "unknown")
            feature_type = feature_def.get("feature_type", FeatureType.RAW)
            source_columns = feature_def.get("source_columns", [])
            params = feature_def.get("params", {})
            
            try:
                # Check if all source columns are available
                if not all(col in data.columns for col in source_columns):
                    logger.warning(f"Missing source columns for feature {name}: {source_columns}")
                    continue
                
                # Compute feature based on type
                if feature_type == FeatureType.RAW:
                    # Just copy the source column
                    if len(source_columns) == 1:
                        result[name] = data[source_columns[0]]
                    else:
                        for i, col in enumerate(source_columns):
                            result[f"{name}_{i}"] = data[col]
                
                elif feature_type == FeatureType.NORMALIZED:
                    # Min-max normalization
                    for i, col in enumerate(source_columns):
                        col_name = name if len(source_columns) == 1 else f"{name}_{i}"
                        min_val = data[col].min()
                        max_val = data[col].max()
                        if max_val > min_val:
                            result[col_name] = (data[col] - min_val) / (max_val - min_val)
                        else:
                            result[col_name] = 0.5
                
                elif feature_type == FeatureType.STANDARDIZED:
                    # Z-score standardization
                    for i, col in enumerate(source_columns):
                        col_name = name if len(source_columns) == 1 else f"{name}_{i}"
                        mean = data[col].mean()
                        std = data[col].std()
                        if std > 0:
                            result[col_name] = (data[col] - mean) / std
                        else:
                            result[col_name] = 0
                
                elif feature_type == FeatureType.TREND:
                    # Trend calculation
                    window = params.get("window", 5)
                    for i, col in enumerate(source_columns):
                        col_name = name if len(source_columns) == 1 else f"{name}_{i}"
                        result[col_name] = data[col].diff(window).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                
                elif feature_type == FeatureType.CROSSOVER:
                    # Crossover calculation
                    if len(source_columns) >= 2:
                        method = params.get("method", "binary")
                        if method == "binary":
                            # 1 if first > second, -1 if first < second, 0 if equal
                            result[name] = np.where(
                                data[source_columns[0]] > data[source_columns[1]], 1,
                                np.where(data[source_columns[0]] < data[source_columns[1]], -1, 0)
                            )
                        elif method == "distance":
                            # Normalized distance between the two
                            result[name] = (data[source_columns[0]] - data[source_columns[1]]) / data[source_columns[1]]
                
                elif feature_type == FeatureType.CATEGORICAL:
                    # One-hot encoding
                    for i, col in enumerate(source_columns):
                        categories = params.get("categories", [])
                        if not categories:
                            categories = data[col].unique().tolist()
                        
                        for category in categories:
                            result[f"{name}_{category}"] = (data[col] == category).astype(int)
                
                else:
                    # For any other type, just copy the source columns
                    for i, col in enumerate(source_columns):
                        col_name = name if len(source_columns) == 1 else f"{name}_{i}"
                        result[col_name] = data[col]
            
            except Exception as e:
                logger.warning(f"Error computing feature {name}: {str(e)}")
        
        return result
    
    def get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """
        Get metadata for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with feature metadata
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return self.provider.get_feature_metadata(feature_name=feature_name)
            except Exception as e:
                logger.warning(f"Error getting feature metadata: {str(e)}")
        
        # Check if we have cached metadata for this feature
        if feature_name in self.metadata_cache:
            return self.metadata_cache[feature_name]
        
        # Fallback to default metadata if no provider available
        metadata = {
            "name": feature_name,
            "description": f"Feature {feature_name}",
            "type": "numeric",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "tags": []
        }
        
        # Add specific metadata based on feature name
        if feature_name.startswith("sma_") or feature_name.startswith("ema_"):
            window = feature_name.split("_")[1]
            indicator_type = "Simple Moving Average" if feature_name.startswith("sma_") else "Exponential Moving Average"
            metadata.update({
                "description": f"{indicator_type} with window {window}",
                "category": "trend",
                "parameters": {"window": int(window)},
                "tags": ["moving_average", "trend"]
            })
        
        elif feature_name == "rsi_14":
            metadata.update({
                "description": "Relative Strength Index with period 14",
                "category": "oscillator",
                "parameters": {"period": 14},
                "tags": ["oscillator", "overbought", "oversold"],
                "min_value": 0,
                "max_value": 100
            })
        
        elif feature_name in ["macd", "macd_signal", "macd_histogram"]:
            metadata.update({
                "description": f"MACD {feature_name.split('_')[1] if '_' in feature_name else 'line'}",
                "category": "momentum",
                "parameters": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "tags": ["momentum", "trend"]
            })
        
        elif feature_name in ["bb_upper", "bb_middle", "bb_lower"]:
            band_type = feature_name.split("_")[1]
            metadata.update({
                "description": f"Bollinger Band {band_type} line",
                "category": "volatility",
                "parameters": {"period": 20, "std_dev": 2},
                "tags": ["volatility", "bands"]
            })
        
        # Cache the metadata
        self.metadata_cache[feature_name] = metadata
        
        return metadata


class FeatureTransformerAdapter(IFeatureTransformer):
    """
    Adapter for feature transformers that implements the common interface.
    
    This adapter can either wrap an actual transformer instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, transformer_instance=None):
        """
        Initialize the adapter.
        
        Args:
            transformer_instance: Optional actual transformer instance to wrap
        """
        self.transformer = transformer_instance
        self.scaler_params = {}
    
    def transform(
        self,
        data: pd.DataFrame,
        feature_definitions: List[Dict[str, Any]] = None,
        fit: bool = False
    ) -> pd.DataFrame:
        """
        Transform features.
        
        Args:
            data: DataFrame with raw data
            feature_definitions: Optional list of feature definitions
            fit: Whether to fit the transformer
            
        Returns:
            DataFrame with transformed features
        """
        if self.transformer:
            try:
                # Try to use the wrapped transformer if available
                return self.transformer.transform(
                    data=data,
                    feature_definitions=feature_definitions,
                    fit=fit
                )
            except Exception as e:
                logger.warning(f"Error transforming features: {str(e)}")
        
        # Fallback to simple transformation if no transformer available
        result = data.copy()
        
        # If no feature definitions, transform all numeric columns
        if not feature_definitions:
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            feature_definitions = [
                {
                    "name": col,
                    "source_columns": [col],
                    "feature_type": FeatureType.NORMALIZED,
                    "params": {}
                }
                for col in numeric_cols
            ]
        
        # Apply transformations based on feature definitions
        for feature_def in feature_definitions:
            name = feature_def.get("name", "unknown")
            feature_type = feature_def.get("feature_type", FeatureType.RAW)
            source_columns = feature_def.get("source_columns", [])
            params = feature_def.get("params", {})
            
            try:
                # Check if all source columns are available
                if not all(col in data.columns for col in source_columns):
                    logger.warning(f"Missing source columns for feature {name}: {source_columns}")
                    continue
                
                # Apply transformation based on type
                if feature_type == FeatureType.NORMALIZED:
                    # Min-max normalization
                    for i, col in enumerate(source_columns):
                        col_name = name if len(source_columns) == 1 else f"{name}_{i}"
                        
                        if fit or col not in self.scaler_params:
                            min_val = data[col].min()
                            max_val = data[col].max()
                            self.scaler_params[col] = {"min": min_val, "max": max_val}
                        
                        min_val = self.scaler_params[col]["min"]
                        max_val = self.scaler_params[col]["max"]
                        
                        if max_val > min_val:
                            result[col_name] = (data[col] - min_val) / (max_val - min_val)
                        else:
                            result[col_name] = 0.5
                
                elif feature_type == FeatureType.STANDARDIZED:
                    # Z-score standardization
                    for i, col in enumerate(source_columns):
                        col_name = name if len(source_columns) == 1 else f"{name}_{i}"
                        
                        if fit or col not in self.scaler_params:
                            mean = data[col].mean()
                            std = data[col].std()
                            self.scaler_params[col] = {"mean": mean, "std": std}
                        
                        mean = self.scaler_params[col]["mean"]
                        std = self.scaler_params[col]["std"]
                        
                        if std > 0:
                            result[col_name] = (data[col] - mean) / std
                        else:
                            result[col_name] = 0
            
            except Exception as e:
                logger.warning(f"Error transforming feature {name}: {str(e)}")
        
        return result
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the transformer to data.
        
        Args:
            data: DataFrame with raw data
        """
        if self.transformer:
            try:
                # Try to use the wrapped transformer if available
                self.transformer.fit(data=data)
                return
            except Exception as e:
                logger.warning(f"Error fitting transformer: {str(e)}")
        
        # Fallback to simple fitting if no transformer available
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        
        # Calculate and store scaling parameters
        for col in numeric_cols:
            # Min-max parameters
            min_val = data[col].min()
            max_val = data[col].max()
            
            # Z-score parameters
            mean = data[col].mean()
            std = data[col].std()
            
            self.scaler_params[col] = {
                "min": min_val,
                "max": max_val,
                "mean": mean,
                "std": std
            }
    
    def save_state(self, path: str) -> None:
        """
        Save the transformer state.
        
        Args:
            path: Path to save the state to
        """
        if self.transformer:
            try:
                # Try to use the wrapped transformer if available
                self.transformer.save_state(path=path)
                return
            except Exception as e:
                logger.warning(f"Error saving transformer state: {str(e)}")
        
        # Fallback to simple state saving if no transformer available
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.scaler_params, f)
        except Exception as e:
            logger.error(f"Error saving transformer state: {str(e)}")
    
    def load_state(self, path: str) -> None:
        """
        Load the transformer state.
        
        Args:
            path: Path to load the state from
        """
        if self.transformer:
            try:
                # Try to use the wrapped transformer if available
                self.transformer.load_state(path=path)
                return
            except Exception as e:
                logger.warning(f"Error loading transformer state: {str(e)}")
        
        # Fallback to simple state loading if no transformer available
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.scaler_params = json.load(f)
        except Exception as e:
            logger.error(f"Error loading transformer state: {str(e)}")


class FeatureSelectorAdapter(IFeatureSelector):
    """
    Adapter for feature selectors that implements the common interface.
    
    This adapter can either wrap an actual selector instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, selector_instance=None):
        """
        Initialize the adapter.
        
        Args:
            selector_instance: Optional actual selector instance to wrap
        """
        self.selector = selector_instance
        self.feature_importances = {}
        self.selected_features = []
    
    def select_features(
        self,
        features: pd.DataFrame,
        target: pd.Series = None,
        method: str = None,
        n_features: int = None
    ) -> pd.DataFrame:
        """
        Select features.
        
        Args:
            features: DataFrame with features
            target: Optional target variable
            method: Optional selection method
            n_features: Optional number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if self.selector:
            try:
                # Try to use the wrapped selector if available
                return self.selector.select_features(
                    features=features,
                    target=target,
                    method=method,
                    n_features=n_features
                )
            except Exception as e:
                logger.warning(f"Error selecting features: {str(e)}")
        
        # Fallback to simple feature selection if no selector available
        if n_features is None:
            n_features = min(10, features.shape[1])
        
        # Default to correlation method if target is provided, otherwise variance
        if method is None:
            method = SelectionMethod.CORRELATION if target is not None else SelectionMethod.VARIANCE
        
        # Select features based on method
        if method == SelectionMethod.CORRELATION and target is not None:
            # Calculate correlation with target
            correlations = {}
            for col in features.columns:
                if features[col].dtype in [np.float64, np.int64]:
                    corr = abs(features[col].corr(target))
                    if not np.isnan(corr):
                        correlations[col] = corr
            
            # Sort by absolute correlation
            sorted_correlations = sorted(
                correlations.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Store feature importances
            self.feature_importances = {f: c for f, c in sorted_correlations}
            
            # Select top features
            self.selected_features = [f for f, c in sorted_correlations[:n_features]]
        
        elif method == SelectionMethod.VARIANCE:
            # Calculate variance
            variances = {}
            for col in features.columns:
                if features[col].dtype in [np.float64, np.int64]:
                    var = features[col].var()
                    if not np.isnan(var):
                        variances[col] = var
            
            # Sort by variance
            sorted_variances = sorted(
                variances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Store feature importances
            self.feature_importances = {f: v for f, v in sorted_variances}
            
            # Select top features
            self.selected_features = [f for f, v in sorted_variances[:n_features]]
        
        else:
            # For any other method, just select the first n_features
            self.selected_features = list(features.columns[:n_features])
            self.feature_importances = {f: 1.0 for f in self.selected_features}
        
        return features[self.selected_features]
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.selector:
            try:
                # Try to use the wrapped selector if available
                return self.selector.get_feature_importances()
            except Exception as e:
                logger.warning(f"Error getting feature importances: {str(e)}")
        
        # Fallback to stored importances if no selector available
        return self.feature_importances
    
    def save_state(self, path: str) -> None:
        """
        Save the selector state.
        
        Args:
            path: Path to save the state to
        """
        if self.selector:
            try:
                # Try to use the wrapped selector if available
                self.selector.save_state(path=path)
                return
            except Exception as e:
                logger.warning(f"Error saving selector state: {str(e)}")
        
        # Fallback to simple state saving if no selector available
        try:
            state = {
                "feature_importances": self.feature_importances,
                "selected_features": self.selected_features
            }
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Error saving selector state: {str(e)}")
    
    def load_state(self, path: str) -> None:
        """
        Load the selector state.
        
        Args:
            path: Path to load the state from
        """
        if self.selector:
            try:
                # Try to use the wrapped selector if available
                self.selector.load_state(path=path)
                return
            except Exception as e:
                logger.warning(f"Error loading selector state: {str(e)}")
        
        # Fallback to simple state loading if no selector available
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    state = json.load(f)
                    self.feature_importances = state.get("feature_importances", {})
                    self.selected_features = state.get("selected_features", [])
        except Exception as e:
            logger.error(f"Error loading selector state: {str(e)}")
