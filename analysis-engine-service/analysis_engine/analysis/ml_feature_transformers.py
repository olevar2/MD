"""
Feature Transformers for ML Integration

This module provides specialized transformers to prepare indicator data for machine learning models:
- Time-series specific transformations
- Feature normalization and standardization
- Feature selection and dimensionality reduction
- Feature cross-validation utilities
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from analysis_engine.utils.validation import ensure_dataframe
from enum import Enum
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Configure logging
logger = logging.getLogger(__name__)


class TransformerType(Enum):
    """Types of feature transformers"""
    SCALER = "scaler"
    SELECTOR = "selector"
    DECOMPOSER = "decomposer"
    TIME_SERIES = "time_series"
    CUSTOM = "custom"


@dataclass
class TransformerConfig:
    """Configuration for a feature transformer"""
    name: str
    type: TransformerType
    params: Dict[str, Any] = field(default_factory=dict)
    

class FeatureTransformerRegistry:
    """Registry of available feature transformers"""
    
    def __init__(self):
        """Initialize the transformer registry"""
        self._transformers = {}
        self._register_default_transformers()
    
    def _register_default_transformers(self):
        """Register built-in transformers"""
        # Scalers
        self.register("standard_scaler", TransformerType.SCALER, 
                     lambda params: StandardScaler(**params))
        
        self.register("minmax_scaler", TransformerType.SCALER,
                     lambda params: MinMaxScaler(**params))
        
        self.register("robust_scaler", TransformerType.SCALER,
                     lambda params: RobustScaler(**params))
        
        # Feature selectors
        self.register("select_k_best", TransformerType.SELECTOR,
                     lambda params: SelectKBest(
                         score_func=params.pop('score_func', f_regression),
                         **params
                     ))
        
        # Decomposers
        self.register("pca", TransformerType.DECOMPOSER,
                     lambda params: PCA(**params))
        
        # Time series transformers
        self.register("lag_features", TransformerType.TIME_SERIES,
                     lambda params: LagFeatureTransformer(**params))
        
        self.register("rolling_features", TransformerType.TIME_SERIES,
                     lambda params: RollingFeatureTransformer(**params))
        
        self.register("diff_features", TransformerType.TIME_SERIES,
                     lambda params: DiffFeatureTransformer(**params))
    
    def register(self, 
                name: str, 
                transformer_type: TransformerType, 
                factory_func: Callable[[Dict[str, Any]], Any]):
        """
        Register a new transformer
        
        Args:
            name: Name of the transformer
            transformer_type: Type of transformer
            factory_func: Function to create transformer instance
        """
        self._transformers[name] = {
            'type': transformer_type,
            'factory': factory_func
        }
        logger.debug(f"Registered transformer: {name} ({transformer_type.value})")
    
    def get_transformer(self, name: str, params: Dict[str, Any] = None) -> Any:
        """
        Get a transformer instance by name
        
        Args:
            name: Name of the transformer
            params: Parameters for the transformer
            
        Returns:
            Transformer instance
            
        Raises:
            ValueError: If transformer not found
        """
        if name not in self._transformers:
            raise ValueError(f"Transformer not found: {name}")
        
        if params is None:
            params = {}
            
        transformer_info = self._transformers[name]
        return transformer_info['factory'](params)
    
    def list_transformers(self, transformer_type: TransformerType = None) -> List[str]:
        """
        List available transformers, optionally filtered by type
        
        Args:
            transformer_type: Type of transformers to list
            
        Returns:
            List of transformer names
        """
        if transformer_type is None:
            return list(self._transformers.keys())
            
        return [name for name, info in self._transformers.items() 
                if info['type'] == transformer_type]
    
    def build_pipeline(self, configs: List[TransformerConfig]) -> Pipeline:
        """
        Build a scikit-learn pipeline from transformer configs
        
        Args:
            configs: List of transformer configurations
            
        Returns:
            Scikit-learn pipeline
        """
        steps = []
        
        for i, config in enumerate(configs):
            try:
                transformer = self.get_transformer(config.name, config.params)
                steps.append((f"{config.type.value}_{i}", transformer))
            except Exception as e:
                logger.error(f"Error creating transformer {config.name}: {str(e)}")
                # Skip this transformer
        
        return Pipeline(steps)


class LagFeatureTransformer:
    """Creates lagged versions of features"""
    
    def __init__(self, lags: List[int] = None, drop_na: bool = True):
        """
        Initialize lag transformer
        
        Args:
            lags: List of lag periods to create
            drop_na: Whether to drop rows with NaN values
        """
        self.lags = lags if lags is not None else [1, 2, 3]
        self.drop_na = drop_na
        self.feature_names_in_ = None
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        """Fit the transformer (just stores feature names)"""
        self.feature_names_in_ = X.columns if hasattr(X, 'columns') else None
        self.feature_names_out_ = self._get_output_feature_names()
        return self
    
    def transform(self, X):
        """Transform the input data by adding lagged features"""
        X_df = ensure_dataframe(X, copy=True)
        if X_df is None:
            logging.error("Failed to ensure DataFrame in LagFeatureTransformer.transform")
            return None # Or raise appropriate error
        
        # Create lag features
        for lag in self.lags:
            for col in X_df.columns:
                X_df[f"{col}_lag_{lag}"] = X_df[col].shift(lag)
        
        # Handle NaN values
        if self.drop_na:
            X_df = X_df.dropna()
        else:
            X_df = X_df.fillna(0)
            
        return X_df
    
    def _get_output_feature_names(self):
        """Get names of output features"""
        if self.feature_names_in_ is None:
            return None
            
        output_names = list(self.feature_names_in_)
        
        for lag in self.lags:
            for col in self.feature_names_in_:
                output_names.append(f"{col}_lag_{lag}")
                
        return output_names


class RollingFeatureTransformer:
    """Creates rolling window features (mean, std, min, max, etc.)"""
    
    def __init__(self, 
                windows: List[int] = None, 
                functions: List[str] = None,
                min_periods: int = 1):
        """
        Initialize rolling window transformer
        
        Args:
            windows: List of window sizes
            functions: List of functions to apply to windows
            min_periods: Minimum observations required for calculation
        """
        self.windows = windows if windows is not None else [5, 10, 20]
        self.functions = functions if functions is not None else ['mean', 'std']
        self.min_periods = min_periods
        self.feature_names_in_ = None
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        """Fit the transformer (just stores feature names)"""
        self.feature_names_in_ = X.columns if hasattr(X, 'columns') else None
        self.feature_names_out_ = self._get_output_feature_names()
        return self
    
    def transform(self, X):
        """Transform the input data by adding rolling window features"""
        X_df = ensure_dataframe(X, copy=True)
        if X_df is None:
            logging.error("Failed to ensure DataFrame in RollingFeatureTransformer.transform")
            return None # Or raise appropriate error
        
        # Create rolling window features
        for window in self.windows:
            for func in self.functions:
                for col in X_df.columns:
                    # Get the rolling window
                    rolling = X_df[col].rolling(window=window, min_periods=self.min_periods)
                    
                    # Apply the function
                    if func == 'mean':
                        X_df[f"{col}_rolling_{window}_{func}"] = rolling.mean()
                    elif func == 'std':
                        X_df[f"{col}_rolling_{window}_{func}"] = rolling.std()
                    elif func == 'min':
                        X_df[f"{col}_rolling_{window}_{func}"] = rolling.min()
                    elif func == 'max':
                        X_df[f"{col}_rolling_{window}_{func}"] = rolling.max()
                    elif func == 'median':
                        X_df[f"{col}_rolling_{window}_{func}"] = rolling.median()
                    elif func == 'sum':
                        X_df[f"{col}_rolling_{window}_{func}"] = rolling.sum()
                    elif func == 'var':
                        X_df[f"{col}_rolling_{window}_{func}"] = rolling.var()
                    elif func == 'skew':
                        X_df[f"{col}_rolling_{window}_{func}"] = rolling.skew()
                    elif func == 'kurt':
                        X_df[f"{col}_rolling_{window}_{func}"] = rolling.kurt()
                    # Add more functions as needed
        
        # Fill NaN values with 0
        X_df = X_df.fillna(0)
        
        return X_df
    
    def _get_output_feature_names(self):
        """Get names of output features"""
        if self.feature_names_in_ is None:
            return None
            
        output_names = list(self.feature_names_in_)
        
        for window in self.windows:
            for func in self.functions:
                for col in self.feature_names_in_:
                    output_names.append(f"{col}_rolling_{window}_{func}")
                    
        return output_names


class DiffFeatureTransformer:
    """Creates differenced versions of features"""
    
    def __init__(self, periods: List[int] = None, drop_na: bool = True):
        """
        Initialize differencing transformer
        
        Args:
            periods: List of differencing periods
            drop_na: Whether to drop rows with NaN values
        """
        self.periods = periods if periods is not None else [1]
        self.drop_na = drop_na
        self.feature_names_in_ = None
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        """Fit the transformer (just stores feature names)"""
        self.feature_names_in_ = X.columns if hasattr(X, 'columns') else None
        self.feature_names_out_ = self._get_output_feature_names()
        return self
    
    def transform(self, X):
        """Transform the input data by adding differenced features"""
        X_df = ensure_dataframe(X, copy=True)
        if X_df is None:
            logging.error("Failed to ensure DataFrame in DiffFeatureTransformer.transform")
            return None # Or raise appropriate error
        
        # Create differenced features
        for period in self.periods:
            for col in X_df.columns:
                X_df[f"{col}_diff_{period}"] = X_df[col].diff(period)
                
                # Percentage change
                if not (X_df[col] == 0).any():  # Avoid division by zero
                    X_df[f"{col}_pct_{period}"] = X_df[col].pct_change(period)
        
        # Handle NaN values
        if self.drop_na:
            X_df = X_df.dropna()
        else:
            X_df = X_df.fillna(0)
            
        return X_df
    
    def _get_output_feature_names(self):
        """Get names of output features"""
        if self.feature_names_in_ is None:
            return None
            
        output_names = list(self.feature_names_in_)
        
        for period in self.periods:
            for col in self.feature_names_in_:
                output_names.append(f"{col}_diff_{period}")
                output_names.append(f"{col}_pct_{period}")
                
        return output_names


class FeatureSelector:
    """Utility for selecting and evaluating features"""
    
    def __init__(self, feature_names: List[str] = None):
        """
        Initialize feature selector
        
        Args:
            feature_names: Initial list of feature names
        """
        self.feature_names = feature_names or []
        self.feature_importance = {}
        self.selected_features = []
        self.selector = None
    
    def calculate_feature_importance(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series,
                                   method: str = 'mutual_info') -> Dict[str, float]:
        """
        Calculate feature importance scores
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Method to use for importance calculation
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if method == 'mutual_info':
            scores = mutual_info_regression(X, y)
        elif method == 'f_regression':
            scores, _ = f_regression(X, y)
        elif method == 'correlation':
            scores = [abs(X[col].corr(y)) for col in X.columns]
        else:
            raise ValueError(f"Unsupported feature importance method: {method}")
        
        # Create feature importance dictionary
        feature_names = X.columns if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        self.feature_importance = {name: score for name, score in zip(feature_names, scores)}
        
        # Sort by importance (descending)
        return dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def select_k_best(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """
        Select k best features based on mutual information
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        self.selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_new = self.selector.fit_transform(X, y)
        
        # Get selected feature names
        feature_names = X.columns if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
        mask = self.selector.get_support()
        self.selected_features = [name for name, selected in zip(feature_names, mask) if selected]
        
        # Return as DataFrame
        X_df = ensure_dataframe(X, copy=False) # Don't need a copy here
        if X_df is not None:
            return X_df[self.selected_features]
        else:
            # If original wasn't DataFrame or convertible, return the transformed numpy array
            # Potentially wrap in DataFrame if needed, but depends on downstream usage
            logging.warning("Input to select_k_best was not a DataFrame, returning NumPy array subset.")
            # Returning the numpy array directly might be expected by sklearn pipelines
            return X_new
    
    def select_by_threshold(self, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          threshold: float = 0.01,
                          method: str = 'mutual_info') -> pd.DataFrame:
        """
        Select features by importance threshold
        
        Args:
            X: Feature matrix
            y: Target variable
            threshold: Minimum importance threshold
            method: Method to use for importance calculation
            
        Returns:
            DataFrame with selected features
        """
        # Calculate feature importance
        importance = self.calculate_feature_importance(X, y, method)
        
        # Select features above threshold
        self.selected_features = [name for name, score in importance.items() if score >= threshold]
        
        # Return selected features
        X_df = ensure_dataframe(X, copy=False)
        if X_df is not None:
            # Ensure columns exist before selecting
            missing_cols = [col for col in self.selected_features if col not in X_df.columns]
            if missing_cols:
                logging.error(f"Missing selected features in DataFrame: {missing_cols}")
                # Decide handling: return original, raise error, or return available columns?
                # Returning available columns for now
                available_features = [col for col in self.selected_features if col in X_df.columns]
                return X_df[available_features]
            return X_df[self.selected_features]
        elif isinstance(X, np.ndarray):
             # Handle NumPy array case if needed
             logging.warning("Input to select_by_threshold was a NumPy array. Index-based selection requires feature names mapping.")
             # Cannot reliably select by name from a NumPy array without metadata.
             # Returning the original array.
             return X
        else:
            logging.error(f"Unsupported input type {type(X)} for select_by_threshold.")
            return X # Or raise error
    
    def pca_transform(self, X: pd.DataFrame, n_components: int = None, variance: float = 0.95) -> pd.DataFrame:
        """
        Apply PCA transformation to reduce dimensionality
        
        Args:
            X: Feature matrix
            n_components: Number of components to keep (if None, use variance)
            variance: Minimum explained variance to maintain
            
        Returns:
            DataFrame with PCA-transformed features
        """
        # If n_components not specified, use variance
        if n_components is None:
            pca = PCA(n_components=variance, svd_solver='full')
        else:
            pca = PCA(n_components=n_components)
        
        # Fit and transform
        X_pca = pca.fit_transform(X)
        
        # Create column names
        columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        
        # Create DataFrame
        result = pd.DataFrame(X_pca, columns=columns)
        
        # Store explained variance
        result.attrs['explained_variance_ratio'] = pca.explained_variance_ratio_
        result.attrs['explained_variance'] = pca.explained_variance_
        result.attrs['n_components'] = pca.n_components_
        
        return result


class TimeSeriesFeatureGenerator:
    """Generates time series specific features for machine learning models"""
    
    def __init__(self):
        """Initialize the time series feature generator"""
        self.transformers = []
    
    def add_lag_features(self, lags: List[int] = None) -> 'TimeSeriesFeatureGenerator':
        """Add lag feature transformer"""
        self.transformers.append(LagFeatureTransformer(lags=lags))
        return self
    
    def add_rolling_features(self, 
                           windows: List[int] = None, 
                           functions: List[str] = None) -> 'TimeSeriesFeatureGenerator':
        """Add rolling window feature transformer"""
        self.transformers.append(RollingFeatureTransformer(windows=windows, functions=functions))
        return self
    
    def add_diff_features(self, periods: List[int] = None) -> 'TimeSeriesFeatureGenerator':
        """Add differencing feature transformer"""
        self.transformers.append(DiffFeatureTransformer(periods=periods))
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformers to the input data"""
        X_transformed = X.copy()
        
        for transformer in self.transformers:
            transformer.fit(X_transformed)
            X_transformed = transformer.transform(X_transformed)
        
        return X_transformed


class TechnicalIndicatorMLFeatureExtractor:
    """Extracts machine learning features from technical indicators"""
    
    def __init__(self, registry=None):
        """
        Initialize feature extractor
        
        Args:
            registry: Indicator registry to use (if None, uses global)
        """
        from analysis_engine.analysis.indicator_interface import indicator_registry as global_registry
        self.registry = registry or global_registry
        self.base_indicators = [
            "RSI", "MACD", "BollingerBands", "ATR", "OBV", "CCI",
            "StochasticOscillator", "ADX", "EMA", "SMA", "WilliamsR"
        ]
    
    def extract_features(self, 
                        data: pd.DataFrame, 
                        indicator_names: List[str] = None,
                        include_price_features: bool = True,
                        **params) -> pd.DataFrame:
        """
        Extract features from indicators
        
        Args:
            data: Price/volume data
            indicator_names: List of indicators to use (if None, uses base_indicators)
            include_price_features: Whether to include basic price features
            **params: Parameters for indicators
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=data.index)
        
        # Use provided indicators or defaults
        indicators_to_use = indicator_names or self.base_indicators
        
        # Add basic price features if requested
        if include_price_features:
            # Price change features
            if 'close' in data.columns:
                features['price_return_1d'] = data['close'].pct_change()
                features['price_log_return_1d'] = np.log(data['close'] / data['close'].shift(1))
                
                # Volatility
                if 'high' in data.columns and 'low' in data.columns:
                    features['daily_volatility'] = (data['high'] - data['low']) / data['close']
                    features['hl_to_close'] = (data['high'] - data['low']) / (data['close'] - data['low'])
                    
            # Volume features
            if 'volume' in data.columns:
                features['volume_change'] = data['volume'].pct_change()
                features['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
                
            # Time features
            if isinstance(data.index, pd.DatetimeIndex):
                features['hour'] = data.index.hour
                features['day_of_week'] = data.index.dayofweek
        
        # Calculate indicators
        for name in indicators_to_use:
            try:
                # Extract parameters for this indicator
                indicator_params = params.get(name, {})
                
                # Calculate the indicator
                result = self.registry.calculate_indicator(name, data, **indicator_params)
                
                # Add all columns from the result
                for col in result.data.columns:
                    # Skip columns that are already in the original data
                    if col not in data.columns:
                        features[f"{name}_{col}"] = result.data[col]
                        
            except Exception as e:
                logger.error(f"Error calculating indicator {name}: {str(e)}")
                continue
        
        return features
    
    def extract_and_transform(self, 
                            data: pd.DataFrame,
                            indicator_names: List[str] = None,
                            include_ts_features: bool = True,
                            **params) -> pd.DataFrame:
        """
        Extract indicator features and apply time series transformations
        
        Args:
            data: Price/volume data
            indicator_names: List of indicators to use
            include_ts_features: Whether to include time series features
            **params: Parameters for indicators
            
        Returns:
            DataFrame with extracted and transformed features
        """
        # Extract base features
        features = self.extract_features(data, indicator_names, **params)
        
        # Apply time series transformations if requested
        if include_ts_features:
            ts_generator = TimeSeriesFeatureGenerator()
            ts_generator.add_lag_features([1, 2, 3])
            ts_generator.add_rolling_features([5, 10], ['mean', 'std'])
            
            features = ts_generator.transform(features)
        
        # Drop NaN values
        features = features.dropna()
        
        return features


# Create global instances for easy import
transformer_registry = FeatureTransformerRegistry()
indicator_feature_extractor = TechnicalIndicatorMLFeatureExtractor()
