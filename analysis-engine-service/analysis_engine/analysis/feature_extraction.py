"""
Feature Extraction Framework

TODO: ARCHITECTURE - This module overlaps significantly with ml-integration-service/ml_integration_service/feature_extraction.py.
      The roles should be clarified, and consolidation should be considered.
      This FeatureStore approach seems more robust for managing features, but the ml-integration
      module contains more complex feature transformation logic (trends, patterns, etc.)
      that could potentially be integrated here as custom transformers or extractors.

This module provides a comprehensive framework for extracting machine learning features
from technical indicators with standardized normalization and transformation functions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from analysis_engine.utils.validation import is_data_valid
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
import statsmodels.api as sm # Add this import for rolling OLS
from scipy import stats # Ensure scipy.stats is imported
import re # Import re for timeframe parsing

# Local imports
from analysis_engine.analysis.indicator_interface import indicator_registry, CalculationMode

# Configure logging
logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features"""
    # Basic types
    RAW = "raw"                     # Raw indicator values
    NORMALIZED = "normalized"       # Normalized indicator value (0-1)
    STANDARDIZED = "standardized"   # Standardized indicator value (mean=0, std=1)
    CATEGORICAL = "categorical"     # Categorical feature
    BINARY = "binary"               # Binary feature (0/1)

    # Advanced types (merged from ml-integration-service)
    RELATIVE = "relative"           # Relative to a reference (e.g., % change)
    TREND = "trend"                 # Trend features (slope, etc.)
    MOMENTUM = "momentum"           # Rate of change features
    CROSSOVER = "crossover"         # Indicator crossovers
    VOLATILITY = "volatility"       # Volatility-related features
    DIVERGENCE = "divergence"       # Divergence between indicators and price
    PATTERN = "pattern"             # Pattern detection features
    COMPOSITE = "composite"         # Composite features from multiple indicators
    STATISTICAL = "statistical"     # Statistical features (e.g., skew, kurtosis)
    SENTIMENT = "sentiment"         # Sentiment-derived features (requires external data)
    REGIME = "regime"               # Market regime detection features
    CUSTOM = "custom"               # Custom feature type


class FeatureScope(Enum):
    """Scope of feature calculation"""
    POINT = "point"            # Single data point
    WINDOW = "window"          # Window of data points
    HISTORICAL = "historical"  # Full historical data


@dataclass
class Feature:
    """Definition of a feature extracted from an indicator"""
    name: str                          # Feature name
    indicator_name: str                # Source indicator name
    output_column: str                 # Column from indicator output to use
    feature_type: FeatureType          # Type of feature
    scope: FeatureScope                # Scope of feature calculation
    parameters: Dict[str, Any] = field(default_factory=dict)  # Parameters for feature extraction
    transform_func: Optional[Callable] = None  # Custom transform function
    description: str = ""              # Description of the feature
    source_columns: List[str] = field(default_factory=list)  # Additional source columns (for composite features)
    lookback_periods: int = 1          # Lookback period for time series features
    is_sequential: bool = False        # Whether feature is a sequence (for RNN/LSTM)
    timeframe: Optional[str] = None    # Optional source/target timeframe (e.g., '1H', '15min') for multi-timeframe features

    def __post_init__(self):
        """Initialize with defaults if needed"""
        if not self.description:
            self.description = f"{self.feature_type.value} feature from {self.indicator_name}.{self.output_column}"

        # Initialize source_columns list if empty
        if not self.source_columns and self.output_column:
            self.source_columns = [self.output_column]


class FeatureExtractor:
    """Base class for feature extractors"""

    def __init__(self, name: str):
        """
        Initialize the feature extractor

        Args:
            name: Name of the extractor
        """
        self.name = name
        self._features = {}
        self._transformers = {}

    def register_feature(self, feature: Feature) -> None:
        """
        Register a feature with this extractor

        Args:
            feature: Feature definition
        """
        self._features[feature.name] = feature
        logger.debug(f"Registered feature: {feature.name}")

    def register_transformer(self, name: str, func: Callable) -> None:
        """
        Register a transformer function

        Args:
            name: Name of the transformer
            func: Transformer function
        """
        self._transformers[name] = func
        logger.debug(f"Registered transformer: {name}")

    def _parse_timeframe(self, timeframe_str: Optional[str]) -> Optional[pd.Timedelta]:
        """
        Parse a timeframe string (e.g., '1H', '15min', 'D') into a pandas Timedelta or offset alias.
        """
        if timeframe_str is None:
            return None
        try:
            # Attempt direct conversion using pandas frequency strings
            return pd.tseries.frequencies.to_offset(timeframe_str)
        except ValueError:
            logger.warning(f"Could not parse timeframe string: {timeframe_str}")
            return None

    def _resample_data(self, data: pd.DataFrame, target_timeframe: str, aggregation: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Resample data to the target timeframe."""
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Cannot resample data without a DatetimeIndex.")
            return data

        offset = self._parse_timeframe(target_timeframe)
        if offset is None:
            logger.warning(f"Invalid target timeframe for resampling: {target_timeframe}")
            return data

        if aggregation is None:
            # Default aggregation: OHLCV
            aggregation = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            # Only apply aggregations for columns that exist
            aggregation = {k: v for k, v in aggregation.items() if k in data.columns}

        if not aggregation:
             logger.warning(f"No valid columns found for default OHLCV aggregation when resampling to {target_timeframe}. Using 'last'.")
             # Fallback to taking the last value for all columns if OHLCV aren't present
             resampled_data = data.resample(offset).last()
        else:
            try:
                resampled_data = data.resample(offset).agg(aggregation)
            except Exception as e:
                logger.error(f"Error during resampling to {target_timeframe} with aggregation {aggregation}: {e}")
                # Fallback to 'last' on error
                resampled_data = data.resample(offset).last()

        # Forward fill NaNs that might result from resampling periods with no trades
        resampled_data = resampled_data.ffill()
        logger.info(f"Resampled data from {data.index.freq or 'unknown freq'} to {target_timeframe}")
        return resampled_data


    def extract(self,
               data: pd.DataFrame,
               indicators: Dict[str, pd.DataFrame] = None,
               feature_names: List[str] = None,
               target_timeframe: Optional[str] = None) -> pd.DataFrame: # Added target_timeframe
        """
        Extract features from data

        Args:
            data: Input data (expected to be at the base timeframe)
            indicators: Pre-calculated indicators (optional)
            feature_names: Names of features to extract (None for all)
            target_timeframe: The target timeframe for the output features (if resampling is needed)

        Returns:
            DataFrame with extracted features, potentially resampled
        """
        # Select features to extract
        if feature_names is None:
            features_to_extract = list(self._features.values())
        else:
            features_to_extract = [self._features[name] for name in feature_names
                                 if name in self._features]

        if not features_to_extract:
            logger.warning("No features selected for extraction")
            return pd.DataFrame(index=data.index)

        # Create result DataFrame, initially with the base data index
        results = pd.DataFrame(index=data.index)
        base_timeframe_data = data # Keep original data reference

        # --- Multi-Timeframe Handling ---
        resample_final_output = False
        if target_timeframe:
            if isinstance(data.index, pd.DatetimeIndex):
                logger.info(f"Target timeframe specified: {target_timeframe}. Final results will be resampled if needed.")
                resample_final_output = True # Flag to resample at the end
            else:
                 logger.warning("Data index is not DatetimeIndex, cannot resample to target timeframe.")
                 target_timeframe = None # Disable resampling

        # Calculate indicators if not provided
        calculated_indicators = indicators.copy() if indicators is not None else {} # Use a copy
        needed_indicator_defs = {} # Store {indicator_name: (feature_timeframe or base_timeframe)}

        # Determine which indicators are needed and at which timeframes
        for feature in features_to_extract:
            if feature.indicator_name and feature.indicator_name not in calculated_indicators:
                # Determine the timeframe required for the indicator based on the feature
                # If feature.timeframe is set, use that, otherwise use the base timeframe of input data
                req_timeframe = feature.timeframe # Feature might specify its required source timeframe
                # Store the most granular timeframe needed for each indicator
                if feature.indicator_name not in needed_indicator_defs or \
                   self._is_more_granular(req_timeframe, needed_indicator_defs[feature.indicator_name]):
                    needed_indicator_defs[feature.indicator_name] = req_timeframe

        # Calculate needed indicators at their required timeframes
        for indicator_name, req_timeframe_str in needed_indicator_defs.items():
            try:
                data_for_indicator = base_timeframe_data
                if req_timeframe_str:
                    # Resample input data *before* calculating the indicator if a specific timeframe is required
                    logger.info(f"Resampling data to {req_timeframe_str} for indicator {indicator_name}")
                    data_for_indicator = self._resample_data(base_timeframe_data, req_timeframe_str)
                    if data_for_indicator.empty:
                        logger.warning(f"Resampling data to {req_timeframe_str} resulted in empty DataFrame for {indicator_name}.")
                        continue

                # Calculate indicator on potentially resampled data
                logger.debug(f"Calculating indicator {indicator_name} on timeframe {req_timeframe_str or 'base'}")
                result = indicator_registry.calculate_indicator(indicator_name, data_for_indicator)

                # Store the result, potentially indexed by timeframe if multiple versions are needed
                # For simplicity now, store only one version per indicator name (the one calculated)
                # A more complex setup might store results['indicator_name']['1H'], results['indicator_name']['15min'] etc.
                calculated_indicators[indicator_name] = result.data

            except Exception as e:
                logger.error(f"Error calculating indicator {indicator_name} for timeframe {req_timeframe_str}: {str(e)}")
                # Continue with other indicators

        # Extract each feature
        for feature in features_to_extract:
            try:
                # --- Get Data for Feature ---
                indicator_data = calculated_indicators.get(feature.indicator_name)
                if indicator_data is None and feature.indicator_name:
                     logger.warning(f"Indicator {feature.indicator_name} data not available for feature {feature.name}")
                     continue

                # Determine the source data series/dataframe for the feature
                source_data_for_feature = None
                if feature.feature_type == FeatureType.COMPOSITE or not feature.indicator_name:
                    # Composite features or features directly from input data
                    current_data_source = base_timeframe_data # Start with base data
                    if feature.timeframe: # If feature needs a specific timeframe
                        current_data_source = self._resample_data(base_timeframe_data, feature.timeframe)

                    combined_data = pd.DataFrame(index=current_data_source.index)
                    cols_to_use = feature.source_columns or [feature.output_column]
                    for col in cols_to_use:
                        if col in current_data_source.columns:
                            combined_data[col] = current_data_source[col]
                        elif indicator_data is not None and col in indicator_data.columns: # Check calculated indicator data
                             # Need to align indicator data index if it was calculated on a different timeframe
                             if indicator_data.index.equals(combined_data.index):
                                 combined_data[col] = indicator_data[col]
                             else:
                                 # Reindex/resample indicator data to match combined_data index
                                 try:
                                     # Use ffill to propagate values across the feature's timeframe
                                     combined_data[col] = indicator_data[col].reindex(combined_data.index, method='ffill')
                                     logger.debug(f"Reindexed indicator column {col} to match feature timeframe {feature.timeframe}")
                                 except Exception as reindex_err:
                                     logger.warning(f"Failed to reindex indicator column {col} for feature {feature.name}: {reindex_err}")
                                     combined_data[col] = np.nan
                        else:
                            logger.warning(f"Source column {col} not found in base data or indicator {feature.indicator_name} for feature {feature.name}")
                            combined_data[col] = np.nan
                    source_data_for_feature = combined_data # Use the combined dataframe

                elif indicator_data is not None:
                    # Feature from a single indicator
                    if feature.output_column in indicator_data.columns:
                        source_data_for_feature = indicator_data[[feature.output_column]] # Keep as DataFrame initially
                        # Align index if necessary (indicator might be on different timeframe than base data)
                        if not source_data_for_feature.index.equals(results.index):
                             try:
                                 source_data_for_feature = source_data_for_feature.reindex(results.index, method='ffill')
                                 logger.debug(f"Reindexed indicator {feature.indicator_name} to base timeframe for feature {feature.name}")
                             except Exception as reindex_err:
                                 logger.warning(f"Failed to reindex indicator {feature.indicator_name} for feature {feature.name}: {reindex_err}")
                                 source_data_for_feature = pd.DataFrame(np.nan, index=results.index, columns=[feature.output_column])

                    else:
                        logger.warning(f"Column {feature.output_column} not found in indicator {feature.indicator_name} output")
                        continue
                else:
                     logger.warning(f"Could not determine source data for feature {feature.name}")
                     continue # Skip this feature if source data is missing


                # --- Extract Feature Value ---
                extracted_value = None
                # Pass the appropriate data (Series or DataFrame) to extraction methods
                feature_input_data = source_data_for_feature[feature.output_column] if isinstance(source_data_for_feature, pd.DataFrame) and feature.output_column in source_data_for_feature.columns and feature.feature_type != FeatureType.COMPOSITE else source_data_for_feature


                # Apply different extraction based on feature type
                if feature.feature_type in [FeatureType.RELATIVE, FeatureType.TREND, FeatureType.MOMENTUM,
                                          FeatureType.CROSSOVER, FeatureType.VOLATILITY,
                                          FeatureType.DIVERGENCE, FeatureType.PATTERN, FeatureType.COMPOSITE,
                                          FeatureType.STATISTICAL, FeatureType.SENTIMENT, FeatureType.REGIME]: # Added new types
                    # These are advanced feature types requiring special handling
                    extracted_value = self._extract_advanced_feature(feature, feature_input_data, base_timeframe_data, calculated_indicators) # Pass base_timeframe_data for context if needed

                # Get indicator data for basic feature types
                indicator_data = calculated_indicators.get(feature.indicator_name)
                if indicator_data is None:
                    logger.warning(f"Indicator {feature.indicator_name} data not available for feature {feature.name}")
                    continue

                # Get column from indicator data
                if feature.output_column not in indicator_data.columns:
                    logger.warning(f"Column {feature.output_column} not found in indicator {feature.indicator_name} output")
                    continue

                # Get raw feature data
                raw_feature = indicator_data[feature.output_column]

                # Apply transformation based on feature type
                if feature.transform_func:
                    # Use custom transformation function
                    transformed = feature.transform_func(raw_feature, feature.parameters)

                elif feature.feature_type == FeatureType.RAW:
                    # Use raw values
                    transformed = raw_feature

                elif feature.feature_type == FeatureType.NORMALIZED:
                    # Normalize to 0-1 range
                    min_val = feature.parameters.get('min_value', raw_feature.min())
                    max_val = feature.parameters.get('max_value', raw_feature.max())

                    if max_val > min_val:
                        transformed = (raw_feature - min_val) / (max_val - min_val)
                    else:
                        transformed = pd.Series(0.5, index=raw_feature.index)

                elif feature.feature_type == FeatureType.STANDARDIZED:
                    # Standardize to mean=0, std=1
                    mean_val = feature.parameters.get('mean', raw_feature.mean())
                    std_val = feature.parameters.get('std', raw_feature.std())

                    if std_val > 0:
                        transformed = (raw_feature - mean_val) / std_val
                    else:
                        transformed = raw_feature - mean_val

                elif feature.feature_type == FeatureType.BINARY:
                    # Create binary feature
                    threshold = feature.parameters.get('threshold', 0)
                    transformed = (raw_feature > threshold).astype(int)

                elif feature.feature_type == FeatureType.CATEGORICAL:
                    # Create categorical feature
                    categories = feature.parameters.get('categories', {})
                    if categories:
                        transformed = raw_feature.map(categories).fillna(feature.parameters.get('default_category', np.nan)) # Handle unmapped values
                    else:
                        transformed = raw_feature # Or apply default categorization if needed

                else:
                    # Default to raw values
                    transformed = raw_feature

                # Add to results DataFrame
                results[feature.name] = transformed

            except Exception as e:
                logger.error(f"Error extracting feature {feature.name}: {str(e)}")
                continue

        # --- Final Resampling ---
        if resample_final_output and target_timeframe:
            logger.info(f"Resampling final feature DataFrame to target timeframe: {target_timeframe}")
            # Define aggregation rules for features - default to 'last' or 'mean'?
            # This needs careful consideration based on feature types.
            # Using 'last' as a general default for now.
            feature_aggregation = {col: 'last' for col in results.columns}
            # Example: override for specific types
            # for feature in features_to_extract:
            #     if feature.feature_type == FeatureType.VOLATILITY:
            #         feature_aggregation[feature.name] = 'mean' # Average volatility over the target period

            results = self._resample_data(results, target_timeframe, aggregation=feature_aggregation)


        return results

    def _is_more_granular(self, tf1: Optional[str], tf2: Optional[str]) -> bool:
        """Check if timeframe tf1 is more granular (shorter) than tf2."""
        if tf1 is None: return False # Base timeframe is not more granular than a specific one
        if tf2 is None: return True  # A specific timeframe is more granular than base

        try:
            delta1 = pd.Timedelta(self._parse_timeframe(tf1).freqstr)
            delta2 = pd.Timedelta(self._parse_timeframe(tf2).freqstr)
            return delta1 < delta2
        except Exception:
            # Fallback comparison if parsing fails
            return tf1 < tf2 # Simple string comparison as fallback

    def _extract_advanced_feature(self,
                                  feature: Feature,
                                  feature_input_data: Union[pd.Series, pd.DataFrame], # Can be Series or DataFrame
                                  base_data: pd.DataFrame, # Original base timeframe data
                                  indicators: Dict[str, pd.DataFrame]) -> Optional[pd.Series]:
        """
        Extract advanced feature types from data

        Args:
            feature: Feature definition
            feature_input_data: Input data for the feature (can be Series or DataFrame)
            base_data: Original base timeframe data
            indicators: Pre-calculated indicators

        Returns:
            Series with extracted feature values or None if extraction fails
        """
        # Initial checks
        if not is_data_valid(feature_input_data, context=f"input data for feature {feature.name}"):
            return None

        try:
            # Route to specific extraction methods based on type
            if feature.feature_type == FeatureType.RELATIVE:
                # Expects Series input
                if isinstance(feature_input_data, pd.Series):
                     return self._extract_relative_feature(feature, feature_input_data, base_data)
                else: logger.warning(f"Invalid input type for RELATIVE feature {feature.name}"); return None

            elif feature.feature_type == FeatureType.TREND:
                 # Expects Series input
                if isinstance(feature_input_data, pd.Series):
                    return self._extract_trend_feature(feature, feature_input_data)
                else: logger.warning(f"Invalid input type for TREND feature {feature.name}"); return None

            elif feature.feature_type == FeatureType.MOMENTUM:
                 # Expects Series input
                if isinstance(feature_input_data, pd.Series):
                    return self._extract_momentum_feature(feature, feature_input_data)
                else: logger.warning(f"Invalid input type for MOMENTUM feature {feature.name}"); return None

            elif feature.feature_type == FeatureType.CROSSOVER:
                # Expects DataFrame with 2+ columns specified in source_columns
                if isinstance(feature_input_data, pd.DataFrame) and len(feature.source_columns) >= 2:
                    series1_name = feature.source_columns[0]
                    series2_name = feature.source_columns[1]
                    if series1_name in feature_input_data and series2_name in feature_input_data:
                        return self._extract_crossover_feature(feature, feature_input_data[series1_name], feature_input_data[series2_name])
                    else: logger.warning(f"Missing source columns in input data for CROSSOVER feature {feature.name}"); return None
                else: logger.warning(f"Invalid input type or insufficient columns for CROSSOVER feature {feature.name}"); return None

            elif feature.feature_type == FeatureType.VOLATILITY:
                 # Expects Series input, but might need full_data context
                if isinstance(feature_input_data, pd.Series):
                    # Pass base_data which contains OHLCV etc.
                    return self._extract_volatility_feature(feature, feature_input_data, base_data)
                else: logger.warning(f"Invalid input type for VOLATILITY feature {feature.name}"); return None

            elif feature.feature_type == FeatureType.DIVERGENCE:
                 # Expects Series input, needs full_data context for price
                if isinstance(feature_input_data, pd.Series):
                     # Pass base_data which contains price column
                    return self._extract_divergence_feature(feature, feature_input_data, base_data)
                else: logger.warning(f"Invalid input type for DIVERGENCE feature {feature.name}"); return None

            elif feature.feature_type == FeatureType.PATTERN:
                 # Expects Series or DataFrame input depending on pattern
                 # Pass base_data for context if needed (e.g., OHLC for candlestick patterns)
                return self._extract_pattern_feature(feature, feature_input_data, base_data) # Modified signature

            elif feature.feature_type == FeatureType.COMPOSITE:
                 # Expects DataFrame input
                if isinstance(feature_input_data, pd.DataFrame):
                    return self._extract_composite_feature(feature, feature_input_data)
                else: logger.warning(f"Invalid input type for COMPOSITE feature {feature.name}"); return None

            # --- Placeholders for New Advanced Types ---
            elif feature.feature_type == FeatureType.STATISTICAL:
                # Expects Series input
                if isinstance(feature_input_data, pd.Series):
                    return self._extract_statistical_feature(feature, feature_input_data)
                else: logger.warning(f"Invalid input type for STATISTICAL feature {feature.name}"); return None

            elif feature.feature_type == FeatureType.SENTIMENT:
                # Needs external sentiment data source, potentially passed via 'indicators' or 'base_data'
                return self._extract_sentiment_feature(feature, base_data, indicators) # Pass context

            elif feature.feature_type == FeatureType.REGIME:
                # Might use price, volatility, or other indicators
                return self._extract_regime_feature(feature, base_data, indicators) # Pass context

            else:
                logger.warning(f"Unsupported advanced feature type {feature.feature_type} for feature {feature.name}")
                return None

        except Exception as e:
            logger.exception(f"Error extracting advanced feature {feature.name}: {str(e)}") # Use exception logging
            return None

    def _extract_relative_feature(
        self,
        feature: Feature,
        source_data: pd.Series,
        full_data: pd.DataFrame
    ) -> pd.Series:
        """Extract relative feature (e.g., percent change)"""
        # Get parameters
        method = feature.parameters.get("method", "pct_change")
        periods = feature.parameters.get("periods", 1)
        reference_col = feature.parameters.get("reference_col", None)

        if method == "pct_change":
            # Percent change from n periods ago
            return source_data.pct_change(periods=periods)

        elif method == "diff":
            # Absolute difference from n periods ago
            return source_data.diff(periods=periods)

        elif method == "ratio" and reference_col is not None:
            # Ratio to another column
            if reference_col in full_data.columns:
                return source_data / full_data[reference_col]
            else:
                logger.warning(f"Reference column {reference_col} not found in data")
                return source_data

        elif method == "z_score":
            # Rolling z-score
            window = feature.parameters.get("window", 20)
            return (source_data - source_data.rolling(window=window).mean()) / source_data.rolling(window=window).std()

        else:
            # Default to percent change
            return source_data.pct_change()

    def _extract_trend_feature(
        self,
        feature: Feature,
        source_data: pd.Series
    ) -> pd.Series:
        """Extract trend features like slope, direction, etc."""
        # Get parameters
        method = feature.parameters.get("method", "slope")
        window = feature.parameters.get("window", 5)

        if method == "slope":
            # Calculate slope using linear regression on rolling window
            result = pd.Series(index=source_data.index, dtype=float)

            for i in range(window - 1, len(source_data)):
                window_data = source_data.iloc[i-window+1:i+1]
                x = np.arange(window)
                if window_data.isna().any():
                    result.iloc[i] = np.nan
                else:
                    try:
                        from scipy import stats
                        slope, _, _, _, _ = stats.linregress(x, window_data)
                        result.iloc[i] = slope
                    except:
                        # Fallback if scipy not available
                        y = window_data.values
                        x_mean = np.mean(x)
                        y_mean = np.mean(y)
                        numerator = np.sum((x - x_mean) * (y - y_mean))
                        denominator = np.sum((x - x_mean) ** 2)
                        slope = numerator / denominator if denominator != 0 else 0
                        result.iloc[i] = slope

            return result

        elif method == "direction":
            # -1 for downtrend, 0 for sideways, 1 for uptrend
            return np.sign(source_data.diff(window))

        elif method == "acceleration":
            # Acceleration (change in slope)
            slopes = self._extract_trend_feature(
                Feature(
                    name=f"{feature.name}_slope",
                    indicator_name=feature.indicator_name,
                    output_column=feature.output_column,
                    feature_type=FeatureType.TREND,
                    scope=feature.scope,
                    parameters={"method": "slope", "window": window},
                    source_columns=feature.source_columns
                ),
                source_data
            )
            return slopes.diff()

        else:
            # Default to simple difference as trend indicator
            return source_data.diff()

    def _extract_momentum_feature(
        self,
        feature: Feature,
        source_data: pd.Series
    ) -> pd.Series:
        """Extract momentum features like ROC"""
        # Get parameters
        method = feature.parameters.get("method", "roc")
        periods = feature.parameters.get("periods", 10)

        if method == "roc":
            # Rate of change
            return (source_data / source_data.shift(periods) - 1) * 100

        elif method == "momentum":
            # Simple momentum (current - n periods ago)
            return source_data - source_data.shift(periods)

        elif method == "tsi":
            # True Strength Index
            long_period = feature.parameters.get("long_period", 25)
            short_period = feature.parameters.get("short_period", 13)

            momentum = source_data.diff()
            # Double EMA smoothing of momentum
            smooth1 = momentum.ewm(span=long_period).mean()
            smooth2 = smooth1.ewm(span=short_period).mean()
            # Double EMA smoothing of absolute momentum
            abs_smooth1 = momentum.abs().ewm(span=long_period).mean()
            abs_smooth2 = abs_smooth1.ewm(span=short_period).mean()

            # TSI calculation
            tsi = 100 * smooth2 / abs_smooth2
            return tsi

        else:
            # Default to simple ROC
            return (source_data / source_data.shift(1) - 1) * 100

    def _extract_crossover_feature(
        self,
        feature: Feature,
        series1: pd.Series,
        series2: pd.Series
    ) -> pd.Series:
        """Extract crossover features between indicators"""
        # Get parameters
        method = feature.parameters.get("method", "binary")

        if method == "binary":
            # Binary crossover indicator (1 for above, -1 for below)
            current_above = series1 > series2
            prev_above = series1.shift(1) > series2.shift(1)

            # Crossover happens when current and previous states differ
            crossover_up = current_above & ~prev_above  # Crossed above
            crossover_down = ~current_above & prev_above  # Crossed below

            result = pd.Series(0, index=series1.index)
            result[crossover_up] = 1
            result[crossover_down] = -1
            return result

        elif method == "distance":
            # Distance between indicators
            return series1 - series2

        elif method == "ratio":
            # Ratio between indicators
            return series1 / series2

        else:
            # Default to distance
            return series1 - series2

    def _extract_volatility_feature(
        self,
        feature: Feature,
        source_data: pd.Series,
        full_data: pd.DataFrame
    ) -> pd.Series:
        """Extract volatility features"""
        # Get parameters
        method = feature.parameters.get("method", "std")
        window = feature.parameters.get("window", 20)

        if method == "std":
            # Standard deviation
            return source_data.rolling(window=window).std()

        elif method == "atr_ratio":
            # Need ATR and close price
            atr_col = feature.parameters.get("atr_column", None)
            close_col = feature.parameters.get("close_column", "close")

            if atr_col is not None and atr_col in full_data.columns and close_col in full_data.columns:
                # ATR as percentage of price
                return full_data[atr_col] / full_data[close_col] * 100
            else:
                logger.warning("Missing ATR or close columns for ATR ratio")
                return source_data.rolling(window=window).std()

        elif method == "bollinger_width":
            # Need upper and lower bands
            upper_col = feature.parameters.get("upper_column", None)
            lower_col = feature.parameters.get("lower_column", None)
            middle_col = feature.parameters.get("middle_column", None)

            if upper_col is not None and lower_col is not None:
                if upper_col in full_data.columns and lower_col in full_data.columns:
                    # Check if we have a middle band
                    if middle_col is not None and middle_col in full_data.columns:
                        # (upper - lower) / middle
                        return (full_data[upper_col] - full_data[lower_col]) / full_data[middle_col]
                    else:
                        # Just upper - lower
                        return full_data[upper_col] - full_data[lower_col]
                else:
                    logger.warning("Missing Bollinger Band columns")

            # Default to standard deviation
            return source_data.rolling(window=window).std()

        else:
            # Default to rolling standard deviation
            return source_data.rolling(window=window).std()

    def _extract_divergence_feature(
        self,
        feature: Feature,
        source_data: pd.Series,
        full_data: pd.DataFrame
    ) -> pd.Series:
        """Extract divergence features between indicator and price"""
        # Get price column
        price_col = feature.parameters.get("price_column", "close")
        if price_col not in full_data.columns:
            logger.warning(f"Price column {price_col} not found for divergence")
            return pd.Series(index=source_data.index, dtype=float)

        # Get parameters
        method = feature.parameters.get("method", "correlation")
        window = feature.parameters.get("window", 20)

        if method == "correlation":
            # Rolling correlation
            return full_data[price_col].rolling(window).corr(source_data)

        elif method == "slope_diff":
            # Difference in slope direction
            price_slope = self._extract_trend_feature(
                Feature(
                    name="price_slope",
                    indicator_name="",
                    output_column="",
                    feature_type=FeatureType.TREND,
                    scope=FeatureScope.WINDOW,
                    parameters={"method": "slope", "window": window},
                    source_columns=[price_col]
                ),
                full_data[price_col]
            )

            indicator_slope = self._extract_trend_feature(
                Feature(
                    name="indicator_slope",
                    indicator_name="",
                    output_column="",
                    feature_type=FeatureType.TREND,
                    scope=FeatureScope.WINDOW,
                    parameters={"method": "slope", "window": window},
                    source_columns=[]
                ),
                source_data
            )

            # Opposite signs indicate divergence
            return price_slope * indicator_slope

        else:
            # Default to correlation
            return full_data[price_col].rolling(window).corr(source_data)

    def _extract_pattern_feature(self,
                                 feature: Feature,
                                 feature_input_data: Union[pd.Series, pd.DataFrame], # Can be Series or DataFrame
                                 full_data: pd.DataFrame) -> Optional[pd.Series]: # Added full_data
        """Extract pattern features (placeholder)"""
        # Example: Candlestick patterns would need OHLC from full_data
        # Example: Chart patterns (head & shoulders) are complex and might need dedicated libraries
        logger.warning(f"Pattern feature extraction not fully implemented for {feature.name}. Returning NaNs.")
        index = feature_input_data.index if isinstance(feature_input_data, (pd.Series, pd.DataFrame)) else full_data.index
        return pd.Series(np.nan, index=index)

    def _extract_composite_feature(self, feature: Feature, combined_data: pd.DataFrame) -> Optional[pd.Series]:
        """Extract composite features from multiple sources (placeholder)"""
        # Example: MACD Histogram (MACD line - Signal line)
        # Example: RSI > 70 AND ADX > 25
        # Requires custom logic based on feature.parameters['formula'] or similar
        formula = feature.parameters.get("formula")
        if formula:
            try:
                # VERY UNSAFE - Use a safer evaluation method like numexpr or asteval in production
                # This is just a conceptual example
                # result = combined_data.eval(formula)
                # Example: Explicit MACD Histogram
                if 'macd_line' in combined_data and 'macd_signal' in combined_data and formula == 'macd_hist':
                     result = combined_data['macd_line'] - combined_data['macd_signal']
                     return result
                else:
                     logger.warning(f"Unsupported composite formula or missing columns for {feature.name}")
                     return pd.Series(np.nan, index=combined_data.index)

            except Exception as e:
                logger.error(f"Error evaluating composite formula for {feature.name}: {e}")
                return pd.Series(np.nan, index=combined_data.index)
        else:
            logger.warning(f"No formula provided for composite feature {feature.name}")
            return pd.Series(np.nan, index=combined_data.index)


    # --- Placeholder Methods for New Advanced Types ---

    def _extract_statistical_feature(self, feature: Feature, source_data: pd.Series) -> Optional[pd.Series]:
        """Extract statistical features (e.g., skew, kurtosis)"""
        method = feature.parameters.get("method", "skew")
        window = feature.parameters.get("window", 20)

        try:
            if method == "skew":
                return source_data.rolling(window=window).skew()
            elif method == "kurtosis":
                return source_data.rolling(window=window).kurt()
            # Add other statistical measures (e.g., variance, median absolute deviation)
            else:
                logger.warning(f"Unsupported statistical method '{method}' for feature {feature.name}")
                return pd.Series(np.nan, index=source_data.index)
        except Exception as e:
            logger.error(f"Error calculating statistical feature {feature.name}: {e}")
            return pd.Series(np.nan, index=source_data.index)

    def _extract_sentiment_feature(self, feature: Feature, base_data: pd.DataFrame, indicators: Dict[str, pd.DataFrame]) -> Optional[pd.Series]:
        """Extract sentiment features (placeholder)"""
        # This would require integrating an external sentiment data source.
        # The sentiment data could be passed in 'base_data' or 'indicators'.
        sentiment_col = feature.parameters.get("sentiment_column", "news_sentiment_score")
        source_df = indicators.get(feature.indicator_name, base_data) # Check indicators first, then base_data

        if sentiment_col in source_df.columns:
            # Apply transformations (e.g., smoothing, thresholding)
            sentiment_data = source_df[sentiment_col]
            transform_method = feature.parameters.get("transform", "raw")
            if transform_method == "sma":
                window = feature.parameters.get("window", 10)
                return sentiment_data.rolling(window=window).mean()
            elif transform_method == "binary":
                threshold = feature.parameters.get("threshold", 0)
                return (sentiment_data > threshold).astype(int)
            else: # raw
                return sentiment_data
        else:
            logger.warning(f"Sentiment column '{sentiment_col}' not found for feature {feature.name}")
            return pd.Series(np.nan, index=base_data.index)


    def _extract_regime_feature(self, feature: Feature, base_data: pd.DataFrame, indicators: Dict[str, pd.DataFrame]) -> Optional[pd.Series]:
        """Extract market regime features (placeholder)"""
        # Example: Volatility-based regime (high/low based on ATR or StdDev)
        # Example: Trend-based regime (trending/ranging based on ADX or MA slope)
        method = feature.parameters.get("method", "volatility_threshold")
        vol_indicator = feature.parameters.get("volatility_indicator", "atr_14") # Example dependency
        threshold = feature.parameters.get("threshold", 1.5) # Example threshold

        if method == "volatility_threshold":
            if vol_indicator in indicators:
                vol_data = indicators[vol_indicator]
                # Assuming the indicator output has a standard column name like 'atr' or 'stddev'
                vol_col_name = feature.parameters.get("volatility_column", vol_data.columns[0]) # Use first column if not specified
                if vol_col_name in vol_data.columns:
                    # Simple binary regime: 1 for high vol, 0 for low vol
                    regime = (vol_data[vol_col_name] > vol_data[vol_col_name].rolling(50).mean() * threshold).astype(int) # Compare to rolling mean * threshold
                    return regime.reindex(base_data.index, method='ffill') # Align index
                else:
                    logger.warning(f"Volatility column '{vol_col_name}' not found in indicator {vol_indicator}")
                    return pd.Series(np.nan, index=base_data.index)
            else:
                logger.warning(f"Volatility indicator '{vol_indicator}' not found for regime feature {feature.name}")
                return pd.Series(np.nan, index=base_data.index)
        else:
            logger.warning(f"Unsupported regime detection method '{method}' for feature {feature.name}")
            return pd.Series(np.nan, index=base_data.index)


    def list_features(self) -> List[Dict[str, Any]]:
        """
        List all available features

        Returns:
            List of feature information dictionaries
        """
        features = [
            {
                'name': feature.name,
                'indicator': feature.indicator_name,
                'output_column': feature.output_column,
                'type': feature.feature_type.value,
                'scope': feature.scope.value,
                'parameters': feature.parameters,
                'description': feature.description
            }
            for feature in self._features.values()
        ]

        return sorted(features, key=lambda x: x['name'])

    def get_feature_definition(self, name: str) -> Optional[Feature]:
        """
        Get the feature definition by name

        Args:
            name: Name of the feature

        Returns:
            The feature definition or None if not found
        """
        return self._features.get(name)

# Default feature extractor instance
default_feature_extractor = FeatureExtractor(name="default")

# Example feature registrations (can be loaded from config)
# default_feature_extractor.register_feature(
#     Feature(name=\"rsi_raw\", indicator_name=\"rsi_14\", output_column=\"rsi\", feature_type=FeatureType.RAW, scope=FeatureScope.POINT)
# )
# default_feature_extractor.register_feature(
#     Feature(name=\"rsi_normalized\", indicator_name=\"rsi_14\", output_column=\"rsi\", feature_type=FeatureType.NORMALIZED, scope=FeatureScope.HISTORICAL, parameters={'min_value': 0, 'max_value': 100})
# )
# default_feature_extractor.register_feature(
#     Feature(name=\"macd_hist_trend\", indicator_name=\"macd_12_26_9\", output_column=\"macd_histogram\", feature_type=FeatureType.TREND, scope=FeatureScope.WINDOW, parameters={'method': 'slope', 'window': 5})
# )
# default_feature_extractor.register_feature(
#     Feature(name=\"ma_crossover\", indicator_name=\"\", output_column=\"\", feature_type=FeatureType.CROSSOVER, scope=FeatureScope.POINT, source_columns=['sma_10', 'sma_50'], parameters={'method': 'binary'})
# )
# default_feature_extractor.register_feature(
#     Feature(name=\"close_skewness\", indicator_name=\"\", output_column=\"close\", feature_type=FeatureType.STATISTICAL, scope=FeatureScope.WINDOW, parameters={'method': 'skew', 'window': 20})
# )
# default_feature_extractor.register_feature(
#     Feature(name=\"high_vol_regime\", indicator_name=\"atr_14\", output_column=\"atr\", feature_type=FeatureType.REGIME, scope=FeatureScope.POINT, parameters={'method': 'volatility_threshold', 'threshold': 1.5, 'volatility_column': 'atr'})
# )
