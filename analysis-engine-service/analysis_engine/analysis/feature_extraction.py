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
import statsmodels.api as sm
from scipy import stats
import re
from analysis_engine.analysis.indicator_interface import indicator_registry, CalculationMode
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class FeatureType(Enum):
    """Types of features"""
    RAW = 'raw'
    NORMALIZED = 'normalized'
    STANDARDIZED = 'standardized'
    CATEGORICAL = 'categorical'
    BINARY = 'binary'
    RELATIVE = 'relative'
    TREND = 'trend'
    MOMENTUM = 'momentum'
    CROSSOVER = 'crossover'
    VOLATILITY = 'volatility'
    DIVERGENCE = 'divergence'
    PATTERN = 'pattern'
    COMPOSITE = 'composite'
    STATISTICAL = 'statistical'
    SENTIMENT = 'sentiment'
    REGIME = 'regime'
    CUSTOM = 'custom'


class FeatureScope(Enum):
    """Scope of feature calculation"""
    POINT = 'point'
    WINDOW = 'window'
    HISTORICAL = 'historical'


@dataclass
class Feature:
    """Definition of a feature extracted from an indicator"""
    name: str
    indicator_name: str
    output_column: str
    feature_type: FeatureType
    scope: FeatureScope
    parameters: Dict[str, Any] = field(default_factory=dict)
    transform_func: Optional[Callable] = None
    description: str = ''
    source_columns: List[str] = field(default_factory=list)
    lookback_periods: int = 1
    is_sequential: bool = False
    timeframe: Optional[str] = None

    def __post_init__(self):
        """Initialize with defaults if needed"""
        if not self.description:
            self.description = (
                f'{self.feature_type.value} feature from {self.indicator_name}.{self.output_column}'
                )
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

    def register_feature(self, feature: Feature) ->None:
        """
        Register a feature with this extractor

        Args:
            feature: Feature definition
        """
        self._features[feature.name] = feature
        logger.debug(f'Registered feature: {feature.name}')

    def register_transformer(self, name: str, func: Callable) ->None:
        """
        Register a transformer function

        Args:
            name: Name of the transformer
            func: Transformer function
        """
        self._transformers[name] = func
        logger.debug(f'Registered transformer: {name}')

    @with_exception_handling
    def _parse_timeframe(self, timeframe_str: Optional[str]) ->Optional[pd.
        Timedelta]:
        """
        Parse a timeframe string (e.g., '1H', '15min', 'D') into a pandas Timedelta or offset alias.
        """
        if timeframe_str is None:
            return None
        try:
            return pd.tseries.frequencies.to_offset(timeframe_str)
        except ValueError:
            logger.warning(f'Could not parse timeframe string: {timeframe_str}'
                )
            return None

    @with_exception_handling
    def _resample_data(self, data: pd.DataFrame, target_timeframe: str,
        aggregation: Optional[Dict[str, Any]]=None) ->pd.DataFrame:
        """Resample data to the target timeframe."""
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning('Cannot resample data without a DatetimeIndex.')
            return data
        offset = self._parse_timeframe(target_timeframe)
        if offset is None:
            logger.warning(
                f'Invalid target timeframe for resampling: {target_timeframe}')
            return data
        if aggregation is None:
            aggregation = {'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'}
            aggregation = {k: v for k, v in aggregation.items() if k in
                data.columns}
        if not aggregation:
            logger.warning(
                f"No valid columns found for default OHLCV aggregation when resampling to {target_timeframe}. Using 'last'."
                )
            resampled_data = data.resample(offset).last()
        else:
            try:
                resampled_data = data.resample(offset).agg(aggregation)
            except Exception as e:
                logger.error(
                    f'Error during resampling to {target_timeframe} with aggregation {aggregation}: {e}'
                    )
                resampled_data = data.resample(offset).last()
        resampled_data = resampled_data.ffill()
        logger.info(
            f"Resampled data from {data.index.freq or 'unknown freq'} to {target_timeframe}"
            )
        return resampled_data

    @with_exception_handling
    def extract(self, data: pd.DataFrame, indicators: Dict[str, pd.
        DataFrame]=None, feature_names: List[str]=None, target_timeframe:
        Optional[str]=None) ->pd.DataFrame:
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
        if feature_names is None:
            features_to_extract = list(self._features.values())
        else:
            features_to_extract = [self._features[name] for name in
                feature_names if name in self._features]
        if not features_to_extract:
            logger.warning('No features selected for extraction')
            return pd.DataFrame(index=data.index)
        results = pd.DataFrame(index=data.index)
        base_timeframe_data = data
        resample_final_output = False
        if target_timeframe:
            if isinstance(data.index, pd.DatetimeIndex):
                logger.info(
                    f'Target timeframe specified: {target_timeframe}. Final results will be resampled if needed.'
                    )
                resample_final_output = True
            else:
                logger.warning(
                    'Data index is not DatetimeIndex, cannot resample to target timeframe.'
                    )
                target_timeframe = None
        calculated_indicators = indicators.copy(
            ) if indicators is not None else {}
        needed_indicator_defs = {}
        for feature in features_to_extract:
            if (feature.indicator_name and feature.indicator_name not in
                calculated_indicators):
                req_timeframe = feature.timeframe
                if (feature.indicator_name not in needed_indicator_defs or
                    self._is_more_granular(req_timeframe,
                    needed_indicator_defs[feature.indicator_name])):
                    needed_indicator_defs[feature.indicator_name
                        ] = req_timeframe
        for indicator_name, req_timeframe_str in needed_indicator_defs.items():
            try:
                data_for_indicator = base_timeframe_data
                if req_timeframe_str:
                    logger.info(
                        f'Resampling data to {req_timeframe_str} for indicator {indicator_name}'
                        )
                    data_for_indicator = self._resample_data(
                        base_timeframe_data, req_timeframe_str)
                    if data_for_indicator.empty:
                        logger.warning(
                            f'Resampling data to {req_timeframe_str} resulted in empty DataFrame for {indicator_name}.'
                            )
                        continue
                logger.debug(
                    f"Calculating indicator {indicator_name} on timeframe {req_timeframe_str or 'base'}"
                    )
                result = indicator_registry.calculate_indicator(indicator_name,
                    data_for_indicator)
                calculated_indicators[indicator_name] = result.data
            except Exception as e:
                logger.error(
                    f'Error calculating indicator {indicator_name} for timeframe {req_timeframe_str}: {str(e)}'
                    )
        for feature in features_to_extract:
            try:
                indicator_data = calculated_indicators.get(feature.
                    indicator_name)
                if indicator_data is None and feature.indicator_name:
                    logger.warning(
                        f'Indicator {feature.indicator_name} data not available for feature {feature.name}'
                        )
                    continue
                source_data_for_feature = None
                if (feature.feature_type == FeatureType.COMPOSITE or not
                    feature.indicator_name):
                    current_data_source = base_timeframe_data
                    if feature.timeframe:
                        current_data_source = self._resample_data(
                            base_timeframe_data, feature.timeframe)
                    combined_data = pd.DataFrame(index=current_data_source.
                        index)
                    cols_to_use = feature.source_columns or [feature.
                        output_column]
                    for col in cols_to_use:
                        if col in current_data_source.columns:
                            combined_data[col] = current_data_source[col]
                        elif indicator_data is not None and col in indicator_data.columns:
                            if indicator_data.index.equals(combined_data.index
                                ):
                                combined_data[col] = indicator_data[col]
                            else:
                                try:
                                    combined_data[col] = indicator_data[col
                                        ].reindex(combined_data.index,
                                        method='ffill')
                                    logger.debug(
                                        f'Reindexed indicator column {col} to match feature timeframe {feature.timeframe}'
                                        )
                                except Exception as reindex_err:
                                    logger.warning(
                                        f'Failed to reindex indicator column {col} for feature {feature.name}: {reindex_err}'
                                        )
                                    combined_data[col] = np.nan
                        else:
                            logger.warning(
                                f'Source column {col} not found in base data or indicator {feature.indicator_name} for feature {feature.name}'
                                )
                            combined_data[col] = np.nan
                    source_data_for_feature = combined_data
                elif indicator_data is not None:
                    if feature.output_column in indicator_data.columns:
                        source_data_for_feature = indicator_data[[feature.
                            output_column]]
                        if not source_data_for_feature.index.equals(results
                            .index):
                            try:
                                source_data_for_feature = (
                                    source_data_for_feature.reindex(results
                                    .index, method='ffill'))
                                logger.debug(
                                    f'Reindexed indicator {feature.indicator_name} to base timeframe for feature {feature.name}'
                                    )
                            except Exception as reindex_err:
                                logger.warning(
                                    f'Failed to reindex indicator {feature.indicator_name} for feature {feature.name}: {reindex_err}'
                                    )
                                source_data_for_feature = pd.DataFrame(np.
                                    nan, index=results.index, columns=[
                                    feature.output_column])
                    else:
                        logger.warning(
                            f'Column {feature.output_column} not found in indicator {feature.indicator_name} output'
                            )
                        continue
                else:
                    logger.warning(
                        f'Could not determine source data for feature {feature.name}'
                        )
                    continue
                extracted_value = None
                feature_input_data = (source_data_for_feature[feature.
                    output_column] if isinstance(source_data_for_feature,
                    pd.DataFrame) and feature.output_column in
                    source_data_for_feature.columns and feature.
                    feature_type != FeatureType.COMPOSITE else
                    source_data_for_feature)
                if feature.feature_type in [FeatureType.RELATIVE,
                    FeatureType.TREND, FeatureType.MOMENTUM, FeatureType.
                    CROSSOVER, FeatureType.VOLATILITY, FeatureType.
                    DIVERGENCE, FeatureType.PATTERN, FeatureType.COMPOSITE,
                    FeatureType.STATISTICAL, FeatureType.SENTIMENT,
                    FeatureType.REGIME]:
                    extracted_value = self._extract_advanced_feature(feature,
                        feature_input_data, base_timeframe_data,
                        calculated_indicators)
                indicator_data = calculated_indicators.get(feature.
                    indicator_name)
                if indicator_data is None:
                    logger.warning(
                        f'Indicator {feature.indicator_name} data not available for feature {feature.name}'
                        )
                    continue
                if feature.output_column not in indicator_data.columns:
                    logger.warning(
                        f'Column {feature.output_column} not found in indicator {feature.indicator_name} output'
                        )
                    continue
                raw_feature = indicator_data[feature.output_column]
                if feature.transform_func:
                    transformed = feature.transform_func(raw_feature,
                        feature.parameters)
                elif feature.feature_type == FeatureType.RAW:
                    transformed = raw_feature
                elif feature.feature_type == FeatureType.NORMALIZED:
                    min_val = feature.parameters.get('min_value',
                        raw_feature.min())
                    max_val = feature.parameters.get('max_value',
                        raw_feature.max())
                    if max_val > min_val:
                        transformed = (raw_feature - min_val) / (max_val -
                            min_val)
                    else:
                        transformed = pd.Series(0.5, index=raw_feature.index)
                elif feature.feature_type == FeatureType.STANDARDIZED:
                    mean_val = feature.parameters.get('mean', raw_feature.
                        mean())
                    std_val = feature.parameters.get('std', raw_feature.std())
                    if std_val > 0:
                        transformed = (raw_feature - mean_val) / std_val
                    else:
                        transformed = raw_feature - mean_val
                elif feature.feature_type == FeatureType.BINARY:
                    threshold = feature.parameters.get('threshold', 0)
                    transformed = (raw_feature > threshold).astype(int)
                elif feature.feature_type == FeatureType.CATEGORICAL:
                    categories = feature.parameters.get('categories', {})
                    if categories:
                        transformed = raw_feature.map(categories).fillna(
                            feature.parameters.get('default_category', np.nan))
                    else:
                        transformed = raw_feature
                else:
                    transformed = raw_feature
                results[feature.name] = transformed
            except Exception as e:
                logger.error(
                    f'Error extracting feature {feature.name}: {str(e)}')
                continue
        if resample_final_output and target_timeframe:
            logger.info(
                f'Resampling final feature DataFrame to target timeframe: {target_timeframe}'
                )
            feature_aggregation = {col: 'last' for col in results.columns}
            results = self._resample_data(results, target_timeframe,
                aggregation=feature_aggregation)
        return results

    @with_exception_handling
    def _is_more_granular(self, tf1: Optional[str], tf2: Optional[str]) ->bool:
        """Check if timeframe tf1 is more granular (shorter) than tf2."""
        if tf1 is None:
            return False
        if tf2 is None:
            return True
        try:
            delta1 = pd.Timedelta(self._parse_timeframe(tf1).freqstr)
            delta2 = pd.Timedelta(self._parse_timeframe(tf2).freqstr)
            return delta1 < delta2
        except Exception:
            return tf1 < tf2

    @with_exception_handling
    def _extract_advanced_feature(self, feature: Feature,
        feature_input_data: Union[pd.Series, pd.DataFrame], base_data: pd.
        DataFrame, indicators: Dict[str, pd.DataFrame]) ->Optional[pd.Series]:
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
        if not is_data_valid(feature_input_data, context=
            f'input data for feature {feature.name}'):
            return None
        try:
            if feature.feature_type == FeatureType.RELATIVE:
                if isinstance(feature_input_data, pd.Series):
                    return self._extract_relative_feature(feature,
                        feature_input_data, base_data)
                else:
                    logger.warning(
                        f'Invalid input type for RELATIVE feature {feature.name}'
                        )
                    return None
            elif feature.feature_type == FeatureType.TREND:
                if isinstance(feature_input_data, pd.Series):
                    return self._extract_trend_feature(feature,
                        feature_input_data)
                else:
                    logger.warning(
                        f'Invalid input type for TREND feature {feature.name}')
                    return None
            elif feature.feature_type == FeatureType.MOMENTUM:
                if isinstance(feature_input_data, pd.Series):
                    return self._extract_momentum_feature(feature,
                        feature_input_data)
                else:
                    logger.warning(
                        f'Invalid input type for MOMENTUM feature {feature.name}'
                        )
                    return None
            elif feature.feature_type == FeatureType.CROSSOVER:
                if isinstance(feature_input_data, pd.DataFrame) and len(feature
                    .source_columns) >= 2:
                    series1_name = feature.source_columns[0]
                    series2_name = feature.source_columns[1]
                    if (series1_name in feature_input_data and series2_name in
                        feature_input_data):
                        return self._extract_crossover_feature(feature,
                            feature_input_data[series1_name],
                            feature_input_data[series2_name])
                    else:
                        logger.warning(
                            f'Missing source columns in input data for CROSSOVER feature {feature.name}'
                            )
                        return None
                else:
                    logger.warning(
                        f'Invalid input type or insufficient columns for CROSSOVER feature {feature.name}'
                        )
                    return None
            elif feature.feature_type == FeatureType.VOLATILITY:
                if isinstance(feature_input_data, pd.Series):
                    return self._extract_volatility_feature(feature,
                        feature_input_data, base_data)
                else:
                    logger.warning(
                        f'Invalid input type for VOLATILITY feature {feature.name}'
                        )
                    return None
            elif feature.feature_type == FeatureType.DIVERGENCE:
                if isinstance(feature_input_data, pd.Series):
                    return self._extract_divergence_feature(feature,
                        feature_input_data, base_data)
                else:
                    logger.warning(
                        f'Invalid input type for DIVERGENCE feature {feature.name}'
                        )
                    return None
            elif feature.feature_type == FeatureType.PATTERN:
                return self._extract_pattern_feature(feature,
                    feature_input_data, base_data)
            elif feature.feature_type == FeatureType.COMPOSITE:
                if isinstance(feature_input_data, pd.DataFrame):
                    return self._extract_composite_feature(feature,
                        feature_input_data)
                else:
                    logger.warning(
                        f'Invalid input type for COMPOSITE feature {feature.name}'
                        )
                    return None
            elif feature.feature_type == FeatureType.STATISTICAL:
                if isinstance(feature_input_data, pd.Series):
                    return self._extract_statistical_feature(feature,
                        feature_input_data)
                else:
                    logger.warning(
                        f'Invalid input type for STATISTICAL feature {feature.name}'
                        )
                    return None
            elif feature.feature_type == FeatureType.SENTIMENT:
                return self._extract_sentiment_feature(feature, base_data,
                    indicators)
            elif feature.feature_type == FeatureType.REGIME:
                return self._extract_regime_feature(feature, base_data,
                    indicators)
            else:
                logger.warning(
                    f'Unsupported advanced feature type {feature.feature_type} for feature {feature.name}'
                    )
                return None
        except Exception as e:
            logger.exception(
                f'Error extracting advanced feature {feature.name}: {str(e)}')
            return None

    def _extract_relative_feature(self, feature: Feature, source_data: pd.
        Series, full_data: pd.DataFrame) ->pd.Series:
        """Extract relative feature (e.g., percent change)"""
        method = feature.parameters.get('method', 'pct_change')
        periods = feature.parameters.get('periods', 1)
        reference_col = feature.parameters.get('reference_col', None)
        if method == 'pct_change':
            return source_data.pct_change(periods=periods)
        elif method == 'diff':
            return source_data.diff(periods=periods)
        elif method == 'ratio' and reference_col is not None:
            if reference_col in full_data.columns:
                return source_data / full_data[reference_col]
            else:
                logger.warning(
                    f'Reference column {reference_col} not found in data')
                return source_data
        elif method == 'z_score':
            window = feature.parameters.get('window', 20)
            return (source_data - source_data.rolling(window=window).mean()
                ) / source_data.rolling(window=window).std()
        else:
            return source_data.pct_change()

    @with_exception_handling
    def _extract_trend_feature(self, feature: Feature, source_data: pd.Series
        ) ->pd.Series:
        """Extract trend features like slope, direction, etc."""
        method = feature.parameters.get('method', 'slope')
        window = feature.parameters.get('window', 5)
        if method == 'slope':
            result = pd.Series(index=source_data.index, dtype=float)
            for i in range(window - 1, len(source_data)):
                window_data = source_data.iloc[i - window + 1:i + 1]
                x = np.arange(window)
                if window_data.isna().any():
                    result.iloc[i] = np.nan
                else:
                    try:
                        from scipy import stats
                        slope, _, _, _, _ = stats.linregress(x, window_data)
                        result.iloc[i] = slope
                    except:
                        y = window_data.values
                        x_mean = np.mean(x)
                        y_mean = np.mean(y)
                        numerator = np.sum((x - x_mean) * (y - y_mean))
                        denominator = np.sum((x - x_mean) ** 2)
                        slope = (numerator / denominator if denominator != 
                            0 else 0)
                        result.iloc[i] = slope
            return result
        elif method == 'direction':
            return np.sign(source_data.diff(window))
        elif method == 'acceleration':
            slopes = self._extract_trend_feature(Feature(name=
                f'{feature.name}_slope', indicator_name=feature.
                indicator_name, output_column=feature.output_column,
                feature_type=FeatureType.TREND, scope=feature.scope,
                parameters={'method': 'slope', 'window': window},
                source_columns=feature.source_columns), source_data)
            return slopes.diff()
        else:
            return source_data.diff()

    def _extract_momentum_feature(self, feature: Feature, source_data: pd.
        Series) ->pd.Series:
        """Extract momentum features like ROC"""
        method = feature.parameters.get('method', 'roc')
        periods = feature.parameters.get('periods', 10)
        if method == 'roc':
            return (source_data / source_data.shift(periods) - 1) * 100
        elif method == 'momentum':
            return source_data - source_data.shift(periods)
        elif method == 'tsi':
            long_period = feature.parameters.get('long_period', 25)
            short_period = feature.parameters.get('short_period', 13)
            momentum = source_data.diff()
            smooth1 = momentum.ewm(span=long_period).mean()
            smooth2 = smooth1.ewm(span=short_period).mean()
            abs_smooth1 = momentum.abs().ewm(span=long_period).mean()
            abs_smooth2 = abs_smooth1.ewm(span=short_period).mean()
            tsi = 100 * smooth2 / abs_smooth2
            return tsi
        else:
            return (source_data / source_data.shift(1) - 1) * 100

    def _extract_crossover_feature(self, feature: Feature, series1: pd.
        Series, series2: pd.Series) ->pd.Series:
        """Extract crossover features between indicators"""
        method = feature.parameters.get('method', 'binary')
        if method == 'binary':
            current_above = series1 > series2
            prev_above = series1.shift(1) > series2.shift(1)
            crossover_up = current_above & ~prev_above
            crossover_down = ~current_above & prev_above
            result = pd.Series(0, index=series1.index)
            result[crossover_up] = 1
            result[crossover_down] = -1
            return result
        elif method == 'distance':
            return series1 - series2
        elif method == 'ratio':
            return series1 / series2
        else:
            return series1 - series2

    def _extract_volatility_feature(self, feature: Feature, source_data: pd
        .Series, full_data: pd.DataFrame) ->pd.Series:
        """Extract volatility features"""
        method = feature.parameters.get('method', 'std')
        window = feature.parameters.get('window', 20)
        if method == 'std':
            return source_data.rolling(window=window).std()
        elif method == 'atr_ratio':
            atr_col = feature.parameters.get('atr_column', None)
            close_col = feature.parameters.get('close_column', 'close')
            if (atr_col is not None and atr_col in full_data.columns and 
                close_col in full_data.columns):
                return full_data[atr_col] / full_data[close_col] * 100
            else:
                logger.warning('Missing ATR or close columns for ATR ratio')
                return source_data.rolling(window=window).std()
        elif method == 'bollinger_width':
            upper_col = feature.parameters.get('upper_column', None)
            lower_col = feature.parameters.get('lower_column', None)
            middle_col = feature.parameters.get('middle_column', None)
            if upper_col is not None and lower_col is not None:
                if (upper_col in full_data.columns and lower_col in
                    full_data.columns):
                    if (middle_col is not None and middle_col in full_data.
                        columns):
                        return (full_data[upper_col] - full_data[lower_col]
                            ) / full_data[middle_col]
                    else:
                        return full_data[upper_col] - full_data[lower_col]
                else:
                    logger.warning('Missing Bollinger Band columns')
            return source_data.rolling(window=window).std()
        else:
            return source_data.rolling(window=window).std()

    def _extract_divergence_feature(self, feature: Feature, source_data: pd
        .Series, full_data: pd.DataFrame) ->pd.Series:
        """Extract divergence features between indicator and price"""
        price_col = feature.parameters.get('price_column', 'close')
        if price_col not in full_data.columns:
            logger.warning(f'Price column {price_col} not found for divergence'
                )
            return pd.Series(index=source_data.index, dtype=float)
        method = feature.parameters.get('method', 'correlation')
        window = feature.parameters.get('window', 20)
        if method == 'correlation':
            return full_data[price_col].rolling(window).corr(source_data)
        elif method == 'slope_diff':
            price_slope = self._extract_trend_feature(Feature(name=
                'price_slope', indicator_name='', output_column='',
                feature_type=FeatureType.TREND, scope=FeatureScope.WINDOW,
                parameters={'method': 'slope', 'window': window},
                source_columns=[price_col]), full_data[price_col])
            indicator_slope = self._extract_trend_feature(Feature(name=
                'indicator_slope', indicator_name='', output_column='',
                feature_type=FeatureType.TREND, scope=FeatureScope.WINDOW,
                parameters={'method': 'slope', 'window': window},
                source_columns=[]), source_data)
            return price_slope * indicator_slope
        else:
            return full_data[price_col].rolling(window).corr(source_data)

    def _extract_pattern_feature(self, feature: Feature, feature_input_data:
        Union[pd.Series, pd.DataFrame], full_data: pd.DataFrame) ->Optional[pd
        .Series]:
        """Extract pattern features (placeholder)"""
        logger.warning(
            f'Pattern feature extraction not fully implemented for {feature.name}. Returning NaNs.'
            )
        index = feature_input_data.index if isinstance(feature_input_data,
            (pd.Series, pd.DataFrame)) else full_data.index
        return pd.Series(np.nan, index=index)

    @with_exception_handling
    def _extract_composite_feature(self, feature: Feature, combined_data:
        pd.DataFrame) ->Optional[pd.Series]:
        """Extract composite features from multiple sources (placeholder)"""
        formula = feature.parameters.get('formula')
        if formula:
            try:
                if (formula == 'macd_hist' and 'macd_line' in combined_data and
                    'macd_signal' in combined_data):
                    result = combined_data['macd_line'] - combined_data[
                        'macd_signal']
                    return result
                elif formula == 'rsi_overbought' and 'rsi' in combined_data:
                    threshold = feature.parameters.get('threshold', 70)
                    result = (combined_data['rsi'] > threshold).astype(int)
                    return result
                elif formula == 'rsi_oversold' and 'rsi' in combined_data:
                    threshold = feature.parameters.get('threshold', 30)
                    result = (combined_data['rsi'] < threshold).astype(int)
                    return result
                elif formula == 'sma_crossover' and 'sma_fast' in combined_data and 'sma_slow' in combined_data:
                    fast_above = combined_data['sma_fast'] > combined_data[
                        'sma_slow']
                    prev_fast_above = fast_above.shift(1)
                    result = (fast_above & ~prev_fast_above).astype(int)
                    return result
                else:
                    logger.warning(
                        f"Unsupported composite formula '{formula}' for feature {feature.name}"
                        )
                    return pd.Series(np.nan, index=combined_data.index)
            except Exception as e:
                logger.error(
                    f'Error evaluating composite formula for {feature.name}: {e}'
                    )
                return pd.Series(np.nan, index=combined_data.index)
        else:
            logger.warning(
                f'No formula provided for composite feature {feature.name}')
            return pd.Series(np.nan, index=combined_data.index)

    @with_exception_handling
    def _extract_statistical_feature(self, feature: Feature, source_data:
        pd.Series) ->Optional[pd.Series]:
        """Extract statistical features (e.g., skew, kurtosis)"""
        method = feature.parameters.get('method', 'skew')
        window = feature.parameters.get('window', 20)
        try:
            if method == 'skew':
                return source_data.rolling(window=window).skew()
            elif method == 'kurtosis':
                return source_data.rolling(window=window).kurt()
            else:
                logger.warning(
                    f"Unsupported statistical method '{method}' for feature {feature.name}"
                    )
                return pd.Series(np.nan, index=source_data.index)
        except Exception as e:
            logger.error(
                f'Error calculating statistical feature {feature.name}: {e}')
            return pd.Series(np.nan, index=source_data.index)

    def _extract_sentiment_feature(self, feature: Feature, base_data: pd.
        DataFrame, indicators: Dict[str, pd.DataFrame]) ->Optional[pd.Series]:
        """Extract sentiment features (placeholder)"""
        sentiment_col = feature.parameters.get('sentiment_column',
            'news_sentiment_score')
        source_df = indicators.get(feature.indicator_name, base_data)
        if sentiment_col in source_df.columns:
            sentiment_data = source_df[sentiment_col]
            transform_method = feature.parameters.get('transform', 'raw')
            if transform_method == 'sma':
                window = feature.parameters.get('window', 10)
                return sentiment_data.rolling(window=window).mean()
            elif transform_method == 'binary':
                threshold = feature.parameters.get('threshold', 0)
                return (sentiment_data > threshold).astype(int)
            else:
                return sentiment_data
        else:
            logger.warning(
                f"Sentiment column '{sentiment_col}' not found for feature {feature.name}"
                )
            return pd.Series(np.nan, index=base_data.index)

    def _extract_regime_feature(self, feature: Feature, base_data: pd.
        DataFrame, indicators: Dict[str, pd.DataFrame]) ->Optional[pd.Series]:
        """Extract market regime features (placeholder)"""
        method = feature.parameters.get('method', 'volatility_threshold')
        vol_indicator = feature.parameters.get('volatility_indicator', 'atr_14'
            )
        threshold = feature.parameters.get('threshold', 1.5)
        if method == 'volatility_threshold':
            if vol_indicator in indicators:
                vol_data = indicators[vol_indicator]
                vol_col_name = feature.parameters.get('volatility_column',
                    vol_data.columns[0])
                if vol_col_name in vol_data.columns:
                    regime = (vol_data[vol_col_name] > vol_data[
                        vol_col_name].rolling(50).mean() * threshold).astype(
                        int)
                    return regime.reindex(base_data.index, method='ffill')
                else:
                    logger.warning(
                        f"Volatility column '{vol_col_name}' not found in indicator {vol_indicator}"
                        )
                    return pd.Series(np.nan, index=base_data.index)
            else:
                logger.warning(
                    f"Volatility indicator '{vol_indicator}' not found for regime feature {feature.name}"
                    )
                return pd.Series(np.nan, index=base_data.index)
        else:
            logger.warning(
                f"Unsupported regime detection method '{method}' for feature {feature.name}"
                )
            return pd.Series(np.nan, index=base_data.index)

    def list_features(self) ->List[Dict[str, Any]]:
        """
        List all available features

        Returns:
            List of feature information dictionaries
        """
        features = [{'name': feature.name, 'indicator': feature.
            indicator_name, 'output_column': feature.output_column, 'type':
            feature.feature_type.value, 'scope': feature.scope.value,
            'parameters': feature.parameters, 'description': feature.
            description} for feature in self._features.values()]
        return sorted(features, key=lambda x: x['name'])

    @with_resilience('get_feature_definition')
    def get_feature_definition(self, name: str) ->Optional[Feature]:
        """
        Get the feature definition by name

        Args:
            name: Name of the feature

        Returns:
            The feature definition or None if not found
        """
        return self._features.get(name)


default_feature_extractor = FeatureExtractor(name='default')
