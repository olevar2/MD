"""
ML Integration: Feature Extraction Module

This module provides tools for extracting ML-ready features from technical indicators,
transforming raw indicator values into features that are optimized for machine learning models.

from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

NOTE: This implementation uses the legacy feature extraction system within this service.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Union, Optional, Any, Callable
from enum import Enum
import logging
from datetime import datetime, timedelta
from scipy import stats
import pytz
import warnings
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from common_lib.ml.feature_interfaces import FeatureType, FeatureScope, SelectionMethod, IFeatureProvider
from ml_integration_service.adapters.feature_store_adapter import FeatureProviderAdapter
from ml_integration_service.caching.feature_vector_cache import cache_feature_vector
logger = logging.getLogger(__name__)
FEATURE_TYPE_MAPPING = {'raw': FeatureType.RAW, 'normalized': FeatureType.
    NORMALIZED, 'relative': FeatureType.RAW, 'trend': FeatureType.TREND,
    'momentum': FeatureType.RAW, 'crossover': FeatureType.CROSSOVER,
    'volatility': FeatureType.RAW, 'divergence': FeatureType.DIVERGENCE,
    'pattern': FeatureType.RAW, 'composite': FeatureType.CUSTOM}


class LegacyFeatureType(Enum):
    """Types of features that can be extracted from indicators (legacy)"""
    RAW = 'raw'
    NORMALIZED = 'normalized'
    RELATIVE = 'relative'
    TREND = 'trend'
    MOMENTUM = 'momentum'
    CROSSOVER = 'crossover'
    VOLATILITY = 'volatility'
    DIVERGENCE = 'divergence'
    PATTERN = 'pattern'
    COMPOSITE = 'composite'


@dataclass
class FeatureDefinition:
    """
    Definition of a feature to extract from technical indicators
    """
    name: str
    source_columns: List[str]
    feature_type: Union[FeatureType, LegacyFeatureType]
    params: Dict[str, Any] = None
    scope: FeatureScope = FeatureScope.INDICATOR
    lookback_periods: int = 1
    is_sequential: bool = False
    description: str = ''

    def __post_init__(self):
        """Initialize default values after initialization"""
        if self.params is None:
            self.params = {}
        if not self.description:
            feature_type_value = self.feature_type.value if isinstance(self
                .feature_type, Enum) else str(self.feature_type)
            self.description = (
                f"{feature_type_value.capitalize()} feature from {', '.join(self.source_columns)}"
                )
        if isinstance(self.feature_type, LegacyFeatureType):
            legacy_value = self.feature_type.value
            if legacy_value in FEATURE_TYPE_MAPPING:
                self.feature_type = FEATURE_TYPE_MAPPING[legacy_value]


class FeatureExtractor:
    """
    Extract machine learning features from technical indicators
    """

    def __init__(self, timezone: str='UTC'):
        """
        Initialize feature extractor

        Args:
            timezone: Timezone for time-based features
        """
        self.timezone = pytz.timezone(timezone)
        self.scalers: Dict[str, Any] = {}
        self.feature_history: Dict[str, pd.Series] = {}
        self.logger = logging.getLogger(__name__)

    @cache_feature_vector(ttl=1800)
    @with_exception_handling
    def extract_features(self, model_name: str, symbol: str, timeframe: str,
        data: pd.DataFrame, feature_definitions: List[FeatureDefinition],
        fit_scalers: bool=False) ->pd.DataFrame:
        """
        Extract features from technical indicator data

        Args:
            model_name: Name of the model for which features are being extracted
            symbol: Trading symbol
            timeframe: Chart timeframe
            data: DataFrame with indicator data
            feature_definitions: List of feature definitions
            fit_scalers: Whether to fit or use pre-fitted scalers

        Returns:
            DataFrame with extracted features
        """
        if data.empty:
            return pd.DataFrame()
        all_required_columns = set()
        for feature_def in feature_definitions:
            all_required_columns.update(feature_def.source_columns)
        missing_columns = all_required_columns - set(data.columns)
        if missing_columns:
            self.logger.warning(
                f'Missing columns for feature extraction: {missing_columns}')
            feature_definitions = [fd for fd in feature_definitions if not
                any(col in missing_columns for col in fd.source_columns)]
        if not feature_definitions:
            self.logger.warning(
                'No valid feature definitions after filtering missing columns')
            return pd.DataFrame(index=data.index)
        result = pd.DataFrame(index=data.index)
        for feature_def in feature_definitions:
            try:
                feature_series = self._extract_single_feature(data,
                    feature_def, fit_scalers)
                if feature_def.is_sequential:
                    for i in range(feature_def.lookback_periods):
                        col_name = f'{feature_def.name}_{i}'
                        result[col_name] = feature_series.shift(i)
                else:
                    result[feature_def.name] = feature_series
            except Exception as e:
                self.logger.error(
                    f'Error extracting feature {feature_def.name}: {str(e)}')
                result[feature_def.name] = np.nan
        return result

    @staticmethod
    def create(timezone: str='UTC'):
        """
        Factory method to create a feature extractor

        Args:
            timezone: Timezone for time-based features

        Returns:
            Feature extractor instance (always the legacy FeatureExtractor)
        """
        logger.info('Using legacy FeatureExtractor implementation.')
        return FeatureExtractor(timezone=timezone)

    def _extract_single_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition, fit_scalers: bool) ->pd.Series:
        """
        Extract a single feature based on its definition

        Args:
            data: DataFrame with indicator data
            feature_def: Feature definition
            fit_scalers: Whether to fit scalers

        Returns:
            Series with extracted feature
        """
        if feature_def.feature_type == FeatureType.RAW:
            return self._extract_raw_feature(data, feature_def)
        elif feature_def.feature_type == FeatureType.NORMALIZED:
            return self._extract_normalized_feature(data, feature_def,
                fit_scalers)
        elif feature_def.feature_type == FeatureType.RELATIVE:
            return self._extract_relative_feature(data, feature_def)
        elif feature_def.feature_type == FeatureType.TREND:
            return self._extract_trend_feature(data, feature_def)
        elif feature_def.feature_type == FeatureType.MOMENTUM:
            return self._extract_momentum_feature(data, feature_def)
        elif feature_def.feature_type == FeatureType.CROSSOVER:
            return self._extract_crossover_feature(data, feature_def)
        elif feature_def.feature_type == FeatureType.VOLATILITY:
            return self._extract_volatility_feature(data, feature_def)
        elif feature_def.feature_type == FeatureType.DIVERGENCE:
            return self._extract_divergence_feature(data, feature_def)
        elif feature_def.feature_type == FeatureType.PATTERN:
            return self._extract_pattern_feature(data, feature_def)
        elif feature_def.feature_type == FeatureType.COMPOSITE:
            return self._extract_composite_feature(data, feature_def)
        else:
            raise ValueError(
                f'Unsupported feature type: {feature_def.feature_type}')

    def _extract_raw_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition) ->pd.Series:
        """Extract raw value feature"""
        method = feature_def.params.get('method', 'first')
        if method == 'first' and len(feature_def.source_columns) >= 1:
            return data[feature_def.source_columns[0]].copy()
        elif method == 'mean':
            return data[feature_def.source_columns].mean(axis=1)
        elif method == 'sum':
            return data[feature_def.source_columns].sum(axis=1)
        elif method == 'min':
            return data[feature_def.source_columns].min(axis=1)
        elif method == 'max':
            return data[feature_def.source_columns].max(axis=1)
        else:
            return data[feature_def.source_columns[0]].copy()

    def _extract_normalized_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition, fit_scalers: bool) ->pd.Series:
        """Extract normalized/scaled feature"""
        raw_values = self._extract_raw_feature(data, feature_def)
        scaler_type = feature_def.params.get('scaler', 'standard')
        feature_name = feature_def.name
        if fit_scalers or feature_name not in self.scalers:
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f'Unsupported scaler type: {scaler_type}')
            reshaped_values = raw_values.values.reshape(-1, 1)
            scaler.fit(reshaped_values)
            self.scalers[feature_name] = scaler
        else:
            scaler = self.scalers[feature_name]
        reshaped_values = raw_values.values.reshape(-1, 1)
        normalized_values = scaler.transform(reshaped_values).flatten()
        return pd.Series(normalized_values, index=raw_values.index)

    def _extract_relative_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition) ->pd.Series:
        """Extract relative feature (e.g., percent change)"""
        raw_values = self._extract_raw_feature(data, feature_def)
        method = feature_def.params.get('method', 'pct_change')
        periods = feature_def.params.get('periods', 1)
        reference_col = feature_def.params.get('reference_col', None)
        if method == 'pct_change':
            return raw_values.pct_change(periods=periods)
        elif method == 'diff':
            return raw_values.diff(periods=periods)
        elif method == 'ratio' and reference_col is not None:
            if reference_col in data.columns:
                return raw_values / data[reference_col]
            else:
                self.logger.warning(
                    f'Reference column {reference_col} not found in data')
                return raw_values
        elif method == 'z_score':
            window = feature_def.params.get('window', 20)
            return (raw_values - raw_values.rolling(window=window).mean()
                ) / raw_values.rolling(window=window).std()
        else:
            return raw_values.pct_change()

    def _extract_trend_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition) ->pd.Series:
        """Extract trend features like slope, direction, etc."""
        raw_values = self._extract_raw_feature(data, feature_def)
        method = feature_def.params.get('method', 'slope')
        window = feature_def.params.get('window', 5)
        if method == 'slope':
            result = pd.Series(index=raw_values.index, dtype=float)
            for i in range(window - 1, len(raw_values)):
                window_data = raw_values.iloc[i - window + 1:i + 1]
                x = np.arange(window)
                if window_data.isna().any():
                    result.iloc[i] = np.nan
                else:
                    slope, _, _, _, _ = stats.linregress(x, window_data)
                    result.iloc[i] = slope
            return result
        elif method == 'direction':
            return np.sign(raw_values.diff(window))
        elif method == 'acceleration':
            slopes = pd.Series(index=raw_values.index, dtype=float)
            for i in range(window - 1, len(raw_values)):
                window_data = raw_values.iloc[i - window + 1:i + 1]
                x = np.arange(window)
                if window_data.isna().any():
                    slopes.iloc[i] = np.nan
                else:
                    slope, _, _, _, _ = stats.linregress(x, window_data)
                    slopes.iloc[i] = slope
            return slopes.diff()
        else:
            return raw_values.diff()

    def _extract_momentum_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition) ->pd.Series:
        """Extract momentum features like ROC"""
        raw_values = self._extract_raw_feature(data, feature_def)
        method = feature_def.params.get('method', 'roc')
        periods = feature_def.params.get('periods', 10)
        if method == 'roc':
            return (raw_values / raw_values.shift(periods) - 1) * 100
        elif method == 'momentum':
            return raw_values - raw_values.shift(periods)
        elif method == 'tsi':
            long_period = feature_def.params.get('long_period', 25)
            short_period = feature_def.params.get('short_period', 13)
            momentum = raw_values.diff()
            smooth1 = momentum.ewm(span=long_period).mean()
            smooth2 = smooth1.ewm(span=short_period).mean()
            abs_smooth1 = momentum.abs().ewm(span=long_period).mean()
            abs_smooth2 = abs_smooth1.ewm(span=short_period).mean()
            tsi = 100 * smooth2 / abs_smooth2
            return tsi
        else:
            return (raw_values / raw_values.shift(1) - 1) * 100

    def _extract_crossover_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition) ->pd.Series:
        """Extract crossover features between indicators"""
        if len(feature_def.source_columns) < 2:
            self.logger.warning(
                f'Need at least 2 columns for crossover: {feature_def.name}')
            return pd.Series(index=data.index, dtype=float)
        col1, col2 = feature_def.source_columns[0], feature_def.source_columns[
            1]
        method = feature_def.params.get('method', 'binary')
        if method == 'binary':
            current_above = data[col1] > data[col2]
            prev_above = data[col1].shift(1) > data[col2].shift(1)
            crossover_up = current_above & ~prev_above
            crossover_down = ~current_above & prev_above
            result = pd.Series(0, index=data.index)
            result[crossover_up] = 1
            result[crossover_down] = -1
            return result
        elif method == 'distance':
            return data[col1] - data[col2]
        elif method == 'ratio':
            return data[col1] / data[col2]
        else:
            return data[col1] - data[col2]

    def _extract_volatility_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition) ->pd.Series:
        """Extract volatility features"""
        raw_values = self._extract_raw_feature(data, feature_def)
        method = feature_def.params.get('method', 'std')
        window = feature_def.params.get('window', 20)
        if method == 'std':
            return raw_values.rolling(window=window).std()
        elif method == 'atr_ratio':
            if 'atr' in feature_def.source_columns and 'close' in data.columns:
                atr_col = next(col for col in feature_def.source_columns if
                    'atr' in col.lower())
                return data[atr_col] / data['close'] * 100
            else:
                self.logger.warning(
                    'Missing ATR or close columns for ATR ratio')
                return raw_values.rolling(window=window).std()
        elif method == 'bollinger_width':
            if any('upper' in col.lower() for col in data.columns) and any(
                'lower' in col.lower() for col in data.columns):
                upper_col = next(col for col in data.columns if 'upper' in
                    col.lower())
                lower_col = next(col for col in data.columns if 'lower' in
                    col.lower())
                middle_col = next((col for col in data.columns if 'middle' in
                    col.lower()), None)
                if middle_col:
                    return (data[upper_col] - data[lower_col]) / data[
                        middle_col]
                else:
                    return data[upper_col] - data[lower_col]
            else:
                self.logger.warning('Missing Bollinger Band columns')
                return raw_values.rolling(window=window).std()
        else:
            return raw_values.rolling(window=window).std()

    def _extract_divergence_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition) ->pd.Series:
        """Extract divergence features between indicator and price"""
        if len(feature_def.source_columns) < 1:
            self.logger.warning(
                f'Need at least 1 indicator column for divergence: {feature_def.name}'
                )
            return pd.Series(index=data.index, dtype=float)
        indicator_col = feature_def.source_columns[0]
        price_col = feature_def.params.get('price_column', 'close')
        if price_col not in data.columns:
            self.logger.warning(
                f'Price column {price_col} not found for divergence')
            return pd.Series(index=data.index, dtype=float)
        method = feature_def.params.get('method', 'correlation')
        window = feature_def.params.get('window', 20)
        if method == 'correlation':
            return data[price_col].rolling(window).corr(data[indicator_col])
        elif method == 'slope_diff':
            price_slope = self._extract_trend_feature(data,
                FeatureDefinition(name='price_slope', source_columns=[
                price_col], feature_type=FeatureType.TREND, params={
                'method': 'slope', 'window': window}))
            indicator_slope = self._extract_trend_feature(data,
                FeatureDefinition(name='indicator_slope', source_columns=[
                indicator_col], feature_type=FeatureType.TREND, params={
                'method': 'slope', 'window': window}))
            return price_slope * indicator_slope
        else:
            return data[price_col].rolling(window).corr(data[indicator_col])

    def _extract_pattern_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition) ->pd.Series:
        """Extract pattern-based features"""
        raw_values = self._extract_raw_feature(data, feature_def)
        method = feature_def.params.get('method', 'peaks')
        window = feature_def.params.get('window', 5)
        if method == 'peaks':
            result = pd.Series(0, index=raw_values.index)
            for i in range(window, len(raw_values) - window):
                left_window = raw_values.iloc[i - window:i]
                right_window = raw_values.iloc[i + 1:i + window + 1]
                current = raw_values.iloc[i]
                if current > left_window.max() and current > right_window.max(
                    ):
                    result.iloc[i] = 1
                elif current < left_window.min(
                    ) and current < right_window.min():
                    result.iloc[i] = -1
            return result
        elif method == 'oscillator_extremes':
            result = pd.Series(0, index=raw_values.index)
            overbought = feature_def.params.get('overbought', 70)
            oversold = feature_def.params.get('oversold', 30)
            result[raw_values > overbought] = 1
            result[raw_values < oversold] = -1
            return result
        else:
            return pd.Series(0, index=raw_values.index)

    def _extract_composite_feature(self, data: pd.DataFrame, feature_def:
        FeatureDefinition) ->pd.Series:
        """Extract composite features combining multiple indicators"""
        method = feature_def.params.get('method', 'weighted_sum')
        if method == 'weighted_sum':
            weights = feature_def.params.get('weights', None)
            if weights is None:
                weights = [1.0] * len(feature_def.source_columns)
            if len(weights) != len(feature_def.source_columns):
                self.logger.warning(
                    f"Number of weights ({len(weights)}) doesn't match columns ({len(feature_def.source_columns)})"
                    )
                weights = weights[:len(feature_def.source_columns)] if len(
                    weights) > len(feature_def.source_columns) else weights + [
                    1.0] * (len(feature_def.source_columns) - len(weights))
            result = pd.Series(0.0, index=data.index)
            for col, weight in zip(feature_def.source_columns, weights):
                if col in data.columns:
                    normalized = (data[col] - data[col].min()) / (data[col]
                        .max() - data[col].min())
                    result += normalized * weight
            return result
        elif method == 'voting':
            threshold = feature_def.params.get('threshold', 0.0)
            votes = pd.DataFrame(index=data.index)
            for i, col in enumerate(feature_def.source_columns):
                if col not in data.columns:
                    continue
                if i < len(feature_def.params.get('thresholds', [])):
                    col_threshold = feature_def.params['thresholds'][i]
                else:
                    col_threshold = threshold
                votes[f'vote_{i}'] = np.sign(data[col] - col_threshold)
            if votes.empty:
                return pd.Series(0, index=data.index)
            return votes.sum(axis=1) / len(feature_def.source_columns)
        else:
            return data[feature_def.source_columns].mean(axis=1)


class StandardFeatureSets:
    """
    Pre-defined standard feature sets for common ML tasks in trading
    """

    @staticmethod
    def directional_prediction_features() ->List[FeatureDefinition]:
        """
        Standard feature set for directional prediction (up/down)

        Returns:
            List of feature definitions
        """
        features = [FeatureDefinition(name='price_normalized',
            source_columns=['close'], feature_type=FeatureType.NORMALIZED,
            params={'scaler': 'minmax'}, scope=FeatureScope.PRICE),
            FeatureDefinition(name='rsi_normalized', source_columns=[
            'rsi_14'], feature_type=FeatureType.NORMALIZED, scope=
            FeatureScope.INDICATOR), FeatureDefinition(name='rsi_trend',
            source_columns=['rsi_14'], feature_type=FeatureType.TREND,
            params={'window': 5}, scope=FeatureScope.INDICATOR),
            FeatureDefinition(name='ma_crossover', source_columns=['sma_10',
            'sma_50'], feature_type=FeatureType.CROSSOVER, scope=
            FeatureScope.INDICATOR), FeatureDefinition(name='ma_distance',
            source_columns=['sma_10', 'sma_50'], feature_type=FeatureType.
            CROSSOVER, params={'method': 'distance'}, scope=FeatureScope.
            INDICATOR), FeatureDefinition(name='bb_width', source_columns=[
            'bb_upper_20', 'bb_lower_20', 'bb_middle_20'], feature_type=
            FeatureType.VOLATILITY, params={'method': 'bollinger_width'}),
            FeatureDefinition(name='macd_crossover', source_columns=[
            'macd_line', 'macd_signal'], feature_type=FeatureType.CROSSOVER
            ), FeatureDefinition(name='combined_momentum', source_columns=[
            'rsi_14', 'macd_histogram', 'stoch_k'], feature_type=
            FeatureType.COMPOSITE, params={'method': 'weighted_sum',
            'weights': [0.4, 0.4, 0.2]})]
        return features

    @staticmethod
    def volatility_prediction_features() ->List[FeatureDefinition]:
        """
        Standard feature set for volatility prediction

        Returns:
            List of feature definitions
        """
        features = [FeatureDefinition(name='atr_normalized', source_columns
            =['atr_14'], feature_type=FeatureType.NORMALIZED),
            FeatureDefinition(name='atr_ratio', source_columns=['atr_14'],
            feature_type=FeatureType.VOLATILITY, params={'method':
            'atr_ratio'}), FeatureDefinition(name='bb_width',
            source_columns=['bb_upper_20', 'bb_lower_20', 'bb_middle_20'],
            feature_type=FeatureType.VOLATILITY, params={'method':
            'bollinger_width'}), FeatureDefinition(name='price_std',
            source_columns=['close'], feature_type=FeatureType.VOLATILITY,
            params={'window': 20}), FeatureDefinition(name='volume_trend',
            source_columns=['volume'], feature_type=FeatureType.TREND,
            params={'window': 5})]
        return features

    @staticmethod
    def price_level_prediction_features() ->List[FeatureDefinition]:
        """
        Standard feature set for price level (support/resistance) prediction

        Returns:
            List of feature definitions
        """
        features = [FeatureDefinition(name='price_peaks', source_columns=[
            'close'], feature_type=FeatureType.PATTERN, params={'method':
            'peaks', 'window': 10}), FeatureDefinition(name='price_zscore',
            source_columns=['close'], feature_type=FeatureType.RELATIVE,
            params={'method': 'z_score', 'window': 50}), FeatureDefinition(
            name='bb_position', source_columns=['close', 'bb_upper_20',
            'bb_lower_20'], feature_type=FeatureType.COMPOSITE, params={
            'method': 'custom', 'function': lambda df: (df['close'] - df[
            'bb_lower_20']) / (df['bb_upper_20'] - df['bb_lower_20'])}),
            FeatureDefinition(name='atr_normalized', source_columns=[
            'atr_14'], feature_type=FeatureType.NORMALIZED)]
        return features

    @staticmethod
    def get_feature_set(task_name: str) ->List[FeatureDefinition]:
        """
        Get a standard feature set by task name

        Args:
            task_name: Name of the prediction task

        Returns:
            List of feature definitions
        """
        if task_name == 'direction':
            return StandardFeatureSets.directional_prediction_features()
        elif task_name == 'volatility':
            return StandardFeatureSets.volatility_prediction_features()
        elif task_name == 'price_level':
            return StandardFeatureSets.price_level_prediction_features()
        else:
            warnings.warn(
                f'Unknown task name: {task_name}. Using directional prediction features.'
                )
            return StandardFeatureSets.directional_prediction_features()


def create_rolling_window_features(base_features: List[FeatureDefinition],
    window_sizes: List[int]=[5, 10, 20], feature_types: List[FeatureType]=[
    FeatureType.TREND, FeatureType.MOMENTUM]) ->List[FeatureDefinition]:
    """
    Create rolling window features from base features

    Args:
        base_features: List of base feature definitions
        window_sizes: List of window sizes to create features for
        feature_types: Types of features to create

    Returns:
        List of expanded feature definitions
    """
    result = list(base_features)
    for base_feature in base_features:
        if base_feature.feature_type in [FeatureType.RAW, FeatureType.
            NORMALIZED]:
            for window in window_sizes:
                for feature_type in feature_types:
                    name = f'{base_feature.name}_{feature_type.value}_{window}'
                    new_feature = FeatureDefinition(name=name,
                        source_columns=base_feature.source_columns,
                        feature_type=feature_type, params={'window': window})
                    result.append(new_feature)
    return result
