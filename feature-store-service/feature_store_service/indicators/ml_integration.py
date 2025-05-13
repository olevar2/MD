"""
ML Integration Enhancement Module.

This module provides integration points between technical indicators and
machine learning models, including feature extraction, feature importance
analysis, and model-indicator feedback loops.
"""
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from enum import Enum
import json
import datetime
from pathlib import Path
import logging
import warnings
from collections import defaultdict
import pickle
from feature_store_service.indicators.base_indicator import BaseIndicator
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeatureType(Enum):
    """Enum representing different types of features for ML models."""
    RAW = 'raw'
    NORMALIZED = 'normalized'
    CATEGORICAL = 'categorical'
    TREND = 'trend'
    DIVERGENCE = 'divergence'
    CROSSOVER = 'crossover'
    CONSOLIDATED = 'consolidated'


class FeatureExtractor:
    """
    Feature Extractor for ML Models
    
    Extracts machine learning features from technical indicators,
    providing various transformations and normalizations suitable
    for different model architectures.
    """

    def __init__(self, feature_config: Optional[Dict[str, Any]]=None,
        normalization_lookback: int=100, **kwargs):
        """
        Initialize Feature Extractor.
        
        Args:
            feature_config: Configuration for feature extraction
            normalization_lookback: Lookback period for normalization
            **kwargs: Additional parameters
        """
        self.feature_config = feature_config or {}
        self.normalization_lookback = normalization_lookback
        self.normalization_params = {}

    def extract_features(self, data: pd.DataFrame, indicators: Dict[str,
        Any], include_categorical: bool=True, include_divergences: bool=
        True, include_crossovers: bool=True, horizon_periods: List[int]=None
        ) ->pd.DataFrame:
        """
        Extract features from indicator data for ML models.
        
        Args:
            data: DataFrame with indicator values and price data
            indicators: Dictionary of indicator information
            include_categorical: Whether to include categorical features
            include_divergences: Whether to include divergence features
            include_crossovers: Whether to include crossover features
            horizon_periods: List of forecast horizons for lagged features
            
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=data.index)
        horizon_periods = horizon_periods or [1, 5, 10, 20]
        for indicator_name, indicator_info in indicators.items():
            indicator_columns = self._get_indicator_columns(data,
                indicator_name)
            if not indicator_columns:
                logger.warning(
                    f'No columns found for indicator {indicator_name}')
                continue
            for col in indicator_columns:
                if data[col].isna().all():
                    continue
                features = self._add_normalized_features(features, data, col)
                features = self._add_trend_features(features, data, col)
                if include_categorical:
                    features = self._add_categorical_features(features,
                        data, col)
                if include_divergences and 'close' in data.columns:
                    features = self._add_divergence_features(features, data,
                        col)
                if include_crossovers:
                    features = self._add_crossover_features(features, data,
                        col, indicator_columns)
                for horizon in horizon_periods:
                    if horizon < len(data):
                        features[f'{col}_lag_{horizon}'] = data[col].shift(
                            horizon)
        features.dropna(inplace=True)
        return features

    def _get_indicator_columns(self, data: pd.DataFrame, indicator_name: str
        ) ->List[str]:
        """Get columns related to a specific indicator."""
        return [col for col in data.columns if indicator_name.lower() in
            col.lower()]

    def _add_normalized_features(self, features: pd.DataFrame, data: pd.
        DataFrame, column: str) ->pd.DataFrame:
        """Add normalized versions of the indicator values."""
        values = data[column].values
        if column not in self.normalization_params:
            lookback_values = values[-self.normalization_lookback:]
            lookback_values = lookback_values[~np.isnan(lookback_values)]
            if len(lookback_values) > 0:
                min_val = np.percentile(lookback_values, 5)
                max_val = np.percentile(lookback_values, 95)
                self.normalization_params[column] = {'min': min_val, 'max':
                    max_val}
            else:
                self.normalization_params[column] = {'min': 0, 'max': 1}
        min_val = self.normalization_params[column]['min']
        max_val = self.normalization_params[column]['max']
        if max_val == min_val:
            normalized = np.zeros_like(values)
        else:
            normalized = 2 * ((values - min_val) / (max_val - min_val) - 0.5)
            normalized = np.clip(normalized, -1, 1)
        features[f'{column}_norm'] = normalized
        return features

    def _add_trend_features(self, features: pd.DataFrame, data: pd.
        DataFrame, column: str) ->pd.DataFrame:
        """Add trend-related features from indicator values."""
        features[f'{column}_direction'] = np.sign(data[column].diff())
        pct_change = data[column].pct_change()
        features[f'{column}_strength'] = pct_change.abs()
        features[f'{column}_acceleration'] = pct_change.diff()
        features[f'{column}_ma_fast'] = data[column].rolling(window=5).mean()
        features[f'{column}_ma_slow'] = data[column].rolling(window=20).mean()
        features[f'{column}_ma_cross'] = np.where(features[
            f'{column}_ma_fast'] > features[f'{column}_ma_slow'], 1, np.
            where(features[f'{column}_ma_fast'] < features[
            f'{column}_ma_slow'], -1, 0))
        return features

    def _add_categorical_features(self, features: pd.DataFrame, data: pd.
        DataFrame, column: str) ->pd.DataFrame:
        """Add categorical features from indicator values."""
        values = data[column].values
        if len(values[~np.isnan(values)]) > 0:
            p25 = np.nanpercentile(values, 25)
            p50 = np.nanpercentile(values, 50)
            p75 = np.nanpercentile(values, 75)
            categories = np.zeros_like(values)
            categories = np.where(values <= p25, -2, categories)
            categories = np.where((values > p25) & (values <= p50), -1,
                categories)
            categories = np.where((values > p50) & (values <= p75), 1,
                categories)
            categories = np.where(values > p75, 2, categories)
            features[f'{column}_category'] = categories
        return features

    def _add_divergence_features(self, features: pd.DataFrame, data: pd.
        DataFrame, column: str) ->pd.DataFrame:
        """Add price divergence features."""
        price_direction = np.sign(data['close'].diff())
        indicator_direction = np.sign(data[column].diff())
        divergence = price_direction * indicator_direction
        features[f'{column}_divergence'] = np.where(divergence < 0, 1, 0)
        correlation = data[column].rolling(window=20).corr(data['close'])
        features[f'{column}_price_corr'] = correlation
        return features

    def _add_crossover_features(self, features: pd.DataFrame, data: pd.
        DataFrame, column: str, related_columns: List[str]) ->pd.DataFrame:
        """Add crossover features between related indicator components."""
        for other_col in related_columns:
            if other_col != column:
                if data[column].isna().all() or data[other_col].isna().all():
                    continue
                crossover = np.zeros(len(data))
                crossover_up = (data[column] > data[other_col]) & (data[
                    column].shift(1) <= data[other_col].shift(1))
                crossover_down = (data[column] < data[other_col]) & (data[
                    column].shift(1) >= data[other_col].shift(1))
                crossover = np.where(crossover_up, 1, np.where(
                    crossover_down, -1, 0))
                features[f'{column}_cross_{other_col}'] = crossover
        return features

    def save_normalization_params(self, path: str) ->None:
        """Save normalization parameters to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.normalization_params, f)

    def load_normalization_params(self, path: str) ->None:
        """Load normalization parameters from file."""
        with open(path, 'rb') as f:
            self.normalization_params = pickle.load(f)


class FeatureSelector:
    """
    Feature Selector for ML Models
    
    Selects the most relevant features for machine learning models
    using various feature importance and selection algorithms.
    """

    def __init__(self, selection_method: str='importance', max_features:
        int=50, selection_threshold: float=0.01, **kwargs):
        """
        Initialize Feature Selector.
        
        Args:
            selection_method: Method for feature selection
                             ("importance", "correlation", "variance", "recursive")
            max_features: Maximum number of features to select
            selection_threshold: Threshold for feature selection
            **kwargs: Additional parameters
        """
        self.selection_method = selection_method
        self.max_features = max_features
        self.selection_threshold = selection_threshold
        self.selected_features = []
        self.feature_importances = {}

    def select_features(self, features: pd.DataFrame, target: pd.Series,
        method: Optional[str]=None) ->pd.DataFrame:
        """
        Select the most relevant features for modeling.
        
        Args:
            features: DataFrame with extracted features
            target: Target variable series
            method: Override the selection method
            
        Returns:
            DataFrame with selected features
        """
        method = method or self.selection_method
        if method == 'importance':
            selected = self._select_by_importance(features, target)
        elif method == 'correlation':
            selected = self._select_by_correlation(features, target)
        elif method == 'variance':
            selected = self._select_by_variance(features)
        elif method == 'recursive':
            selected = self._select_by_recursive_elimination(features, target)
        else:
            logger.warning(
                f'Unknown selection method: {method}. Using importance.')
            selected = self._select_by_importance(features, target)
        self.selected_features = selected.columns.tolist()
        return selected

    @with_exception_handling
    def _select_by_importance(self, features: pd.DataFrame, target: pd.Series
        ) ->pd.DataFrame:
        """Select features based on importance to target."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, max_depth=5,
                random_state=42)
            model.fit(features, target)
            importances = model.feature_importances_
            self.feature_importances = {feature: importance for feature,
                importance in zip(features.columns, importances)}
            sorted_features = sorted(self.feature_importances.items(), key=
                lambda x: x[1], reverse=True)
            top_features = [f for f, i in sorted_features[:self.
                max_features] if i > self.selection_threshold]
            return features[top_features]
        except ImportError:
            logger.warning(
                'scikit-learn not available. Falling back to correlation method.'
                )
            return self._select_by_correlation(features, target)

    def _select_by_correlation(self, features: pd.DataFrame, target: pd.Series
        ) ->pd.DataFrame:
        """Select features based on correlation with target."""
        correlations = {}
        for col in features.columns:
            if features[col].dtype in [np.float64, np.int64]:
                corr = abs(features[col].corr(target))
                if not np.isnan(corr):
                    correlations[col] = corr
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[
            1], reverse=True)
        self.feature_importances = {f: c for f, c in sorted_correlations}
        top_features = [f for f, c in sorted_correlations[:self.
            max_features] if c > self.selection_threshold]
        return features[top_features]

    def _select_by_variance(self, features: pd.DataFrame) ->pd.DataFrame:
        """Select features based on variance."""
        from sklearn.feature_selection import VarianceThreshold
        variances = {}
        for col in features.columns:
            if features[col].dtype in [np.float64, np.int64]:
                variances[col] = features[col].var()
        sorted_variances = sorted(variances.items(), key=lambda x: x[1],
            reverse=True)
        self.feature_importances = {f: v for f, v in sorted_variances}
        top_features = [f for f, v in sorted_variances[:self.max_features] if
            v > self.selection_threshold]
        return features[top_features]

    @with_exception_handling
    def _select_by_recursive_elimination(self, features: pd.DataFrame,
        target: pd.Series) ->pd.DataFrame:
        """Select features using recursive feature elimination."""
        try:
            from sklearn.feature_selection import RFECV
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=100, random_state=42
                )
            selector = RFECV(estimator, step=1, cv=5,
                min_features_to_select=min(10, len(features.columns)))
            try:
                selector.fit(features, target)
                selected_mask = selector.support_
                selected_indices = np.where(selected_mask)[0]
                selected_columns = features.columns[selected_indices].tolist()
                self.feature_importances = {feature: (1.0 if i in
                    selected_indices else 0.0) for i, feature in enumerate(
                    features.columns)}
                return features[selected_columns]
            except Exception as e:
                logger.warning(
                    f'RFE failed: {str(e)}. Falling back to importance method.'
                    )
                return self._select_by_importance(features, target)
        except ImportError:
            logger.warning(
                'scikit-learn not available. Falling back to correlation method.'
                )
            return self._select_by_correlation(features, target)

    def get_feature_importances(self) ->Dict[str, float]:
        """Get the calculated feature importances."""
        return self.feature_importances


class IndicatorMLFeedback:
    """
    Indicator-ML Feedback System
    
    Creates a feedback loop between ML models and technical indicators,
    allowing indicators to be adjusted based on model performance and
    model features to be refined based on indicator performance.
    """

    def __init__(self, performance_threshold: float=0.6, feedback_period:
        int=20, **kwargs):
        """
        Initialize Indicator-ML Feedback System.
        
        Args:
            performance_threshold: Threshold for considering performance good/bad
            feedback_period: Number of periods to evaluate before feedback
            **kwargs: Additional parameters
        """
        self.performance_threshold = performance_threshold
        self.feedback_period = feedback_period
        self.indicator_performance = {}
        self.model_performance = {}
        self.feedback_history = []

    def evaluate_indicators(self, indicators: Dict[str, BaseIndicator],
        feature_importances: Dict[str, float], prediction_accuracy: float
        ) ->Dict[str, Dict[str, Any]]:
        """
        Evaluate indicators based on their contribution to model performance.
        
        Args:
            indicators: Dictionary of active indicators (name -> indicator)
            feature_importances: Dictionary of feature importances (feature -> importance)
            prediction_accuracy: Overall model prediction accuracy
            
        Returns:
            Dictionary with indicator evaluations and recommendations
        """
        indicator_importances = defaultdict(list)
        for feature, importance in feature_importances.items():
            for ind_name in indicators:
                if ind_name.lower() in feature.lower():
                    indicator_importances[ind_name].append(importance)
                    break
        evaluations = {}
        for ind_name, indicator in indicators.items():
            if (ind_name not in indicator_importances or not
                indicator_importances[ind_name]):
                continue
            avg_importance = sum(indicator_importances[ind_name]) / len(
                indicator_importances[ind_name])
            contribution_score = avg_importance * prediction_accuracy
            if ind_name not in self.indicator_performance:
                self.indicator_performance[ind_name] = []
            self.indicator_performance[ind_name].append({'timestamp':
                datetime.datetime.now().isoformat(), 'importance':
                avg_importance, 'contribution_score': contribution_score})
            max_history = 100
            if len(self.indicator_performance[ind_name]) > max_history:
                self.indicator_performance[ind_name
                    ] = self.indicator_performance[ind_name][-max_history:]
            recommendations = []
            if contribution_score > self.performance_threshold:
                recommendations.append('Maintain current settings')
            elif avg_importance < 0.05:
                recommendations.append(
                    'Consider replacing with alternative indicator')
            else:
                params = {}
                if hasattr(indicator, 'get_info'):
                    info = indicator.get_info()
                    if 'parameters' in info:
                        for param in info['parameters']:
                            if hasattr(indicator, param['name']):
                                params[param['name']] = getattr(indicator,
                                    param['name'])
                if params:
                    recommendations.append(
                        f'Consider adjusting parameters: {list(params.keys())}'
                        )
            evaluations[ind_name] = {'importance': avg_importance,
                'contribution_score': contribution_score, 'recommendations':
                recommendations}
        return evaluations

    def generate_indicator_feedback(self, evaluations: Dict[str, Dict[str,
        Any]]) ->Dict[str, Dict[str, Any]]:
        """
        Generate feedback for indicators based on evaluations.
        
        Args:
            evaluations: Dictionary of indicator evaluations
            
        Returns:
            Dictionary with indicator feedback and parameter suggestions
        """
        feedback = {}
        for ind_name, eval_data in evaluations.items():
            if eval_data['contribution_score'] > self.performance_threshold:
                continue
            if ind_name in self.indicator_performance and len(self.
                indicator_performance[ind_name]) > 1:
                history = self.indicator_performance[ind_name]
                recent_scores = [h['contribution_score'] for h in history[-
                    min(self.feedback_period, len(history)):]]
                if len(recent_scores) > 1:
                    trend = recent_scores[-1] - recent_scores[0]
                    if trend < -0.05:
                        feedback[ind_name] = {'status': 'declining',
                            'message':
                            f'Performance declining over last {len(recent_scores)} periods'
                            , 'suggestions': {'action': 'replace', 'reason':
                            'Consistent decline in predictive value'}}
                    elif eval_data['contribution_score'
                        ] < self.performance_threshold / 2:
                        feedback[ind_name] = {'status': 'poor', 'message':
                            f'Consistently low predictive value',
                            'suggestions': {'action': 'replace', 'reason':
                            'Low contribution to model performance'}}
                    else:
                        feedback[ind_name] = {'status': 'underperforming',
                            'message': f'Below threshold performance',
                            'suggestions': {'action': 'adjust', 'reason':
                            'Moderate contribution could be improved'}}
            else:
                feedback[ind_name] = {'status': 'insufficient_data',
                    'message': f'Not enough performance data',
                    'suggestions': {'action': 'monitor', 'reason':
                    'Need more data to evaluate'}}
        if feedback:
            self.feedback_history.append({'timestamp': datetime.datetime.
                now().isoformat(), 'feedback': feedback})
        return feedback

    def update_model_performance(self, model_name: str, performance_metrics:
        Dict[str, float]) ->None:
        """
        Update model performance history.
        
        Args:
            model_name: Name of the model
            performance_metrics: Dictionary of performance metrics
        """
        if model_name not in self.model_performance:
            self.model_performance[model_name] = []
        self.model_performance[model_name].append({'timestamp': datetime.
            datetime.now().isoformat(), 'metrics': performance_metrics})
        max_history = 100
        if len(self.model_performance[model_name]) > max_history:
            self.model_performance[model_name] = self.model_performance[
                model_name][-max_history:]

    def get_indicator_importance_trend(self, indicator_name: str) ->Optional[
        Dict[str, Any]]:
        """
        Get the trend of an indicator's importance over time.
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Dictionary with trend analysis or None if insufficient data
        """
        if indicator_name not in self.indicator_performance:
            return None
        history = self.indicator_performance[indicator_name]
        if len(history) < 2:
            return None
        importances = [h['importance'] for h in history]
        contributions = [h['contribution_score'] for h in history]
        importance_trend = importances[-1] - importances[0]
        contribution_trend = contributions[-1] - contributions[0]
        return {'first_importance': importances[0], 'last_importance':
            importances[-1], 'importance_trend': importance_trend,
            'contribution_trend': contribution_trend, 'data_points': len(
            history), 'trend_direction': 'improving' if contribution_trend >
            0 else 'declining'}

    def save_history(self, path: str) ->None:
        """Save feedback history to file."""
        data = {'indicator_performance': self.indicator_performance,
            'model_performance': self.model_performance, 'feedback_history':
            self.feedback_history}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_history(self, path: str) ->None:
        """Load feedback history from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.indicator_performance = data.get('indicator_performance', {})
        self.model_performance = data.get('model_performance', {})
        self.feedback_history = data.get('feedback_history', [])


class ModelInputPreparation:
    """
    Model Input Preparation
    
    Prepares indicator data for use with different types of machine learning models,
    handling different model requirements and data formats.
    """

    def __init__(self, feature_extractor: FeatureExtractor,
        feature_selector: Optional[FeatureSelector]=None, sequence_length:
        int=10, forecast_horizon: int=5, **kwargs):
        """
        Initialize Model Input Preparation.
        
        Args:
            feature_extractor: Feature extractor instance
            feature_selector: Optional feature selector instance
            sequence_length: Length of sequence for sequential models (LSTM, etc.)
            forecast_horizon: Number of periods to forecast
            **kwargs: Additional parameters
        """
        self.feature_extractor = feature_extractor
        self.feature_selector = feature_selector or FeatureSelector()
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

    def prepare_standard_inputs(self, data: pd.DataFrame, indicators: Dict[
        str, Any], target_column: str='close', target_transformation: str=
        'returns', train_ratio: float=0.8) ->Dict[str, Any]:
        """
        Prepare standard tabular inputs for ML models like Random Forests, GBM, etc.
        
        Args:
            data: DataFrame with OHLCV data and indicator values
            indicators: Dictionary of indicator information
            target_column: Column to use as prediction target
            target_transformation: Transformation to apply to target ('returns', 'log_returns', 'direction')
            train_ratio: Ratio of data to use for training (0.0-1.0)
            
        Returns:
            Dictionary with prepared inputs and related information
        """
        features_df = self.feature_extractor.extract_features(data, indicators)
        if target_transformation == 'returns':
            target = data[target_column].pct_change(self.forecast_horizon
                ).shift(-self.forecast_horizon)
        elif target_transformation == 'log_returns':
            target = np.log(data[target_column] / data[target_column].shift
                (self.forecast_horizon)).shift(-self.forecast_horizon)
        elif target_transformation == 'direction':
            target = np.sign(data[target_column].diff(self.forecast_horizon
                ).shift(-self.forecast_horizon))
        else:
            target = data[target_column].shift(-self.forecast_horizon)
        aligned_data = pd.concat([features_df, target.rename('target')], axis=1
            ).dropna()
        if len(aligned_data) == 0:
            return {'error': 'No valid data after alignment'}
        X = aligned_data.drop('target', axis=1)
        y = aligned_data['target']
        if self.feature_selector:
            X = self.feature_selector.select_features(X, y)
        train_size = int(len(X) * train_ratio)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test,
            'y_test': y_test, 'feature_names': X.columns.tolist(),
            'target_transformation': target_transformation,
            'forecast_horizon': self.forecast_horizon}

    def prepare_sequence_inputs(self, data: pd.DataFrame, indicators: Dict[
        str, Any], target_column: str='close', target_transformation: str=
        'returns', train_ratio: float=0.8) ->Dict[str, Any]:
        """
        Prepare sequential inputs for recurrent models like LSTM, GRU, etc.
        
        Args:
            data: DataFrame with OHLCV data and indicator values
            indicators: Dictionary of indicator information
            target_column: Column to use as prediction target
            target_transformation: Transformation to apply to target ('returns', 'log_returns', 'direction')
            train_ratio: Ratio of data to use for training (0.0-1.0)
            
        Returns:
            Dictionary with prepared inputs and related information
        """
        features_df = self.feature_extractor.extract_features(data, indicators)
        if target_transformation == 'returns':
            target = data[target_column].pct_change(self.forecast_horizon
                ).shift(-self.forecast_horizon)
        elif target_transformation == 'log_returns':
            target = np.log(data[target_column] / data[target_column].shift
                (self.forecast_horizon)).shift(-self.forecast_horizon)
        elif target_transformation == 'direction':
            target = np.sign(data[target_column].diff(self.forecast_horizon
                ).shift(-self.forecast_horizon))
        else:
            target = data[target_column].shift(-self.forecast_horizon)
        aligned_data = pd.concat([features_df, target.rename('target')], axis=1
            ).dropna()
        if len(aligned_data) == 0:
            return {'error': 'No valid data after alignment'}
        X = aligned_data.drop('target', axis=1)
        y = aligned_data['target']
        if self.feature_selector:
            X = self.feature_selector.select_features(X, y)
        sequences = []
        targets = []
        for i in range(len(X) - self.sequence_length + 1):
            seq = X.iloc[i:i + self.sequence_length].values
            target_val = y.iloc[i + self.sequence_length - 1]
            sequences.append(seq)
            targets.append(target_val)
        X_seq = np.array(sequences)
        y_seq = np.array(targets)
        train_size = int(len(X_seq) * train_ratio)
        X_train = X_seq[:train_size]
        y_train = y_seq[:train_size]
        X_test = X_seq[train_size:]
        y_test = y_seq[train_size:]
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test,
            'y_test': y_test, 'feature_names': X.columns.tolist(),
            'target_transformation': target_transformation,
            'forecast_horizon': self.forecast_horizon, 'sequence_length':
            self.sequence_length}

    def prepare_multistep_forecast_inputs(self, data: pd.DataFrame,
        indicators: Dict[str, Any], target_column: str='close',
        target_transformation: str='returns', forecast_steps: List[int]=
        None, train_ratio: float=0.8) ->Dict[str, Any]:
        """
        Prepare inputs for multi-step forecasting models.
        
        Args:
            data: DataFrame with OHLCV data and indicator values
            indicators: Dictionary of indicator information
            target_column: Column to use as prediction target
            target_transformation: Transformation to apply to target ('returns', 'log_returns', 'direction')
            forecast_steps: List of steps ahead to forecast
            train_ratio: Ratio of data to use for training (0.0-1.0)
            
        Returns:
            Dictionary with prepared inputs and related information
        """
        forecast_steps = forecast_steps or [1, 3, 5, 10, 20]
        features_df = self.feature_extractor.extract_features(data, indicators)
        targets = {}
        for step in forecast_steps:
            if target_transformation == 'returns':
                targets[f'target_{step}'] = data[target_column].pct_change(step
                    ).shift(-step)
            elif target_transformation == 'log_returns':
                targets[f'target_{step}'] = np.log(data[target_column] /
                    data[target_column].shift(step)).shift(-step)
            elif target_transformation == 'direction':
                targets[f'target_{step}'] = np.sign(data[target_column].
                    diff(step).shift(-step))
            else:
                targets[f'target_{step}'] = data[target_column].shift(-step)
        targets_df = pd.DataFrame(targets)
        aligned_data = pd.concat([features_df, targets_df], axis=1).dropna()
        if len(aligned_data) == 0:
            return {'error': 'No valid data after alignment'}
        X = aligned_data[[col for col in aligned_data.columns if not col.
            startswith('target_')]]
        y = aligned_data[[col for col in aligned_data.columns if col.
            startswith('target_')]]
        if self.feature_selector:
            mid_step = forecast_steps[len(forecast_steps) // 2]
            X = self.feature_selector.select_features(X, y[
                f'target_{mid_step}'])
        train_size = int(len(X) * train_ratio)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]
        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test,
            'y_test': y_test, 'feature_names': X.columns.tolist(),
            'target_columns': y.columns.tolist(), 'target_transformation':
            target_transformation, 'forecast_steps': forecast_steps}

    def save_preparation_config(self, path: str) ->None:
        """Save model input preparation configuration to file."""
        config = {'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon}
        with open(path, 'wb') as f:
            pickle.dump(config, f)

    def load_preparation_config(self, path: str) ->None:
        """Load model input preparation configuration from file."""
        with open(path, 'rb') as f:
            config = pickle.load(f)
        self.sequence_length = config.get('sequence_length', self.
            sequence_length)
        self.forecast_horizon = config.get('forecast_horizon', self.
            forecast_horizon)
