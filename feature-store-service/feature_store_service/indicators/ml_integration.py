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


class FeatureType(Enum):
    """Enum representing different types of features for ML models."""
    RAW = "raw"  # Raw indicator values
    NORMALIZED = "normalized"  # Normalized to a range (e.g., -1 to 1)
    CATEGORICAL = "categorical"  # Discretized into categories
    TREND = "trend"  # Direction (up/down)
    DIVERGENCE = "divergence"  # Divergence with price
    CROSSOVER = "crossover"  # Crossover events
    CONSOLIDATED = "consolidated"  # Consolidated from multiple indicators


class FeatureExtractor:
    """
    Feature Extractor for ML Models
    
    Extracts machine learning features from technical indicators,
    providing various transformations and normalizations suitable
    for different model architectures.
    """
    
    def __init__(
        self, 
        feature_config: Optional[Dict[str, Any]] = None,
        normalization_lookback: int = 100,
        **kwargs
    ):
        """
        Initialize Feature Extractor.
        
        Args:
            feature_config: Configuration for feature extraction
            normalization_lookback: Lookback period for normalization
            **kwargs: Additional parameters
        """
        self.feature_config = feature_config or {}
        self.normalization_lookback = normalization_lookback
        self.normalization_params = {}  # Store normalization parameters
        
    def extract_features(
        self, 
        data: pd.DataFrame, 
        indicators: Dict[str, Any],
        include_categorical: bool = True,
        include_divergences: bool = True,
        include_crossovers: bool = True,
        horizon_periods: List[int] = None
    ) -> pd.DataFrame:
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
        # Initialize the features DataFrame
        features = pd.DataFrame(index=data.index)
        
        # Default horizon periods if not provided
        horizon_periods = horizon_periods or [1, 5, 10, 20]
        
        # Process each indicator
        for indicator_name, indicator_info in indicators.items():
            # Extract indicator columns from data
            indicator_columns = self._get_indicator_columns(data, indicator_name)
            
            if not indicator_columns:
                logger.warning(f"No columns found for indicator {indicator_name}")
                continue
                
            # Process each column for this indicator
            for col in indicator_columns:
                # Only process columns with data
                if data[col].isna().all():
                    continue
                    
                # 1. Raw Values (normalized)
                features = self._add_normalized_features(features, data, col)
                
                # 2. Trend Features
                features = self._add_trend_features(features, data, col)
                
                # 3. Categorical Features (if requested)
                if include_categorical:
                    features = self._add_categorical_features(features, data, col)
                    
                # 4. Divergence Features (if requested)
                if include_divergences and 'close' in data.columns:
                    features = self._add_divergence_features(features, data, col)
                    
                # 5. Crossover Features (if requested)
                if include_crossovers:
                    features = self._add_crossover_features(features, data, col, indicator_columns)
                    
                # 6. Add lagged features for different horizons
                for horizon in horizon_periods:
                    if horizon < len(data):
                        features[f"{col}_lag_{horizon}"] = data[col].shift(horizon)
                        
        # Drop rows with NaN values (mostly due to lagged features)
        features.dropna(inplace=True)
        
        return features
        
    def _get_indicator_columns(self, data: pd.DataFrame, indicator_name: str) -> List[str]:
        """Get columns related to a specific indicator."""
        # This is a heuristic approach - assumes indicator columns contain the indicator name
        return [col for col in data.columns if indicator_name.lower() in col.lower()]
        
    def _add_normalized_features(
        self, features: pd.DataFrame, data: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """Add normalized versions of the indicator values."""
        # Standard min-max normalization to range (-1, 1)
        values = data[column].values
        
        # Calculate or retrieve normalization parameters
        if column not in self.normalization_params:
            # Use the lookback period for calculating normalization parameters
            lookback_values = values[-self.normalization_lookback:]
            lookback_values = lookback_values[~np.isnan(lookback_values)]
            
            if len(lookback_values) > 0:
                # Calculate percentiles instead of min/max to handle outliers
                min_val = np.percentile(lookback_values, 5)
                max_val = np.percentile(lookback_values, 95)
                
                # Store parameters
                self.normalization_params[column] = {
                    'min': min_val,
                    'max': max_val
                }
            else:
                # No valid data, use default parameters
                self.normalization_params[column] = {
                    'min': 0,
                    'max': 1
                }
                
        # Get normalization parameters
        min_val = self.normalization_params[column]['min']
        max_val = self.normalization_params[column]['max']
        
        # Avoid division by zero
        if max_val == min_val:
            normalized = np.zeros_like(values)
        else:
            # Normalize to (-1, 1) range
            normalized = 2 * ((values - min_val) / (max_val - min_val) - 0.5)
            
            # Clip to handle outliers
            normalized = np.clip(normalized, -1, 1)
            
        # Add to features
        features[f"{column}_norm"] = normalized
        
        return features
        
    def _add_trend_features(
        self, features: pd.DataFrame, data: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """Add trend-related features from indicator values."""
        # Calculate trend direction (1 for up, -1 for down, 0 for flat)
        features[f"{column}_direction"] = np.sign(data[column].diff())
        
        # Calculate trend strength (absolute percentage change)
        pct_change = data[column].pct_change()
        features[f"{column}_strength"] = pct_change.abs()
        
        # Calculate trend acceleration (change in trend)
        features[f"{column}_acceleration"] = pct_change.diff()
        
        # Calculate moving average crossovers (common in trading strategies)
        # Fast MA (5-period)
        features[f"{column}_ma_fast"] = data[column].rolling(window=5).mean()
        
        # Slow MA (20-period)
        features[f"{column}_ma_slow"] = data[column].rolling(window=20).mean()
        
        # MA crossover signal (1 when fast crosses above slow, -1 when below)
        features[f"{column}_ma_cross"] = np.where(
            features[f"{column}_ma_fast"] > features[f"{column}_ma_slow"], 1, 
            np.where(features[f"{column}_ma_fast"] < features[f"{column}_ma_slow"], -1, 0)
        )
        
        return features
        
    def _add_categorical_features(
        self, features: pd.DataFrame, data: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """Add categorical features from indicator values."""
        # Discretize the indicator value into categories (e.g., low, medium, high)
        values = data[column].values
        
        # Calculate percentiles for categorization
        if len(values[~np.isnan(values)]) > 0:
            p25 = np.nanpercentile(values, 25)
            p50 = np.nanpercentile(values, 50)
            p75 = np.nanpercentile(values, 75)
            
            # Create categorical feature
            categories = np.zeros_like(values)
            categories = np.where(values <= p25, -2, categories)
            categories = np.where((values > p25) & (values <= p50), -1, categories)
            categories = np.where((values > p50) & (values <= p75), 1, categories)
            categories = np.where(values > p75, 2, categories)
            
            features[f"{column}_category"] = categories
            
        return features
        
    def _add_divergence_features(
        self, features: pd.DataFrame, data: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """Add price divergence features."""
        # Calculate if indicator and price are moving in different directions
        price_direction = np.sign(data['close'].diff())
        indicator_direction = np.sign(data[column].diff())
        
        # Divergence occurs when directions are opposite
        divergence = price_direction * indicator_direction
        features[f"{column}_divergence"] = np.where(divergence < 0, 1, 0)
        
        # Calculate rolling correlation between indicator and price
        correlation = data[column].rolling(window=20).corr(data['close'])
        features[f"{column}_price_corr"] = correlation
        
        return features
        
    def _add_crossover_features(
        self, features: pd.DataFrame, data: pd.DataFrame, column: str, related_columns: List[str]
    ) -> pd.DataFrame:
        """Add crossover features between related indicator components."""
        # Find potential crossover pairs
        for other_col in related_columns:
            if other_col != column:
                # Skip if either column has all NaN values
                if data[column].isna().all() or data[other_col].isna().all():
                    continue
                    
                # Calculate crossovers
                crossover = np.zeros(len(data))
                
                # Current value is above other, but previous was below (bullish crossover)
                crossover_up = (data[column] > data[other_col]) & (data[column].shift(1) <= data[other_col].shift(1))
                
                # Current value is below other, but previous was above (bearish crossover)
                crossover_down = (data[column] < data[other_col]) & (data[column].shift(1) >= data[other_col].shift(1))
                
                crossover = np.where(crossover_up, 1, np.where(crossover_down, -1, 0))
                
                features[f"{column}_cross_{other_col}"] = crossover
                
        return features
        
    def save_normalization_params(self, path: str) -> None:
        """Save normalization parameters to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.normalization_params, f)
            
    def load_normalization_params(self, path: str) -> None:
        """Load normalization parameters from file."""
        with open(path, 'rb') as f:
            self.normalization_params = pickle.load(f)


class FeatureSelector:
    """
    Feature Selector for ML Models
    
    Selects the most relevant features for machine learning models
    using various feature importance and selection algorithms.
    """
    
    def __init__(
        self, 
        selection_method: str = "importance",
        max_features: int = 50,
        selection_threshold: float = 0.01,
        **kwargs
    ):
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
        
    def select_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        method: Optional[str] = None
    ) -> pd.DataFrame:
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
        
        if method == "importance":
            selected = self._select_by_importance(features, target)
        elif method == "correlation":
            selected = self._select_by_correlation(features, target)
        elif method == "variance":
            selected = self._select_by_variance(features)
        elif method == "recursive":
            selected = self._select_by_recursive_elimination(features, target)
        else:
            logger.warning(f"Unknown selection method: {method}. Using importance.")
            selected = self._select_by_importance(features, target)
            
        # Store the selected features
        self.selected_features = selected.columns.tolist()
        
        return selected
        
    def _select_by_importance(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select features based on importance to target."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Train a simple random forest to get feature importances
            model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            model.fit(features, target)
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Create a dictionary of feature importances
            self.feature_importances = {
                feature: importance 
                for feature, importance in zip(features.columns, importances)
            }
            
            # Sort features by importance
            sorted_features = sorted(
                self.feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select top features
            top_features = [f for f, i in sorted_features[:self.max_features] 
                          if i > self.selection_threshold]
            
            return features[top_features]
            
        except ImportError:
            logger.warning("scikit-learn not available. Falling back to correlation method.")
            return self._select_by_correlation(features, target)
            
    def _select_by_correlation(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select features based on correlation with target."""
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
        
        # Store feature importances based on correlation
        self.feature_importances = {f: c for f, c in sorted_correlations}
        
        # Select top features
        top_features = [f for f, c in sorted_correlations[:self.max_features] 
                       if c > self.selection_threshold]
        
        return features[top_features]
        
    def _select_by_variance(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select features based on variance."""
        from sklearn.feature_selection import VarianceThreshold
        
        # Calculate variance for each feature
        variances = {}
        for col in features.columns:
            if features[col].dtype in [np.float64, np.int64]:
                variances[col] = features[col].var()
                
        # Sort by variance
        sorted_variances = sorted(
            variances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Store feature importances based on variance
        self.feature_importances = {f: v for f, v in sorted_variances}
        
        # Select top features
        top_features = [f for f, v in sorted_variances[:self.max_features] 
                       if v > self.selection_threshold]
        
        return features[top_features]
        
    def _select_by_recursive_elimination(self, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select features using recursive feature elimination."""
        try:
            from sklearn.feature_selection import RFECV
            from sklearn.ensemble import RandomForestRegressor
            
            # Create estimator
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Create RFE selector
            selector = RFECV(estimator, step=1, cv=5, min_features_to_select=min(10, len(features.columns)))
            
            try:
                selector.fit(features, target)
                
                # Get selected feature mask
                selected_mask = selector.support_
                
                # Get selected feature indices
                selected_indices = np.where(selected_mask)[0]
                
                # Get selected feature names
                selected_columns = features.columns[selected_indices].tolist()
                
                # Store feature importances
                self.feature_importances = {
                    feature: 1.0 if i in selected_indices else 0.0
                    for i, feature in enumerate(features.columns)
                }
                
                return features[selected_columns]
                
            except Exception as e:
                logger.warning(f"RFE failed: {str(e)}. Falling back to importance method.")
                return self._select_by_importance(features, target)
                
        except ImportError:
            logger.warning("scikit-learn not available. Falling back to correlation method.")
            return self._select_by_correlation(features, target)
            
    def get_feature_importances(self) -> Dict[str, float]:
        """Get the calculated feature importances."""
        return self.feature_importances


class IndicatorMLFeedback:
    """
    Indicator-ML Feedback System
    
    Creates a feedback loop between ML models and technical indicators,
    allowing indicators to be adjusted based on model performance and
    model features to be refined based on indicator performance.
    """
    
    def __init__(
        self, 
        performance_threshold: float = 0.6,
        feedback_period: int = 20,
        **kwargs
    ):
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
        
    def evaluate_indicators(
        self, 
        indicators: Dict[str, BaseIndicator],
        feature_importances: Dict[str, float],
        prediction_accuracy: float
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate indicators based on their contribution to model performance.
        
        Args:
            indicators: Dictionary of active indicators (name -> indicator)
            feature_importances: Dictionary of feature importances (feature -> importance)
            prediction_accuracy: Overall model prediction accuracy
            
        Returns:
            Dictionary with indicator evaluations and recommendations
        """
        # Group feature importances by indicator
        indicator_importances = defaultdict(list)
        
        for feature, importance in feature_importances.items():
            # Find which indicator this feature belongs to
            for ind_name in indicators:
                if ind_name.lower() in feature.lower():
                    indicator_importances[ind_name].append(importance)
                    break
                    
        # Calculate indicator contributions
        evaluations = {}
        
        for ind_name, indicator in indicators.items():
            # Skip indicators with no importance data
            if ind_name not in indicator_importances or not indicator_importances[ind_name]:
                continue
                
            # Calculate average importance for this indicator
            avg_importance = sum(indicator_importances[ind_name]) / len(indicator_importances[ind_name])
            
            # Calculate contribution score (importance * prediction_accuracy)
            contribution_score = avg_importance * prediction_accuracy
            
            # Update indicator performance history
            if ind_name not in self.indicator_performance:
                self.indicator_performance[ind_name] = []
                
            self.indicator_performance[ind_name].append({
                'timestamp': datetime.datetime.now().isoformat(),
                'importance': avg_importance,
                'contribution_score': contribution_score
            })
            
            # Limit history size
            max_history = 100
            if len(self.indicator_performance[ind_name]) > max_history:
                self.indicator_performance[ind_name] = self.indicator_performance[ind_name][-max_history:]
                
            # Generate recommendations
            recommendations = []
            
            if contribution_score > self.performance_threshold:
                recommendations.append("Maintain current settings")
            else:
                if avg_importance < 0.05:
                    recommendations.append("Consider replacing with alternative indicator")
                else:
                    # Get indicator parameters if available
                    params = {}
                    if hasattr(indicator, 'get_info'):
                        info = indicator.get_info()
                        if 'parameters' in info:
                            for param in info['parameters']:
                                if hasattr(indicator, param['name']):
                                    params[param['name']] = getattr(indicator, param['name'])
                                    
                    if params:
                        recommendations.append(f"Consider adjusting parameters: {list(params.keys())}")
                        
            # Add evaluation to results
            evaluations[ind_name] = {
                'importance': avg_importance,
                'contribution_score': contribution_score,
                'recommendations': recommendations
            }
            
        return evaluations
        
    def generate_indicator_feedback(
        self, 
        evaluations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate feedback for indicators based on evaluations.
        
        Args:
            evaluations: Dictionary of indicator evaluations
            
        Returns:
            Dictionary with indicator feedback and parameter suggestions
        """
        feedback = {}
        
        for ind_name, eval_data in evaluations.items():
            # Skip indicators with good performance
            if eval_data['contribution_score'] > self.performance_threshold:
                continue
                
            # Generate feedback based on historical performance
            if ind_name in self.indicator_performance and len(self.indicator_performance[ind_name]) > 1:
                history = self.indicator_performance[ind_name]
                
                # Check if performance is declining
                recent_scores = [h['contribution_score'] for h in history[-min(self.feedback_period, len(history)):]]
                if len(recent_scores) > 1:
                    trend = recent_scores[-1] - recent_scores[0]
                    
                    if trend < -0.05:
                        feedback[ind_name] = {
                            'status': 'declining',
                            'message': f"Performance declining over last {len(recent_scores)} periods",
                            'suggestions': {
                                'action': 'replace',
                                'reason': 'Consistent decline in predictive value'
                            }
                        }
                    elif eval_data['contribution_score'] < self.performance_threshold / 2:
                        feedback[ind_name] = {
                            'status': 'poor',
                            'message': f"Consistently low predictive value",
                            'suggestions': {
                                'action': 'replace',
                                'reason': 'Low contribution to model performance'
                            }
                        }
                    else:
                        feedback[ind_name] = {
                            'status': 'underperforming',
                            'message': f"Below threshold performance",
                            'suggestions': {
                                'action': 'adjust',
                                'reason': 'Moderate contribution could be improved'
                            }
                        }
            else:
                feedback[ind_name] = {
                    'status': 'insufficient_data',
                    'message': f"Not enough performance data",
                    'suggestions': {
                        'action': 'monitor',
                        'reason': 'Need more data to evaluate'
                    }
                }
                
        # Record feedback
        if feedback:
            self.feedback_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'feedback': feedback
            })
            
        return feedback
        
    def update_model_performance(
        self, 
        model_name: str,
        performance_metrics: Dict[str, float]
    ) -> None:
        """
        Update model performance history.
        
        Args:
            model_name: Name of the model
            performance_metrics: Dictionary of performance metrics
        """
        if model_name not in self.model_performance:
            self.model_performance[model_name] = []
            
        self.model_performance[model_name].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': performance_metrics
        })
        
        # Limit history size
        max_history = 100
        if len(self.model_performance[model_name]) > max_history:
            self.model_performance[model_name] = self.model_performance[model_name][-max_history:]
            
    def get_indicator_importance_trend(
        self, 
        indicator_name: str
    ) -> Optional[Dict[str, Any]]:
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
            
        # Extract importance values and calculate trend
        importances = [h['importance'] for h in history]
        contributions = [h['contribution_score'] for h in history]
        
        # Calculate trend
        importance_trend = importances[-1] - importances[0]
        contribution_trend = contributions[-1] - contributions[0]
        
        return {
            'first_importance': importances[0],
            'last_importance': importances[-1],
            'importance_trend': importance_trend,
            'contribution_trend': contribution_trend,
            'data_points': len(history),
            'trend_direction': 'improving' if contribution_trend > 0 else 'declining'
        }
        
    def save_history(self, path: str) -> None:
        """Save feedback history to file."""
        data = {
            'indicator_performance': self.indicator_performance,
            'model_performance': self.model_performance,
            'feedback_history': self.feedback_history
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_history(self, path: str) -> None:
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
    
    def __init__(
        self, 
        feature_extractor: FeatureExtractor,
        feature_selector: Optional[FeatureSelector] = None,
        sequence_length: int = 10,
        forecast_horizon: int = 5,
        **kwargs
    ):
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
        
    def prepare_standard_inputs(
        self, 
        data: pd.DataFrame, 
        indicators: Dict[str, Any],
        target_column: str = 'close',
        target_transformation: str = 'returns',
        train_ratio: float = 0.8
    ) -> Dict[str, Any]:
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
        # Extract features from indicators
        features_df = self.feature_extractor.extract_features(data, indicators)
        
        # Create target variable
        if target_transformation == 'returns':
            target = data[target_column].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
        elif target_transformation == 'log_returns':
            target = np.log(data[target_column] / data[target_column].shift(self.forecast_horizon)).shift(-self.forecast_horizon)
        elif target_transformation == 'direction':
            target = np.sign(data[target_column].diff(self.forecast_horizon).shift(-self.forecast_horizon))
        else:
            target = data[target_column].shift(-self.forecast_horizon)
            
        # Align features and target
        aligned_data = pd.concat([features_df, target.rename('target')], axis=1).dropna()
        
        if len(aligned_data) == 0:
            return {'error': 'No valid data after alignment'}
            
        # Split features and target
        X = aligned_data.drop('target', axis=1)
        y = aligned_data['target']
        
        # Select features
        if self.feature_selector:
            X = self.feature_selector.select_features(X, y)
            
        # Split into train and test sets
        train_size = int(len(X) * train_ratio)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'target_transformation': target_transformation,
            'forecast_horizon': self.forecast_horizon
        }
        
    def prepare_sequence_inputs(
        self, 
        data: pd.DataFrame, 
        indicators: Dict[str, Any],
        target_column: str = 'close',
        target_transformation: str = 'returns',
        train_ratio: float = 0.8
    ) -> Dict[str, Any]:
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
        # Extract features from indicators
        features_df = self.feature_extractor.extract_features(data, indicators)
        
        # Create target variable
        if target_transformation == 'returns':
            target = data[target_column].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
        elif target_transformation == 'log_returns':
            target = np.log(data[target_column] / data[target_column].shift(self.forecast_horizon)).shift(-self.forecast_horizon)
        elif target_transformation == 'direction':
            target = np.sign(data[target_column].diff(self.forecast_horizon).shift(-self.forecast_horizon))
        else:
            target = data[target_column].shift(-self.forecast_horizon)
            
        # Align features and target
        aligned_data = pd.concat([features_df, target.rename('target')], axis=1).dropna()
        
        if len(aligned_data) == 0:
            return {'error': 'No valid data after alignment'}
            
        # Split features and target
        X = aligned_data.drop('target', axis=1)
        y = aligned_data['target']
        
        # Select features
        if self.feature_selector:
            X = self.feature_selector.select_features(X, y)
            
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(X) - self.sequence_length + 1):
            seq = X.iloc[i:i+self.sequence_length].values
            target_val = y.iloc[i+self.sequence_length-1]
            
            sequences.append(seq)
            targets.append(target_val)
            
        # Convert to numpy arrays
        X_seq = np.array(sequences)
        y_seq = np.array(targets)
        
        # Split into train and test sets
        train_size = int(len(X_seq) * train_ratio)
        X_train = X_seq[:train_size]
        y_train = y_seq[:train_size]
        X_test = X_seq[train_size:]
        y_test = y_seq[train_size:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'target_transformation': target_transformation,
            'forecast_horizon': self.forecast_horizon,
            'sequence_length': self.sequence_length
        }
        
    def prepare_multistep_forecast_inputs(
        self, 
        data: pd.DataFrame, 
        indicators: Dict[str, Any],
        target_column: str = 'close',
        target_transformation: str = 'returns',
        forecast_steps: List[int] = None,
        train_ratio: float = 0.8
    ) -> Dict[str, Any]:
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
        # Default forecast steps if not provided
        forecast_steps = forecast_steps or [1, 3, 5, 10, 20]
        
        # Extract features from indicators
        features_df = self.feature_extractor.extract_features(data, indicators)
        
        # Create multiple target variables
        targets = {}
        
        for step in forecast_steps:
            if target_transformation == 'returns':
                targets[f'target_{step}'] = data[target_column].pct_change(step).shift(-step)
            elif target_transformation == 'log_returns':
                targets[f'target_{step}'] = np.log(data[target_column] / data[target_column].shift(step)).shift(-step)
            elif target_transformation == 'direction':
                targets[f'target_{step}'] = np.sign(data[target_column].diff(step).shift(-step))
            else:
                targets[f'target_{step}'] = data[target_column].shift(-step)
                
        # Combine targets into a DataFrame
        targets_df = pd.DataFrame(targets)
        
        # Align features and targets
        aligned_data = pd.concat([features_df, targets_df], axis=1).dropna()
        
        if len(aligned_data) == 0:
            return {'error': 'No valid data after alignment'}
            
        # Split features and targets
        X = aligned_data[[col for col in aligned_data.columns if not col.startswith('target_')]]
        y = aligned_data[[col for col in aligned_data.columns if col.startswith('target_')]]
        
        # Select features
        if self.feature_selector:
            # Use the target for the middle step for feature selection
            mid_step = forecast_steps[len(forecast_steps) // 2]
            X = self.feature_selector.select_features(X, y[f'target_{mid_step}'])
            
        # Split into train and test sets
        train_size = int(len(X) * train_ratio)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'target_columns': y.columns.tolist(),
            'target_transformation': target_transformation,
            'forecast_steps': forecast_steps
        }
        
    def save_preparation_config(self, path: str) -> None:
        """Save model input preparation configuration to file."""
        config = {
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon
        }
        
        with open(path, 'wb') as f:
            pickle.dump(config, f)
            
    def load_preparation_config(self, path: str) -> None:
        """Load model input preparation configuration from file."""
        with open(path, 'rb') as f:
            config = pickle.load(f)
            
        self.sequence_length = config.get('sequence_length', self.sequence_length)
        self.forecast_horizon = config.get('forecast_horizon', self.forecast_horizon)
