"""
ML Integration: Feature Importance Analysis Module

This module provides tools for analyzing and ranking feature importance
for machine learning models in financial time series contexts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Union, Optional, Any, Callable
from enum import Enum
import logging
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LogisticRegression
import shap
import eli5
from eli5.sklearn import PermutationImportance
import joblib
import os
from pathlib import Path
from datetime import datetime
import json

# Configure logging
logger = logging.getLogger(__name__)


class ImportanceMethod(Enum):
    """Methods for calculating feature importance"""
    PERMUTATION = "permutation"  # Permutation-based importance
    MODEL = "model"  # Model-specific importance (e.g., feature_importances_)
    SHAP = "shap"  # SHAP values
    MUTUAL_INFO = "mutual_info"  # Mutual information
    CORRELATION = "correlation"  # Correlation with target
    LASSO = "lasso"  # Lasso coefficients


@dataclass
class FeatureImportanceResult:
    """Class to store feature importance results"""
    feature_names: List[str]
    importance_scores: List[float]
    method: ImportanceMethod
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize after construction"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
            
    @property
    def ranked_features(self) -> List[Tuple[str, float]]:
        """Get features ranked by importance"""
        return sorted(zip(self.feature_names, self.importance_scores), 
                     key=lambda x: abs(x[1]), reverse=True)
    
    @property
    def top_features(self) -> List[str]:
        """Get top features by importance"""
        return [f[0] for f in self.ranked_features]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "feature_names": self.feature_names,
            "importance_scores": [float(score) for score in self.importance_scores],
            "method": self.method.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureImportanceResult':
        """Create from dictionary"""
        return cls(
            feature_names=data["feature_names"],
            importance_scores=data["importance_scores"],
            method=ImportanceMethod(data["method"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )
    
    def save(self, filepath: str) -> None:
        """Save to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureImportanceResult':
        """Load from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using multiple methods
    """
    
    def __init__(
        self, 
        n_jobs: int = -1,
        n_repeats: int = 10,
        random_state: int = 42
    ):
        """
        Initialize feature importance analyzer
        
        Args:
            n_jobs: Number of parallel jobs for computations
            n_repeats: Number of repeats for permutation importance
            random_state: Random seed for reproducibility
        """
        self.n_jobs = n_jobs
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.results = {}  # Store results from different methods
        
    def analyze(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        method: Union[ImportanceMethod, str, List[Union[ImportanceMethod, str]]] = None,
        model=None
    ) -> Dict[str, FeatureImportanceResult]:
        """
        Analyze feature importance using specified method(s)
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: Method or list of methods to use
            model: Pre-trained model to use (required for some methods)
            
        Returns:
            Dictionary mapping method names to importance results
        """
        # Handle empty inputs
        if X.empty or len(y) == 0:
            logger.warning("Empty input data, cannot analyze feature importance")
            return {}
            
        # Default to all methods if none specified
        if method is None:
            methods = [
                ImportanceMethod.PERMUTATION,
                ImportanceMethod.MUTUAL_INFO,
                ImportanceMethod.CORRELATION
            ]
        elif isinstance(method, (str, ImportanceMethod)):
            # Convert to list with single method
            if isinstance(method, str):
                try:
                    method = ImportanceMethod(method)
                except ValueError:
                    logger.warning(f"Unknown method {method}, using PERMUTATION instead")
                    method = ImportanceMethod.PERMUTATION
            methods = [method]
        else:
            # Already a list, convert any string methods to enum
            methods = []
            for m in method:
                if isinstance(m, str):
                    try:
                        methods.append(ImportanceMethod(m))
                    except ValueError:
                        logger.warning(f"Unknown method {m}, skipping")
                else:
                    methods.append(m)
        
        # Clear previous results for these methods
        for m in methods:
            if m.value in self.results:
                del self.results[m.value]
                
        # Run each method
        results = {}
        
        for method in methods:
            try:
                # Check if model is required but not provided
                if method in [ImportanceMethod.MODEL, ImportanceMethod.SHAP] and model is None:
                    # Create default model based on target type
                    model = self._create_default_model(y)
                    # Train the model
                    model.fit(X, y)
                
                # Calculate importance
                importance_result = self._calculate_importance(X, y, method, model)
                
                # Store result
                if importance_result is not None:
                    self.results[method.value] = importance_result
                    results[method.value] = importance_result
            except Exception as e:
                logger.error(f"Error analyzing feature importance with method {method}: {str(e)}")
                
        return results
    
    def _calculate_importance(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        method: ImportanceMethod,
        model=None
    ) -> Optional[FeatureImportanceResult]:
        """
        Calculate feature importance using specified method
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: Method to use
            model: Pre-trained model to use
            
        Returns:
            Feature importance result or None on failure
        """
        feature_names = list(X.columns)
        
        # Check for classification vs regression
        is_classifier = self._is_classification_task(y)
        
        if method == ImportanceMethod.PERMUTATION:
            # Permutation importance
            if model is None:
                model = self._create_default_model(y)
                model.fit(X, y)
                
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            
            return FeatureImportanceResult(
                feature_names=feature_names,
                importance_scores=perm_importance.importances_mean,
                method=method,
                metadata={
                    "std": perm_importance.importances_std.tolist(),
                    "is_classifier": is_classifier
                }
            )
            
        elif method == ImportanceMethod.MODEL:
            # Model-specific importance
            if model is None:
                model = self._create_default_model(y)
                model.fit(X, y)
                
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                if len(model.coef_.shape) > 1:
                    # For multi-class, take the mean across classes
                    importance_scores = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importance_scores = np.abs(model.coef_)
            else:
                logger.warning(f"Model doesn't have feature_importances_ or coef_ attribute")
                return None
                
            return FeatureImportanceResult(
                feature_names=feature_names,
                importance_scores=importance_scores,
                method=method,
                metadata={"model_type": type(model).__name__}
            )
            
        elif method == ImportanceMethod.SHAP:
            # SHAP values
            if model is None:
                model = self._create_default_model(y)
                model.fit(X, y)
            
            try:
                # Initialize SHAP explainer based on model type
                if hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model) if hasattr(model, 'estimators_') else shap.Explainer(model)
                else:
                    explainer = shap.Explainer(model)
                
                # Calculate SHAP values (use a subset if data is large)
                max_samples = min(len(X), 1000)  # Limit for performance
                if len(X) > max_samples:
                    X_sample = X.sample(max_samples, random_state=self.random_state)
                else:
                    X_sample = X
                
                shap_values = explainer(X_sample)
                
                # Use mean absolute SHAP value as importance
                if len(shap_values.shape) > 2:  # Multi-class
                    importance_scores = np.mean(np.mean(np.abs(shap_values.values), axis=0), axis=0)
                else:
                    importance_scores = np.mean(np.abs(shap_values.values), axis=0)
                
                return FeatureImportanceResult(
                    feature_names=feature_names,
                    importance_scores=importance_scores,
                    method=method,
                    metadata={"model_type": type(model).__name__}
                )
            except Exception as e:
                logger.error(f"Error calculating SHAP values: {str(e)}")
                return None
                
        elif method == ImportanceMethod.MUTUAL_INFO:
            # Mutual information
            if is_classifier:
                importance_scores = mutual_info_classif(
                    X, y,
                    random_state=self.random_state,
                    n_neighbors=3
                )
            else:
                importance_scores = mutual_info_regression(
                    X, y,
                    random_state=self.random_state,
                    n_neighbors=3
                )
                
            return FeatureImportanceResult(
                feature_names=feature_names,
                importance_scores=importance_scores,
                method=method,
                metadata={"is_classifier": is_classifier}
            )
            
        elif method == ImportanceMethod.CORRELATION:
            # Correlation with target
            try:
                # Ensure y is a Series with a name
                if not isinstance(y, pd.Series):
                    y = pd.Series(y, name='target')
                elif y.name is None:
                    y = y.rename('target')
                
                # Create a DataFrame with features and target
                df = pd.concat([X, y], axis=1)
                
                # Calculate correlation with target
                correlations = df.corr()[y.name].drop(y.name)
                
                return FeatureImportanceResult(
                    feature_names=list(correlations.index),
                    importance_scores=correlations.values,
                    method=method,
                    metadata={
                        "is_classifier": is_classifier,
                        "correlation_method": "pearson"
                    }
                )
            except Exception as e:
                logger.error(f"Error calculating correlation importance: {str(e)}")
                return None
                
        elif method == ImportanceMethod.LASSO:
            # Lasso coefficients
            try:
                # Scale features for better Lasso performance
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                if is_classifier:
                    # Use Logistic Regression with L1 penalty
                    lasso = LogisticRegression(
                        penalty='l1', 
                        solver='liblinear',
                        C=0.1,  # Higher regularization
                        random_state=self.random_state
                    )
                else:
                    # Use Lasso regression
                    lasso = Lasso(
                        alpha=0.1,
                        random_state=self.random_state
                    )
                
                lasso.fit(X_scaled, y)
                
                if is_classifier and len(lasso.coef_.shape) > 1:
                    # Multi-class case, take mean of absolute coefficients
                    importance_scores = np.mean(np.abs(lasso.coef_), axis=0)
                else:
                    importance_scores = np.abs(lasso.coef_)
                
                return FeatureImportanceResult(
                    feature_names=feature_names,
                    importance_scores=importance_scores,
                    method=method,
                    metadata={"is_classifier": is_classifier}
                )
            except Exception as e:
                logger.error(f"Error calculating Lasso importance: {str(e)}")
                return None
                
        else:
            logger.warning(f"Unknown importance method: {method}")
            return None
    
    def _create_default_model(self, y: Union[pd.Series, np.ndarray]):
        """Create a default model based on target type"""
        is_classifier = self._is_classification_task(y)
        
        if is_classifier:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
    
    def _is_classification_task(self, y: Union[pd.Series, np.ndarray]) -> bool:
        """Determine if this is a classification task based on target values"""
        unique_values = pd.Series(y).nunique()
        
        # Heuristic: if few unique values, likely classification
        if unique_values <= 10 and unique_values < len(y) / 10:
            return True
            
        # Check if values are all integers or boolean
        y_series = pd.Series(y)
        if pd.api.types.is_bool_dtype(y_series) or pd.api.types.is_integer_dtype(y_series) and y_series.nunique() <= 100:
            return True
            
        return False
        
    def get_combined_importance(
        self, 
        top_k: int = None, 
        methods: List[ImportanceMethod] = None,
        weights: List[float] = None
    ) -> pd.DataFrame:
        """
        Get combined feature importance across multiple methods
        
        Args:
            top_k: Number of top features to include (None for all)
            methods: Methods to include (None for all available)
            weights: Weights for each method (None for equal weights)
            
        Returns:
            DataFrame with combined importance scores
        """
        if not self.results:
            logger.warning("No feature importance results available")
            return pd.DataFrame()
            
        # Default to all available methods
        if methods is None:
            methods = [ImportanceMethod(m) for m in self.results.keys()]
            
        # Filter to methods with results
        methods = [m for m in methods if m.value in self.results]
        
        if not methods:
            logger.warning("No results available for specified methods")
            return pd.DataFrame()
            
        # Default to equal weights
        if weights is None:
            weights = [1.0] * len(methods)
        elif len(weights) != len(methods):
            logger.warning("Number of weights doesn't match methods, using equal weights")
            weights = [1.0] * len(methods)
            
        # Normalize weights to sum to 1
        weights = np.array(weights) / np.sum(weights)
        
        # Get all unique features across methods
        all_features = set()
        for m in methods:
            if m.value in self.results:
                all_features.update(self.results[m.value].feature_names)
        
        # Create result DataFrame
        result = pd.DataFrame(index=sorted(all_features))
        
        # Add individual method scores
        for method in methods:
            if method.value not in self.results:
                continue
                
            importance = self.results[method.value]
            # Create a Series from the result
            method_scores = pd.Series(
                dict(zip(importance.feature_names, importance.importance_scores)),
                name=method.value
            )
            
            # Add to result
            result = result.join(method_scores)
        
        # Fill NaN values with 0
        result = result.fillna(0)
        
        # Calculate combined score
        combined_scores = pd.Series(0, index=result.index)
        for i, method in enumerate(methods):
            if method.value in result.columns:
                # Normalize scores for the method (0-1 range)
                method_col = result[method.value]
                if method_col.max() > method_col.min():
                    normalized = (method_col - method_col.min()) / (method_col.max() - method_col.min())
                else:
                    normalized = method_col / method_col.max() if method_col.max() > 0 else method_col
                
                # Add weighted normalized scores
                combined_scores += normalized * weights[i]
        
        # Add combined score to result
        result['combined_score'] = combined_scores
        
        # Sort by combined score
        result = result.sort_values('combined_score', ascending=False)
        
        # Limit to top_k if specified
        if top_k is not None and top_k < len(result):
            result = result.head(top_k)
            
        return result
    
    def plot_importance(
        self,
        result: Union[FeatureImportanceResult, str] = None,
        top_k: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            result: Importance result or method name (None for combined)
            top_k: Number of top features to show
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        
        if result is None:
            # Use combined importance
            df = self.get_combined_importance(top_k=top_k)
            if df.empty:
                logger.warning("No feature importance results available to plot")
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, "No feature importance data available", 
                        ha='center', va='center', fontsize=12)
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight')
                return fig
                
            title = "Combined Feature Importance"
            importance_scores = df['combined_score']
            feature_names = df.index
            
        elif isinstance(result, str):
            # Use specified method
            if result not in self.results:
                logger.warning(f"No results available for method {result}")
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(0.5, 0.5, f"No results for method {result}", 
                        ha='center', va='center', fontsize=12)
                if save_path:
                    plt.savefig(save_path, bbox_inches='tight')
                return fig
                
            importance = self.results[result]
            title = f"Feature Importance ({result})"
            
            # Get top features
            ranked = sorted(zip(importance.feature_names, importance.importance_scores),
                         key=lambda x: abs(x[1]), reverse=True)
            if top_k is not None:
                ranked = ranked[:top_k]
                
            feature_names, importance_scores = zip(*ranked)
            
        else:
            # Use provided result
            importance = result
            title = f"Feature Importance ({importance.method.value})"
            
            # Get top features
            ranked = sorted(zip(importance.feature_names, importance.importance_scores),
                         key=lambda x: abs(x[1]), reverse=True)
            if top_k is not None:
                ranked = ranked[:top_k]
                
            feature_names, importance_scores = zip(*ranked)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, importance_scores, align='center')
        
        # Color bars based on importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        for i, bar in enumerate(bars):
            bar.set_color(colors[i])
        
        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        return fig
    
    def save_results(self, directory: str) -> None:
        """
        Save all importance results to files
        
        Args:
            directory: Directory to save results
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save individual results
        for method, result in self.results.items():
            filepath = os.path.join(directory, f"importance_{method}.json")
            result.save(filepath)
            
        # Save combined result if available
        if len(self.results) > 1:
            combined = self.get_combined_importance()
            if not combined.empty:
                combined_path = os.path.join(directory, "importance_combined.csv")
                combined.to_csv(combined_path)
                
                # Save plot
                plot_path = os.path.join(directory, "importance_plot.png")
                self.plot_importance(save_path=plot_path)
    
    def load_results(self, directory: str) -> None:
        """
        Load importance results from files
        
        Args:
            directory: Directory with saved results
        """
        if not os.path.isdir(directory):
            logger.warning(f"Directory not found: {directory}")
            return
            
        # Clear current results
        self.results = {}
        
        # Find all JSON files
        json_files = list(Path(directory).glob("importance_*.json"))
        
        for file_path in json_files:
            try:
                result = FeatureImportanceResult.load(str(file_path))
                method = result.method.value
                self.results[method] = result
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")


class FeatureSelector:
    """
    Select features based on importance analysis
    """
    
    def __init__(
        self, 
        importance_analyzer: FeatureImportanceAnalyzer = None,
        selection_threshold: float = 0.01,
        max_features: int = None,
        use_cumulative: bool = True
    ):
        """
        Initialize feature selector
        
        Args:
            importance_analyzer: Feature importance analyzer
            selection_threshold: Minimum importance threshold for selection
            max_features: Maximum number of features to select
            use_cumulative: Whether to use cumulative importance threshold
        """
        self.importance_analyzer = importance_analyzer or FeatureImportanceAnalyzer()
        self.selection_threshold = selection_threshold
        self.max_features = max_features
        self.use_cumulative = use_cumulative
        self.selected_features = None
        
    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        importance_methods: List[ImportanceMethod] = None
    ) -> 'FeatureSelector':
        """
        Fit selector to training data
        
        Args:
            X: Feature DataFrame
            y: Target variable
            importance_methods: Methods to use for importance analysis
            
        Returns:
            Self for method chaining
        """
        # Default importance methods
        if importance_methods is None:
            importance_methods = [
                ImportanceMethod.PERMUTATION,
                ImportanceMethod.MUTUAL_INFO
            ]
            
        # Analyze feature importance
        self.importance_analyzer.analyze(X, y, method=importance_methods)
        
        # Get combined importance
        importance_df = self.importance_analyzer.get_combined_importance()
        
        if importance_df.empty:
            logger.warning("No feature importance results available")
            self.selected_features = list(X.columns)
            return self
            
        # Normalize scores
        combined_score = importance_df['combined_score']
        if combined_score.max() > 0:
            normalized_scores = combined_score / combined_score.max()
        else:
            normalized_scores = combined_score
        
        # Select features
        if self.use_cumulative:
            # Sort by importance
            sorted_importance = normalized_scores.sort_values(ascending=False)
            
            # Calculate cumulative importance
            cumulative_importance = sorted_importance.cumsum()
            
            # Select features based on cumulative threshold
            selected = cumulative_importance[cumulative_importance <= 0.95].index.tolist()
            
            # Ensure we have at least one feature
            if not selected:
                selected = [sorted_importance.index[0]]
        else:
            # Select based on threshold
            selected = normalized_scores[normalized_scores >= self.selection_threshold].index.tolist()
        
        # Limit to max_features if specified
        if self.max_features is not None and len(selected) > self.max_features:
            # Get top features by importance
            top_indices = np.argsort(-normalized_scores.loc[selected].values)[:self.max_features]
            selected = [selected[i] for i in top_indices]
        
        self.selected_features = selected
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to include only selected features
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with selected features
        """
        if self.selected_features is None:
            logger.warning("Selector not fitted, returning all features")
            return X
            
        # Select columns that exist in X
        valid_features = [f for f in self.selected_features if f in X.columns]
        
        if len(valid_features) == 0:
            logger.warning("No selected features found in input data")
            return X
            
        return X[valid_features]
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        importance_methods: List[ImportanceMethod] = None
    ) -> pd.DataFrame:
        """
        Fit to data, then transform it
        
        Args:
            X: Feature DataFrame
            y: Target variable
            importance_methods: Methods to use for importance analysis
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y, importance_methods).transform(X)
        
    @property
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance DataFrame"""
        if not hasattr(self.importance_analyzer, 'results') or not self.importance_analyzer.results:
            logger.warning("No feature importance results available")
            return pd.DataFrame()
            
        return self.importance_analyzer.get_combined_importance()
