"""
Feature Importance Analysis Module

This module implements tools for analyzing and interpreting the importance of features
in machine learning models to provide explainability and transparency.
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from pydantic import BaseModel, Field
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
try:
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
try:
    import eli5
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False
from ml_workbench_service.model_registry.model_registry_service import ModelRegistryService
from ml_workbench_service.model_registry.registry import ModelFramework
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeatureImportance(BaseModel):
    """Model for feature importance data."""
    feature_name: str
    importance_score: float
    std_dev: Optional[float] = None
    confidence_interval_low: Optional[float] = None
    confidence_interval_high: Optional[float] = None
    rank: int


class ModelFeatureImportance(BaseModel):
    """Model for complete feature importance analysis results."""
    model_id: str
    version_id: str
    features: List[FeatureImportance]
    method: str
    explanation: Optional[str] = None
    visualization_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeatureImportanceAnalyzer:
    """
    Analyzer for calculating and visualizing feature importance across 
    different model types. Supports various techniques:
    - Built-in importance (tree-based models)
    - SHAP values
    - Permutation importance
    - Partial dependence plots
    - Feature interaction strength
    """

    def __init__(self, model_registry_service: ModelRegistryService):
        """Initialize the analyzer with a model registry service."""
        self.model_registry = model_registry_service

    @with_exception_handling
    def analyze_feature_importance(self, model_id: str, version_id: str,
        X_sample: pd.DataFrame, y_sample: Optional[pd.Series]=None, method:
        str='auto', n_samples: int=100, random_state: int=42
        ) ->ModelFeatureImportance:
        """
        Analyze feature importance for the specified model and version.
        
        Args:
            model_id: ID of the model
            version_id: ID of the model version
            X_sample: Sample data for feature importance analysis
            y_sample: Sample target data (required for some methods)
            method: Method to use ('auto', 'built_in', 'shap', 'permutation', 'eli5')
            n_samples: Number of samples to use for permutation importance
            random_state: Random state for reproducibility
            
        Returns:
            ModelFeatureImportance: Feature importance analysis results
        """
        try:
            model_version = self.model_registry.get_model_version(model_id,
                version_id)
            model_path = self.model_registry.get_model_artifact_path(model_id,
                version_id)
            model_framework = model_version.framework
            model = self._load_model(model_path, model_framework)
            if model is None:
                raise ValueError(
                    f'Failed to load model with framework {model_framework}')
            if method == 'auto':
                method = self._auto_select_method(model_framework, model)
            if method == 'built_in':
                feature_importances = self._get_built_in_importance(model,
                    X_sample)
            elif method == 'shap' and SHAP_AVAILABLE:
                feature_importances = self._get_shap_importance(model, X_sample
                    )
            elif method == 'permutation' and SKLEARN_AVAILABLE:
                if y_sample is None:
                    raise ValueError(
                        'y_sample is required for permutation importance')
                feature_importances = self._get_permutation_importance(model,
                    X_sample, y_sample, n_samples, random_state)
            elif method == 'eli5' and ELI5_AVAILABLE:
                if y_sample is None:
                    raise ValueError('y_sample is required for ELI5 importance'
                        )
                feature_importances = self._get_eli5_importance(model,
                    X_sample, y_sample, n_samples, random_state)
            else:
                raise ValueError(
                    f'Method {method} not supported or required package not available'
                    )
            visualization_data = self._create_visualization(feature_importances
                , method)
            result = ModelFeatureImportance(model_id=model_id, version_id=
                version_id, features=feature_importances, method=method,
                explanation=self._get_method_explanation(method),
                visualization_data=visualization_data)
            return result
        except Exception as e:
            logger.error(f'Error analyzing feature importance: {str(e)}')
            raise

    @with_exception_handling
    def _load_model(self, model_path: str, framework: ModelFramework):
        """Load a model based on its framework."""
        try:
            if framework == ModelFramework.SKLEARN:
                import joblib
                return joblib.load(model_path)
            elif framework == ModelFramework.XGBOOST:
                import xgboost as xgb
                return xgb.Booster(model_file=model_path)
            elif framework == ModelFramework.LIGHTGBM:
                import lightgbm as lgb
                return lgb.Booster(model_file=model_path)
            elif framework == ModelFramework.PYTORCH:
                import torch
                return torch.load(model_path)
            elif framework == ModelFramework.TENSORFLOW:
                import tensorflow as tf
                return tf.keras.models.load_model(model_path)
            else:
                logger.warning(f'Unsupported model framework: {framework}')
                return None
        except Exception as e:
            logger.error(f'Error loading model: {str(e)}')
            return None

    def _auto_select_method(self, framework: ModelFramework, model) ->str:
        """Automatically select the best feature importance method based on model type."""
        if framework in [ModelFramework.SKLEARN, ModelFramework.XGBOOST,
            ModelFramework.LIGHTGBM]:
            if hasattr(model, 'feature_importances_'):
                return 'built_in'
        if SHAP_AVAILABLE:
            return 'shap'
        if SKLEARN_AVAILABLE:
            return 'permutation'
        if ELI5_AVAILABLE:
            return 'eli5'
        return 'built_in'

    @with_exception_handling
    def _get_built_in_importance(self, model, X_sample: pd.DataFrame) ->List[
        FeatureImportance]:
        """Get built-in feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if len(model.coef_.shape
                    ) > 1 else np.abs(model.coef_)
            else:
                raise AttributeError(
                    'Model does not have built-in feature importance')
            feature_names = X_sample.columns
            features = []
            sorted_idx = np.argsort(importances)[::-1]
            for rank, idx in enumerate(sorted_idx):
                features.append(FeatureImportance(feature_name=
                    feature_names[idx], importance_score=float(importances[
                    idx]), rank=rank + 1))
            return features
        except Exception as e:
            logger.error(f'Error getting built-in feature importance: {str(e)}'
                )
            raise

    @with_exception_handling
    def _get_shap_importance(self, model, X_sample: pd.DataFrame) ->List[
        FeatureImportance]:
        """Get SHAP-based feature importance."""
        try:
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model)
            else:
                explainer = shap.Explainer(model)
            shap_values = explainer(X_sample)
            if hasattr(shap_values, 'values'):
                mean_shap = np.abs(shap_values.values).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
            feature_names = X_sample.columns
            features = []
            sorted_idx = np.argsort(mean_shap)[::-1]
            for rank, idx in enumerate(sorted_idx):
                features.append(FeatureImportance(feature_name=
                    feature_names[idx], importance_score=float(mean_shap[
                    idx]), rank=rank + 1))
            return features
        except Exception as e:
            logger.error(f'Error getting SHAP importance: {str(e)}')
            raise

    @with_exception_handling
    def _get_permutation_importance(self, model, X_sample: pd.DataFrame,
        y_sample: pd.Series, n_samples: int, random_state: int) ->List[
        FeatureImportance]:
        """Get permutation-based feature importance."""
        try:
            perm_importance = permutation_importance(model, X_sample,
                y_sample, n_repeats=n_samples, random_state=random_state)
            feature_names = X_sample.columns
            importances = perm_importance.importances_mean
            stds = perm_importance.importances_std
            features = []
            sorted_idx = np.argsort(importances)[::-1]
            for rank, idx in enumerate(sorted_idx):
                features.append(FeatureImportance(feature_name=
                    feature_names[idx], importance_score=float(importances[
                    idx]), std_dev=float(stds[idx]), rank=rank + 1))
            return features
        except Exception as e:
            logger.error(f'Error getting permutation importance: {str(e)}')
            raise

    @with_exception_handling
    def _get_eli5_importance(self, model, X_sample: pd.DataFrame, y_sample:
        pd.Series, n_samples: int, random_state: int) ->List[FeatureImportance
        ]:
        """Get ELI5-based feature importance."""
        try:
            perm = PermutationImportance(model, random_state=random_state,
                n_iter=n_samples).fit(X_sample, y_sample)
            feature_names = X_sample.columns
            importances = perm.feature_importances_
            features = []
            sorted_idx = np.argsort(importances)[::-1]
            for rank, idx in enumerate(sorted_idx):
                features.append(FeatureImportance(feature_name=
                    feature_names[idx], importance_score=float(importances[
                    idx]), rank=rank + 1))
            return features
        except Exception as e:
            logger.error(f'Error getting ELI5 importance: {str(e)}')
            raise

    @with_exception_handling
    def _create_visualization(self, features: List[FeatureImportance],
        method: str) ->Dict[str, Any]:
        """Create visualization data for feature importance."""
        try:
            sorted_features = sorted(features, key=lambda x: x.
                importance_score, reverse=True)
            feature_names = [f.feature_name for f in sorted_features[:15]]
            importance_scores = [f.importance_score for f in
                sorted_features[:15]]
            plt.figure(figsize=(10, 6))
            bars = plt.barh(feature_names, importance_scores)
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance ({method})')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            visualization = {'plot_type': 'bar', 'base64_image': img_str,
                'data': {'feature_names': feature_names,
                'importance_scores': importance_scores}, 'method': method}
            return visualization
        except Exception as e:
            logger.error(f'Error creating visualization: {str(e)}')
            return {}

    def _get_method_explanation(self, method: str) ->str:
        """Get explanation text for the feature importance method."""
        explanations = {'built_in':
            "This analysis uses the model's built-in feature importance, which is typically based on how much each feature contributes to decreasing the loss function during training. For tree-based models, this represents the total reduction of the criterion brought by each feature."
            , 'shap':
            'SHAP (SHapley Additive exPlanations) values show the impact of each feature on individual predictions. These values represent the contribution of each feature to the difference between the actual prediction and the mean prediction. Higher absolute values indicate stronger impact.'
            , 'permutation':
            "Permutation importance measures the increase in prediction error when a feature's values are randomly shuffled. This breaks the relationship between the feature and the true outcome, indicating how much the model depends on that feature for accuracy. Higher values indicate more important features."
            , 'eli5':
            "This analysis uses ELI5's permutation importance, which measures how model performance decreases when a single feature is randomly shuffled. Features that cause larger performance drops when shuffled are considered more important."
            }
        return explanations.get(method,
            'Feature importance analysis showing the relative impact of each feature on model predictions.'
            )


class PartialDependenceAnalyzer:
    """
    Analyzer for calculating partial dependence plots to show how features
    influence model predictions, controlling for other features.
    """

    def __init__(self, model_registry_service: ModelRegistryService):
        """Initialize the analyzer with a model registry service."""
        self.model_registry = model_registry_service

    def compute_partial_dependence(self, model_id: str, version_id: str,
        X_sample: pd.DataFrame, features: List[str], feature_pairs:
        Optional[List[Tuple[str, str]]]=None, grid_resolution: int=20) ->Dict[
        str, Any]:
        """
        Compute partial dependence for specified features or feature pairs.
        
        Args:
            model_id: ID of the model
            version_id: ID of the model version
            X_sample: Sample data for partial dependence analysis
            features: List of feature names for individual PDPs
            feature_pairs: List of feature name pairs for 2D PDPs
            grid_resolution: Number of points in the grid for each feature
            
        Returns:
            dict: Partial dependence analysis results
        """
        return {'message': 'Partial dependence calculation not yet implemented'
            }


class FeatureInteractionAnalyzer:
    """
    Analyzer for calculating feature interactions to identify which features
    work together to influence predictions.
    """

    def __init__(self, model_registry_service: ModelRegistryService):
        """Initialize the analyzer with a model registry service."""
        self.model_registry = model_registry_service

    def analyze_feature_interactions(self, model_id: str, version_id: str,
        X_sample: pd.DataFrame, method: str='shap') ->Dict[str, Any]:
        """
        Analyze interactions between features.
        
        Args:
            model_id: ID of the model
            version_id: ID of the model version
            X_sample: Sample data for interaction analysis
            method: Method to use for interaction analysis
            
        Returns:
            dict: Feature interaction analysis results
        """
        return {'message': 'Feature interaction analysis not yet implemented'}
