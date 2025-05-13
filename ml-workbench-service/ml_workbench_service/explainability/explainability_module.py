"""
Explainability Module Implementation

This module provides explainability tools for understanding the predictions
made by machine learning models in the forex trading platform.
"""
import os
import json
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import shap
from lime import lime_tabular
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from core_foundations.utils.logger import get_logger
from ml_workbench_service.models.multitask.multitask_model import MultitaskModel, TaskType
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ModelExplainer:
    """
    Base class for model explainers that provide interpretability
    for machine learning model predictions.
    
    Attributes:
        model: The model to explain
        feature_names: Names of input features
        class_names: Names of output classes (for classification)
        explainer_type: Type of explainer being used
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]]=None,
        class_names: Optional[List[str]]=None, explainer_type: str='base'):
        """
        Initialize the ModelExplainer.
        
        Args:
            model: The model to be explained
            feature_names: List of feature names
            class_names: List of class names (for classification)
            explainer_type: Type of explainer being used
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer_type = explainer_type

    def explain(self, X: Union[np.ndarray, pd.DataFrame]) ->Dict[str, Any]:
        """
        Abstract method to explain model predictions.
        
        Args:
            X: Input data to generate explanations for
            
        Returns:
            Dictionary with explanation results
        """
        raise NotImplementedError('Subclasses must implement explain method')

    def plot(self, explanation: Dict[str, Any], **kwargs) ->plt.Figure:
        """
        Abstract method to plot model explanations.
        
        Args:
            explanation: The explanation to plot
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib Figure with the explanation visualization
        """
        raise NotImplementedError('Subclasses must implement plot method')

    def save(self, path: str) ->None:
        """
        Save the explainer.
        
        Args:
            path: Directory path to save the explainer
        """
        os.makedirs(path, exist_ok=True)
        metadata = {'explainer_type': self.explainer_type, 'feature_names':
            self.feature_names, 'class_names': self.class_names,
            'date_saved': datetime.now().isoformat()}
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        logger.info(f'Saved {self.explainer_type} explainer metadata to {path}'
            )

    @classmethod
    def load(cls, path: str) ->'ModelExplainer':
        """
        Load a saved explainer. Abstract method to be implemented by subclasses.
        
        Args:
            path: Path to the saved explainer
        
        Returns:
            Loaded ModelExplainer instance
        """
        raise NotImplementedError('Subclasses must implement load method')


class SHAPExplainer(ModelExplainer):
    """
    SHAP (SHapley Additive exPlanations) explainer implementation.
    
    SHAP values represent the contribution of each feature to a prediction,
    based on cooperative game theory (Shapley values).
    """

    @with_exception_handling
    def __init__(self, model: Any, feature_names: Optional[List[str]]=None,
        class_names: Optional[List[str]]=None, model_output: Union[str, int
        ]=0, background_data: Optional[Union[np.ndarray, pd.DataFrame]]=None):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: The model to explain
            feature_names: Names of input features
            class_names: Names of output classes
            model_output: For multi-output models, which output to explain
            background_data: Reference dataset for explainer initialization
        """
        super().__init__(model, feature_names, class_names, 'shap')
        self.model_output = model_output
        if isinstance(model, tf.keras.Model):
            self.wrapped_model = self._create_keras_wrapper(model, model_output
                )
            if background_data is not None:
                self.explainer = shap.DeepExplainer(self.wrapped_model,
                    background_data)
            else:
                self.explainer = None
        else:
            try:
                self.explainer = shap.TreeExplainer(model)
            except:
                if background_data is not None:
                    self.explainer = shap.KernelExplainer(model.predict if
                        hasattr(model, 'predict') else model, background_data)
                else:
                    self.explainer = None

    def _create_keras_wrapper(self, model: tf.keras.Model, output_idx:
        Union[str, int]) ->Callable:
        """
        Create a wrapper function for Keras models to extract specific outputs.
        
        Args:
            model: Keras model
            output_idx: Index or name of the output to explain
            
        Returns:
            Wrapper function that returns the specified output
        """
        if isinstance(model.output, list) or hasattr(model.output,
            '_keras_shape') and len(model.output.shape) > 0:
            if isinstance(output_idx, str):
                for i, output in enumerate(model.outputs):
                    if output.name.split('/')[0] == output_idx:
                        output_idx = i
                        break

            def wrapped_predict(x):
    """
    Wrapped predict.
    
    Args:
        x: Description of x
    
    """

                preds = model.predict(x)
                return preds[output_idx]
            return wrapped_predict
        return model.predict

    def explain(self, X: Union[np.ndarray, pd.DataFrame], background_data:
        Optional[Union[np.ndarray, pd.DataFrame]]=None) ->Dict[str, Any]:
        """
        Generate SHAP values to explain model predictions.
        
        Args:
            X: Input data to explain
            background_data: Reference dataset (used if explainer not initialized)
            
        Returns:
            Dictionary with SHAP explanations
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
        if self.explainer is None:
            if background_data is None:
                background_data = X_values[:min(100, len(X_values))]
            if isinstance(self.model, tf.keras.Model):
                self.explainer = shap.DeepExplainer(self.wrapped_model,
                    background_data)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict if
                    hasattr(self.model, 'predict') else self.model,
                    background_data)
        shap_values = self.explainer.shap_values(X_values)
        if isinstance(shap_values, list) and not isinstance(self.model, tf.
            keras.Model):
            explanation = {'shap_values': shap_values, 'expected_value':
                self.explainer.expected_value, 'data': X_values,
                'feature_names': self.feature_names, 'class_names': self.
                class_names}
        else:
            explanation = {'shap_values': shap_values, 'expected_value':
                self.explainer.expected_value, 'data': X_values,
                'feature_names': self.feature_names}
        return explanation

    def plot(self, explanation: Dict[str, Any], plot_type: str='summary',
        max_display: int=20, **kwargs) ->plt.Figure:
        """
        Plot SHAP explanations.
        
        Args:
            explanation: SHAP explanation from the explain method
            plot_type: Type of plot ('summary', 'bar', 'waterfall', 'force', 'decision')
            max_display: Maximum number of features to show
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib Figure with the SHAP visualization
        """
        plt.figure(figsize=kwargs.get('figsize', (10, 8)))
        shap_values = explanation['shap_values']
        data = explanation['data']
        feature_names = explanation.get('feature_names')
        if plot_type == 'summary':
            if isinstance(shap_values, list) and len(shap_values) > 1:
                class_idx = kwargs.get('class_idx', 0)
                shap.summary_plot(shap_values[class_idx], data,
                    feature_names=feature_names, max_display=max_display,
                    show=False)
            else:
                shap.summary_plot(shap_values, data, feature_names=
                    feature_names, max_display=max_display, show=False)
        elif plot_type == 'bar':
            if isinstance(shap_values, list) and len(shap_values) > 1:
                class_idx = kwargs.get('class_idx', 0)
                shap.summary_plot(shap_values[class_idx], data,
                    feature_names=feature_names, plot_type='bar',
                    max_display=max_display, show=False)
            else:
                shap.summary_plot(shap_values, data, feature_names=
                    feature_names, plot_type='bar', max_display=max_display,
                    show=False)
        elif plot_type == 'waterfall':
            instance_idx = kwargs.get('instance_idx', 0)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                class_idx = kwargs.get('class_idx', 0)
                shap.waterfall_plot(shap.Explanation(values=shap_values[
                    class_idx][instance_idx], base_values=explanation[
                    'expected_value'][class_idx], data=data[instance_idx],
                    feature_names=feature_names), max_display=max_display,
                    show=False)
            else:
                base_value = explanation['expected_value']
                if isinstance(base_value, list):
                    base_value = base_value[0]
                shap.waterfall_plot(shap.Explanation(values=shap_values[
                    instance_idx], base_values=base_value, data=data[
                    instance_idx], feature_names=feature_names),
                    max_display=max_display, show=False)
        elif plot_type == 'force':
            instance_idx = kwargs.get('instance_idx', 0)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                class_idx = kwargs.get('class_idx', 0)
                shap.force_plot(explanation['expected_value'][class_idx],
                    shap_values[class_idx][instance_idx], data[instance_idx
                    ], feature_names=feature_names, show=False)
            else:
                base_value = explanation['expected_value']
                if isinstance(base_value, list):
                    base_value = base_value[0]
                shap.force_plot(base_value, shap_values[instance_idx], data
                    [instance_idx], feature_names=feature_names, show=False)
        fig = plt.gcf()
        return fig

    @with_exception_handling
    def save(self, path: str) ->None:
        """
        Save the SHAP explainer.
        
        Args:
            path: Directory path to save the explainer
        """
        super().save(path)
        try:
            with open(os.path.join(path, 'explainer.pkl'), 'wb') as f:
                pickle.dump(self.explainer, f)
        except:
            logger.warning(
                'Could not pickle the SHAP explainer. Will need to reinitialize on load.'
                )

    @classmethod
    @with_exception_handling
    def load(cls, path: str, model: Any, background_data: Optional[Union[np
        .ndarray, pd.DataFrame]]=None) ->'SHAPExplainer':
        """
        Load a saved SHAP explainer.
        
        Args:
            path: Path to the saved explainer
            model: Model to use with the loaded explainer
            background_data: Background data if the explainer needs to be reinitialized
            
        Returns:
            Loaded SHAPExplainer instance
        """
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        instance = cls(model=model, feature_names=metadata.get(
            'feature_names'), class_names=metadata.get('class_names'))
        try:
            with open(os.path.join(path, 'explainer.pkl'), 'rb') as f:
                instance.explainer = pickle.load(f)
        except:
            logger.warning(
                'Could not load the pickled SHAP explainer. Will initialize on the first call to explain().'
                )
        return instance


class LIMEExplainer(ModelExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations) explainer implementation.
    
    LIME creates a local surrogate model that approximates the behavior of the
    original model in the neighborhood of a specific instance.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]]=None,
        class_names: Optional[List[str]]=None, mode: str='classification',
        training_data: Optional[Union[np.ndarray, pd.DataFrame]]=None):
        """
        Initialize the LIME explainer.
        
        Args:
            model: The model to explain
            feature_names: Names of input features
            class_names: Names of output classes
            mode: 'classification' or 'regression'
            training_data: Training data for the explainer
        """
        super().__init__(model, feature_names, class_names, 'lime')
        self.mode = mode
        if training_data is not None:
            if isinstance(training_data, pd.DataFrame):
                if self.feature_names is None:
                    self.feature_names = training_data.columns.tolist()
                training_data = training_data.values
            self.explainer = lime_tabular.LimeTabularExplainer(training_data,
                feature_names=self.feature_names, class_names=self.
                class_names, mode=mode, verbose=False)
        else:
            self.explainer = None
        if isinstance(model, tf.keras.Model):
            self.predict_fn = model.predict
        else:
            self.predict_fn = model.predict if hasattr(model, 'predict'
                ) else model

    def explain(self, X: Union[np.ndarray, pd.DataFrame], num_features: int
        =10, num_samples: int=5000, training_data: Optional[Union[np.
        ndarray, pd.DataFrame]]=None) ->Dict[str, Any]:
        """
        Generate LIME explanations for model predictions.
        
        Args:
            X: Input data to explain
            num_features: Maximum number of features to include in explanation
            num_samples: Number of samples to generate for the surrogate model
            training_data: Training data if explainer not initialized
            
        Returns:
            Dictionary with LIME explanations
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
        if self.explainer is None:
            if training_data is None:
                raise ValueError(
                    'Training data must be provided if explainer is not initialized'
                    )
            if isinstance(training_data, pd.DataFrame):
                if self.feature_names is None:
                    self.feature_names = training_data.columns.tolist()
                training_data = training_data.values
            self.explainer = lime_tabular.LimeTabularExplainer(training_data,
                feature_names=self.feature_names, class_names=self.
                class_names, mode=self.mode, verbose=False)
        explanations = []
        for i in range(len(X_values)):
            exp = self.explainer.explain_instance(X_values[i], self.
                predict_fn, num_features=num_features, num_samples=num_samples)
            explanations.append(exp)
        return {'explanations': explanations, 'data': X_values,
            'feature_names': self.feature_names, 'class_names': self.
            class_names}

    def plot(self, explanation: Dict[str, Any], instance_idx: int=0,
        label_idx: int=0, **kwargs) ->plt.Figure:
        """
        Plot LIME explanation.
        
        Args:
            explanation: LIME explanation from the explain method
            instance_idx: Index of the instance to plot
            label_idx: Index of the label/class to explain
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib Figure with the LIME visualization
        """
        lime_exp = explanation['explanations'][instance_idx]
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        lime_exp.as_pyplot_figure(label=label_idx)
        plt.title(f'LIME Explanation for Instance {instance_idx}', fontsize=14)
        plt.tight_layout()
        return fig

    @with_exception_handling
    def save(self, path: str) ->None:
        """
        Save the LIME explainer.
        
        Args:
            path: Directory path to save the explainer
        """
        super().save(path)
        try:
            with open(os.path.join(path, 'explainer.pkl'), 'wb') as f:
                pickle.dump(self.explainer, f)
        except:
            logger.warning(
                'Could not pickle the LIME explainer. Will need to reinitialize on load.'
                )

    @classmethod
    @with_exception_handling
    def load(cls, path: str, model: Any, training_data: Optional[Union[np.
        ndarray, pd.DataFrame]]=None) ->'LIMEExplainer':
        """
        Load a saved LIME explainer.
        
        Args:
            path: Path to the saved explainer
            model: Model to use with the loaded explainer
            training_data: Training data if the explainer needs to be reinitialized
            
        Returns:
            Loaded LIMEExplainer instance
        """
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        instance = cls(model=model, feature_names=metadata.get(
            'feature_names'), class_names=metadata.get('class_names'), mode
            =metadata.get('mode', 'classification'))
        try:
            with open(os.path.join(path, 'explainer.pkl'), 'rb') as f:
                instance.explainer = pickle.load(f)
        except:
            logger.warning(
                'Could not load the pickled LIME explainer. Will initialize on the first call to explain().'
                )
            if training_data is None:
                logger.warning(
                    'Training data should be provided to initialize the explainer'
                    )
        return instance


class PermutationImportanceExplainer(ModelExplainer):
    """
    Permutation Importance explainer implementation.
    
    This explainer measures feature importance by calculating the decrease in
    model performance when a single feature is randomly shuffled.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]]=None,
        scoring: Optional[Union[str, Callable]]=None):
        """
        Initialize the Permutation Importance explainer.
        
        Args:
            model: The model to explain
            feature_names: Names of input features
            scoring: Scoring metric for evaluating model performance
        """
        super().__init__(model, feature_names, None, 'permutation_importance')
        self.scoring = scoring

    def explain(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union
        [np.ndarray, pd.Series]]=None, n_repeats: int=10, random_state: int=42
        ) ->Dict[str, Any]:
        """
        Calculate permutation importance for features.
        
        Args:
            X: Input data
            y: Target values (required for supervised models)
            n_repeats: Number of times to permute each feature
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with permutation importance results
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            result = permutation_importance(self.model, X_values, y,
                n_repeats=n_repeats, random_state=random_state, scoring=
                self.scoring)
            importances = result.importances_mean
            importances_std = result.importances_std
        else:
            predictions = self._predict(X_values)
            baseline_score = np.var(predictions)
            importances = np.zeros(X_values.shape[1])
            importances_std = np.zeros(X_values.shape[1])
            for i in range(X_values.shape[1]):
                feature_importances = []
                for _ in range(n_repeats):
                    X_permuted = X_values.copy()
                    np.random.shuffle(X_permuted[:, i])
                    perm_predictions = self._predict(X_permuted)
                    perm_score = np.var(perm_predictions)
                    importance = baseline_score - perm_score
                    feature_importances.append(importance)
                importances[i] = np.mean(feature_importances)
                importances_std[i] = np.std(feature_importances)
        if np.sum(np.abs(importances)) > 0:
            importances = importances / np.sum(np.abs(importances))
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names or [
            f'feature_{i}' for i in range(len(importances))]):
            feature_importance[feature_name] = importances[i]
        return {'importances': importances, 'importances_std':
            importances_std, 'feature_importance': feature_importance,
            'feature_names': self.feature_names}

    def _predict(self, X: np.ndarray) ->np.ndarray:
        """
        Get model predictions for input data.
        
        Args:
            X: Input data
            
        Returns:
            Model predictions
        """
        if isinstance(self.model, tf.keras.Model):
            return self.model.predict(X)
        elif hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            return self.model(X)

    def plot(self, explanation: Dict[str, Any], max_features: int=20, **kwargs
        ) ->plt.Figure:
        """
        Plot feature importances.
        
        Args:
            explanation: Permutation importance explanation from the explain method
            max_features: Maximum number of features to show
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib Figure with the feature importance visualization
        """
        importances = explanation['importances']
        importances_std = explanation.get('importances_std')
        feature_names = explanation.get('feature_names', [f'feature_{i}' for
            i in range(len(importances))])
        indices = np.argsort(importances)[::-1]
        indices = indices[:max_features]
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
        y_pos = np.arange(len(indices))
        if importances_std is not None:
            ax.barh(y_pos, importances[indices], xerr=importances_std[
                indices], align='center', alpha=0.7)
        else:
            ax.barh(y_pos, importances[indices], align='center', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title('Permutation Feature Importance')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        return fig

    def save(self, path: str) ->None:
        """
        Save the permutation importance explainer.
        
        Args:
            path: Directory path to save the explainer
        """
        super().save(path)
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        if isinstance(self.scoring, str):
            metadata['scoring'] = self.scoring
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str, model: Any) ->'PermutationImportanceExplainer':
        """
        Load a saved permutation importance explainer.
        
        Args:
            path: Path to the saved explainer
            model: Model to use with the loaded explainer
            
        Returns:
            Loaded PermutationImportanceExplainer instance
        """
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        instance = cls(model=model, feature_names=metadata.get(
            'feature_names'), scoring=metadata.get('scoring'))
        return instance


class ExplainabilityModule:
    """
    Main module for providing explanations for model predictions.
    
    This module manages different explainers and provides a uniform
    interface for explaining predictions across different model types.
    """

    def __init__(self):
        """Initialize the explainability module."""
        self.explainers = {}

    def add_explainer(self, name: str, explainer: ModelExplainer) ->None:
        """
        Add an explainer to the module.
        
        Args:
            name: Name to identify the explainer
            explainer: ModelExplainer instance
        """
        self.explainers[name] = explainer
        logger.info(f"Added {explainer.explainer_type} explainer as '{name}'")

    def explain(self, name: str, X: Union[np.ndarray, pd.DataFrame], **kwargs
        ) ->Dict[str, Any]:
        """
        Generate explanations using the specified explainer.
        
        Args:
            name: Name of the explainer to use
            X: Input data to explain
            **kwargs: Additional parameters for the explainer
            
        Returns:
            Dictionary with explanation results
        """
        if name not in self.explainers:
            raise ValueError(f"Explainer '{name}' not found")
        explainer = self.explainers[name]
        explanation = explainer.explain(X, **kwargs)
        return explanation

    def explain_model(self, model: Any, X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]]=None, feature_names:
        Optional[List[str]]=None, class_names: Optional[List[str]]=None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]]=None,
        methods: Optional[List[str]]=None) ->Dict[str, Dict[str, Any]]:
        """
        Explain a model using multiple explainability methods.
        
        Args:
            model: Model to explain
            X: Input data to explain
            y: Target values (for supervised methods)
            feature_names: Names of input features
            class_names: Names of output classes
            background_data: Reference data for SHAP explainer
            methods: List of explainability methods to use
                    (defaults to ['shap', 'lime', 'permutation'])
            
        Returns:
            Dictionary with explanations from all methods
        """
        if methods is None:
            methods = ['shap', 'permutation']
        if isinstance(X, pd.DataFrame) and feature_names is None:
            feature_names = X.columns.tolist()
        explanations = {}
        if 'shap' in methods:
            if 'shap' not in self.explainers:
                shap_explainer = SHAPExplainer(model=model, feature_names=
                    feature_names, class_names=class_names, background_data
                    =background_data)
                self.add_explainer('shap', shap_explainer)
            explanations['shap'] = self.explain('shap', X)
        if 'lime' in methods:
            if 'lime' not in self.explainers:
                lime_explainer = LIMEExplainer(model=model, feature_names=
                    feature_names, class_names=class_names, training_data=
                    background_data or X)
                self.add_explainer('lime', lime_explainer)
            explanations['lime'] = self.explain('lime', X)
        if 'permutation' in methods:
            if 'permutation' not in self.explainers:
                perm_explainer = PermutationImportanceExplainer(model=model,
                    feature_names=feature_names)
                self.add_explainer('permutation', perm_explainer)
            explanations['permutation'] = self.explain('permutation', X, y=y)
        return explanations

    def plot_explanation(self, name: str, explanation: Dict[str, Any], **kwargs
        ) ->plt.Figure:
        """
        Plot an explanation using the specified explainer.
        
        Args:
            name: Name of the explainer
            explanation: Explanation to plot
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib Figure with the visualization
        """
        if name not in self.explainers:
            raise ValueError(f"Explainer '{name}' not found")
        explainer = self.explainers[name]
        return explainer.plot(explanation, **kwargs)

    def save_explainers(self, path: str) ->None:
        """
        Save all explainers in the module.
        
        Args:
            path: Directory path to save explainers
        """
        os.makedirs(path, exist_ok=True)
        for name, explainer in self.explainers.items():
            explainer_path = os.path.join(path, name)
            os.makedirs(explainer_path, exist_ok=True)
            explainer.save(explainer_path)
        metadata = {'explainers': list(self.explainers.keys()),
            'date_saved': datetime.now().isoformat()}
        with open(os.path.join(path, 'module_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        logger.info(
            f'Saved ExplainabilityModule with {len(self.explainers)} explainers to {path}'
            )

    @classmethod
    def load(cls, path: str, models: Dict[str, Any]) ->'ExplainabilityModule':
        """
        Load a saved explainability module.
        
        Args:
            path: Path to the saved module
            models: Dictionary of models to use with the explainers
            
        Returns:
            Loaded ExplainabilityModule instance
        """
        with open(os.path.join(path, 'module_metadata.json'), 'r') as f:
            metadata = json.load(f)
        module = cls()
        for explainer_name in metadata.get('explainers', []):
            explainer_path = os.path.join(path, explainer_name)
            with open(os.path.join(explainer_path, 'metadata.json'), 'r') as f:
                explainer_metadata = json.load(f)
            explainer_type = explainer_metadata.get('explainer_type')
            model = models.get(explainer_name, next(iter(models.values())))
            if explainer_type == 'shap':
                explainer = SHAPExplainer.load(explainer_path, model)
                module.add_explainer(explainer_name, explainer)
            elif explainer_type == 'lime':
                explainer = LIMEExplainer.load(explainer_path, model)
                module.add_explainer(explainer_name, explainer)
            elif explainer_type == 'permutation_importance':
                explainer = PermutationImportanceExplainer.load(explainer_path,
                    model)
                module.add_explainer(explainer_name, explainer)
        logger.info(
            f'Loaded ExplainabilityModule with {len(module.explainers)} explainers from {path}'
            )
        return module


def explain_multitask_model(model: MultitaskModel, X: Union[np.ndarray, pd.
    DataFrame], task: TaskType, background_data: Optional[Union[np.ndarray,
    pd.DataFrame]]=None, methods: Optional[List[str]]=None) ->Dict[str,
    Dict[str, Any]]:
    """
    Generate explanations for a specific task in a MultitaskModel.
    
    Args:
        model: MultitaskModel instance
        X: Input data to explain
        task: Specific task to explain
        background_data: Reference data for explainers
        methods: List of explainability methods to use
        
    Returns:
        Dictionary with explanations from all methods
    """

    def predict_wrapper(x):
    """
    Predict wrapper.
    
    Args:
        x: Description of x
    
    """

        predictions = model.predict_single_task(x, task)
        return predictions
    explainability = ExplainabilityModule()
    feature_names = model.feature_names
    class_names = None
    task_config = model.task_configs[task]
    if task_config['activation'] == 'softmax':
        if task == TaskType.PRICE_DIRECTION:
            class_names = ['down', 'sideways', 'up']
        elif task == TaskType.REGIME_CLASSIFICATION:
            class_names = ['trending_up', 'trending_down', 'volatile',
                'range_bound']
    return explainability.explain_model(model=predict_wrapper, X=X,
        feature_names=feature_names, class_names=class_names,
        background_data=background_data, methods=methods)
