"""
MultitaskModel Implementation

This module implements a multitask learning model for forex trading that can simultaneously predict
multiple related targets including price direction, volatility, and support/resistance levels.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from enum import Enum
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
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

class TaskType(str, Enum):
    """Types of prediction tasks supported by MultitaskModel"""
    PRICE_DIRECTION = 'price_direction'
    VOLATILITY = 'volatility'
    SUPPORT_RESISTANCE = 'support_resistance'
    REGIME_CLASSIFICATION = 'regime_classification'
    CORRELATION_STABILITY = 'correlation_stability'


class MultitaskModel:
    """
    A multitask learning model for forex trading that simultaneously predicts multiple targets.
    
    This model architecture uses shared layers to learn common representations 
    and task-specific layers to specialize for different prediction targets.
    
    Attributes:
        model: The underlying TensorFlow model
        tasks: List of prediction tasks this model is trained for
        config: Model configuration parameters
        feature_names: Names of input features
        task_configs: Configuration for each task (output shape, loss function, etc.)
    """

    def __init__(self, tasks: List[TaskType], config: Dict[str, Any],
        feature_names: Optional[List[str]]=None):
        """
        Initialize the MultitaskModel.
        
        Args:
            tasks: List of tasks this model should predict
            config: Model configuration including architecture parameters
            feature_names: Names of input features (optional)
        """
        self.tasks = tasks
        self.config = config
        self.feature_names = feature_names
        self.model = None
        self.task_configs = self._setup_task_configs()
        for task in tasks:
            if task not in TaskType:
                raise ValueError(f'Unsupported task: {task}')

    def _setup_task_configs(self) ->Dict[str, Dict[str, Any]]:
        """
        Set up configuration for each supported task.
        
        Returns:
            Dictionary with task configurations
        """
        task_configs = {TaskType.PRICE_DIRECTION: {'units': self.config.get
            ('price_direction_units', 3), 'activation': 'softmax', 'loss':
            'categorical_crossentropy', 'metrics': ['accuracy']}, TaskType.
            VOLATILITY: {'units': self.config_manager.get('volatility_units', 1),
            'activation': 'sigmoid' if self.config.get('volatility_binary',
            True) else 'linear', 'loss': 'binary_crossentropy' if self.
            config_manager.get('volatility_binary', True) else 'mean_squared_error',
            'metrics': ['accuracy'] if self.config.get('volatility_binary',
            True) else ['mean_absolute_error']}, TaskType.
            SUPPORT_RESISTANCE: {'units': self.config.get(
            'support_resistance_units', 2), 'activation': 'linear', 'loss':
            'mean_squared_error', 'metrics': ['mean_absolute_error']},
            TaskType.REGIME_CLASSIFICATION: {'units': self.config.get(
            'regime_units', 4), 'activation': 'softmax', 'loss':
            'categorical_crossentropy', 'metrics': ['accuracy']}, TaskType.
            CORRELATION_STABILITY: {'units': self.config.get(
            'correlation_units', 1), 'activation': 'sigmoid', 'loss':
            'binary_crossentropy', 'metrics': ['accuracy']}}
        return task_configs

    def build_model(self, input_shape: Tuple[int, int]) ->None:
        """
        Build the multitask model architecture.
        
        This creates a model with shared layers and task-specific output heads.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
        """
        inputs = Input(shape=input_shape, name='shared_input')
        shared_lstm_units = self.config_manager.get('shared_lstm_units', [128, 64])
        x = inputs
        for i, units in enumerate(shared_lstm_units):
            return_sequences = i < len(shared_lstm_units) - 1
            x = LSTM(units, return_sequences=return_sequences, name=
                f'shared_lstm_{i + 1}')(x)
            x = BatchNormalization(name=f'shared_bn_{i + 1}')(x)
            x = Dropout(self.config_manager.get('dropout_rate', 0.2), name=
                f'shared_dropout_{i + 1}')(x)
        shared_output = x
        outputs = {}
        losses = {}
        metrics = {}
        for task in self.tasks:
            task_config = self.task_configs[task]
            task_dense = Dense(self.config.get(f'{task.value}_dense_units',
                32), activation='relu', name=f'{task.value}_dense')(
                shared_output)
            task_dropout = Dropout(self.config.get('task_dropout_rate', 0.1
                ), name=f'{task.value}_dropout')(task_dense)
            task_output = Dense(task_config['units'], activation=
                task_config['activation'], name=f'{task.value}_output')(
                task_dropout)
            outputs[f'{task.value}_output'] = task_output
            losses[f'{task.value}_output'] = task_config['loss']
            metrics[f'{task.value}_output'] = task_config['metrics']
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=self.config.get(
            'learning_rate', 0.001)), loss=losses, metrics=metrics,
            loss_weights={f'{task.value}_output': self.config.get(
            f'{task.value}_weight', 1.0) for task in self.tasks})
        logger.info(
            f'Built MultitaskModel with {len(self.tasks)} tasks: {[task.value for task in self.tasks]}'
            )

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Dict[
        str, Union[np.ndarray, pd.DataFrame]], X_val: Optional[Union[np.
        ndarray, pd.DataFrame]]=None, y_val: Optional[Dict[str, Union[np.
        ndarray, pd.DataFrame]]]=None, epochs: int=100, batch_size: int=32,
        early_stopping_patience: int=10, model_checkpoint_path: Optional[
        str]=None, **kwargs) ->Dict[str, Any]:
        """
        Train the multitask model.
        
        Args:
            X_train: Training features
            y_train: Dictionary of training targets, keyed by task name
            X_val: Validation features (optional)
            y_val: Dictionary of validation targets (optional)
            epochs: Number of epochs to train
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
            model_checkpoint_path: Path to save model checkpoints
            **kwargs: Additional arguments to pass to model.fit()
            
        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if X_val is not None and isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        callbacks = []
        if early_stopping_patience > 0:
            early_stopping = EarlyStopping(monitor='val_loss' if X_val is not
                None else 'loss', patience=early_stopping_patience,
                restore_best_weights=True)
            callbacks.append(early_stopping)
        if model_checkpoint_path:
            os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
            checkpoint = ModelCheckpoint(model_checkpoint_path, monitor=
                'val_loss' if X_val is not None else 'loss', save_best_only
                =True, save_weights_only=False)
            callbacks.append(checkpoint)
        y_train_dict = {f'{task.value}_output': y_train[task.value] for
            task in self.tasks}
        validation_data = None
        if X_val is not None and y_val is not None:
            y_val_dict = {f'{task.value}_output': y_val[task.value] for
                task in self.tasks}
            validation_data = X_val, y_val_dict
        history = self.model.fit(X_train, y_train_dict, validation_data=
            validation_data, epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=1, **kwargs)
        logger.info(f'MultitaskModel training completed for {epochs} epochs')
        return history.history

    def evaluate(self, X_test: Union[np.ndarray, pd.DataFrame], y_test:
        Dict[str, Union[np.ndarray, pd.DataFrame]]) ->Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Dictionary of test targets, keyed by task name
            
        Returns:
            Dictionary of evaluation metrics for each task
        """
        if self.model is None:
            raise ValueError('Model must be trained before evaluation')
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        y_test_dict = {f'{task.value}_output': y_test[task.value] for task in
            self.tasks}
        results = self.model.evaluate(X_test, y_test_dict, verbose=1,
            return_dict=True)
        for key, value in results.items():
            if not key.startswith('val_'):
                logger.info(f'Test {key}: {value:.4f}')
        return results

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) ->Dict[str, np.
        ndarray]:
        """
        Generate predictions for all tasks.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of predictions for each task
        """
        if self.model is None:
            raise ValueError('Model must be trained before prediction')
        if isinstance(X, pd.DataFrame):
            X = X.values
        raw_predictions = self.model.predict(X)
        predictions = {}
        for task in self.tasks:
            task_output_name = f'{task.value}_output'
            predictions[task.value] = raw_predictions[task_output_name]
        return predictions

    def predict_single_task(self, X: Union[np.ndarray, pd.DataFrame], task:
        TaskType) ->np.ndarray:
        """
        Generate predictions for a single task.
        
        Args:
            X: Input features
            task: The specific task to predict
            
        Returns:
            Predictions for the specified task
        """
        if task not in self.tasks:
            raise ValueError(f'Model not trained for task: {task}')
        all_predictions = self.predict(X)
        return all_predictions[task.value]

    def save(self, path: str) ->None:
        """
        Save the model to disk.
        
        This saves both the model weights/architecture and the model configuration.
        
        Args:
            path: Directory path to save the model
        """
        if self.model is None:
            raise ValueError('Model must be built before saving')
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'model.h5')
        self.model.save(model_path)
        config = {'tasks': [task.value for task in self.tasks], 'config':
            self.config, 'feature_names': self.feature_names, 'date_saved':
            datetime.now().isoformat()}
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        logger.info(f'Saved MultitaskModel to {path}')

    @classmethod
    def load(cls, path: str) ->'MultitaskModel':
        """
        Load a saved model from disk.
        
        Args:
            path: Path to the saved model directory
            
        Returns:
            Loaded MultitaskModel instance
        """
        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        tasks = [TaskType(task) for task in config_data['tasks']]
        model = cls(tasks=tasks, config=config_data['config'],
            feature_names=config_data.get('feature_names'))
        model_path = os.path.join(path, 'model.h5')
        model.model = tf.keras.models.load_model(model_path)
        logger.info(f'Loaded MultitaskModel from {path} with tasks: {tasks}')
        return model

    def calculate_feature_importance(self, X: Union[np.ndarray, pd.
        DataFrame], task: TaskType, n_repeats: int=10) ->Dict[str, float]:
        """
        Calculate feature importance using permutation importance.
        
        Permutation importance measures how much the model performance decreases
        when a single feature is randomly shuffled.
        
        Args:
            X: Input features
            task: The task to calculate feature importance for
            n_repeats: Number of times to repeat the permutation
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError(
                'Model must be trained before calculating feature importance')
        if task not in self.tasks:
            raise ValueError(f'Model not trained for task: {task}')
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        baseline_pred = self.predict_single_task(X_np, task)
        importance_scores = {}
        feature_names = self.feature_names or [f'feature_{i}' for i in
            range(X_np.shape[2])]
        for i in range(X_np.shape[2]):
            feature_name = feature_names[i]
            scores = []
            for _ in range(n_repeats):
                X_permuted = X_np.copy()
                np.random.shuffle(X_permuted[:, :, i])
                permuted_pred = self.predict_single_task(X_permuted, task)
                task_config = self.task_configs[task]
                if task_config['activation'] == 'softmax':
                    score = np.mean(np.abs(baseline_pred - permuted_pred))
                else:
                    score = np.mean((baseline_pred - permuted_pred) ** 2)
                scores.append(score)
            importance_scores[feature_name] = np.mean(scores)
        if sum(importance_scores.values()) > 0:
            total = sum(importance_scores.values())
            importance_scores = {k: (v / total) for k, v in
                importance_scores.items()}
        return importance_scores

    @with_exception_handling
    def visualize(self, output_path: Optional[str]=None) ->None:
        """
        Visualize the model architecture.
        
        Args:
            output_path: Path to save the visualization (optional)
        """
        if self.model is None:
            raise ValueError('Model must be built before visualization')
        if output_path:
            plot_model(self.model, to_file=output_path, show_shapes=True,
                show_layer_names=True)
        else:
            try:
                from IPython.display import display
                display(tf.keras.utils.model_to_dot(self.model, show_shapes
                    =True))
            except ImportError:
                logger.warning(
                    'Visualization requires IPython for display or an output path'
                    )
