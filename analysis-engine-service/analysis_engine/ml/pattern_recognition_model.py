"""
Pattern Recognition Model

This module provides machine learning models for recognizing patterns in financial data.
It uses TensorFlow/Keras to implement deep learning models for pattern recognition.
"""
import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class PatternRecognitionModel:
    """
    Machine learning model for recognizing patterns in financial data.
    
    This class implements a deep learning model for recognizing common chart patterns
    such as double tops, head and shoulders, triangles, etc.
    """

    def __init__(self, model_path: Optional[str]=None, window_size: int=30,
        feature_columns: Optional[List[str]]=None, num_patterns: int=8,
        use_gpu: bool=True):
        """
        Initialize the pattern recognition model.
        
        Args:
            model_path: Path to saved model (if None, a new model will be created)
            window_size: Size of the window for pattern recognition
            feature_columns: List of feature columns to use (if None, use all OHLCV columns)
            num_patterns: Number of patterns to recognize
            use_gpu: Whether to use GPU for training and inference
        """
        self.window_size = window_size
        self.feature_columns = feature_columns or ['open', 'high', 'low',
            'close', 'volume']
        self.num_patterns = num_patterns
        self.use_gpu = use_gpu
        if not use_gpu:
            logger.info('Disabling GPU for pattern recognition model')
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f'Using GPU for pattern recognition model: {gpus}')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.warning('No GPU available, falling back to CPU')
        if model_path and os.path.exists(model_path):
            logger.info(f'Loading model from {model_path}')
            self.model = models.load_model(model_path)
        else:
            logger.info('Creating new pattern recognition model')
            self.model = self._create_model()
        self.scaler = MinMaxScaler()
        self.pattern_names = ['double_top', 'double_bottom',
            'head_and_shoulders', 'inverse_head_and_shoulders', 'triangle',
            'wedge', 'rectangle', 'flag'][:num_patterns]
        logger.info(
            f'Pattern recognition model initialized with {num_patterns} patterns'
            )

    def _create_model(self) ->tf.keras.Model:
        """
        Create a new pattern recognition model.
        
        Returns:
            TensorFlow/Keras model
        """
        input_layer = layers.Input(shape=(self.window_size, len(self.
            feature_columns)))
        x = layers.Conv1D(filters=64, kernel_size=3, padding='same',
            activation='relu')(input_layer)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same',
            activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same',
            activation='relu')(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output_layer = layers.Dense(self.num_patterns, activation='sigmoid')(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=
            'binary_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_data(self, df: pd.DataFrame) ->np.ndarray:
        """
        Preprocess data for the model.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Preprocessed data as numpy array
        """
        data = df[self.feature_columns].values
        data = self.scaler.fit_transform(data)
        windows = []
        for i in range(len(data) - self.window_size + 1):
            windows.append(data[i:i + self.window_size])
        return np.array(windows)

    def train(self, training_data: Dict[str, pd.DataFrame], labels: Dict[
        str, List[List[int]]], validation_split: float=0.2, epochs: int=50,
        batch_size: int=32, early_stopping: bool=True) ->Dict[str, Any]:
        """
        Train the model on labeled data.
        
        Args:
            training_data: Dictionary mapping symbols to DataFrames with OHLCV data
            labels: Dictionary mapping symbols to lists of pattern labels
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping: Whether to use early stopping
            
        Returns:
            Training history
        """
        logger.info(
            f'Training pattern recognition model on {len(training_data)} symbols'
            )
        X = []
        y = []
        for symbol, df in training_data.items():
            windows = self.preprocess_data(df)
            symbol_labels = labels.get(symbol, [])
            X.append(windows)
            y.append(np.array(symbol_labels))
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=
            validation_split, random_state=42)
        callbacks_list = []
        if early_stopping:
            callbacks_list.append(callbacks.EarlyStopping(monitor=
                'val_loss', patience=5, restore_best_weights=True))
        start_time = time.time()
        history = self.model.fit(X_train, y_train, validation_data=(X_val,
            y_val), epochs=epochs, batch_size=batch_size, callbacks=
            callbacks_list, verbose=1)
        training_time = time.time() - start_time
        logger.info(f'Model training completed in {training_time:.2f} seconds')
        evaluation = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info(
            f'Validation loss: {evaluation[0]:.4f}, accuracy: {evaluation[1]:.4f}'
            )
        return {'history': history.history, 'training_time': training_time,
            'validation_loss': evaluation[0], 'validation_accuracy':
            evaluation[1]}

    def predict(self, df: pd.DataFrame) ->Dict[str, List[float]]:
        """
        Predict patterns in the given data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping pattern names to lists of probabilities
        """
        windows = self.preprocess_data(df)
        start_time = time.time()
        predictions = self.model.predict(windows, verbose=0)
        prediction_time = time.time() - start_time
        result = {}
        for i, pattern in enumerate(self.pattern_names):
            result[pattern] = predictions[:, i].tolist()
        logger.debug(
            f'Pattern prediction completed in {prediction_time:.4f} seconds')
        return result

    def save_model(self, model_path: str) ->None:
        """
        Save the model to disk.
        
        Args:
            model_path: Path to save the model
        """
        logger.info(f'Saving model to {model_path}')
        self.model.save(model_path)

    @with_resilience('get_summary')
    def get_summary(self) ->str:
        """
        Get a summary of the model.
        
        Returns:
            Model summary as string
        """
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            self.model.summary()
        return f.getvalue()
