"""
Price Prediction Model

This module provides machine learning models for predicting future price movements.
It uses TensorFlow/Keras to implement deep learning models for time series forecasting.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class PricePredictionModel:
    """
    Machine learning model for predicting future price movements.
    
    This class implements a deep learning model for time series forecasting
    of financial data, with a focus on short-term price predictions.
    """

    def __init__(self, model_path: Optional[str]=None, input_window: int=60,
        output_window: int=10, feature_columns: Optional[List[str]]=None,
        target_column: str='close', use_gpu: bool=True):
        """
        Initialize the price prediction model.
        
        Args:
            model_path: Path to saved model (if None, a new model will be created)
            input_window: Size of the input window for prediction
            output_window: Size of the output window (prediction horizon)
            feature_columns: List of feature columns to use (if None, use all OHLCV columns)
            target_column: Column to predict
            use_gpu: Whether to use GPU for training and inference
        """
        self.input_window = input_window
        self.output_window = output_window
        self.feature_columns = feature_columns or ['open', 'high', 'low',
            'close', 'volume']
        self.target_column = target_column
        self.use_gpu = use_gpu
        if not use_gpu:
            logger.info('Disabling GPU for price prediction model')
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f'Using GPU for price prediction model: {gpus}')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.warning('No GPU available, falling back to CPU')
        if model_path and os.path.exists(model_path):
            logger.info(f'Loading model from {model_path}')
            self.model = models.load_model(model_path)
        else:
            logger.info('Creating new price prediction model')
            self.model = self._create_model()
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        logger.info(
            f'Price prediction model initialized with input_window={input_window}, output_window={output_window}'
            )

    def _create_model(self) ->tf.keras.Model:
        """
        Create a new price prediction model.
        
        Returns:
            TensorFlow/Keras model
        """
        input_layer = layers.Input(shape=(self.input_window, len(self.
            feature_columns)))
        encoder = layers.Conv1D(filters=64, kernel_size=3, padding='same',
            activation='relu')(input_layer)
        encoder = layers.MaxPooling1D(pool_size=2)(encoder)
        encoder = layers.Conv1D(filters=128, kernel_size=3, padding='same',
            activation='relu')(encoder)
        encoder = layers.MaxPooling1D(pool_size=2)(encoder)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(
            encoder)
        x = layers.Bidirectional(layers.LSTM(64))(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        output_layer = layers.Dense(self.output_window)(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=
            'mse', metrics=['mae'])
        return model

    def preprocess_data(self, df: pd.DataFrame, is_training: bool=True
        ) ->Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for the model.
        
        Args:
            df: DataFrame with OHLCV data
            is_training: Whether this is for training (True) or prediction (False)
            
        Returns:
            Tuple of (X, y) where X is input data and y is target data (None for prediction)
        """
        feature_data = df[self.feature_columns].values
        if is_training:
            feature_data = self.feature_scaler.fit_transform(feature_data)
        else:
            feature_data = self.feature_scaler.transform(feature_data)
        target_data = df[[self.target_column]].values
        if is_training:
            target_data = self.target_scaler.fit_transform(target_data)
        else:
            target_data = self.target_scaler.transform(target_data)
        X = []
        for i in range(len(feature_data) - self.input_window - self.
            output_window + 1):
            X.append(feature_data[i:i + self.input_window])
        if not is_training:
            return np.array([feature_data[-self.input_window:]]), None
        y = []
        for i in range(len(target_data) - self.input_window - self.
            output_window + 1):
            y.append(target_data[i + self.input_window:i + self.
                input_window + self.output_window, 0])
        return np.array(X), np.array(y)

    def train(self, training_data: Dict[str, pd.DataFrame],
        validation_split: float=0.2, epochs: int=100, batch_size: int=32,
        early_stopping: bool=True) ->Dict[str, Any]:
        """
        Train the model on historical data.
        
        Args:
            training_data: Dictionary mapping symbols to DataFrames with OHLCV data
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping: Whether to use early stopping
            
        Returns:
            Training history
        """
        logger.info(
            f'Training price prediction model on {len(training_data)} symbols')
        X_all = []
        y_all = []
        for symbol, df in training_data.items():
            X, y = self.preprocess_data(df, is_training=True)
            X_all.append(X)
            y_all.append(y)
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
        train_size = int((1 - validation_split) * len(X_all))
        X_train, X_val = X_all[:train_size], X_all[train_size:]
        y_train, y_val = y_all[:train_size], y_all[train_size:]
        callbacks_list = []
        if early_stopping:
            callbacks_list.append(callbacks.EarlyStopping(monitor=
                'val_loss', patience=10, restore_best_weights=True))
        start_time = time.time()
        history = self.model.fit(X_train, y_train, validation_data=(X_val,
            y_val), epochs=epochs, batch_size=batch_size, callbacks=
            callbacks_list, verbose=1)
        training_time = time.time() - start_time
        logger.info(f'Model training completed in {training_time:.2f} seconds')
        evaluation = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info(
            f'Validation loss (MSE): {evaluation[0]:.4f}, MAE: {evaluation[1]:.4f}'
            )
        y_pred = self.model.predict(X_val)
        y_val_reshaped = y_val.reshape(-1, 1)
        y_pred_reshaped = y_pred.reshape(-1, 1)
        y_val_actual = self.target_scaler.inverse_transform(y_val_reshaped)
        y_pred_actual = self.target_scaler.inverse_transform(y_pred_reshaped)
        mse = mean_squared_error(y_val_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_actual, y_pred_actual)
        r2 = r2_score(y_val_actual, y_pred_actual)
        logger.info(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
        return {'history': history.history, 'training_time': training_time,
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

    def predict(self, df: pd.DataFrame) ->Dict[str, Any]:
        """
        Predict future prices based on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with prediction results
        """
        X, _ = self.preprocess_data(df, is_training=False)
        start_time = time.time()
        predictions = self.model.predict(X, verbose=0)
        prediction_time = time.time() - start_time
        predictions_reshaped = predictions.reshape(-1, 1)
        predictions_actual = self.target_scaler.inverse_transform(
            predictions_reshaped)
        predictions_actual = predictions_actual.reshape(-1, self.output_window)
        last_price = df[self.target_column].iloc[-1]
        percentage_changes = []
        for i in range(self.output_window):
            if i == 0:
                change = (predictions_actual[0, i] - last_price
                    ) / last_price * 100
            else:
                change = (predictions_actual[0, i] - predictions_actual[0, 
                    i - 1]) / predictions_actual[0, i - 1] * 100
            percentage_changes.append(change)
        std_dev = df[self.target_column].std()
        lower_bound = predictions_actual[0] - 1.96 * std_dev
        upper_bound = predictions_actual[0] + 1.96 * std_dev
        logger.debug(
            f'Price prediction completed in {prediction_time:.4f} seconds')
        return {'predictions': predictions_actual[0].tolist(),
            'lower_bound': lower_bound.tolist(), 'upper_bound': upper_bound
            .tolist(), 'percentage_changes': percentage_changes,
            'prediction_time': prediction_time}

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
