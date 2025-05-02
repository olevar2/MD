"""
Forex Baseline LSTM Model Module.

This module contains the implementation of a baseline LSTM neural network
model for Forex price prediction using advanced technical analysis features.
"""

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Any, Tuple, Optional, Union

from ml_workbench_service.models.model import ModelMetadata

logger = logging.getLogger(__name__)

class ForexLSTMModel:
    """
    Baseline LSTM model for Forex price prediction.
    
    This class implements a Long Short-Term Memory (LSTM) neural network
    designed specifically for Forex price prediction using advanced
    technical analysis features. It includes functionality for data
    preprocessing, model creation, training, evaluation, and prediction.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 forecast_horizon: int = 5,
                 lstm_units: List[int] = [128, 64],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 patience: int = 10,
                 target_column: str = 'close',
                 model_path: Optional[str] = None):
        """
        Initialize the Forex LSTM model.
        
        Args:
            sequence_length: Number of time steps to use for each prediction
            forecast_horizon: Number of time steps to predict ahead
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            patience: Patience for early stopping
            target_column: Target column for prediction (e.g., 'close', 'high', 'low')
            model_path: Path to load existing model (if None, create new model)
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.target_column = target_column
        
        # Preprocessing components
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = None
        
        # Model
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            self.model = load_model(model_path)
        else:
            self.model = None
            
        # Training history
        self.history = None
            
        logger.info("ForexLSTMModel initialized")
    
    def build_model(self, input_dim: int) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            input_dim: Number of input features
        """
        logger.info(f"Building LSTM model with input_dim={input_dim}, "
                  f"sequence_length={self.sequence_length}")
        
        self.model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        self.model.add(LSTM(units=self.lstm_units[0], 
                           return_sequences=len(self.lstm_units) > 1,
                           input_shape=(self.sequence_length, input_dim)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, len(self.lstm_units)):
            return_sequences = i < len(self.lstm_units) - 1
            self.model.add(LSTM(units=self.lstm_units[i], return_sequences=return_sequences))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(Dense(self.forecast_horizon))
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        logger.info(f"Model built with {len(self.lstm_units)} LSTM layers")
        self.model.summary(print_fn=logger.info)
    
    def create_sequences(self, 
                        data: pd.DataFrame, 
                        target_column: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and target values from time series data.
        
        Args:
            data: DataFrame containing time series data
            target_column: Target column name (if None, uses self.target_column)
            
        Returns:
            Tuple of (X, y) arrays for model training or prediction
        """
        if target_column is None:
            target_column = self.target_column
            
        X = []
        y = []
        
        # Get the target data
        target_data = data[target_column].values
        
        # Create sequences
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X.append(data.iloc[i:i+self.sequence_length].values)
            
            # Target sequence (future values of target_column)
            y.append(target_data[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def preprocess_data(self, 
                       train_data: pd.DataFrame, 
                       validation_data: Optional[pd.DataFrame] = None,
                       feature_columns: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Preprocess data for training or prediction.
        
        Args:
            train_data: Training data DataFrame
            validation_data: Optional validation data DataFrame
            feature_columns: Optional list of feature column names to use
            
        Returns:
            Dictionary containing preprocessed X_train, y_train, X_val, y_val arrays
        """
        # Store feature columns for later use
        if feature_columns:
            self.feature_columns = feature_columns
        else:
            # Use all columns except datetime if exists
            self.feature_columns = [col for col in train_data.columns 
                                  if col != 'datetime' and col != 'date']
        
        logger.info(f"Preprocessing data with {len(self.feature_columns)} features")
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Scale features
        train_features = train_data[self.feature_columns].copy()
        train_features_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(train_features),
            columns=self.feature_columns
        )
        
        # Scale target
        train_target = train_data[[self.target_column]].copy()
        self.target_scaler.fit(train_target)
        
        # Create sequences for training
        X_train, y_train = self.create_sequences(train_features_scaled)
        
        # Scale target sequences
        y_train_reshaped = y_train.reshape(-1, 1)
        y_train_scaled = self.target_scaler.transform(y_train_reshaped)
        y_train = y_train_scaled.reshape(y_train.shape)
        
        result = {
            'X_train': X_train,
            'y_train': y_train
        }
        
        # Process validation data if provided
        if validation_data is not None:
            val_features = validation_data[self.feature_columns].copy()
            val_features_scaled = pd.DataFrame(
                self.feature_scaler.transform(val_features),
                columns=self.feature_columns
            )
            
            X_val, y_val = self.create_sequences(val_features_scaled)
            
            # Scale validation target sequences
            y_val_reshaped = y_val.reshape(-1, 1)
            y_val_scaled = self.target_scaler.transform(y_val_reshaped)
            y_val = y_val_scaled.reshape(y_val.shape)
            
            result['X_val'] = X_val
            result['y_val'] = y_val
        
        logger.info(f"Preprocessing complete. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return result
    
    def train(self, 
             train_data: pd.DataFrame, 
             validation_split: float = 0.2,
             validation_data: Optional[pd.DataFrame] = None,
             feature_columns: Optional[List[str]] = None,
             callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
             model_save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            train_data: Training data DataFrame
            validation_split: Fraction of training data to use for validation
            validation_data: Optional separate validation DataFrame
            feature_columns: List of feature column names to use
            callbacks: Additional Keras callbacks
            model_save_path: Path to save the trained model
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Starting model training with {len(train_data)} rows")
        
        # Preprocess data
        if validation_data is not None:
            data = self.preprocess_data(train_data, validation_data, feature_columns)
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            data = self.preprocess_data(train_data, None, feature_columns)
            X_train, y_train = data['X_train'], data['y_train']
        
        # Build model if not already built
        if self.model is None:
            input_dim = X_train.shape[2]  # Number of features
            self.build_model(input_dim)
        
        # Set up callbacks
        if callbacks is None:
            callbacks = []
        
        # Add default callbacks
        callbacks.extend([
            EarlyStopping(
                monitor='val_loss', 
                patience=self.patience, 
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ])
        
        # Add model checkpoint if save path is provided
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=model_save_path,
                    monitor='val_loss',
                    save_best_only=True
                )
            )
        
        # Train the model
        start_time = pd.Timestamp.now()
        
        if validation_data is not None:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=2
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=2
            )
        
        training_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Get training results
        results = {
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
            'best_val_loss': min(self.history.history['val_loss']),
            'training_time_seconds': training_time
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds. "
                  f"Best validation loss: {results['best_val_loss']:.6f}")
        
        return results
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data DataFrame
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() and train() first.")
        
        # Preprocess test data
        test_features = test_data[self.feature_columns].copy()
        test_features_scaled = pd.DataFrame(
            self.feature_scaler.transform(test_features),
            columns=self.feature_columns
        )
        
        X_test, y_test = self.create_sequences(test_features_scaled)
        
        # Scale test target sequences
        y_test_reshaped = y_test.reshape(-1, 1)
        y_test_scaled = self.target_scaler.transform(y_test_reshaped)
        y_test = y_test_scaled.reshape(y_test.shape)
        
        # Evaluate model
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        evaluation = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        y_test_flat = y_test.reshape(-1, 1)
        y_pred_flat = y_pred.reshape(-1, 1)
        
        y_test_inv = self.target_scaler.inverse_transform(y_test_flat).reshape(y_test.shape)
        y_pred_inv = self.target_scaler.inverse_transform(y_pred_flat).reshape(y_pred.shape)
        
        # Calculate additional metrics
        mae = np.mean(np.abs(y_test_inv - y_pred_inv))
        rmse = np.sqrt(np.mean((y_test_inv - y_pred_inv) ** 2))
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
        
        metrics = {
            'loss': evaluation[0],
            'mae': evaluation[1],
            'mae_original_scale': mae,
            'rmse_original_scale': rmse,
            'mape': mape
        }
        
        # Direction accuracy (for first step prediction)
        actual_direction = np.sign(y_test_inv[:, 0] - test_data[self.target_column].values[-len(y_test):])
        pred_direction = np.sign(y_pred_inv[:, 0] - test_data[self.target_column].values[-len(y_pred):])
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        metrics['direction_accuracy'] = direction_accuracy
        
        logger.info(f"Evaluation metrics: MAE={mae:.6f}, RMSE={rmse:.6f}, "
                  f"MAPE={mape:.2f}%, Direction Accuracy={direction_accuracy:.2f}%")
        
        return metrics
    
    def predict(self, data: pd.DataFrame, return_unscaled: bool = True) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: Input data for prediction (should contain self.feature_columns)
            return_unscaled: Whether to return predictions in original scale
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call build_model() and train() first.")
        
        if len(data) < self.sequence_length:
            raise ValueError(f"Input data length ({len(data)}) must be at least sequence_length ({self.sequence_length})")
        
        # Preprocess input data
        features = data[self.feature_columns].copy()
        features_scaled = pd.DataFrame(
            self.feature_scaler.transform(features),
            columns=self.feature_columns
        )
        
        # Create sequence (only one sequence for the most recent data)
        X = np.array([features_scaled.values[-self.sequence_length:]])
        
        # Get prediction
        prediction = self.model.predict(X)
        
        # Return unscaled prediction if requested
        if return_unscaled:
            prediction_flat = prediction.reshape(-1, 1)
            prediction_unscaled = self.target_scaler.inverse_transform(prediction_flat)
            return prediction_unscaled.reshape(prediction.shape)
        
        return prediction
    
    def save(self, path: str, include_scalers: bool = True) -> None:
        """
        Save the model and optionally scalers.
        
        Args:
            path: Directory path to save the model
            include_scalers: Whether to save scalers along with model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model.h5")
        
        # Save Keras model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scalers if requested
        if include_scalers:
            import pickle
            scalers_path = os.path.join(path, "scalers.pkl")
            with open(scalers_path, 'wb') as f:
                pickle.dump({
                    'feature_scaler': self.feature_scaler,
                    'target_scaler': self.target_scaler,
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'sequence_length': self.sequence_length,
                    'forecast_horizon': self.forecast_horizon
                }, f)
            logger.info(f"Scalers and metadata saved to {scalers_path}")
    
    @classmethod
    def load(cls, path: str) -> 'ForexLSTMModel':
        """
        Load a saved model and scalers.
        
        Args:
            path: Directory path containing the saved model
            
        Returns:
            Loaded ForexLSTMModel instance
        """
        import pickle
        
        model_path = os.path.join(path, "model.h5")
        scalers_path = os.path.join(path, "scalers.pkl")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
            
        if not os.path.exists(scalers_path):
            raise ValueError(f"Scalers file not found: {scalers_path}")
        
        # Load scalers and metadata
        with open(scalers_path, 'rb') as f:
            data = pickle.load(f)
            
        # Create model instance with loaded parameters
        instance = cls(
            sequence_length=data['sequence_length'],
            forecast_horizon=data['forecast_horizon'],
            target_column=data['target_column'],
            model_path=model_path
        )
        
        # Set scalers and feature columns
        instance.feature_scaler = data['feature_scaler']
        instance.target_scaler = data['target_scaler']
        instance.feature_columns = data['feature_columns']
        
        logger.info(f"Model loaded from {path} with {len(instance.feature_columns)} features")
        return instance
    
    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata for the model registry.
        
        Returns:
            ModelMetadata object
        """
        if self.model is None:
            raise ValueError("Model not initialized")
            
        metadata = {
            'model_type': 'LSTM',
            'target_column': self.target_column,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'feature_columns': self.feature_columns,
            'lstm_layers': len(self.lstm_units),
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'input_shape': [self.sequence_length, len(self.feature_columns)],
            'output_shape': [self.forecast_horizon],
            'framework': 'tensorflow',
            'keras_version': tf.keras.__version__,
            'tensorflow_version': tf.__version__
        }
        
        # Add training history if available
        if self.history is not None:
            train_metrics = {}
            for metric_name, values in self.history.history.items():
                train_metrics[f'final_{metric_name}'] = values[-1]
                train_metrics[f'best_{metric_name}'] = min(values) if 'loss' in metric_name else max(values)
            
            metadata.update({
                'epochs_trained': len(self.history.history['loss']),
                'training_metrics': train_metrics
            })
        
        return ModelMetadata(metadata=metadata)
