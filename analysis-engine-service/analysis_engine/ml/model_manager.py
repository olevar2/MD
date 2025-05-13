"""
Model Manager

This module provides a centralized manager for machine learning models.
It handles model loading, saving, training, and inference.
"""
import os
import time
import logging
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import tensorflow as tf
from analysis_engine.ml.pattern_recognition_model import PatternRecognitionModel
from analysis_engine.ml.price_prediction_model import PricePredictionModel
from analysis_engine.ml.ml_confluence_detector import MLConfluenceDetector
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ModelManager:
    """
    Centralized manager for machine learning models.
    
    This class provides a unified interface for managing machine learning models,
    including loading, saving, training, and inference.
    """

    def __init__(self, model_dir: str='models', use_gpu: bool=True,
        correlation_service: Optional[Any]=None, currency_strength_analyzer:
        Optional[Any]=None):
        """
        Initialize the model manager.
        
        Args:
            model_dir: Directory for storing models
            use_gpu: Whether to use GPU for ML models
            correlation_service: Service for getting correlations between pairs
            currency_strength_analyzer: Analyzer for calculating currency strength
        """
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.correlation_service = correlation_service
        self.currency_strength_analyzer = currency_strength_analyzer
        os.makedirs(model_dir, exist_ok=True)
        self.model_registry = {}
        registry_path = os.path.join(model_dir, 'model_registry.json')
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                self.model_registry = json.load(f)
        logger.info(
            f'Model manager initialized with model_dir={model_dir}, use_gpu={use_gpu}'
            )

    def save_registry(self) ->None:
        """Save the model registry to disk."""
        registry_path = os.path.join(self.model_dir, 'model_registry.json')
        with open(registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)

    @with_database_resilience('load_pattern_model')
    def load_pattern_model(self, model_name: str='pattern_recognition',
        window_size: int=30, feature_columns: Optional[List[str]]=None,
        num_patterns: int=8) ->PatternRecognitionModel:
        """
        Load a pattern recognition model.
        
        Args:
            model_name: Name of the model
            window_size: Size of the window for pattern recognition
            feature_columns: List of feature columns to use
            num_patterns: Number of patterns to recognize
            
        Returns:
            Pattern recognition model
        """
        model_path = None
        if model_name in self.model_registry:
            model_info = self.model_registry[model_name]
            if model_info['type'] == 'pattern_recognition':
                model_path = os.path.join(self.model_dir, model_info['path'])
                logger.info(
                    f'Loading pattern recognition model from registry: {model_path}'
                    )
        model = PatternRecognitionModel(model_path=model_path, window_size=
            window_size, feature_columns=feature_columns, num_patterns=
            num_patterns, use_gpu=self.use_gpu)
        return model

    @with_analysis_resilience('load_prediction_model')
    def load_prediction_model(self, model_name: str='price_prediction',
        input_window: int=60, output_window: int=10, feature_columns:
        Optional[List[str]]=None, target_column: str='close'
        ) ->PricePredictionModel:
        """
        Load a price prediction model.
        
        Args:
            model_name: Name of the model
            input_window: Size of the input window for prediction
            output_window: Size of the output window (prediction horizon)
            feature_columns: List of feature columns to use
            target_column: Column to predict
            
        Returns:
            Price prediction model
        """
        model_path = None
        if model_name in self.model_registry:
            model_info = self.model_registry[model_name]
            if model_info['type'] == 'price_prediction':
                model_path = os.path.join(self.model_dir, model_info['path'])
                logger.info(
                    f'Loading price prediction model from registry: {model_path}'
                    )
        model = PricePredictionModel(model_path=model_path, input_window=
            input_window, output_window=output_window, feature_columns=
            feature_columns, target_column=target_column, use_gpu=self.use_gpu)
        return model

    @with_database_resilience('load_ml_confluence_detector')
    def load_ml_confluence_detector(self, pattern_model_name: str=
        'pattern_recognition', prediction_model_name: str=
        'price_prediction', correlation_threshold: float=0.7,
        lookback_periods: int=20, cache_ttl_minutes: int=60
        ) ->MLConfluenceDetector:
        """
        Load an ML confluence detector.
        
        Args:
            pattern_model_name: Name of the pattern recognition model
            prediction_model_name: Name of the price prediction model
            correlation_threshold: Minimum correlation for related pairs
            lookback_periods: Number of periods to look back for analysis
            cache_ttl_minutes: Cache time-to-live in minutes
            
        Returns:
            ML confluence detector
        """
        pattern_model_path = None
        prediction_model_path = None
        if pattern_model_name in self.model_registry:
            model_info = self.model_registry[pattern_model_name]
            if model_info['type'] == 'pattern_recognition':
                pattern_model_path = os.path.join(self.model_dir,
                    model_info['path'])
        if prediction_model_name in self.model_registry:
            model_info = self.model_registry[prediction_model_name]
            if model_info['type'] == 'price_prediction':
                prediction_model_path = os.path.join(self.model_dir,
                    model_info['path'])
        detector = MLConfluenceDetector(correlation_service=self.
            correlation_service, currency_strength_analyzer=self.
            currency_strength_analyzer, pattern_model_path=
            pattern_model_path, prediction_model_path=prediction_model_path,
            correlation_threshold=correlation_threshold, lookback_periods=
            lookback_periods, cache_ttl_minutes=cache_ttl_minutes, use_gpu=
            self.use_gpu)
        return detector

    def save_model(self, model: Union[PatternRecognitionModel,
        PricePredictionModel], model_name: str, model_type: str,
        description: str='', metadata: Optional[Dict[str, Any]]=None) ->str:
        """
        Save a model to disk and update the registry.
        
        Args:
            model: Model to save
            model_name: Name of the model
            model_type: Type of the model ("pattern_recognition" or "price_prediction")
            description: Description of the model
            metadata: Additional metadata for the model
            
        Returns:
            Path to the saved model
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = os.path.join(self.model_dir, f'{model_name}_{timestamp}')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model')
        model.save_model(model_path)
        model_info = {'name': model_name, 'type': model_type, 'path': os.
            path.relpath(model_path, self.model_dir), 'created_at':
            timestamp, 'description': description, 'metadata': metadata or {}}
        self.model_registry[model_name] = model_info
        self.save_registry()
        logger.info(f'Model {model_name} saved to {model_path}')
        return model_path

    def train_pattern_model(self, training_data: Dict[str, pd.DataFrame],
        labels: Dict[str, List[List[int]]], model_name: str=
        'pattern_recognition', window_size: int=30, feature_columns:
        Optional[List[str]]=None, num_patterns: int=8, validation_split:
        float=0.2, epochs: int=50, batch_size: int=32, description: str='',
        metadata: Optional[Dict[str, Any]]=None) ->Tuple[
        PatternRecognitionModel, Dict[str, Any]]:
        """
        Train a pattern recognition model.
        
        Args:
            training_data: Dictionary mapping symbols to DataFrames with OHLCV data
            labels: Dictionary mapping symbols to lists of pattern labels
            model_name: Name of the model
            window_size: Size of the window for pattern recognition
            feature_columns: List of feature columns to use
            num_patterns: Number of patterns to recognize
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            description: Description of the model
            metadata: Additional metadata for the model
            
        Returns:
            Tuple of (model, training_results)
        """
        logger.info(f'Training pattern recognition model {model_name}')
        model = PatternRecognitionModel(window_size=window_size,
            feature_columns=feature_columns, num_patterns=num_patterns,
            use_gpu=self.use_gpu)
        training_results = model.train(training_data=training_data, labels=
            labels, validation_split=validation_split, epochs=epochs,
            batch_size=batch_size)
        metadata = metadata or {}
        metadata.update({'window_size': window_size, 'num_patterns':
            num_patterns, 'training_results': {'validation_loss':
            training_results['validation_loss'], 'validation_accuracy':
            training_results['validation_accuracy'], 'training_time':
            training_results['training_time']}})
        self.save_model(model=model, model_name=model_name, model_type=
            'pattern_recognition', description=description, metadata=metadata)
        return model, training_results

    def train_prediction_model(self, training_data: Dict[str, pd.DataFrame],
        model_name: str='price_prediction', input_window: int=60,
        output_window: int=10, feature_columns: Optional[List[str]]=None,
        target_column: str='close', validation_split: float=0.2, epochs:
        int=100, batch_size: int=32, description: str='', metadata:
        Optional[Dict[str, Any]]=None) ->Tuple[PricePredictionModel, Dict[
        str, Any]]:
        """
        Train a price prediction model.
        
        Args:
            training_data: Dictionary mapping symbols to DataFrames with OHLCV data
            model_name: Name of the model
            input_window: Size of the input window for prediction
            output_window: Size of the output window (prediction horizon)
            feature_columns: List of feature columns to use
            target_column: Column to predict
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            description: Description of the model
            metadata: Additional metadata for the model
            
        Returns:
            Tuple of (model, training_results)
        """
        logger.info(f'Training price prediction model {model_name}')
        model = PricePredictionModel(input_window=input_window,
            output_window=output_window, feature_columns=feature_columns,
            target_column=target_column, use_gpu=self.use_gpu)
        training_results = model.train(training_data=training_data,
            validation_split=validation_split, epochs=epochs, batch_size=
            batch_size)
        metadata = metadata or {}
        metadata.update({'input_window': input_window, 'output_window':
            output_window, 'target_column': target_column,
            'training_results': {'rmse': training_results['rmse'], 'mae':
            training_results['mae'], 'r2': training_results['r2'],
            'training_time': training_results['training_time']}})
        self.save_model(model=model, model_name=model_name, model_type=
            'price_prediction', description=description, metadata=metadata)
        return model, training_results

    @with_resilience('get_model_info')
    def get_model_info(self, model_name: str) ->Optional[Dict[str, Any]]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information, or None if model not found
        """
        return self.model_registry.get(model_name)

    def list_models(self, model_type: Optional[str]=None) ->List[Dict[str, Any]
        ]:
        """
        List all models in the registry.
        
        Args:
            model_type: Type of models to list (None for all)
            
        Returns:
            List of model information dictionaries
        """
        if model_type:
            return [info for name, info in self.model_registry.items() if 
                info['type'] == model_type]
        else:
            return list(self.model_registry.values())

    @with_resilience('delete_model')
    def delete_model(self, model_name: str) ->bool:
        """
        Delete a model from the registry and disk.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model was deleted, False otherwise
        """
        if model_name not in self.model_registry:
            return False
        model_info = self.model_registry[model_name]
        model_path = os.path.join(self.model_dir, model_info['path'])
        if os.path.exists(model_path):
            import shutil
            shutil.rmtree(os.path.dirname(model_path))
        del self.model_registry[model_name]
        self.save_registry()
        logger.info(f'Model {model_name} deleted')
        return True
