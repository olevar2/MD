"""
ML Backtester Integration Module.

This module enhances the existing backtester with the ability to
incorporate machine learning predictions into trading strategies.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger
from strategy_execution_engine.backtesting.backtest_engine import BacktestEngine
from ml_workbench_service.clients.ml_prediction_client import MLPredictionClient

logger = get_logger("ml_backtester")

class MLBacktesterIntegration:
    """
    Integration layer between the backtester and ML prediction service.
    
    This class adds ML prediction capabilities to the standard backtester,
    allowing strategies to incorporate ML signals into their decision-making.
    """
    
    def __init__(self, 
                backtest_engine: BacktestEngine,
                ml_client: Optional[MLPredictionClient] = None,
                api_base_url: str = "http://localhost:8002"):
        """
        Initialize the ML backtester integration.
        
        Args:
            backtest_engine: Existing backtest engine instance
            ml_client: Optional ML prediction client (creates one if None)
            api_base_url: Base URL for the ML prediction API
        """
        self.backtest_engine = backtest_engine
        
        # Create ML client if not provided
        self.ml_client = ml_client or MLPredictionClient(api_base_url=api_base_url)
        
        # Cache for predictions to avoid redundant API calls
        self._prediction_cache = {}
        
        # Hook into the backtest engine
        self._patch_backtest_engine()
        
        logger.info("ML Backtester Integration initialized")
    
    def _patch_backtest_engine(self):
        """
        Patch the backtest engine with ML prediction capabilities.
        This extends the original engine without modifying its source code.
        """
        # Store original methods that will be extended
        self._original_init_backtest = self.backtest_engine.init_backtest
        self._original_process_bar = self.backtest_engine.process_bar
        
        # Patch methods
        self.backtest_engine.init_backtest = self._patched_init_backtest
        self.backtest_engine.process_bar = self._patched_process_bar
        
        # Add new methods
        self.backtest_engine.get_ml_prediction = self.get_prediction
        self.backtest_engine.get_ml_forecast = self.get_forecast
        
        logger.info("Backtest engine patched with ML prediction capabilities")
    
    def _patched_init_backtest(self, *args, **kwargs):
        """
        Patched initialization method that adds ML-specific setup.
        """
        # Call the original method first
        result = self._original_init_backtest(*args, **kwargs)
        
        # Initialize prediction cache
        self._prediction_cache = {}
        
        # Log the enhancement
        logger.info("Backtest initialized with ML prediction capabilities")
        
        return result
    
    def _patched_process_bar(self, *args, **kwargs):
        """
        Patched bar processing method that clears prediction cache
        for the current bar to ensure fresh predictions.
        """
        # Clear per-bar prediction cache if using one
        current_bar_time = self.backtest_engine.current_bar_time
        if current_bar_time:
            cache_keys = list(self._prediction_cache.keys())
            for key in cache_keys:
                if key.startswith(f"{current_bar_time}:"):
                    del self._prediction_cache[key]
        
        # Call the original method
        return self._original_process_bar(*args, **kwargs)
    
    def get_prediction(self, 
                     model_name: str,
                     inputs: Dict[str, Any] = None,
                     version_id: Optional[str] = None,
                     use_cache: bool = True) -> Dict[str, Any]:
        """
        Get a prediction from an ML model.
        
        Args:
            model_name: Name of the model to use
            inputs: Input data for prediction (if None, uses current context)
            version_id: Optional specific model version to use
            use_cache: Whether to use cached predictions
            
        Returns:
            Dictionary containing prediction results
            
        Raises:
            Exception: If the prediction fails
        """
        # If no inputs provided, use current market data
        if inputs is None:
            inputs = self._get_current_market_context()
        
        # Generate cache key
        current_time = self.backtest_engine.current_bar_time
        cache_key = f"{current_time}:{model_name}:{version_id or 'default'}"
        
        # Return cached prediction if available
        if use_cache and cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        try:
            # Get prediction from ML service
            result = self.ml_client.get_prediction(
                model_name=model_name,
                inputs=inputs,
                version_id=version_id
            )
            
            # Cache the prediction
            if use_cache:
                self._prediction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting ML prediction: {str(e)}")
            
            # In backtesting, we don't want to crash the entire backtest
            # Return a placeholder with error information
            return {
                "prediction": None,
                "error": str(e),
                "metadata": {
                    "model_name": model_name,
                    "version_id": version_id,
                    "status": "error"
                }
            }
    
    def get_forecast(self, 
                    model_name: str, 
                    horizon: int = 1, 
                    return_direction: bool = False,
                    threshold: float = 0.0,
                    version_id: Optional[str] = None) -> Union[float, bool]:
        """
        Get a simplified forecast from an ML model.
        
        This is a convenience method that returns just the forecasted value
        or direction, suitable for direct use in strategy logic.
        
        Args:
            model_name: Name of the model to use
            horizon: How many steps ahead to forecast (1-indexed)
            return_direction: If True, returns directional signal (bool) instead of value
            threshold: Threshold for directional signal (if return_direction=True)
            version_id: Optional specific model version to use
            
        Returns:
            Forecasted value (float) or direction (bool)
            
        Raises:
            Exception: If the forecast fails
        """
        try:
            # Get the full prediction
            result = self.get_prediction(
                model_name=model_name,
                version_id=version_id
            )
            
            # Check if prediction was successful
            if result.get("prediction") is None:
                raise ValueError(result.get("error", "Unknown prediction error"))
            
            # Extract the forecast for the specified horizon
            prediction_data = result["prediction"]
            
            # Handle different prediction formats
            if isinstance(prediction_data, dict) and "predictions" in prediction_data:
                forecasts = prediction_data["predictions"]
                if isinstance(forecasts, list) and horizon <= len(forecasts):
                    forecast_value = forecasts[horizon - 1]
                else:
                    forecast_value = forecasts
            elif isinstance(prediction_data, list) and horizon <= len(prediction_data):
                forecast_value = prediction_data[horizon - 1]
            else:
                forecast_value = prediction_data
            
            # Return directional signal if requested
            if return_direction:
                # Get current price
                current_price = self.backtest_engine.get_current_price()
                # Return directional boolean based on threshold
                return (forecast_value - current_price) > threshold
            else:
                # Return the raw forecast value
                return forecast_value
                
        except Exception as e:
            logger.error(f"Error getting forecast: {str(e)}")
            # Return a neutral value
            return 0.0 if not return_direction else False
    
    def _get_current_market_context(self) -> Dict[str, Any]:
        """
        Get the current market context from the backtest engine.
        
        Returns:
            Dictionary with market data suitable for ML model input
        """
        # Extract relevant data for ML input
        engine = self.backtest_engine
        
        # Current symbol and timestamp
        symbol = engine.current_symbol
        timestamp = engine.current_bar_time
        
        # Get historical data (last N bars)
        lookback = 60  # Default lookback for ML models
        historical_data = engine.get_historical_bars(lookback)
        
        # Get technical indicators (if available)
        indicators = {}
        if hasattr(engine, 'get_indicators'):
            indicators = engine.get_indicators()
        
        # Format data for ML input
        context = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat() if timestamp else None,
            "historical_data": historical_data.to_dict(orient='records') if isinstance(historical_data, pd.DataFrame) else [],
            "indicators": indicators,
            # Add any other context needed by the ML model
        }
        
        return context
