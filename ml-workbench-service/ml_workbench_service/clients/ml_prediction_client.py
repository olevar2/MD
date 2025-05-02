"""
ML Prediction Client for Backtester Integration.

This module provides a client for integration between the backtesting engine
and ML prediction API, allowing strategies to leverage ML predictions
during backtesting or live execution.
"""

import logging
import requests
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from core_foundations.utils.logger import get_logger

logger = get_logger("ml_prediction_client")

class MLPredictionClient:
    """
    Client for interacting with the ML prediction API.
    
    This client allows the backtester and strategies to request predictions
    from ML models served by the ModelServingEngine. It handles communication
    with the API, error handling, and data formatting.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8002", timeout: int = 5):
        """
        Initialize the ML prediction client.
        
        Args:
            api_base_url: Base URL for the ML prediction API
            timeout: Request timeout in seconds
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.timeout = timeout
        self.prediction_endpoint = f"{self.api_base_url}/api/v1/predict"
        self.models_endpoint = f"{self.api_base_url}/api/v1/models"
        
        logger.info(f"MLPredictionClient initialized with API at {self.api_base_url}")
    
    def get_prediction(self, 
                     model_name: str, 
                     inputs: Dict[str, Any],
                     version_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a prediction from a model.
        
        Args:
            model_name: Name of the model to use
            inputs: Input data for prediction
            version_id: Optional specific model version to use
            
        Returns:
            Dictionary containing prediction results
            
        Raises:
            Exception: If the request fails
        """
        try:
            endpoint = self.prediction_endpoint
            
            payload = {
                "model_name": model_name,
                "inputs": inputs
            }
            
            if version_id:
                payload["version_id"] = version_id
            
            headers = {"Content-Type": "application/json"}
            
            start_time = datetime.now()
            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            request_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"Prediction from {model_name} received in {request_time:.2f}ms")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"Request to ML prediction API timed out after {self.timeout}s")
            raise Exception(f"ML prediction request timed out after {self.timeout}s")
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from ML prediction API: {str(e)}")
            if response.status_code == 404:
                raise Exception(f"Model {model_name} not found")
            else:
                raise Exception(f"ML prediction API error: {str(e)}")
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to ML prediction API: {str(e)}")
            raise Exception("Could not connect to ML prediction API")
            
        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
            raise
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get the list of available models from the API.
        
        Returns:
            List of dictionaries with model information
            
        Raises:
            Exception: If the request fails
        """
        try:
            response = requests.get(
                self.models_endpoint,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json().get("models", [])
            
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            raise
    
    def get_model_info(self, model_name: str, version_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            version_id: Optional specific model version
            
        Returns:
            Dictionary with model information
            
        Raises:
            Exception: If the request fails
        """
        try:
            endpoint = f"{self.models_endpoint}/{model_name}"
            if version_id:
                endpoint += f"/{version_id}"
                
            response = requests.get(
                endpoint,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            raise
