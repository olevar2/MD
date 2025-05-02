"""
Transfer Learning Client

This module provides a client for interacting with the transfer learning API,
making it easy for other services to use transfer learning functionality.
"""

import logging
import requests
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class TransferLearningClient:
    """
    Client for interacting with the transfer learning API.
    
    This client provides methods to:
    1. List available models
    2. Find transfer learning opportunities
    3. Create and manage transfer learning models
    4. Evaluate transfer learning effectiveness
    5. Transform features using transfer learning models
    """
    
    def __init__(self, 
                base_url: str = "http://ml-workbench-service:8030/api/v1",
                api_key: Optional[str] = None):
        """
        Initialize the transfer learning client
        
        Args:
            base_url: Base URL of the ML Workbench Service API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("ML_WORKBENCH_API_KEY")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     data: Optional[Dict[str, Any]] = None,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the API
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request data
            params: URL parameters
            
        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        try:
            response = None
            
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, json=data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for successful response
            response.raise_for_status()
            
            # Parse and return response data
            if response.content:
                return response.json()
            return {}
            
        except requests.RequestException as e:
            self.logger.error(f"API request error: {str(e)}")
            
            # Try to extract error details if available
            error_detail = "Unknown error"
            if response is not None:
                try:
                    error_data = response.json()
                    if "detail" in error_data:
                        error_detail = error_data["detail"]
                except:
                    error_detail = response.text
            
            raise Exception(f"API request failed: {error_detail}")
    
    def list_models(self,
                   symbol: Optional[str] = None,
                   timeframe: Optional[str] = None,
                   is_transfer_model: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        List available models with optional filtering
        
        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            is_transfer_model: Filter by transfer model status
            
        Returns:
            List of model information dictionaries
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        if timeframe:
            params["timeframe"] = timeframe
        if is_transfer_model is not None:
            params["is_transfer_model"] = str(is_transfer_model).lower()
            
        response = self._make_request("GET", "/transfer-learning/models", params=params)
        return response.get("models", [])
    
    def find_transfer_candidates(self,
                               target_symbol: str,
                               target_timeframe: str,
                               min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find suitable source models for transfer learning
        
        Args:
            target_symbol: Target instrument
            target_timeframe: Target timeframe
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of transfer candidates with similarity scores
        """
        data = {
            "target_symbol": target_symbol,
            "target_timeframe": target_timeframe,
            "min_similarity": min_similarity
        }
        
        response = self._make_request("POST", "/transfer-learning/candidates", data=data)
        return response.get("candidates", [])
    
    def create_transfer_model(self,
                            source_model_id: str,
                            source_data: Union[pd.DataFrame, List[Dict[str, Any]]],
                            target_data: Union[pd.DataFrame, List[Dict[str, Any]]],
                            target_symbol: str,
                            target_timeframe: str,
                            adapt_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a transfer learning model
        
        Args:
            source_model_id: ID of the source model
            source_data: Training data from source domain
            target_data: Training data from target domain
            target_symbol: Target instrument symbol
            target_timeframe: Target timeframe
            adapt_layers: Optional list of layers to adapt
            
        Returns:
            Result dictionary with model information
        """
        # Convert DataFrame to list of dictionaries if needed
        if isinstance(source_data, pd.DataFrame):
            source_data = source_data.to_dict(orient="records")
            
        if isinstance(target_data, pd.DataFrame):
            target_data = target_data.to_dict(orient="records")
            
        data = {
            "source_model_id": source_model_id,
            "source_data": source_data,
            "target_data": target_data,
            "target_symbol": target_symbol,
            "target_timeframe": target_timeframe,
        }
        
        if adapt_layers:
            data["adapt_layers"] = adapt_layers
        
        return self._make_request("POST", "/transfer-learning/models", data=data)
    
    def evaluate_model(self,
                      model_id: str,
                      test_data: Union[pd.DataFrame, List[Dict[str, Any]]],
                      test_labels: Union[pd.Series, List[Any]]) -> Dict[str, Any]:
        """
        Evaluate a transfer model on test data
        
        Args:
            model_id: ID of the transfer model
            test_data: Test data for evaluation
            test_labels: Ground truth labels
            
        Returns:
            Evaluation metrics
        """
        # Convert DataFrame/Series to list if needed
        if isinstance(test_data, pd.DataFrame):
            test_data = test_data.to_dict(orient="records")
            
        if isinstance(test_labels, pd.Series):
            test_labels = test_labels.tolist()
            
        data = {
            "model_id": model_id,
            "test_data": test_data,
            "test_labels": test_labels
        }
        
        return self._make_request("POST", f"/transfer-learning/models/{model_id}/evaluate", data=data)
    
    def transform_features(self,
                          model_id: str,
                          features: Union[pd.DataFrame, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Transform features using a transfer model
        
        Args:
            model_id: ID of the transfer model
            features: Features to transform
            
        Returns:
            Transformed features and metadata
        """
        # Convert DataFrame to list if needed
        if isinstance(features, pd.DataFrame):
            features = features.to_dict(orient="records")
            
        data = {
            "model_id": model_id,
            "features": features
        }
        
        result = self._make_request("POST", "/transfer-learning/transform", data=data)
        
        # Convert transformed features back to DataFrame if results exist
        if result.get("success") and "transformed_features" in result and result["transformed_features"]:
            result["transformed_features_df"] = pd.DataFrame(result["transformed_features"])
            
        return result
    
    def upload_data(self, filepath: str) -> Dict[str, Any]:
        """
        Upload data file for transfer learning analysis
        
        Args:
            filepath: Path to CSV or JSON file
            
        Returns:
            Upload result information
        """
        url = f"{self.base_url}/transfer-learning/upload-data"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            with open(filepath, 'rb') as f:
                files = {'file': (os.path.basename(filepath), f)}
                response = requests.post(url, headers=headers, files=files)
                
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error uploading file: {str(e)}")
            raise Exception(f"Failed to upload file: {str(e)}")

    
class TransferLearningExample:
    """
    Example usage of the transfer learning client
    """
    
    @staticmethod
    def demonstrate_workflow():
        """
        Demonstrate complete transfer learning workflow
        """
        # Create client
        client = TransferLearningClient()
        print("Transfer Learning Client initialized...")
        
        # Example target
        target_symbol = "GBPUSD"
        target_timeframe = "1h"
        print(f"Looking for transfer candidates for {target_symbol} {target_timeframe}...")
        
        # Find transfer candidates
        candidates = client.find_transfer_candidates(target_symbol, target_timeframe)
        
        if not candidates:
            print("No transfer candidates found with sufficient similarity")
            return
            
        print(f"Found {len(candidates)} potential candidates.")
        print(f"Best candidate: {candidates[0]['source_symbol']} with similarity {candidates[0]['similarity']}")
        
        # Create example data (in practice, this would be real data)
        import numpy as np
        
        # Create synthetic source data
        source_data = pd.DataFrame({
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.uniform(-2, 2, 100),
            'ma_diff': np.random.uniform(-1, 1, 100),
            'volatility': np.random.uniform(0.1, 0.5, 100)
        })
        
        # Create synthetic target data
        target_data = pd.DataFrame({
            'rsi': np.random.uniform(30, 70, 100),
            'macd': np.random.uniform(-1.5, 1.5, 100),
            'ma_diff': np.random.uniform(-0.8, 0.8, 100),
            'volatility': np.random.uniform(0.2, 0.6, 100)
        })
        
        # Create test data & labels
        test_data = pd.DataFrame({
            'rsi': np.random.uniform(30, 70, 20),
            'macd': np.random.uniform(-1.5, 1.5, 20),
            'ma_diff': np.random.uniform(-0.8, 0.8, 20),
            'volatility': np.random.uniform(0.2, 0.6, 20)
        })
        test_labels = np.random.choice([0, 1], 20)
        
        # Get source model ID from first candidate if available
        source_model_id = None
        if candidates and candidates[0].get("models"):
            source_model_id = candidates[0]["models"][0]["id"]
            source_symbol = candidates[0]["source_symbol"]
            
        if not source_model_id:
            print("No source model available. Using example model ID.")
            source_model_id = "example_model_id"
            source_symbol = "EURUSD"
            
        print(f"Using source model {source_model_id} from {source_symbol}...")
        
        # Create transfer model
        print("Creating transfer model...")
        try:
            result = client.create_transfer_model(
                source_model_id=source_model_id,
                source_data=source_data,
                target_data=target_data,
                target_symbol=target_symbol,
                target_timeframe=target_timeframe
            )
            
            if result.get("success"):
                model_id = result.get("model_id")
                print(f"Transfer model created successfully: {model_id}")
                
                # Evaluate model
                print("Evaluating transfer model...")
                eval_result = client.evaluate_model(model_id, test_data, test_labels)
                
                if eval_result.get("success"):
                    print("Evaluation metrics:", eval_result)
                    
                    # Transform new features
                    print("Transforming new features...")
                    new_features = pd.DataFrame({
                        'rsi': np.random.uniform(30, 70, 5),
                        'macd': np.random.uniform(-1.5, 1.5, 5),
                        'ma_diff': np.random.uniform(-0.8, 0.8, 5),
                        'volatility': np.random.uniform(0.2, 0.6, 5)
                    })
                    
                    transform_result = client.transform_features(model_id, new_features)
                    
                    if transform_result.get("success"):
                        print("Features transformed successfully.")
                        print(f"Original shape: {transform_result.get('original_shape')}")
                        print(f"Transformed shape: {transform_result.get('transformed_shape')}")
                        
                        if "transformed_features_df" in transform_result:
                            print("Transformed features sample:")
                            print(transform_result["transformed_features_df"].head())
                    else:
                        print(f"Feature transformation failed: {transform_result.get('error')}")
                else:
                    print(f"Model evaluation failed: {eval_result.get('error')}")
            else:
                print(f"Transfer model creation failed: {result.get('error')}")
                
        except Exception as e:
            print(f"Error during demonstration: {str(e)}")
            
        print("Transfer learning workflow demonstration complete.")
