"""
ML Pipeline Client

A client for interacting with external ML services to trigger model retraining,
track job status, and handle model lifecycle management.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
import httpx
from urllib.parse import urljoin
from pybreaker import CircuitBreaker, CircuitBreakerState  # Assuming pybreaker is used

from core_foundations.utils.logger import get_logger
from core_foundations.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerState
from common_lib.resilience import retry_with_policy
from core_foundations.config.configuration import ConfigurationManager
from core_foundations.exceptions.client_exceptions import (
    MLClientConnectionError, 
    MLClientTimeoutError,
    MLClientAuthError,
    MLJobSubmissionError
)

# Base exception class for ML client errors
class MLClientError(Exception):
    """Base exception for ML client errors."""
    pass

logger = get_logger(__name__)


class MLPipelineClient:
    """
    Client for interacting with ML training pipelines and model management services.
    Supports integration with platforms like MLflow, Kubeflow, or custom ML services.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the ML Pipeline client.
        
        Args:
            config_manager: Configuration manager to load ML pipeline settings
        """
        self.config_manager = config_manager or ConfigurationManager() # Use provided or default manager
        self.config = self._load_config()
        
        self.base_url = self.config.get("base_url", "http://ml-integration-service:8000")
        self.timeout = self.config.get("timeout", 30.0)
        self.auth_token = self.config.get("auth_token") # Load auth token if configured

        # Circuit Breaker setup
        failure_threshold = self.config.get("circuit_breaker_failure_threshold", 5)
        recovery_timeout = self.config.get("circuit_breaker_recovery_timeout", 60)
        self.circuit_breaker = CircuitBreaker(fail_max=failure_threshold, reset_timeout=recovery_timeout)

        # Retry policy parameters
        self.max_attempts = self.config.get("max_retries", 3)
        self.base_delay = self.config.get("base_delay", 1.0)
        self.backoff_factor = self.config.get("backoff_factor", 1.5)
        self.max_delay = self.config.get("max_backoff", 30)
        # Define retryable exceptions directly here or ensure they are accessible
        self.retryable_exceptions = [MLClientConnectionError, MLClientTimeoutError] 
        
        logger.info(f"MLPipelineClient initialized. Base URL: {self.base_url}, Timeout: {self.timeout}, Max Retries: {self.max_attempts}")
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the configuration manager or use defaults.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {
            "base_url": "http://ml-integration-service:8000",
            "timeout": 30.0,
            "circuit_breaker_failure_threshold": 5,
            "circuit_breaker_recovery_timeout": 60,
            "max_retries": 3,
            "base_delay": 1.0,
            "backoff_factor": 1.5,
            "max_backoff": 30,
            "auth_token": None # Default auth token to None
        }
        
        if self.config_manager:
            try:
                # Assuming config manager has a method like get_section or get_config
                ml_config = self.config_manager.get_section("ml_pipeline_client") 
                # Merge defaults with loaded config, loaded config takes precedence
                default_config.update(ml_config)
                logger.info("Loaded configuration for MLPipelineClient from ConfigurationManager.")
            except Exception as e:
                logger.warning(f"Could not load MLPipelineClient config from manager: {e}. Using defaults.")
                
        return default_config
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the ML pipeline service with circuit breaker and retry support.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Optional data payload for POST/PUT requests
            
        Returns:
            Dict[str, Any]: Response data
            
        Raises:
            MLClientConnectionError: If connection to ML service fails or circuit breaker is open
            MLClientTimeoutError: If request times out
            MLClientAuthError: If authentication fails
            MLJobSubmissionError: If job submission fails or other HTTP errors occur
            ValueError: If an unsupported HTTP method is used
        """
        # Check circuit breaker state
        if self.circuit_breaker.current_state == CircuitBreakerState.OPEN:
            logger.warning("Circuit breaker is open. Refusing to make request to ML service.")
            raise MLClientConnectionError("Circuit breaker is open")
        
        url = urljoin(self.base_url, endpoint)
        headers = {"Content-Type": "application/json"}
        
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.debug(f"Making {method.upper()} request to {url} with data: {data}")
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers, json=data)
                elif method.upper() == "PUT":
                    response = await client.put(url, headers=headers, json=data)
                # Add DELETE or other methods if needed
                # elif method.upper() == "DELETE":
                #     response = await client.delete(url, headers=headers)
                else:
                    logger.error(f"Unsupported HTTP method requested: {method}")
                    raise ValueError(f"Unsupported HTTP method: {method}")

                logger.debug(f"Received response: {response.status_code}")

                # Check for error status codes before recording success
                if response.status_code == 401 or response.status_code == 403:
                    # Don't record failure for auth errors, as they are not transient network issues
                    logger.error(f"Authentication failed for {method.upper()} {url}: {response.status_code}")
                    raise MLClientAuthError(f"Authentication failed: {response.status_code}")
                
                if response.status_code >= 400:
                    # Record failure for server errors or bad requests that might be transient
                    self.circuit_breaker.record_failure()
                    logger.error(f"ML service request failed for {method.upper()} {url}: {response.status_code}, {response.text}")
                    raise MLJobSubmissionError(f"ML service request failed: {response.status_code}, {response.text}")
                
                # If no errors, record success and return the JSON response
                self.circuit_breaker.record_success()
                logger.debug(f"Request successful: {response.status_code}")
                return response.json()
                
        except httpx.ConnectError as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Connection error to ML service ({url}): {e}")
            raise MLClientConnectionError(f"Failed to connect to ML service: {e}") from e
        except httpx.TimeoutException as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Timeout connecting to ML service ({url}): {e}")
            raise MLClientTimeoutError(f"Request to ML service timed out: {e}") from e
        except Exception as e:
            # Catch unexpected errors and record failure
            self.circuit_breaker.record_failure()
            logger.error(f"Unexpected error making request to ML service ({url}): {e}", exc_info=True)
            # Re-raise the original exception to allow specific handling upstream if needed
            raise MLClientError(f"An unexpected error occurred: {e}") from e
            
    @retry_with_policy(
        # Pass the actual list of exception types
        exceptions=lambda self: self.retryable_exceptions, 
        max_attempts=lambda self: self.max_attempts,
        base_delay=lambda self: self.base_delay,
        max_delay=lambda self: self.max_delay,
        backoff_factor=lambda self: self.backoff_factor,
        service_name="ml-pipeline-client", # More specific service name
        operation_name="start_retraining_job"
    )
    async def start_retraining_job(self, model_id: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a model retraining job via the ML service.
        
        Args:
            model_id: Identifier for the model to retrain.
            params: Optional parameters for the retraining job (e.g., dataset version, hyperparameters).
            
        Returns:
            str: Job ID for the submitted retraining job.
            
        Raises:
            MLJobSubmissionError: If the job submission request fails after retries.
            MLClientConnectionError: If connection fails after retries.
            MLClientTimeoutError: If the request times out after retries.
            MLClientAuthError: If authentication fails (not typically retried).
            MLClientError: For other unexpected errors.
        """
        endpoint = f"/jobs/retrain/{model_id}" # Example endpoint structure
        payload = params if params is not None else {}
        
        logger.info(f"Starting retraining job for model_id: {model_id} with params: {payload}")
        
        try:
            response_data = await self._make_request("POST", endpoint, data=payload)
            
            job_id = response_data.get("job_id")
            if not job_id:
                logger.error(f"ML service response missing 'job_id'. Response: {response_data}")
                raise MLJobSubmissionError("ML service response did not include a job ID.")
                
            logger.info(f"Successfully submitted retraining job for model_id: {model_id}. Job ID: {job_id}")
            return job_id
        except (MLClientConnectionError, MLClientTimeoutError, MLJobSubmissionError, MLClientAuthError) as e:
            # Log specific client errors occurred during the request
            logger.error(f"Failed to start retraining job for model_id {model_id}: {e}")
            raise # Re-raise the specific exception caught by _make_request
        except Exception as e:
            # Catch any other unexpected errors during processing
            logger.error(f"An unexpected error occurred while starting retraining job for model_id {model_id}: {e}", exc_info=True)
            raise MLClientError(f"An unexpected error occurred during job submission: {e}") from e

    # --- Add other client methods below ---

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific job.

        Args:
            job_id: The ID of the job to check.

        Returns:
            Dict[str, Any]: A dictionary containing the job status information.
        
        Raises:
            MLClientError: If the request fails or the job is not found.
        """
        endpoint = f"/jobs/status/{job_id}" # Example endpoint
        logger.info(f"Fetching status for job_id: {job_id}")
        try:
            # Assuming GET request for status check
            status_data = await self._make_request("GET", endpoint) 
            logger.info(f"Successfully fetched status for job_id: {job_id}. Status: {status_data}")
            return status_data
        except Exception as e:
            logger.error(f"Failed to get status for job_id {job_id}: {e}")
            raise # Re-raise exceptions from _make_request or other issues

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models managed by the ML service.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a model.
        
        Raises:
            MLClientError: If the request fails.
        """
        endpoint = "/models" # Example endpoint
        logger.info("Fetching list of models.")
        try:
            models_data = await self._make_request("GET", endpoint)
            if not isinstance(models_data, list): # Basic validation
                 logger.error(f"Expected list of models, but got: {type(models_data)}")
                 raise MLClientError("Invalid response format when listing models.")
            logger.info(f"Successfully fetched {len(models_data)} models.")
            return models_data
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise

    async def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: The ID of the model to retrieve details for.
            
        Returns:
            Dict[str, Any]: A dictionary containing detailed model information.
            
        Raises:
            MLClientError: If the request fails or the model is not found.
        """
        endpoint = f"/models/{model_id}"
        logger.info(f"Fetching details for model_id: {model_id}")
        
        try:
            model_data = await self._make_request("GET", endpoint)
            logger.info(f"Successfully fetched details for model_id: {model_id}")
            return model_data
        except Exception as e:
            logger.error(f"Failed to get details for model_id {model_id}: {e}")
            raise
    
    async def deploy_model(self, model_id: str, environment: str = "production", 
                         config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Deploy a model to the specified environment.
        
        Args:
            model_id: The ID of the model to deploy.
            environment: The target environment (e.g., "staging", "production").
            config: Optional configuration parameters for deployment.
            
        Returns:
            Dict[str, Any]: Deployment response information.
            
        Raises:
            MLClientError: If the deployment request fails.
        """
        endpoint = f"/models/{model_id}/deploy"
        payload = {
            "environment": environment,
            **(config or {})
        }
        
        logger.info(f"Deploying model_id: {model_id} to environment: {environment}")
        
        try:
            response_data = await self._make_request("POST", endpoint, data=payload)
            logger.info(f"Successfully initiated deployment for model_id: {model_id} to {environment}")
            return response_data
        except Exception as e:
            logger.error(f"Failed to deploy model_id {model_id} to environment {environment}: {e}")
            raise
    
    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running job.
        
        Args:
            job_id: The ID of the job to cancel.
            
        Returns:
            Dict[str, Any]: A dictionary containing the cancellation response.
            
        Raises:
            MLClientError: If the request fails or the job cannot be cancelled.
        """
        endpoint = f"/jobs/{job_id}/cancel"
        logger.info(f"Attempting to cancel job_id: {job_id}")
        
        try:
            response_data = await self._make_request("POST", endpoint)
            logger.info(f"Successfully cancelled job_id: {job_id}")
            return response_data
        except Exception as e:
            logger.error(f"Failed to cancel job_id {job_id}: {e}")
            raise

    async def get_prediction(self, model_id: str, features: Dict[str, Any], 
                           version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get prediction from a deployed model.
        
        Args:
            model_id: The ID of the model to use for prediction.
            features: Dictionary containing the feature values needed for prediction.
            version: Optional specific model version to use (defaults to latest if not provided).
            
        Returns:
            Dict[str, Any]: The prediction results from the model.
            
        Raises:
            MLClientError: If the prediction request fails.
        """
        if version:
            endpoint = f"/models/{model_id}/versions/{version}/predict"
        else:
            endpoint = f"/models/{model_id}/predict"
            
        payload = {"features": features}
        
        logger.info(f"Requesting prediction from model_id: {model_id}" + 
                  (f" version: {version}" if version else " (latest version)"))
        
        try:
            start_time = datetime.now(timezone.utc)
            response_data = await self._make_request("POST", endpoint, data=payload)
            end_time = datetime.now(timezone.utc)
            
            # Log the latency for monitoring purposes
            latency_ms = (end_time - start_time).total_seconds() * 1000
            logger.debug(f"Prediction latency: {latency_ms:.2f}ms for model_id: {model_id}")
            
            # Add metadata about the prediction request
            response_data["metadata"] = {
                "request_time": start_time.isoformat(),
                "response_time": end_time.isoformat(),
                "latency_ms": latency_ms,
                "model_id": model_id,
                "version": version or "latest"
            }
            
            logger.info(f"Successfully obtained prediction from model_id: {model_id}")
            return response_data
        except Exception as e:
            logger.error(f"Failed to get prediction from model_id {model_id}: {e}")
            raise

    # Add more methods as needed, e.g., for deploying models, getting predictions, etc.

# Example Usage (requires an async context)
# async def main():
#     # Assuming ConfigurationManager is set up
#     # config_manager = ConfigurationManager(...) 
#     client = MLPipelineClient() # Or pass config_manager
#     try:
#         job_id = await client.start_retraining_job("model-abc-123", params={"epochs": 10, "learning_rate": 0.01})
#         print(f"Started job: {job_id}")
#         # ... wait or poll ...
#         status = await client.get_job_status(job_id)
#         print(f"Job status: {status}")
#         models = await client.list_models()
#         print(f"Available models: {models}")
#     except MLClientError as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     import asyncio
#     # Configure logging basic setup for demonstration
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     asyncio.run(main())
