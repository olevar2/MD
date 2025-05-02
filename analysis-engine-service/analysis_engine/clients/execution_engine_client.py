"""
Execution Engine Client

A client for interacting with the Strategy Execution Engine service to deploy
strategies, update parameters, and manage strategy lifecycle operations.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timezone
import httpx
from urllib.parse import urljoin

from core_foundations.utils.logger import get_logger
from core_foundations.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerState
from common_lib.resilience import retry_with_policy
from core_foundations.config.configuration import ConfigurationManager
from core_foundations.exceptions.client_exceptions import (
    ExecutionEngineConnectionError,
    ExecutionEngineTimeoutError,
    ExecutionEngineAuthError,
    StrategyDeploymentError,
    StrategyParameterUpdateError
)

logger = get_logger(__name__)


class ExecutionEngineClient:
    """
    Client for interacting with the Strategy Execution Engine service.
    Provides methods for deploying strategies and updating strategy parameters.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the Execution Engine client.
        
        Args:
            config_manager: Configuration manager to load execution engine settings
        """
        self.config_manager = config_manager
        self.config = self._load_config()
        self.base_url = self.config.get("base_url", "http://strategy-execution-engine:8080")
        self.auth_token = self.config.get("auth_token")
        self.timeout = self.config.get("timeout", 30.0)
        
        # Initialize circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get("circuit_breaker_failure_threshold", 5),
            recovery_timeout=self.config.get("circuit_breaker_recovery_timeout", 60),
            name="execution_engine_client"
        )
          # Retry policy parameters - to be used directly with retry_with_policy decorator
        self.max_attempts = self.config.get("max_retries", 3)
        self.base_delay = self.config.get("base_delay", 1.0)
        self.backoff_factor = self.config.get("backoff_factor", 1.5)
        self.max_delay = self.config.get("max_backoff", 30)
        self.retryable_exceptions = [ExecutionEngineConnectionError, ExecutionEngineTimeoutError]
        
        logger.info(f"ExecutionEngineClient initialized with base URL: {self.base_url}")
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the configuration manager or use defaults.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {
            "base_url": "http://strategy-execution-engine:8080",
            "timeout": 30.0,
            "circuit_breaker_failure_threshold": 5,
            "circuit_breaker_recovery_timeout": 60,
            "max_retries": 3,
            "backoff_factor": 1.5,
            "max_backoff": 30
        }
        
        if self.config_manager:
            try:
                execution_config = self.config_manager.get_config("execution_engine_client")
                if execution_config:
                    return {**default_config, **execution_config}
            except Exception as e:
                logger.warning(f"Failed to load execution engine client config: {e}")
                
        return default_config
      async def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the Strategy Execution Engine with circuit breaker and retry support.
        
        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            endpoint: API endpoint path
            data: Optional data payload for POST/PUT requests
            
        Returns:
            Dict[str, Any]: Response data
            
        Raises:
            ExecutionEngineConnectionError: If connection to service fails
            ExecutionEngineTimeoutError: If request times out
            ExecutionEngineAuthError: If authentication fails
            StrategyDeploymentError: If strategy deployment fails
            StrategyParameterUpdateError: If parameter update fails
        """
        # Check circuit breaker state
        if self.circuit_breaker.state == CircuitBreakerState.OPEN:
            logger.warning("Circuit breaker is open. Refusing to make request to Strategy Execution Engine.")
            raise ExecutionEngineConnectionError("Circuit breaker is open")
        
        url = urljoin(self.base_url, endpoint)
        headers = {"Content-Type": "application/json"}
        
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        try:
            response = await self._execute_request(method, url, headers, data)
            
            # Record success with circuit breaker
            self.circuit_breaker.record_success()
            
            # Handle error status codes
            if response.status_code == 401 or response.status_code == 403:
                raise ExecutionEngineAuthError(f"Authentication failed: {response.status_code}")
            
            if response.status_code >= 400:
                self._handle_error_response(response, endpoint)
                
            return response.json()
                
                return response.json()
                
        except httpx.ConnectError as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Connection error to Strategy Execution Engine: {e}")
            raise ExecutionEngineConnectionError(f"Failed to connect to Strategy Execution Engine: {e}")
        except httpx.TimeoutException as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Timeout connecting to Strategy Execution Engine: {e}")
            raise ExecutionEngineTimeoutError(f"Request to Strategy Execution Engine timed out: {e}")
        except Exception as e:
            if not isinstance(e, (ExecutionEngineAuthError, StrategyDeploymentError, StrategyParameterUpdateError)):
                self.circuit_breaker.record_failure()
            logger.error(f"Error making request to Strategy Execution Engine: {e}")
            raise
    
    @async_retry(retry_config_attr="retry_config")
    async def set_strategy_parameter(self, strategy_id: str, parameter_name: str, new_value: Any) -> bool:
        """
        Update a specific parameter for a running strategy.
        
        Args:
            strategy_id: The identifier of the strategy to update
            parameter_name: The name of the parameter to update
            new_value: The new value for the parameter
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            StrategyParameterUpdateError: If parameter update fails
        """
        logger.info(f"Updating parameter '{parameter_name}' for strategy {strategy_id} to {new_value}")
        
        update_data = {
            "parameter_name": parameter_name,
            "value": new_value,
            "updated_at": datetime.utcnow().isoformat(),
            "update_id": str(uuid.uuid4())
        }
        
        try:
            response = await self._make_request(
                "PUT", 
                f"/api/v1/strategies/{strategy_id}/parameters",
                update_data
            )
            
            success = response.get("success", False)
            if success:
                logger.info(f"Successfully updated parameter '{parameter_name}' for strategy {strategy_id}")
            else:
                logger.warning(f"Failed to update parameter '{parameter_name}' for strategy {strategy_id}: {response.get('message', 'Unknown error')}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error updating parameter for strategy {strategy_id}: {e}")
            if isinstance(e, StrategyParameterUpdateError):
                raise
            raise StrategyParameterUpdateError(f"Failed to update parameter: {e}")

    @async_retry(retry_config_attr="retry_config")
    async def deploy_strategy(self, strategy_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy a new or updated strategy configuration.
        
        Args:
            strategy_id: The identifier of the strategy
            config: The strategy configuration to deploy
            
        Returns:
            Dict[str, Any]: Deployment result information
            
        Raises:
            StrategyDeploymentError: If strategy deployment fails
        """
        logger.info(f"Deploying strategy {strategy_id} with config version {config.get('version', 'unknown')}")
        
        deployment_data = {
            "strategy_id": strategy_id,
            "config": config,
            "deployment_id": str(uuid.uuid4()),
            "deployed_at": datetime.utcnow().isoformat()
        }
        
        try:
            response = await self._make_request("POST", "/api/v1/strategies/deploy", deployment_data)
            
            if response.get("success", False):
                logger.info(f"Successfully deployed strategy {strategy_id}")
                return response
            else:
                error_msg = response.get("message", "Unknown error")
                logger.error(f"Failed to deploy strategy {strategy_id}: {error_msg}")
                raise StrategyDeploymentError(f"Failed to deploy strategy: {error_msg}")
                
        except Exception as e:
            logger.error(f"Error deploying strategy {strategy_id}: {e}")
            if isinstance(e, StrategyDeploymentError):
                raise
            raise StrategyDeploymentError(f"Failed to deploy strategy: {e}")
    
    async def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get the current status of a deployed strategy.
        
        Args:
            strategy_id: The identifier of the strategy
            
        Returns:
            Dict[str, Any]: Strategy status information
        """
        try:
            return await self._make_request("GET", f"/api/v1/strategies/{strategy_id}/status")
        except Exception as e:
            logger.error(f"Error getting status for strategy {strategy_id}: {e}")
            raise
    
    async def list_strategies(self, 
                       status: Optional[str] = None,
                       asset_class: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all strategies, optionally filtered by status or asset class.
        
        Args:
            status: Optional filter by status (active, paused, etc.)
            asset_class: Optional filter by asset class (forex, crypto, etc.)
            
        Returns:
            List[Dict[str, Any]]: List of strategy information
        """
        params = []
        if status:
            params.append(f"status={status}")
        if asset_class:
            params.append(f"asset_class={asset_class}")
            
        endpoint = "/api/v1/strategies"
        if params:
            endpoint = f"{endpoint}?{'&'.join(params)}"
            
        try:
            response = await self._make_request("GET", endpoint)
            return response.get("strategies", [])
        except Exception as e:
            logger.error(f"Error listing strategies: {e}")
            raise
    
    async def pause_strategy(self, strategy_id: str) -> bool:
        """
        Pause a running strategy.
        
        Args:
            strategy_id: The identifier of the strategy
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = await self._make_request(
                "PUT", 
                f"/api/v1/strategies/{strategy_id}/pause", 
                {"paused_at": datetime.utcnow().isoformat()}
            )
            return response.get("success", False)
        except Exception as e:
            logger.error(f"Error pausing strategy {strategy_id}: {e}")
            raise
    
    async def resume_strategy(self, strategy_id: str) -> bool:
        """
        Resume a paused strategy.
        
        Args:
            strategy_id: The identifier of the strategy
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = await self._make_request(
                "PUT", 
                f"/api/v1/strategies/{strategy_id}/resume",
                {"resumed_at": datetime.utcnow().isoformat()}
            )
            return response.get("success", False)
        except Exception as e:
            logger.error(f"Error resuming strategy {strategy_id}: {e}")
            raise
