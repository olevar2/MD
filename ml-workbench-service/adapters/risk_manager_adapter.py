#!/usr/bin/env python3
"""
RiskManagerAdapter - Adapter for IRiskManager
"""

from typing import Dict, List, Optional, Any

from common_lib.interfaces.risk_manager import IRiskManager
from common_lib.errors import ServiceError, NotFoundError
from common_lib.resilience.decorators import with_circuit_breaker, with_retry, with_timeout

class RiskManagerAdapter(IRiskManager):
    """
    Adapter implementation for IRiskManager.
    """
    
    def __init__(self, service_client=None):
        """
        Initialize the adapter with an optional service client.
        
        Args:
            service_client: Client for the risk management service
        """
        self.service_client = service_client
    
    @with_circuit_breaker
    @with_retry
    @with_timeout(seconds=30)
    def validate_risk_limits(self, strategy_id: str, position_size: float, 
                            instrument: str, direction: str) -> Dict[str, Any]:
        """
        Validate if a position complies with risk limits.
        
        Args:
            strategy_id: ID of the strategy
            position_size: Size of the position
            instrument: Trading instrument
            direction: Trade direction (buy/sell)
            
        Returns:
            Dict containing validation result
            
        Raises:
            ServiceError: If the service call fails
            NotFoundError: If the strategy is not found
        """
        try:
            return self.service_client.validate_risk_limits(
                strategy_id=strategy_id,
                position_size=position_size,
                instrument=instrument,
                direction=direction
            )
        except Exception as exc:
            raise ServiceError(f"Failed to validate risk limits: {str(exc)}") from exc
    
    @with_circuit_breaker
    @with_retry
    @with_timeout(seconds=30)
    def get_risk_profile(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get the risk profile for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dict containing risk profile
            
        Raises:
            ServiceError: If the service call fails
            NotFoundError: If the strategy is not found
        """
        try:
            return self.service_client.get_risk_profile(strategy_id=strategy_id)
        except Exception as exc:
            raise ServiceError(f"Failed to get risk profile: {str(exc)}") from exc
    
    @with_circuit_breaker
    @with_retry
    @with_timeout(seconds=30)
    def update_risk_parameters(self, strategy_id: str, 
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update risk parameters for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            parameters: Risk parameters to update
            
        Returns:
            Dict containing updated risk profile
            
        Raises:
            ServiceError: If the service call fails
            NotFoundError: If the strategy is not found
        """
        try:
            return self.service_client.update_risk_parameters(
                strategy_id=strategy_id,
                parameters=parameters
            )
        except Exception as exc:
            raise ServiceError(f"Failed to update risk parameters: {str(exc)}") from exc
    
    @with_circuit_breaker
    @with_retry
    @with_timeout(seconds=30)
    def calculate_position_size(self, strategy_id: str, instrument: str, 
                               risk_percentage: float) -> Dict[str, Any]:
        """
        Calculate position size based on risk percentage.
        
        Args:
            strategy_id: ID of the strategy
            instrument: Trading instrument
            risk_percentage: Percentage of account to risk
            
        Returns:
            Dict containing calculated position size
            
        Raises:
            ServiceError: If the service call fails
            NotFoundError: If the strategy is not found
        """
        try:
            return self.service_client.calculate_position_size(
                strategy_id=strategy_id,
                instrument=instrument,
                risk_percentage=risk_percentage
            )
        except Exception as exc:
            raise ServiceError(f"Failed to calculate position size: {str(exc)}") from exc
