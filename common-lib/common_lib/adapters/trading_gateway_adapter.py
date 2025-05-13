"""
Trading Gateway Service Adapter.

This module provides adapter implementations for the Trading Gateway Service interfaces.
These adapters allow other services to interact with the Trading Gateway Service
without direct dependencies, breaking circular dependencies.
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast
from datetime import datetime

from common_lib.interfaces.trading_gateway import (
    ITradingGateway, IOrderManagement, IPositionManagement, IAccountManagement,
    IMarketDataProvider, ISimulationProvider
)
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.service_client.http_client import AsyncHTTPServiceClient
from common_lib.errors.base_exceptions import (
    ServiceUnavailableError, ResourceNotFoundError, ValidationError
)
from common_lib.models.trading import (
    Order, Position, Account, MarketData, OrderStatus, OrderType, ActionType
)

logger = logging.getLogger(__name__)


class TradingGatewayAdapter(ITradingGateway):
    """Adapter for Trading Gateway operations."""
    
    def __init__(self, client: Optional[AsyncHTTPServiceClient] = None, config: Optional[ServiceClientConfig] = None):
        """
        Initialize the adapter.
        
        Args:
            client: Optional pre-configured HTTP client
            config: Optional client configuration
        """
        self.client = client or AsyncHTTPServiceClient(
            config or ServiceClientConfig(
                base_url="http://trading-gateway-service:8000/api/v1",
                timeout=30
            )
        )
        
        # Initialize sub-adapters
        self.order_management = OrderManagementAdapter(self.client)
        self.position_management = PositionManagementAdapter(self.client)
        self.account_management = AccountManagementAdapter(self.client)
        self.market_data = MarketDataProviderAdapter(self.client)
        self.simulation = SimulationProviderAdapter(self.client)
    
    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get the status of the Trading Gateway Service.
        
        Returns:
            Dictionary containing service status information
        """
        try:
            response = await self.client.get("/status")
            return response
        except Exception as e:
            logger.error(f"Error getting service status: {str(e)}")
            raise ServiceUnavailableError(f"Trading Gateway Service unavailable: {str(e)}")


class OrderManagementAdapter(IOrderManagement):
    """Adapter for Order Management operations."""
    
    def __init__(self, client: AsyncHTTPServiceClient):
        """
        Initialize the adapter.
        
        Args:
            client: HTTP client for API calls
        """
        self.client = client
    
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            order: The order to place
            
        Returns:
            Dictionary containing the created order information
        """
        try:
            payload = order.dict()
            response = await self.client.post("/orders", json=payload)
            return response
        except ValidationError as e:
            logger.error(f"Validation error when placing order: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: The ID of the order to cancel
            
        Returns:
            True if the cancellation was successful, False otherwise
        """
        try:
            response = await self.client.delete(f"/orders/{order_id}")
            return response.get("success", False)
        except ResourceNotFoundError:
            logger.error(f"Order with ID {order_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {str(e)}")
            raise
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get information about a specific order.
        
        Args:
            order_id: The ID of the order
            
        Returns:
            Dictionary containing order information
        """
        try:
            response = await self.client.get(f"/orders/{order_id}")
            return response
        except ResourceNotFoundError:
            logger.error(f"Order with ID {order_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {str(e)}")
            raise
    
    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get orders with optional filtering.
        
        Args:
            symbol: Filter by symbol
            status: Filter by order status
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of dictionaries containing order information
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        if status:
            params["status"] = status.value if isinstance(status, OrderStatus) else status
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        try:
            response = await self.client.get("/orders", params=params)
            return response.get("orders", [])
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            raise


class PositionManagementAdapter(IPositionManagement):
    """Adapter for Position Management operations."""
    
    def __init__(self, client: AsyncHTTPServiceClient):
        """
        Initialize the adapter.
        
        Args:
            client: HTTP client for API calls
        """
        self.client = client
    
    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current position for a symbol.
        
        Args:
            symbol: The symbol to get the position for
            
        Returns:
            Dictionary containing position information
        """
        try:
            response = await self.client.get(f"/positions/{symbol}")
            return response
        except ResourceNotFoundError:
            logger.error(f"Position for symbol {symbol} not found")
            raise
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            raise
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current positions.
        
        Returns:
            List of dictionaries containing position information
        """
        try:
            response = await self.client.get("/positions")
            return response.get("positions", [])
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise
    
    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        Close a position for a symbol.
        
        Args:
            symbol: The symbol to close the position for
            
        Returns:
            Dictionary containing the result of the operation
        """
        try:
            response = await self.client.post(f"/positions/{symbol}/close")
            return response
        except ResourceNotFoundError:
            logger.error(f"Position for symbol {symbol} not found")
            raise
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            raise


class AccountManagementAdapter(IAccountManagement):
    """Adapter for Account Management operations."""
    
    def __init__(self, client: AsyncHTTPServiceClient):
        """
        Initialize the adapter.
        
        Args:
            client: HTTP client for API calls
        """
        self.client = client
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get information about the trading account.
        
        Returns:
            Dictionary containing account information
        """
        try:
            response = await self.client.get("/account")
            return response
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            raise
    
    async def get_account_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get account history with optional time filtering.
        
        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of dictionaries containing account history entries
        """
        params = {}
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        try:
            response = await self.client.get("/account/history", params=params)
            return response.get("history", [])
        except Exception as e:
            logger.error(f"Error getting account history: {str(e)}")
            raise


class MarketDataProviderAdapter(IMarketDataProvider):
    """Adapter for Market Data Provider operations."""
    
    def __init__(self, client: AsyncHTTPServiceClient):
        """
        Initialize the adapter.
        
        Args:
            client: HTTP client for API calls
        """
        self.client = client
    
    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: The symbol to get the price for
            
        Returns:
            Dictionary containing price information
        """
        try:
            response = await self.client.get(f"/market-data/{symbol}/price")
            return response
        except ResourceNotFoundError:
            logger.error(f"Price data for symbol {symbol} not found")
            raise
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            raise
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get historical market data.
        
        Args:
            symbol: The symbol to get data for
            timeframe: The timeframe of the data (e.g., "1m", "1h", "1d")
            start_time: Start time for the data
            end_time: End time for the data
            
        Returns:
            List of dictionaries containing historical market data
        """
        params = {
            "timeframe": timeframe,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
        
        try:
            response = await self.client.get(f"/market-data/{symbol}/history", params=params)
            return response.get("data", [])
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            raise


class SimulationProviderAdapter(ISimulationProvider):
    """Adapter for Simulation Provider operations."""
    
    def __init__(self, client: AsyncHTTPServiceClient):
        """
        Initialize the adapter.
        
        Args:
            client: HTTP client for API calls
        """
        self.client = client
    
    async def create_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new simulation.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Dictionary containing the created simulation information
        """
        try:
            response = await self.client.post("/simulations", json=config)
            return response
        except ValidationError:
            logger.error("Validation error when creating simulation")
            raise
        except Exception as e:
            logger.error(f"Error creating simulation: {str(e)}")
            raise
    
    async def get_simulation(self, simulation_id: str) -> Dict[str, Any]:
        """
        Get information about a specific simulation.
        
        Args:
            simulation_id: The ID of the simulation
            
        Returns:
            Dictionary containing simulation information
        """
        try:
            response = await self.client.get(f"/simulations/{simulation_id}")
            return response
        except ResourceNotFoundError:
            logger.error(f"Simulation with ID {simulation_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error getting simulation {simulation_id}: {str(e)}")
            raise
    
    async def run_simulation(self, simulation_id: str) -> Dict[str, Any]:
        """
        Run a simulation.
        
        Args:
            simulation_id: The ID of the simulation to run
            
        Returns:
            Dictionary containing the simulation run information
        """
        try:
            response = await self.client.post(f"/simulations/{simulation_id}/run")
            return response
        except ResourceNotFoundError:
            logger.error(f"Simulation with ID {simulation_id} not found")
            raise
        except Exception as e:
            logger.error(f"Error running simulation {simulation_id}: {str(e)}")
            raise
