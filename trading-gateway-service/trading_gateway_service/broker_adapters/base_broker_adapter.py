"""
Base implementation for broker adapters with common functionality.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import aiohttp

# Import the centralized retry policy
from common_lib.resilience import retry_with_policy, register_common_retryable_exceptions

from ..interfaces.broker_adapter import (
    BrokerAdapter,
    OrderRequest,
    ExecutionReport,
    PositionUpdate,
    AccountUpdate,
    OrderStatus,
)

logger = logging.getLogger(__name__)

# Register common HTTP client errors for retry
# aiohttp exceptions inherit from ClientError
register_common_retryable_exceptions([aiohttp.ClientError])


class BaseBrokerAdapter(BrokerAdapter):
    """
    Base implementation of the BrokerAdapter interface with common functionality.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base broker adapter.
        
        Args:
            config: Configuration dictionary containing:
                - api_url: Base URL for the broker's API
                - api_key: API key for authentication
                - api_secret: API secret for authentication
                - connection_timeout: Connection timeout in seconds
                - max_retries: Maximum number of retry attempts
                - retry_delay: Initial delay between retries in seconds
        """
        self.config = config
        self.api_url = config["api_url"]
        self.api_key = config["api_key"]
        self.api_secret = config["api_secret"]
        self.session: Optional[aiohttp.ClientSession] = None
        self._is_connected = False
        
        # Local order queue for handling API outages
        self._order_queue: List[OrderRequest] = []
        
        # Heartbeat monitoring
        self._last_heartbeat: Optional[datetime] = None
        self._heartbeat_interval = config.get("heartbeat_interval", 30)
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Real-time update callbacks
        self._execution_callback: Optional[Callable] = None
        self._position_callback: Optional[Callable] = None
        self._account_callback: Optional[Callable] = None

    async def _init_session(self) -> None:
        """Initialize the HTTP session with proper headers and authentication."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                base_url=self.api_url,
                headers=self._get_auth_headers()
            )

    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests. Must be implemented by subclasses."""
        pass

    @retry_with_policy(
        max_attempts=3,
        base_delay=4.0,  # Corresponds to tenacity's 'min' in wait_exponential
        max_delay=10.0,  # Corresponds to tenacity's 'max'
        exceptions=[aiohttp.ClientError]  # Specify exceptions to retry
    )
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Any:
        """
        Make an HTTP request to the broker's API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments for the request
            
        Returns:
            Parsed JSON response
            
        Raises:
            core_foundations.resilience.retry_policy.RetryExhaustedException: If all retries fail.
            aiohttp.ClientResponseError: For non-retryable HTTP errors (e.g., 4xx).
            Exception: For other unexpected errors.
        """
        await self._init_session()
        async with self.session.request(method, endpoint, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def _start_heartbeat(self) -> None:
        """Start the heartbeat monitoring task."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

    async def _heartbeat_monitor(self) -> None:
        """Monitor connection health with periodic heartbeats."""
        while self._is_connected:
            try:
                # Implement specific heartbeat logic in subclasses
                await self._send_heartbeat()
                self._last_heartbeat = datetime.utcnow()
                await asyncio.sleep(self._heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat failed: {str(e)}")
                await self._handle_connection_loss()

    async def _handle_connection_loss(self) -> None:
        """Handle connection loss with reconnection logic."""
        self._is_connected = False
        logger.warning("Connection lost, attempting to reconnect...")
        
        # Close existing session
        if self.session and not self.session.closed:
            await self.session.close()
        
        # Attempt reconnection with exponential backoff
        retry_count = 0
        max_retries = self.config.get("max_retries", 5)
        base_delay = self.config.get("retry_delay", 1)
        
        while retry_count < max_retries and not self._is_connected:
            try:
                delay = base_delay * (2 ** retry_count)
                await asyncio.sleep(delay)
                await self.connect()
                if self._is_connected:
                    logger.info("Successfully reconnected")
                    await self._reconcile_positions()
                    await self._process_queued_orders()
                    return
            except Exception as e:
                logger.error(f"Reconnection attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
        
        logger.error("Failed to reconnect after maximum retries")

    async def _reconcile_positions(self) -> None:
        """
        Reconcile local position state with broker position state after reconnection.
        Subclasses should implement specific reconciliation logic.
        """
        try:
            broker_positions = await self.get_positions()
            # Implement position reconciliation logic
            logger.info(f"Position reconciliation completed. Found {len(broker_positions)} positions.")
        except Exception as e:
            logger.error(f"Position reconciliation failed: {str(e)}")

    async def _process_queued_orders(self) -> None:
        """Process any orders that were queued during the connection loss."""
        while self._order_queue:
            order = self._order_queue.pop(0)
            try:
                await self.place_order(order)
            except Exception as e:
                logger.error(f"Failed to process queued order: {str(e)}")
                # Re-queue the order if it's still valid
                if self._is_order_still_valid(order):
                    self._order_queue.append(order)

    def _is_order_still_valid(self, order: OrderRequest) -> bool:
        """Check if a queued order is still valid to execute."""
        if order.good_till_date and order.good_till_date < datetime.utcnow():
            return False
        return True

    @abstractmethod
    async def _send_heartbeat(self) -> None:
        """Send a heartbeat message to the broker. Must be implemented by subclasses."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if the adapter is currently connected."""
        return self._is_connected

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
