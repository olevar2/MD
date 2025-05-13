"""
Trading Gateway Service adapters for decoupled service communication.
"""
from typing import Dict, Any, Optional
from common_lib.trading.interfaces import ITradingGateway, OrderRequest, TradingStatus
from common_lib.events.event_bus import EventBus, EventType

class TradingGatewayAdapter(ITradingGateway):
    """
    TradingGatewayAdapter class that inherits from ITradingGateway.
    
    Attributes:
        Add attributes here
    """

    def __init__(self):
    """
      init  .
    
    """

        self.event_bus = EventBus()

    async def submit_order(self, order: OrderRequest) -> Dict[str, Any]:
    """
    Submit order.
    
    Args:
        order: Description of order
    
    Returns:
        Dict[str, Any]: Description of return value
    
    """

        # Publish order submission event
        await self.event_bus.publish(
            EventType.ORDER_SUBMITTED,
            {"order": order.__dict__}
        )
        # Implementation would make actual API call to trading gateway
        # but through event-based architecture
        return {"order_id": "test_order_id"}

    async def cancel_order(self, order_id: str) -> bool:
    """
    Cancel order.
    
    Args:
        order_id: Description of order_id
    
    Returns:
        bool: Description of return value
    
    """

        await self.event_bus.publish(
            EventType.ORDER_CANCELLED,
            {"order_id": order_id}
        )
        return True

    async def get_order_status(self, order_id: str) -> TradingStatus:
    """
    Get order status.
    
    Args:
        order_id: Description of order_id
    
    Returns:
        TradingStatus: Description of return value
    
    """

        # Implementation would check actual order status
        return TradingStatus.PENDING
