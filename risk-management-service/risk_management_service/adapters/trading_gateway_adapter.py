"""
Trading Gateway Service adapters for decoupled service communication.
"""
from typing import Dict, Any, Optional
from common_lib.trading.interfaces import ITradingGateway, OrderRequest, TradingStatus
from common_lib.events.event_bus import EventBus, EventType

class TradingGatewayAdapter(ITradingGateway):
    def __init__(self):
        self.event_bus = EventBus()

    async def submit_order(self, order: OrderRequest) -> Dict[str, Any]:
        # Publish order submission event
        await self.event_bus.publish(
            EventType.ORDER_SUBMITTED,
            {"order": order.__dict__}
        )
        # Implementation would make actual API call to trading gateway
        # but through event-based architecture
        return {"order_id": "test_order_id"}

    async def cancel_order(self, order_id: str) -> bool:
        await self.event_bus.publish(
            EventType.ORDER_CANCELLED,
            {"order_id": order_id}
        )
        return True

    async def get_order_status(self, order_id: str) -> TradingStatus:
        # Implementation would check actual order status
        return TradingStatus.PENDING
