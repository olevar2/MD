"""
Risk Management Service adapters for decoupled service communication.
"""
from typing import Dict, Any, Optional
from common_lib.trading.interfaces import IRiskManager, OrderRequest
from common_lib.events.event_bus import EventBus, EventType

class RiskManagerAdapter(IRiskManager):
    def __init__(self):
        self.event_bus = EventBus()

    async def validate_order(self, order: OrderRequest) -> Dict[str, Any]:
        validation_result = {"is_valid": True, "message": "Order validated"}
        await self.event_bus.publish(
            EventType.RISK_VALIDATION,
            {
                "order": order.__dict__,
                "validation": validation_result
            }
        )
        return validation_result

    async def get_position_risk(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "var": 0.0,
            "expected_shortfall": 0.0
        }

    async def get_portfolio_risk(self) -> Dict[str, Any]:
        return {
            "total_var": 0.0,
            "total_exposure": 0.0
        }
