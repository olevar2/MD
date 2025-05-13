"""
Event bus implementation for inter-service communication.
"""
import asyncio
from typing import Dict, Any, Callable, List, Optional
from enum import Enum

class EventType(Enum):
    ORDER_SUBMITTED = "order.submitted"
    ORDER_EXECUTED = "order.executed"
    ORDER_CANCELLED = "order.cancelled"
    RISK_VALIDATION = "risk.validation"
    POSITION_UPDATE = "position.update"
    MARKET_REGIME_CHANGE = "market.regime.change"
    ANALYSIS_SIGNAL = "analysis.signal"

class EventBus:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._handlers = {}
            cls._instance._loop = asyncio.get_event_loop()
        return cls._instance

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def publish(self, event_type: EventType, data: Dict[str, Any]) -> None:
        if event_type in self._handlers:
            tasks = [
                self._loop.create_task(handler(data))
                for handler in self._handlers[event_type]
            ]
            await asyncio.gather(*tasks)

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)
