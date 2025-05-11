"""
API Dependencies Module

This module provides dependency functions for the API endpoints.
"""

from typing import Any, Dict, List, Optional, Union

from common_lib.interfaces.trading import ITradingProvider, IOrderBookProvider
from common_lib.interfaces.risk_management import IRiskManager
from trading_gateway_service.adapters.adapter_factory import adapter_factory


async def get_trading_provider() -> ITradingProvider:
    """
    Get a trading provider adapter instance.

    Returns:
        Trading provider adapter instance
    """
    return adapter_factory.get_trading_provider()


async def get_order_book_provider() -> IOrderBookProvider:
    """
    Get an order book provider adapter instance.

    Returns:
        Order book provider adapter instance
    """
    return adapter_factory.get_order_book_provider()


async def get_risk_manager() -> IRiskManager:
    """
    Get a risk manager adapter instance.

    Returns:
        Risk manager adapter instance
    """
    return adapter_factory.get_risk_manager()
