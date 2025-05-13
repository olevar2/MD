#!/usr/bin/env python3
"""
Adapter factory for trading service.
"""

from typing import Dict, List, Optional, Any

from .riskmanageradapter import RiskManagerAdapter
from .riskmanageradapter import RiskManagerAdapter
from .orderbookprovideradapter import OrderBookProviderAdapter
from .tradingprovideradapter import TradingProviderAdapter

class AdapterFactory:
    """
    Factory for creating adapter instances.
    """

    @classmethod
    def get_riskmanageradapter(cls) -> RiskManagerAdapter:
        """Get an instance of RiskManagerAdapter."""
        return RiskManagerAdapter()
    @classmethod
    def get_riskmanageradapter(cls) -> RiskManagerAdapter:
        """Get an instance of RiskManagerAdapter."""
        return RiskManagerAdapter()
    @classmethod
    def get_orderbookprovideradapter(cls) -> OrderBookProviderAdapter:
        """Get an instance of OrderBookProviderAdapter."""
        return OrderBookProviderAdapter()
    @classmethod
    def get_tradingprovideradapter(cls) -> TradingProviderAdapter:
        """Get an instance of TradingProviderAdapter."""
        return TradingProviderAdapter()
