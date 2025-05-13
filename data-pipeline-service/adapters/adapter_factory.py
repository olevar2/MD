#!/usr/bin/env python3
"""
Adapter factory for data service.
"""

from typing import Dict, List, Optional, Any

from .marketdataprovideradapter import MarketDataProviderAdapter
from .marketdatacacheadapter import MarketDataCacheAdapter

class AdapterFactory:
    """
    Factory for creating adapter instances.
    """

    @classmethod
    def get_marketdataprovideradapter(cls) -> MarketDataProviderAdapter:
        """Get an instance of MarketDataProviderAdapter."""
        return MarketDataProviderAdapter()
    @classmethod
    def get_marketdatacacheadapter(cls) -> MarketDataCacheAdapter:
        """Get an instance of MarketDataCacheAdapter."""
        return MarketDataCacheAdapter()
