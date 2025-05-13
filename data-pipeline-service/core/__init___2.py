"""
Adapters package initialization.

Exports adapter components for use across the data pipeline service.
"""
from adapters.multi_asset_adapter import (
    MultiAssetServiceAdapter,
    AssetRegistryAdapter
)
from adapters.market_data_adapter import (
    MarketDataProviderAdapter,
    MarketDataCacheAdapter
)
from adapters.analysis_engine_adapter import AnalysisEngineAdapter
from adapters.adapter_factory_1 import adapter_factory

__all__ = [
    'MultiAssetServiceAdapter',
    'AssetRegistryAdapter',
    'MarketDataProviderAdapter',
    'MarketDataCacheAdapter',
    'AnalysisEngineAdapter',
    'adapter_factory'
]
