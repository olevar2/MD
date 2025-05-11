"""
Adapters package initialization.

Exports adapter components for use across the data pipeline service.
"""
from data_pipeline_service.adapters.multi_asset_adapter import (
    MultiAssetServiceAdapter,
    AssetRegistryAdapter
)
from data_pipeline_service.adapters.market_data_adapter import (
    MarketDataProviderAdapter,
    MarketDataCacheAdapter
)
from data_pipeline_service.adapters.analysis_engine_adapter import AnalysisEngineAdapter
from data_pipeline_service.adapters.adapter_factory import adapter_factory

__all__ = [
    'MultiAssetServiceAdapter',
    'AssetRegistryAdapter',
    'MarketDataProviderAdapter',
    'MarketDataCacheAdapter',
    'AnalysisEngineAdapter',
    'adapter_factory'
]
