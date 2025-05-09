"""
Adapters package initialization.

Exports adapter components for use across the data pipeline service.
"""
from data_pipeline_service.adapters.multi_asset_adapter import (
    MultiAssetServiceAdapter,
    AssetRegistryAdapter
)

__all__ = [
    'MultiAssetServiceAdapter',
    'AssetRegistryAdapter'
]
