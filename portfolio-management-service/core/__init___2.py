"""
Adapters package initialization.

Exports adapter components for use across the portfolio management service.
"""
from adapters.multi_asset_adapter import MultiAssetServiceAdapter

__all__ = [
    'MultiAssetServiceAdapter'
]
