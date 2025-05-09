"""
Adapters package initialization.

Exports adapter components for use across the portfolio management service.
"""
from portfolio_management_service.adapters.multi_asset_adapter import MultiAssetServiceAdapter

__all__ = [
    'MultiAssetServiceAdapter'
]
