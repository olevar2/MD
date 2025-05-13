#!/usr/bin/env python3
"""
Adapter factory for ml service.
"""

from typing import Dict, List, Optional, Any

from .featureprovideradapter import FeatureProviderAdapter

class AdapterFactory:
    """
    Factory for creating adapter instances.
    """

    @classmethod
    def get_featureprovideradapter(cls) -> FeatureProviderAdapter:
        """Get an instance of FeatureProviderAdapter."""
        return FeatureProviderAdapter()
