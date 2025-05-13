#!/usr/bin/env python3
"""
Adapter factory for feature service.
"""

from typing import Dict, List, Optional, Any

from .featureprovideradapter import FeatureProviderAdapter
from .featurestoreadapter import FeatureStoreAdapter
from .featuregeneratoradapter import FeatureGeneratorAdapter

class AdapterFactory:
    """
    Factory for creating adapter instances.
    """

    @classmethod
    def get_featureprovideradapter(cls) -> FeatureProviderAdapter:
        """Get an instance of FeatureProviderAdapter."""
        return FeatureProviderAdapter()
    @classmethod
    def get_featurestoreadapter(cls) -> FeatureStoreAdapter:
        """Get an instance of FeatureStoreAdapter."""
        return FeatureStoreAdapter()
    @classmethod
    def get_featuregeneratoradapter(cls) -> FeatureGeneratorAdapter:
        """Get an instance of FeatureGeneratorAdapter."""
        return FeatureGeneratorAdapter()
