"""
API Dependencies Module

This module provides dependency functions for the API endpoints.
"""

from typing import Any, Dict, List, Optional, Union

from common_lib.interfaces.feature_store import IFeatureProvider, IFeatureStore, IFeatureGenerator
from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer
from feature_store_service.adapters.adapter_factory import adapter_factory


async def get_feature_provider() -> IFeatureProvider:
    """
    Get a feature provider adapter instance.

    Returns:
        Feature provider adapter instance
    """
    return adapter_factory.get_feature_provider()


async def get_feature_store() -> IFeatureStore:
    """
    Get a feature store adapter instance.

    Returns:
        Feature store adapter instance
    """
    return adapter_factory.get_feature_store()


async def get_feature_generator() -> IFeatureGenerator:
    """
    Get a feature generator adapter instance.

    Returns:
        Feature generator adapter instance
    """
    return adapter_factory.get_feature_generator()


async def get_analysis_provider() -> IAnalysisProvider:
    """
    Get an analysis provider adapter instance.

    Returns:
        Analysis provider adapter instance
    """
    return adapter_factory.get_analysis_provider()


async def get_indicator_provider() -> IIndicatorProvider:
    """
    Get an indicator provider adapter instance.

    Returns:
        Indicator provider adapter instance
    """
    return adapter_factory.get_indicator_provider()


async def get_pattern_recognizer() -> IPatternRecognizer:
    """
    Get a pattern recognizer adapter instance.

    Returns:
        Pattern recognizer adapter instance
    """
    return adapter_factory.get_pattern_recognizer()
