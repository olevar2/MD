"""
Adapters for external service communication.
"""

from market_analysis_service.adapters.data_pipeline_adapter import DataPipelineAdapter
from market_analysis_service.adapters.analysis_coordinator_adapter import AnalysisCoordinatorAdapter
from market_analysis_service.adapters.feature_store_adapter import FeatureStoreAdapter

__all__ = [
    'DataPipelineAdapter',
    'AnalysisCoordinatorAdapter',
    'FeatureStoreAdapter'
]