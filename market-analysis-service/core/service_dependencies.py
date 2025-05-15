from typing import Optional
from functools import lru_cache

from market_analysis_service.services.market_analysis_service import MarketAnalysisService
from market_analysis_service.repositories.analysis_repository import AnalysisRepository
from market_analysis_service.adapters.data_pipeline_adapter import DataPipelineAdapter
from market_analysis_service.adapters.analysis_coordinator_adapter import AnalysisCoordinatorAdapter
from market_analysis_service.adapters.feature_store_adapter import FeatureStoreAdapter

@lru_cache()
def get_data_pipeline_adapter() -> DataPipelineAdapter:
    """
    Get the data pipeline adapter with dependency injection.
    """
    return DataPipelineAdapter()

@lru_cache()
def get_analysis_coordinator_adapter() -> AnalysisCoordinatorAdapter:
    """
    Get the analysis coordinator adapter with dependency injection.
    """
    return AnalysisCoordinatorAdapter()

@lru_cache()
def get_feature_store_adapter() -> FeatureStoreAdapter:
    """
    Get the feature store adapter with dependency injection.
    """
    return FeatureStoreAdapter()

@lru_cache()
def get_analysis_repository() -> AnalysisRepository:
    """
    Get the analysis repository with dependency injection.
    """
    return AnalysisRepository()

@lru_cache()
def get_market_analysis_service() -> MarketAnalysisService:
    """
    Get the market analysis service with dependency injection.
    """
    data_pipeline_adapter = get_data_pipeline_adapter()
    analysis_coordinator_adapter = get_analysis_coordinator_adapter()
    feature_store_adapter = get_feature_store_adapter()
    analysis_repository = get_analysis_repository()
    
    return MarketAnalysisService(
        data_pipeline_adapter=data_pipeline_adapter,
        analysis_coordinator_adapter=analysis_coordinator_adapter,
        feature_store_adapter=feature_store_adapter,
        analysis_repository=analysis_repository
    )