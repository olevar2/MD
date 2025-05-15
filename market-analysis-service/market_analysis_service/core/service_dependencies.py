"""
Service Dependencies for Market Analysis Service.

This module provides functions for creating and retrieving service dependencies.
"""
import logging
from functools import lru_cache
from typing import Dict, List, Any, Optional

from market_analysis_service.config import settings
from market_analysis_service.adapters.data_pipeline_adapter import DataPipelineAdapter
from market_analysis_service.adapters.analysis_coordinator_adapter import AnalysisCoordinatorAdapter
from market_analysis_service.adapters.feature_store_adapter import FeatureStoreAdapter
from market_analysis_service.repositories.analysis_repository import AnalysisRepository
from market_analysis_service.services.market_analysis_service import MarketAnalysisService
from market_analysis_service.repositories.read_repositories import AnalysisReadRepository
from market_analysis_service.repositories.write_repositories import AnalysisWriteRepository

logger = logging.getLogger(__name__)

@lru_cache
def get_data_pipeline_adapter() -> DataPipelineAdapter:
    """
    Get the Data Pipeline Adapter.
    
    Returns:
        Data Pipeline Adapter
    """
    logger.info("Creating Data Pipeline Adapter")
    return DataPipelineAdapter(base_url=settings.DATA_PIPELINE_SERVICE_URL)

@lru_cache
def get_analysis_coordinator_adapter() -> AnalysisCoordinatorAdapter:
    """
    Get the Analysis Coordinator Adapter.
    
    Returns:
        Analysis Coordinator Adapter
    """
    logger.info("Creating Analysis Coordinator Adapter")
    return AnalysisCoordinatorAdapter(base_url=settings.ANALYSIS_COORDINATOR_SERVICE_URL)

@lru_cache
def get_feature_store_adapter() -> FeatureStoreAdapter:
    """
    Get the Feature Store Adapter.
    
    Returns:
        Feature Store Adapter
    """
    logger.info("Creating Feature Store Adapter")
    return FeatureStoreAdapter(base_url=settings.FEATURE_STORE_SERVICE_URL)

@lru_cache
def get_analysis_repository() -> AnalysisRepository:
    """
    Get the Analysis Repository.
    
    Returns:
        Analysis Repository
    """
    logger.info("Creating Analysis Repository")
    return AnalysisRepository(data_dir=settings.DATA_DIR)

@lru_cache
def get_analysis_read_repository() -> AnalysisReadRepository:
    """
    Get the Analysis Read Repository.
    
    Returns:
        Analysis Read Repository
    """
    logger.info("Creating Analysis Read Repository")
    return AnalysisReadRepository(data_dir=settings.DATA_DIR)

@lru_cache
def get_analysis_write_repository() -> AnalysisWriteRepository:
    """
    Get the Analysis Write Repository.
    
    Returns:
        Analysis Write Repository
    """
    logger.info("Creating Analysis Write Repository")
    return AnalysisWriteRepository(data_dir=settings.DATA_DIR)

@lru_cache
def get_market_analysis_service() -> MarketAnalysisService:
    """
    Get the Market Analysis Service.
    
    Returns:
        Market Analysis Service
    """
    logger.info("Creating Market Analysis Service")
    return MarketAnalysisService(
        data_pipeline_adapter=get_data_pipeline_adapter(),
        analysis_coordinator_adapter=get_analysis_coordinator_adapter(),
        feature_store_adapter=get_feature_store_adapter(),
        analysis_repository=get_analysis_repository()
    )