"""
Service Dependencies for Analysis Engine Service.

This module configures the dependencies on other services for the Analysis Engine Service.
It uses the common adapter factory to create adapters for the services that the Analysis Engine
Service depends on, using the common interfaces defined in common-lib.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import Depends
from typing_extensions import Annotated

from common_lib.interfaces.trading_gateway import ITradingGateway
from common_lib.interfaces.ml_integration import IMLModelRegistry, IMLMetricsProvider
from common_lib.interfaces.ml_workbench import IExperimentManager, IModelEvaluator, IDatasetManager
from common_lib.interfaces.risk_management import IRiskManager
from common_lib.interfaces.feature_store import IFeatureProvider, IFeatureStore, IFeatureGenerator

from analysis_engine.adapters.common_adapter_factory import get_common_adapter_factory


logger = logging.getLogger(__name__)


class ServiceDependencies:
    """Service dependencies for the Analysis Engine Service."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize service dependencies.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.adapter_factory = get_common_adapter_factory()
        self.logger = logger
    
    async def get_trading_gateway(self) -> ITradingGateway:
        """
        Get the Trading Gateway adapter.
        
        Returns:
            Trading Gateway adapter
        """
        return self.adapter_factory.get_adapter(ITradingGateway)
    
    async def get_ml_model_registry(self) -> IMLModelRegistry:
        """
        Get the ML Model Registry adapter.
        
        Returns:
            ML Model Registry adapter
        """
        return self.adapter_factory.get_adapter(IMLModelRegistry)
    
    async def get_ml_metrics_provider(self) -> IMLMetricsProvider:
        """
        Get the ML Metrics Provider adapter.
        
        Returns:
            ML Metrics Provider adapter
        """
        return self.adapter_factory.get_adapter(IMLMetricsProvider)
    
    async def get_experiment_manager(self) -> IExperimentManager:
        """
        Get the Experiment Manager adapter.
        
        Returns:
            Experiment Manager adapter
        """
        return self.adapter_factory.get_adapter(IExperimentManager)
    
    async def get_model_evaluator(self) -> IModelEvaluator:
        """
        Get the Model Evaluator adapter.
        
        Returns:
            Model Evaluator adapter
        """
        return self.adapter_factory.get_adapter(IModelEvaluator)
    
    async def get_dataset_manager(self) -> IDatasetManager:
        """
        Get the Dataset Manager adapter.
        
        Returns:
            Dataset Manager adapter
        """
        return self.adapter_factory.get_adapter(IDatasetManager)
    
    async def get_risk_manager(self) -> IRiskManager:
        """
        Get the Risk Manager adapter.
        
        Returns:
            Risk Manager adapter
        """
        return self.adapter_factory.get_adapter(IRiskManager)
    
    async def get_feature_provider(self) -> IFeatureProvider:
        """
        Get the Feature Provider adapter.
        
        Returns:
            Feature Provider adapter
        """
        return self.adapter_factory.get_adapter(IFeatureProvider)
    
    async def get_feature_store(self) -> IFeatureStore:
        """
        Get the Feature Store adapter.
        
        Returns:
            Feature Store adapter
        """
        return self.adapter_factory.get_adapter(IFeatureStore)
    
    async def get_feature_generator(self) -> IFeatureGenerator:
        """
        Get the Feature Generator adapter.
        
        Returns:
            Feature Generator adapter
        """
        return self.adapter_factory.get_adapter(IFeatureGenerator)


# Singleton instance of service dependencies
service_dependencies = ServiceDependencies()


# FastAPI dependency injection
async def get_trading_gateway() -> ITradingGateway:
    """FastAPI dependency for Trading Gateway."""
    return await service_dependencies.get_trading_gateway()


async def get_ml_model_registry() -> IMLModelRegistry:
    """FastAPI dependency for ML Model Registry."""
    return await service_dependencies.get_ml_model_registry()


async def get_ml_metrics_provider() -> IMLMetricsProvider:
    """FastAPI dependency for ML Metrics Provider."""
    return await service_dependencies.get_ml_metrics_provider()


async def get_experiment_manager() -> IExperimentManager:
    """FastAPI dependency for Experiment Manager."""
    return await service_dependencies.get_experiment_manager()


async def get_model_evaluator() -> IModelEvaluator:
    """FastAPI dependency for Model Evaluator."""
    return await service_dependencies.get_model_evaluator()


async def get_dataset_manager() -> IDatasetManager:
    """FastAPI dependency for Dataset Manager."""
    return await service_dependencies.get_dataset_manager()


async def get_risk_manager() -> IRiskManager:
    """FastAPI dependency for Risk Manager."""
    return await service_dependencies.get_risk_manager()


async def get_feature_provider() -> IFeatureProvider:
    """FastAPI dependency for Feature Provider."""
    return await service_dependencies.get_feature_provider()


async def get_feature_store() -> IFeatureStore:
    """FastAPI dependency for Feature Store."""
    return await service_dependencies.get_feature_store()


async def get_feature_generator() -> IFeatureGenerator:
    """FastAPI dependency for Feature Generator."""
    return await service_dependencies.get_feature_generator()


# Type annotations for FastAPI dependency injection
TradingGatewayDep = Annotated[ITradingGateway, Depends(get_trading_gateway)]
MLModelRegistryDep = Annotated[IMLModelRegistry, Depends(get_ml_model_registry)]
MLMetricsProviderDep = Annotated[IMLMetricsProvider, Depends(get_ml_metrics_provider)]
ExperimentManagerDep = Annotated[IExperimentManager, Depends(get_experiment_manager)]
ModelEvaluatorDep = Annotated[IModelEvaluator, Depends(get_model_evaluator)]
DatasetManagerDep = Annotated[IDatasetManager, Depends(get_dataset_manager)]
RiskManagerDep = Annotated[IRiskManager, Depends(get_risk_manager)]
FeatureProviderDep = Annotated[IFeatureProvider, Depends(get_feature_provider)]
FeatureStoreDep = Annotated[IFeatureStore, Depends(get_feature_store)]
FeatureGeneratorDep = Annotated[IFeatureGenerator, Depends(get_feature_generator)]
