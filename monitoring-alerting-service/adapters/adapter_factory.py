"""
Factory for creating service adapters.
"""

from typing import Dict, Any, Type
from common_lib.config import ConfigManager
from ..interfaces.analysis_engine_interface import AnalysisEngineInterface
from ..adapters.analysis_engine_adapter import AnalysisEngineAdapter
from ..interfaces.analysis_engine_service_interface import AnalysisEngineServiceInterface
from ..adapters.analysis_engine_service_adapter import AnalysisEngineServiceAdapter
from ..interfaces.api_gateway_interface import ApiGatewayInterface
from ..adapters.api_gateway_adapter import ApiGatewayAdapter
from ..interfaces.data_management_service_interface import DataManagementServiceInterface
from ..adapters.data_management_service_adapter import DataManagementServiceAdapter
from ..interfaces.data_pipeline_service_interface import DataPipelineServiceInterface
from ..adapters.data_pipeline_service_adapter import DataPipelineServiceAdapter
from ..interfaces.feature_store_service_interface import FeatureStoreServiceInterface
from ..adapters.feature_store_service_adapter import FeatureStoreServiceAdapter
from ..interfaces.feature_store_service_backup_interface import FeatureStoreServiceBackupInterface
from ..adapters.feature_store_service_backup_adapter import FeatureStoreServiceBackupAdapter
from ..interfaces.ml_integration_service_interface import MlIntegrationServiceInterface
from ..adapters.ml_integration_service_adapter import MlIntegrationServiceAdapter
from ..interfaces.ml_workbench_service_interface import MlWorkbenchServiceInterface
from ..adapters.ml_workbench_service_adapter import MlWorkbenchServiceAdapter
from ..interfaces.model_registry_service_interface import ModelRegistryServiceInterface
from ..adapters.model_registry_service_adapter import ModelRegistryServiceAdapter
from ..interfaces.monitoring_alerting_service_interface import MonitoringAlertingServiceInterface
from ..adapters.monitoring_alerting_service_adapter import MonitoringAlertingServiceAdapter
from ..interfaces.portfolio_management_service_interface import PortfolioManagementServiceInterface
from ..adapters.portfolio_management_service_adapter import PortfolioManagementServiceAdapter
from ..interfaces.risk_management_service_interface import RiskManagementServiceInterface
from ..adapters.risk_management_service_adapter import RiskManagementServiceAdapter
from ..interfaces.strategy_execution_engine_interface import StrategyExecutionEngineInterface
from ..adapters.strategy_execution_engine_adapter import StrategyExecutionEngineAdapter
from ..interfaces.trading_gateway_service_interface import TradingGatewayServiceInterface
from ..adapters.trading_gateway_service_adapter import TradingGatewayServiceAdapter
from ..interfaces.ui_service_interface import UiServiceInterface
from ..adapters.ui_service_adapter import UiServiceAdapter


class AdapterFactory:
    """
    Factory for creating service adapters.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the factory.

        Args:
            config_manager: Configuration manager
        """
        self.config_manager = config_manager
        self.adapters = {}

    def get_analysis_engine_adapter(self) -> AnalysisEngineInterface:
        """
        Get an adapter for the analysis-engine service.

        Returns:
            analysis-engine service adapter
        """
        if "analysis_engine" not in self.adapters:
            config = self.config_manager.get_service_config("analysis-engine")
            self.adapters["analysis_engine"] = AnalysisEngineAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["analysis_engine"]

    def get_analysis_engine_service_adapter(self) -> AnalysisEngineServiceInterface:
        """
        Get an adapter for the analysis-engine-service service.

        Returns:
            analysis-engine-service service adapter
        """
        if "analysis_engine_service" not in self.adapters:
            config = self.config_manager.get_service_config("analysis-engine-service")
            self.adapters["analysis_engine_service"] = AnalysisEngineServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["analysis_engine_service"]

    def get_api_gateway_adapter(self) -> ApiGatewayInterface:
        """
        Get an adapter for the api-gateway service.

        Returns:
            api-gateway service adapter
        """
        if "api_gateway" not in self.adapters:
            config = self.config_manager.get_service_config("api-gateway")
            self.adapters["api_gateway"] = ApiGatewayAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["api_gateway"]

    def get_data_management_service_adapter(self) -> DataManagementServiceInterface:
        """
        Get an adapter for the data-management-service service.

        Returns:
            data-management-service service adapter
        """
        if "data_management_service" not in self.adapters:
            config = self.config_manager.get_service_config("data-management-service")
            self.adapters["data_management_service"] = DataManagementServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["data_management_service"]

    def get_data_pipeline_service_adapter(self) -> DataPipelineServiceInterface:
        """
        Get an adapter for the data-pipeline-service service.

        Returns:
            data-pipeline-service service adapter
        """
        if "data_pipeline_service" not in self.adapters:
            config = self.config_manager.get_service_config("data-pipeline-service")
            self.adapters["data_pipeline_service"] = DataPipelineServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["data_pipeline_service"]

    def get_feature_store_service_adapter(self) -> FeatureStoreServiceInterface:
        """
        Get an adapter for the feature-store-service service.

        Returns:
            feature-store-service service adapter
        """
        if "feature_store_service" not in self.adapters:
            config = self.config_manager.get_service_config("feature-store-service")
            self.adapters["feature_store_service"] = FeatureStoreServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["feature_store_service"]

    def get_feature_store_service_backup_adapter(self) -> FeatureStoreServiceBackupInterface:
        """
        Get an adapter for the feature-store-service-backup service.

        Returns:
            feature-store-service-backup service adapter
        """
        if "feature_store_service_backup" not in self.adapters:
            config = self.config_manager.get_service_config("feature-store-service-backup")
            self.adapters["feature_store_service_backup"] = FeatureStoreServiceBackupAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["feature_store_service_backup"]

    def get_ml_integration_service_adapter(self) -> MlIntegrationServiceInterface:
        """
        Get an adapter for the ml-integration-service service.

        Returns:
            ml-integration-service service adapter
        """
        if "ml_integration_service" not in self.adapters:
            config = self.config_manager.get_service_config("ml-integration-service")
            self.adapters["ml_integration_service"] = MlIntegrationServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["ml_integration_service"]

    def get_ml_workbench_service_adapter(self) -> MlWorkbenchServiceInterface:
        """
        Get an adapter for the ml-workbench-service service.

        Returns:
            ml-workbench-service service adapter
        """
        if "ml_workbench_service" not in self.adapters:
            config = self.config_manager.get_service_config("ml-workbench-service")
            self.adapters["ml_workbench_service"] = MlWorkbenchServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["ml_workbench_service"]

    def get_model_registry_service_adapter(self) -> ModelRegistryServiceInterface:
        """
        Get an adapter for the model-registry-service service.

        Returns:
            model-registry-service service adapter
        """
        if "model_registry_service" not in self.adapters:
            config = self.config_manager.get_service_config("model-registry-service")
            self.adapters["model_registry_service"] = ModelRegistryServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["model_registry_service"]

    def get_monitoring_alerting_service_adapter(self) -> MonitoringAlertingServiceInterface:
        """
        Get an adapter for the monitoring-alerting-service service.

        Returns:
            monitoring-alerting-service service adapter
        """
        if "monitoring_alerting_service" not in self.adapters:
            config = self.config_manager.get_service_config("monitoring-alerting-service")
            self.adapters["monitoring_alerting_service"] = MonitoringAlertingServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["monitoring_alerting_service"]

    def get_portfolio_management_service_adapter(self) -> PortfolioManagementServiceInterface:
        """
        Get an adapter for the portfolio-management-service service.

        Returns:
            portfolio-management-service service adapter
        """
        if "portfolio_management_service" not in self.adapters:
            config = self.config_manager.get_service_config("portfolio-management-service")
            self.adapters["portfolio_management_service"] = PortfolioManagementServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["portfolio_management_service"]

    def get_risk_management_service_adapter(self) -> RiskManagementServiceInterface:
        """
        Get an adapter for the risk-management-service service.

        Returns:
            risk-management-service service adapter
        """
        if "risk_management_service" not in self.adapters:
            config = self.config_manager.get_service_config("risk-management-service")
            self.adapters["risk_management_service"] = RiskManagementServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["risk_management_service"]

    def get_strategy_execution_engine_adapter(self) -> StrategyExecutionEngineInterface:
        """
        Get an adapter for the strategy-execution-engine service.

        Returns:
            strategy-execution-engine service adapter
        """
        if "strategy_execution_engine" not in self.adapters:
            config = self.config_manager.get_service_config("strategy-execution-engine")
            self.adapters["strategy_execution_engine"] = StrategyExecutionEngineAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["strategy_execution_engine"]

    def get_trading_gateway_service_adapter(self) -> TradingGatewayServiceInterface:
        """
        Get an adapter for the trading-gateway-service service.

        Returns:
            trading-gateway-service service adapter
        """
        if "trading_gateway_service" not in self.adapters:
            config = self.config_manager.get_service_config("trading-gateway-service")
            self.adapters["trading_gateway_service"] = TradingGatewayServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["trading_gateway_service"]

    def get_ui_service_adapter(self) -> UiServiceInterface:
        """
        Get an adapter for the ui-service service.

        Returns:
            ui-service service adapter
        """
        if "ui_service" not in self.adapters:
            config = self.config_manager.get_service_config("ui-service")
            self.adapters["ui_service"] = UiServiceAdapter(
                base_url=config["base_url"],
                timeout=config.get("timeout", 30)
            )

        return self.adapters["ui_service"]

