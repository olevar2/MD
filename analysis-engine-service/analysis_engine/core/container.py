"""
Service Container Module

This module provides a container for managing services and analyzers in the Analysis Engine Service.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable

from analysis_engine.core.errors import ServiceContainerError
from analysis_engine.monitoring.health_checks import HealthCheck, ComponentHealth, DependencyHealth, HealthStatus
from analysis_engine.adapters.adapter_factory import adapter_factory

class ServiceContainer:
    """Service container for managing services and analyzers."""

    def __init__(self):
        """Initialize the service container."""
        self._services: Dict[str, Any] = {}
        self._analyzers: Dict[str, Any] = {}
        self._initialized = False
        self._logger = logging.getLogger(__name__)
        self._health_check = None

    def register_service(self, name: str, service: Any) -> None:
        """
        Register a service with the container.

        Args:
            name: Name of the service
            service: Service instance

        Raises:
            ServiceContainerError: If a service with the same name is already registered
        """
        if name in self._services:
            raise ServiceContainerError(f"Service '{name}' is already registered")

        self._services[name] = service
        self._logger.debug(f"Registered service: {name}")

    def register_analyzer(self, name: str, analyzer: Any) -> None:
        """
        Register an analyzer with the container.

        Args:
            name: Name of the analyzer
            analyzer: Analyzer instance

        Raises:
            ServiceContainerError: If an analyzer with the same name is already registered
        """
        if name in self._analyzers:
            raise ServiceContainerError(f"Analyzer '{name}' is already registered")

        self._analyzers[name] = analyzer
        self._logger.debug(f"Registered analyzer: {name}")

    def get_service(self, name: str) -> Any:
        """
        Get a service by name.

        Args:
            name: Name of the service

        Returns:
            The service instance

        Raises:
            ServiceContainerError: If the service is not found
        """
        if name not in self._services:
            raise ServiceContainerError(f"Service '{name}' not found")

        return self._services[name]

    def get_analyzer(self, name: str) -> Any:
        """
        Get an analyzer by name.

        Args:
            name: Name of the analyzer

        Returns:
            The analyzer instance

        Raises:
            ServiceContainerError: If the analyzer is not found
        """
        if name not in self._analyzers:
            raise ServiceContainerError(f"Analyzer '{name}' not found")

        return self._analyzers[name]

    def list_services(self) -> List[str]:
        """
        Get a list of registered service names.

        Returns:
            List of service names
        """
        return list(self._services.keys())

    def list_analyzers(self) -> List[str]:
        """
        Get a list of registered analyzer names.

        Returns:
            List of analyzer names
        """
        return list(self._analyzers.keys())

    async def initialize(self) -> None:
        """
        Initialize all registered services and analyzers.

        This method calls the initialize method on all registered services and analyzers.
        Services are initialized before analyzers.

        Raises:
            ServiceContainerError: If initialization fails
        """
        if self._initialized:
            return

        try:
            # Initialize adapter factory
            self.register_service("adapter_factory", adapter_factory)
            self._logger.info("Adapter factory registered")

            # Initialize services first
            for name, service in self._services.items():
                if hasattr(service, 'initialize') and callable(service.initialize):
                    if asyncio.iscoroutinefunction(service.initialize):
                        await service.initialize()
                    else:
                        service.initialize()
                    self._logger.debug(f"Initialized service: {name}")

            # Then initialize analyzers
            for name, analyzer in self._analyzers.items():
                if hasattr(analyzer, 'initialize') and callable(analyzer.initialize):
                    if asyncio.iscoroutinefunction(analyzer.initialize):
                        await analyzer.initialize()
                    else:
                        analyzer.initialize()
                    self._logger.debug(f"Initialized analyzer: {name}")

            # Register adapters with the adapter factory
            from analysis_engine.services.analysis_service import AnalysisService
            from analysis_engine.services.indicator_service import IndicatorService
            from analysis_engine.services.pattern_service import PatternService
            from analysis_engine.adapters.analysis_adapter import AnalysisProviderAdapter
            from analysis_engine.adapters.indicator_adapter import IndicatorProviderAdapter
            from analysis_engine.adapters.pattern_adapter import PatternRecognizerAdapter
            from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer

            # Create service instances if they don't exist
            analysis_service = self.get_service("analysis_service") if "analysis_service" in self._services else AnalysisService()
            indicator_service = self.get_service("indicator_service") if "indicator_service" in self._services else IndicatorService()
            pattern_service = self.get_service("pattern_service") if "pattern_service" in self._services else PatternService()

            # Create adapter instances
            analysis_adapter = AnalysisProviderAdapter(analysis_service)
            indicator_adapter = IndicatorProviderAdapter(indicator_service)
            pattern_adapter = PatternRecognizerAdapter(pattern_service)

            # Register adapters with the adapter factory
            adapter_factory.register_adapter(IAnalysisProvider, analysis_adapter)
            adapter_factory.register_adapter(IIndicatorProvider, indicator_adapter)
            adapter_factory.register_adapter(IPatternRecognizer, pattern_adapter)

            self._logger.info("Adapters registered with adapter factory")

            self._initialized = True
            self._logger.info("Service container initialized")

        except Exception as e:
            self._logger.error(f"Error initializing container: {e}", exc_info=True)
            raise ServiceContainerError(f"Failed to initialize container: {str(e)}")

    @property
    def health_check(self) -> HealthCheck:
        """
        Get the health check instance.

        Returns:
            HealthCheck instance
        """
        if self._health_check is None:
            from analysis_engine.core.config import get_settings
            settings = get_settings()
            self._health_check = HealthCheck(
                service_name="analysis-engine-service",
                version=settings.app_version
            )
            self._setup_health_checks()

        return self._health_check

    def _setup_health_checks(self) -> None:
        """Set up health checks for services and analyzers."""
        from analysis_engine.monitoring.health_checks import check_component_health

        # Add component health checks for services
        for name, service in self._services.items():
            self._health_check.add_component_check(
                name,
                lambda service_name=name, service_instance=service: check_component_health(
                    service_instance, service_name
                )
            )

        # Add component health checks for analyzers
        for name, analyzer in self._analyzers.items():
            self._health_check.add_component_check(
                name,
                lambda analyzer_name=name, analyzer_instance=analyzer: check_component_health(
                    analyzer_instance, analyzer_name
                )
            )

    async def cleanup(self) -> None:
        """
        Clean up all registered services and analyzers.

        This method calls the cleanup method on all registered services and analyzers.
        Analyzers are cleaned up before services.

        Raises:
            ServiceContainerError: If cleanup fails
        """
        try:
            # Clean up analyzers first
            for name, analyzer in self._analyzers.items():
                if hasattr(analyzer, 'cleanup') and callable(analyzer.cleanup):
                    if asyncio.iscoroutinefunction(analyzer.cleanup):
                        await analyzer.cleanup()
                    else:
                        analyzer.cleanup()
                    self._logger.debug(f"Cleaned up analyzer: {name}")

            # Clean up adapter factory
            if "adapter_factory" in self._services:
                adapter_factory = self._services["adapter_factory"]
                adapter_factory.clear_adapters()
                self._logger.info("Adapter factory cleared")

            # Then clean up services
            for name, service in self._services.items():
                if name != "adapter_factory" and hasattr(service, 'cleanup') and callable(service.cleanup):
                    if asyncio.iscoroutinefunction(service.cleanup):
                        await service.cleanup()
                    else:
                        service.cleanup()
                    self._logger.debug(f"Cleaned up service: {name}")

            self._services.clear()
            self._analyzers.clear()
            self._initialized = False
            self._health_check = None
            self._logger.info("Service container cleaned up")

        except Exception as e:
            self._logger.error(f"Error cleaning up container: {e}", exc_info=True)
            raise ServiceContainerError(f"Failed to clean up container: {str(e)}")
