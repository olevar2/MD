"""
Unified Factory for Adaptive Layer Components

This module provides factory functions and classes to create instances of all
adaptive layer components, including feedback loop, adaptation engine,
parameter tracking, strategy mutation, Kafka integration, and related services.
It consolidates logic from previous factory files.
"""

from typing import Dict, Any, Optional, List
import asyncio

from core_foundations.config.configuration import ConfigurationManager
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.events.event_subscriber import EventSubscriber
from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.di.service_factory import ServiceFactory, register_factory
from core_foundations.di.service_container import ServiceContainer

from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from analysis_engine.adaptive_layer.trading_feedback_collector import TradingFeedbackCollector
from analysis_engine.adaptive_layer.feedback_categorizer import FeedbackCategorizer
from analysis_engine.adaptive_layer.feedback_router import FeedbackRouter
from analysis_engine.adaptive_layer.feedback_integration_service import FeedbackIntegrationService
from analysis_engine.adaptive_layer.strategy_mutation import StrategyMutationEngine # Renamed from service
from analysis_engine.adaptive_layer.parameter_feedback import ParameterFeedbackTracker # Renamed from service
from analysis_engine.adaptive_layer.enhanced_feedback_kafka_handler import EnhancedFeedbackKafkaHandler # Using enhanced handler
from analysis_engine.adaptive_layer.event_consumers import (
    TradingOutcomeConsumer, ParameterPerformanceConsumer, StrategyEffectivenessConsumer,
    ModelPredictionConsumer, ExecutionQualityConsumer
)
from analysis_engine.repositories.strategy_repository import StrategyRepositoryBase
from analysis_engine.integration.event_bus import EventBusBase # Assuming this is the correct base

# Import ML clients if available
try:
    from ml_integration_service.client import MLServiceClient
    ML_CLIENT_AVAILABLE = True
except ImportError:
    ML_CLIENT_AVAILABLE = False

try:
    from ml_workbench_service.feedback.model_training_feedback import ModelTrainingFeedbackIntegrator
    ML_WORKBENCH_AVAILABLE = True
except ImportError:
    ML_WORKBENCH_AVAILABLE = False


# --- Core Feedback Loop Components ---

@register_factory
class AdaptationEngineFactory(ServiceFactory):
    """Factory for creating AdaptationEngine instances."""
    async def create(
        self,
        config_manager: ConfigurationManager,
        # Add dependencies needed by AdaptationEngine (e.g., repositories, services)
        # tool_effectiveness_service: ToolEffectivenessService,
        # market_regime_identifier: MarketRegimeIdentifier,
        **kwargs
    ) -> AdaptationEngine:
        config = config_manager.get_configuration("adaptation_engine", {})
        # Replace with actual AdaptationEngine instantiation and dependencies
        # return AdaptationEngine(tool_effectiveness_service, market_regime_identifier, config=config)
        return AdaptationEngine(config=config) # Placeholder


@register_factory
class FeedbackLoopFactory(ServiceFactory):
    """Factory for creating FeedbackLoop instances."""
    async def create(
        self,
        config_manager: ConfigurationManager,
        adaptation_engine: AdaptationEngine, # Requires AdaptationEngine
        **kwargs
    ) -> FeedbackLoop:
        config = config_manager.get_configuration("feedback_loop", {})
        return FeedbackLoop(adaptation_engine=adaptation_engine, config=config)


@register_factory
class FeedbackCategorizerFactory(ServiceFactory):
    """Factory for creating FeedbackCategorizer instances."""
    async def create(
        self,
        config_manager: ConfigurationManager,
        **kwargs
    ) -> FeedbackCategorizer:
        config = config_manager.get_configuration("feedback_categorizer", {})
        return FeedbackCategorizer(config=config)


@register_factory
class FeedbackRouterFactory(ServiceFactory):
    """Factory for creating FeedbackRouter instances."""
    async def create(
        self,
        config_manager: ConfigurationManager,
        # Add dependencies needed by FeedbackRouter (e.g., other services)
        **kwargs
    ) -> FeedbackRouter:
        config = config_manager.get_configuration("feedback_router", {})
        # router = FeedbackRouter(config=config)
        # Example: Register routes if needed
        # router.register_route(FeedbackCategory.PERFORMANCE, parameter_feedback_service.handle_feedback)
        return FeedbackRouter(config=config) # Placeholder


@register_factory
class TradingFeedbackCollectorFactory(ServiceFactory):
    """Factory for creating TradingFeedbackCollector instances."""
    async def create(
        self,
        config_manager: ConfigurationManager,
        event_publisher: EventPublisher,
        feedback_loop: FeedbackLoop, # Requires FeedbackLoop
        **kwargs
    ) -> TradingFeedbackCollector:
        config = config_manager.get_configuration("feedback_collector", {})
        collector = TradingFeedbackCollector(
            feedback_loop=feedback_loop,
            event_publisher=event_publisher,
            config=config
        )
        # Assuming start is now handled by the integration service or main application flow
        # await collector.start()
        return collector


# --- Parameter and Strategy Adaptation ---

@register_factory
class ParameterFeedbackTrackerFactory(ServiceFactory):
    """Factory for creating ParameterFeedbackTracker instances."""
    async def create(
        self,
        config_manager: ConfigurationManager,
        event_publisher: EventPublisher,
        **kwargs
    ) -> ParameterFeedbackTracker:
        config = config_manager.get_configuration("feedback_system", {}).get("parameter_tracking", {})
        return ParameterFeedbackTracker(
            event_publisher=event_publisher,
            config=config
        )


@register_factory
class StrategyMutationEngineFactory(ServiceFactory):
    """Factory for creating StrategyMutationEngine instances."""
    async def create(
        self,
        config_manager: ConfigurationManager,
        parameter_feedback_tracker: ParameterFeedbackTracker, # Renamed dependency
        strategy_repository: StrategyRepositoryBase, # Requires StrategyRepository
        event_bus: EventBusBase, # Requires EventBus
        **kwargs
    ) -> StrategyMutationEngine:
        config = config_manager.get_configuration("feedback_system", {}).get("strategy_mutation", {})
        return StrategyMutationEngine(
            parameter_feedback_service=parameter_feedback_tracker, # Pass tracker instance
            strategy_repository=strategy_repository,
            event_bus=event_bus,
            config=config # Pass config
        )


# --- Kafka Integration ---

@register_factory
class EnhancedFeedbackKafkaHandlerFactory(ServiceFactory):
    """Factory for creating EnhancedFeedbackKafkaHandler instances."""
    async def create(
        self,
        config_manager: ConfigurationManager,
        event_bus: KafkaEventBus, # Requires KafkaEventBus
        feedback_loop: FeedbackLoop, # Requires FeedbackLoop
        parameter_feedback_tracker: ParameterFeedbackTracker, # Requires ParameterFeedbackTracker
        strategy_mutation_engine: StrategyMutationEngine, # Requires StrategyMutationEngine
        feedback_router: FeedbackRouter, # Requires FeedbackRouter
        **kwargs
    ) -> EnhancedFeedbackKafkaHandler:
        feedback_config = config_manager.get_configuration("feedback_system", {})
        service_name = feedback_config.get("service_name", "analysis_engine")
        kafka_config = feedback_config.get("kafka_integration", {})

        handler = EnhancedFeedbackKafkaHandler(
            event_bus=event_bus,
            service_name=service_name,
            config=kafka_config,
            # Pass dependencies needed for handling events
            feedback_loop=feedback_loop,
            parameter_tracking=parameter_feedback_tracker,
            strategy_mutation=strategy_mutation_engine,
            feedback_router=feedback_router
        )

        # Register specific handlers from event_consumers.py or within EnhancedFeedbackKafkaHandler
        # Example: handler.register_handler("feedback.trading_outcome", handler._handle_trading_outcome)
        # This registration logic might need refinement based on EnhancedFeedbackKafkaHandler's design

        await handler.start() # Start the handler after creation
        return handler

    async def dispose(self, instance: EnhancedFeedbackKafkaHandler):
        """Clean up resources when disposing the handler."""
        await instance.stop()


# --- Additional factories can be registered here as needed ---

# Note: Specialized event consumers are implemented directly within the
# EnhancedFeedbackKafkaHandler which handles all routing internally.
# This design eliminates the need for separate consumer factories.


# --- Integration Services ---

@register_factory
class FeedbackIntegrationServiceFactory(ServiceFactory):
    """Factory for creating FeedbackIntegrationService instances."""
    async def create(
        self,
        config_manager: ConfigurationManager,
        event_publisher: EventPublisher,
        event_subscriber: EventSubscriber,
        adaptation_engine: AdaptationEngine, # Requires AdaptationEngine
        feedback_loop: FeedbackLoop, # Requires FeedbackLoop
        # Add optional ML client dependency if needed
        # ml_client: Optional[MLServiceClient] = None,
        **kwargs
    ) -> FeedbackIntegrationService:
        # Create ML client if available and configured
        ml_client = None
        if ML_CLIENT_AVAILABLE:
            ml_config = config_manager.get_configuration("ml_integration", {})
            if ml_config.get("enabled", False):
                 # Assuming MLServiceClient takes config directly
                 ml_client = MLServiceClient(config=ml_config)

        service = FeedbackIntegrationService(
            config_manager=config_manager,
            event_publisher=event_publisher,
            event_subscriber=event_subscriber,
            adaptation_engine=adaptation_engine,
            feedback_loop=feedback_loop,
            ml_client=ml_client
        )
        await service.start()
        return service

    async def dispose(self, instance: FeedbackIntegrationService):
        """Clean up resources when disposing the service."""
        await instance.stop()


# --- Helper function to register all factories ---

def register_adaptive_layer_factories(container: ServiceContainer):
    """Registers all factories defined in this module with the service container."""
    container.register_factory(AdaptationEngine, AdaptationEngineFactory())
    container.register_factory(FeedbackLoop, FeedbackLoopFactory())
    container.register_factory(FeedbackCategorizer, FeedbackCategorizerFactory())
    container.register_factory(FeedbackRouter, FeedbackRouterFactory())
    container.register_factory(TradingFeedbackCollector, TradingFeedbackCollectorFactory())
    container.register_factory(ParameterFeedbackTracker, ParameterFeedbackTrackerFactory())
    container.register_factory(StrategyMutationEngine, StrategyMutationEngineFactory())
    container.register_factory(EnhancedFeedbackKafkaHandler, EnhancedFeedbackKafkaHandlerFactory())
    container.register_factory(FeedbackIntegrationService, FeedbackIntegrationServiceFactory())
    # Register other factories (e.g., consumers if needed)
    # container.register_factory(TradingOutcomeConsumer, TradingOutcomeConsumerFactory())

    # Note: Ensure base services like ConfigurationManager, EventPublisher, EventSubscriber,
    # KafkaEventBus, StrategyRepositoryBase are registered elsewhere in the application setup.

