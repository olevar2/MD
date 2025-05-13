"""
Service Factory

This module provides factory methods for obtaining service instances throughout
the analysis engine service, ensuring proper initialization and dependency injection.
"""

from typing import Optional
from fastapi import Depends

# Import core foundation services
from core_foundations.config.configuration import ConfigurationManager
from core_foundations.events.event_bus import EventBus

# Import analysis engine services
from analysis_engine.adaptive_layer.feedback_integration_service import FeedbackIntegrationService
from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop

# Singleton instances
_feedback_integration_service: Optional[FeedbackIntegrationService] = None


def get_configuration_manager():
    """
    Get the configuration manager instance.
    
    Returns:
        ConfigurationManager: The configuration manager
    """
    # In a real implementation, this would likely access a singleton or dependency injection system
    # For simplicity, we're creating a new instance here
    return ConfigurationManager()


def get_event_bus():
    """
    Get the event bus instance.
    
    Returns:
        EventBus: The event bus
    """
    # In a real implementation, this would likely access a singleton or dependency injection system
    # For simplicity, we're creating a new instance here
    return EventBus()


def get_adaptation_engine():
    """
    Get the adaptation engine instance.
    
    Returns:
        AdaptationEngine: The adaptation engine
    """
    # In a real implementation, this would likely access a singleton or dependency injection system
    # For simplicity, we're creating a new instance here
    return AdaptationEngine()


def get_feedback_loop(adaptation_engine: AdaptationEngine = Depends(get_adaptation_engine)):
    """
    Get the feedback loop instance.
    
    Args:
        adaptation_engine: The adaptation engine
        
    Returns:
        FeedbackLoop: The feedback loop
    """
    # In a real implementation, this would likely access a singleton or dependency injection system
    # For simplicity, we're creating a new instance here
    return FeedbackLoop(adaptation_engine)


def get_feedback_service(
    config_manager: ConfigurationManager = Depends(get_configuration_manager),
    event_bus: EventBus = Depends(get_event_bus),
    adaptation_engine: AdaptationEngine = Depends(get_adaptation_engine),
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
):
    """
    Get the feedback integration service instance.
    
    This factory method ensures a singleton instance of the FeedbackIntegrationService
    is used throughout the application, properly initialized with all dependencies.
    
    Args:
        config_manager: The configuration manager
        event_bus: The event bus for event publishing and subscribing
        adaptation_engine: The adaptation engine for strategy adjustments
        feedback_loop: The feedback loop for learning from outcomes
        
    Returns:
        FeedbackIntegrationService: The feedback integration service
    """
    global _feedback_integration_service
    
    if _feedback_integration_service is None:
        _feedback_integration_service = FeedbackIntegrationService(
            config_manager=config_manager,
            event_publisher=event_bus,
            event_subscriber=event_bus,
            adaptation_engine=adaptation_engine,
            feedback_loop=feedback_loop
        )
    
    return _feedback_integration_service
