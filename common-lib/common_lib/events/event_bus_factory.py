"""
Event Bus Factory Module

This module provides a factory for creating event buses.
"""

import logging
from enum import Enum
from typing import Dict, Optional, Any

from .base import IEventBus
from .in_memory_event_bus import InMemoryEventBus
from .kafka_event_bus_v2 import KafkaEventBusV2, KafkaConfig


class EventBusType(str, Enum):
    """Types of event buses."""

    IN_MEMORY = "in_memory"
    KAFKA = "kafka"


class EventBusFactory:
    """
    Factory for creating event buses.

    This class provides a factory for creating different types of event buses.
    """

    @staticmethod
    def create_event_bus(
        bus_type: EventBusType,
        service_name: str,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ) -> IEventBus:
        """
        Create an event bus.

        Args:
            bus_type: Type of event bus to create
            service_name: Name of the service using the event bus
            config: Configuration for the event bus
            logger: Optional logger

        Returns:
            Event bus

        Raises:
            ValueError: If the bus type is not supported
        """
        config = config or {}
        logger = logger or logging.getLogger(f"{__name__}.EventBusFactory")

        if bus_type == EventBusType.IN_MEMORY:
            logger.info(f"Creating in-memory event bus for service: {service_name}")
            return InMemoryEventBus(logger=logger)

        elif bus_type == EventBusType.KAFKA:
            logger.info(f"Creating Kafka event bus for service: {service_name}")

            # Get Kafka configuration
            bootstrap_servers = config.get("bootstrap_servers", "localhost:9092")
            group_id = config.get("group_id")
            client_id = config.get("client_id")
            auto_create_topics = config.get("auto_create_topics", True)
            producer_config = config.get("producer_config", {})
            consumer_config = config.get("consumer_config", {})
            topic_prefix = config.get("topic_prefix", "forex.")
            num_partitions = config.get("num_partitions", 3)
            replication_factor = config.get("replication_factor", 1)

            # Create Kafka configuration
            kafka_config = KafkaConfig(
                bootstrap_servers=bootstrap_servers,
                service_name=service_name,
                group_id=group_id,
                client_id=client_id,
                auto_create_topics=auto_create_topics,
                producer_config=producer_config,
                consumer_config=consumer_config,
                topic_prefix=topic_prefix,
                num_partitions=num_partitions,
                replication_factor=replication_factor
            )

            # Create Kafka event bus
            return KafkaEventBusV2(config=kafka_config)

        else:
            raise ValueError(f"Unsupported event bus type: {bus_type}")


# Singleton event bus manager
class EventBusManager:
    """
    Manager for event buses.

    This class provides a singleton manager for event buses.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the event bus manager.

        Returns:
            Event bus manager
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the event bus manager.

        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            return

        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._event_buses = {}
        self._initialized = True

    def get_event_bus(
        self,
        name: str = "default",
        bus_type: EventBusType = EventBusType.IN_MEMORY,
        service_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> IEventBus:
        """
        Get an event bus by name.

        Args:
            name: Name of the event bus
            bus_type: Type of event bus to create if it doesn't exist
            service_name: Name of the service using the event bus
            config: Configuration for the event bus

        Returns:
            Event bus
        """
        if name not in self._event_buses:
            service_name = service_name or name
            self._event_buses[name] = EventBusFactory.create_event_bus(
                bus_type=bus_type,
                service_name=service_name,
                config=config,
                logger=self.logger
            )

        return self._event_buses[name]

    async def start_all(self) -> None:
        """
        Start all event buses.
        """
        for name, event_bus in self._event_buses.items():
            self.logger.info(f"Starting event bus: {name}")
            await event_bus.start()

    async def stop_all(self) -> None:
        """
        Stop all event buses.
        """
        for name, event_bus in self._event_buses.items():
            self.logger.info(f"Stopping event bus: {name}")
            await event_bus.stop()
