"""
Event Persistence System for Forex Trading Platform

This module provides persistence capabilities for the event-driven architecture
of the Forex trading platform. It allows events to be stored, queried, and
replayed, which is essential for:

1. Debugging and troubleshooting production issues
2. Recovering from failures by replaying missed events
3. Rebuilding read models and projections
4. Testing using real production event sequences
5. Auditing and compliance

The implementation uses a combination of:
- Persistent Kafka topics with longer retention
- MongoDB for queryable event storage
- Redis for caching and optimizing high-frequency replay scenarios
"""

import datetime
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from abc import ABC, abstractmethod # Added import

# Try to import dependencies
try:
    import pymongo
    from pymongo import MongoClient, IndexModel, ASCENDING, DESCENDING
    from pymongo.errors import PyMongoError
except ImportError:
    raise ImportError(
        "pymongo is required for event persistence. "
        "Install it with 'pip install pymongo'"
    )

try:
    import redis
except ImportError:
    # Redis is optional, but recommended for caching
    redis = None

from .event_schema import Event, EventType
from .kafka_event_bus import KafkaEventBus, TopicNamingStrategy

# Configure logger
logger = logging.getLogger(__name__)


class PersistenceMode(str, Enum):
    """Enum for different persistence modes."""
    ALL = "all"                  # Store all events
    IMPORTANT = "important"      # Store only important events based on criteria
    SAMPLED = "sampled"          # Store a statistical sample of events (e.g., 1 in 10)
    NONE = "none"                # Don't persist (used for high-volume, low-value events)


@dataclass
class EventPersistenceConfig:
    """Configuration for event persistence."""
    persistence_mode: PersistenceMode = PersistenceMode.ALL
    sampling_rate: float = 1.0  # For SAMPLED mode, percentage of events to store
    cache_events: bool = True    # Whether to cache events in Redis
    cache_ttl: int = 86400       # Default TTL for cached events (1 day)
    importance_threshold: float = 0.5  # For IMPORTANT mode, minimum importance score
    
    # Event types to always persist regardless of other settings
    always_persist_types: Set[EventType] = field(default_factory=set)


class EventStore:
    """
    Storage for persisted events with query and replay capabilities.
    
    This class handles event persistence to MongoDB and optional Redis caching,
    with functionality to query and replay events based on various criteria.
    """
    
    def __init__(
        self,
        mongo_uri: str,
        db_name: str = "forex_event_store",
        collection_name: str = "events",
        redis_url: Optional[str] = None,
        default_config: Optional[EventPersistenceConfig] = None,
        per_event_type_config: Optional[Dict[EventType, EventPersistenceConfig]] = None
    ):
        """
        Initialize the event store.
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: MongoDB database name
            collection_name: MongoDB collection name for events
            redis_url: Optional Redis connection URL for caching
            default_config: Default persistence configuration
            per_event_type_config: Override persistence configuration for specific event types
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.redis_url = redis_url
        
        # Set up configurations
        self.default_config = default_config or EventPersistenceConfig()
        self.per_event_type_config = per_event_type_config or {}
        
        # Initialize MongoDB connection
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[db_name]
        self.events_collection = self.db[collection_name]
        
        # Set up indexes for efficient queries
        self._create_indexes()
        
        # Initialize Redis connection if provided
        self.redis_client = None
        if redis_url and redis:
            try:
                self.redis_client = redis.Redis.from_url(
                    redis_url, 
                    decode_responses=True
                )
                # Test the connection
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None

    def _create_indexes(self) -> None:
        """Create necessary indexes on the events collection."""
        try:
            # Create indexes for common query patterns
            indexes = [
                IndexModel([("event_type", ASCENDING)], background=True),
                IndexModel([("event_time", DESCENDING)], background=True),
                IndexModel([("source_service", ASCENDING)], background=True),
                IndexModel([
                    ("event_type", ASCENDING), 
                    ("event_time", DESCENDING)
                ], background=True),
                IndexModel([("correlation_id", ASCENDING)], background=True),
                IndexModel([("causation_id", ASCENDING)], background=True),
                # Compound index for efficient event replay queries
                IndexModel([
                    ("event_type", ASCENDING), 
                    ("event_time", ASCENDING)
                ], background=True),
                # TTL index for auto-expiration if needed
                # IndexModel([("event_time", ASCENDING)], 
                #            expireAfterSeconds=30*24*60*60)  # 30 days
            ]
            
            self.events_collection.create_indexes(indexes)
            logger.info("Event store indexes created or already exist")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")

    def _get_config_for_event(self, event: Event) -> EventPersistenceConfig:
        """
        Get the persistence configuration for an event type.
        
        Args:
            event: The event to check configuration for
            
        Returns:
            The appropriate EventPersistenceConfig
        """
        # Try to get specific config for this event type
        if event.event_type in self.per_event_type_config:
            return self.per_event_type_config[event.event_type]
        return self.default_config

    def _should_persist_event(self, event: Event) -> bool:
        """
        Determine whether an event should be persisted.
        
        Args:
            event: The event to check
            
        Returns:
            bool: True if the event should be persisted
        """
        config = self._get_config_for_event(event)
        
        # Always persist certain event types
        if event.event_type in config.always_persist_types:
            return True
        
        # Check persistence mode
        if config.persistence_mode == PersistenceMode.ALL:
            return True
        elif config.persistence_mode == PersistenceMode.NONE:
            return False
        elif config.persistence_mode == PersistenceMode.SAMPLED:
            # Simple random sampling based on event ID
            # Using last byte of UUID as pseudo-random value for consistent sampling
            uuid_bytes = event.event_id.bytes
            random_byte = uuid_bytes[-1] / 255.0
            return random_byte <= config.sampling_rate
        elif config.persistence_mode == PersistenceMode.IMPORTANT:
            # Check event priority or other importance criteria
            if hasattr(event, 'priority') and event.priority:
                return event.priority.value in ('high', 'critical')
            
            # Default importance estimator based on event type
            important_prefixes = ('risk', 'order', 'position', 'system')
            return any(event.event_type.value.startswith(prefix) for prefix in important_prefixes)
        
        return False

    def store_event(self, event: Event) -> bool:
        """
        Store an event in the persistence store.
        
        Args:
            event: The event to store
            
        Returns:
            bool: True if the event was stored, False otherwise
        """
        if not self._should_persist_event(event):
            return False
        
        try:
            # Convert event to dict for MongoDB storage
            event_dict = json.loads(event.json())
            
            # Add metadata for the event store
            event_dict['_stored_at'] = datetime.datetime.utcnow()
            
            # Store in MongoDB
            self.events_collection.insert_one(event_dict)
            
            # Cache in Redis if available
            if self.redis_client and self._get_config_for_event(event).cache_events:
                try:
                    # Cache by event ID
                    event_key = f"event:{event.event_id}"
                    self.redis_client.set(
                        event_key,
                        event.json(),
                        ex=self._get_config_for_event(event).cache_ttl
                    )
                    
                    # Add to type-based list for efficient replay
                    type_key = f"events:type:{event.event_type.value}"
                    self.redis_client.zadd(
                        type_key, 
                        {str(event.event_id): int(event.event_time.timestamp())}
                    )
                    # Set expiration on the sorted set
                    self.redis_client.expire(
                        type_key,
                        self._get_config_for_event(event).cache_ttl
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache event in Redis: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False

    def get_event(self, event_id: Union[str, uuid.UUID]) -> Optional[Event]:
        """
        Retrieve an event by its ID.
        
        Args:
            event_id: The ID of the event to retrieve
            
        Returns:
            The event if found, None otherwise
        """
        # Convert to string if UUID
        if isinstance(event_id, uuid.UUID):
            event_id = str(event_id)
            
        # Try cache first if available
        if self.redis_client:
            try:
                cached_event = self.redis_client.get(f"event:{event_id}")
                if cached_event:
                    return Event.parse_raw(cached_event)
            except Exception as e:
                logger.warning(f"Failed to retrieve event from Redis: {e}")
        
        # Fallback to MongoDB
        try:
            event_dict = self.events_collection.find_one({"event_id": event_id})
            if event_dict:
                # Remove MongoDB-specific fields
                if '_id' in event_dict:
                    del event_dict['_id']
                if '_stored_at' in event_dict:
                    del event_dict['_stored_at']
                    
                return Event.parse_obj(event_dict)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve event: {e}")
            return None

    def find_events(
        self,
        event_types: Optional[List[EventType]] = None,
        source_services: Optional[List[str]] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        correlation_id: Optional[Union[str, uuid.UUID]] = None,
        limit: int = 100,
        skip: int = 0,
        sort_order: str = "desc"  # "desc" for newest first, "asc" for oldest first
    ) -> List[Event]:
        """
        Find events matching the specified criteria.
        
        Args:
            event_types: Optional list of event types to filter by
            source_services: Optional list of source services to filter by
            start_time: Optional start time for time range filter
            end_time: Optional end time for time range filter
            correlation_id: Optional correlation ID to filter by
            limit: Maximum number of events to return
            skip: Number of events to skip (for pagination)
            sort_order: Sort order for events, "desc" for newest first
            
        Returns:
            List of matching events
        """
        # Build query
        query: Dict[str, Any] = {}
        
        if event_types:
            query["event_type"] = {"$in": [et.value for et in event_types]}
            
        if source_services:
            query["source_service"] = {"$in": source_services}
            
        if start_time or end_time:
            time_query = {}
            if start_time:
                time_query["$gte"] = start_time
            if end_time:
                time_query["$lte"] = end_time
            if time_query:
                query["event_time"] = time_query
                
        if correlation_id:
            query["correlation_id"] = str(correlation_id)
            
        # Set sort direction
        sort_direction = DESCENDING if sort_order == "desc" else ASCENDING
        
        try:
            # Execute query
            cursor = self.events_collection.find(
                query,
                sort=[("event_time", sort_direction)],
                limit=limit,
                skip=skip
            )
            
            # Convert to Event objects
            events = []
            for event_dict in cursor:
                # Remove MongoDB-specific fields
                if '_id' in event_dict:
                    del event_dict['_id']
                if '_stored_at' in event_dict:
                    del event_dict['_stored_at']
                    
                events.append(Event.parse_obj(event_dict))
                
            return events
        except Exception as e:
            logger.error(f"Failed to find events: {e}")
            return []

    def replay_events(
        self,
        event_bus: KafkaEventBus,
        event_types: List[EventType],
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        correlation_id: Optional[Union[str, uuid.UUID]] = None,
        batch_size: int = 100,
        delay_between_batches_ms: int = 100
    ) -> Tuple[int, int]:
        """
        Replay events through the event bus.
        
        Args:
            event_bus: The event bus to publish replayed events to
            event_types: Types of events to replay
            start_time: Optional start time for replay range
            end_time: Optional end time for replay range
            correlation_id: Optional correlation ID to filter events
            batch_size: Number of events to process in each batch
            delay_between_batches_ms: Delay between batches in milliseconds
            
        Returns:
            Tuple of (events_found, events_published)
        """
        total_events_found = 0
        total_events_published = 0
        
        # Build query
        query: Dict[str, Any] = {
            "event_type": {"$in": [et.value for et in event_types]}
        }
        
        if start_time or end_time:
            time_query = {}
            if start_time:
                time_query["$gte"] = start_time
            if end_time:
                time_query["$lte"] = end_time
            if time_query:
                query["event_time"] = time_query
                
        if correlation_id:
            query["correlation_id"] = str(correlation_id)
            
        try:
            # Get count for logging
            total_count = self.events_collection.count_documents(query)
            logger.info(f"Found {total_count} events to replay")
            
            # Process in batches
            skip = 0
            while True:
                cursor = self.events_collection.find(
                    query,
                    sort=[("event_time", ASCENDING)],  # Always replay in chronological order
                    limit=batch_size,
                    skip=skip
                )
                
                batch = list(cursor)
                total_events_found += len(batch)
                
                if not batch:
                    break
                
                # Replay each event
                for event_dict in batch:
                    # Remove MongoDB-specific fields
                    if '_id' in event_dict:
                        del event_dict['_id']
                    if '_stored_at' in event_dict:
                        del event_dict['_stored_at']
                        
                    try:
                        # Parse and republish the event
                        event = Event.parse_obj(event_dict)
                        
                        # Add metadata to indicate this is a replayed event
                        event.metadata["replayed"] = True
                        event.metadata["original_time"] = event.event_time.isoformat()
                        
                        # Publish to event bus
                        event_bus.publish(event)
                        total_events_published += 1
                    except Exception as e:
                        logger.error(f"Failed to replay event: {e}")
                
                # Update skip for next batch
                skip += batch_size
                
                # Add delay between batches
                if delay_between_batches_ms > 0:
                    time.sleep(delay_between_batches_ms / 1000.0)
                
                # Log progress
                logger.info(
                    f"Replayed {total_events_published}/{total_events_found} "
                    f"events ({total_events_found}/{total_count} total)"
                )
                
            return total_events_found, total_events_published
        except Exception as e:
            logger.error(f"Failed to replay events: {e}")
            return total_events_found, total_events_published

    def get_event_stats(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about stored events.
        
        Args:
            start_time: Optional start time for stats range
            end_time: Optional end time for stats range
            
        Returns:
            Dictionary with event statistics
        """
        stats = {}
        
        # Build time range query
        time_query = {}
        if start_time:
            time_query["$gte"] = start_time
        if end_time:
            time_query["$lte"] = end_time
            
        query = {}
        if time_query:
            query["event_time"] = time_query
        
        try:
            # Total events
            stats["total_events"] = self.events_collection.count_documents(query)
            
            # Events by type
            pipeline = [
                {"$match": query},
                {"$group": {"_id": "$event_type", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            event_types = self.events_collection.aggregate(pipeline)
            stats["events_by_type"] = {
                item["_id"]: item["count"] for item in event_types
            }
            
            # Events by service
            pipeline = [
                {"$match": query},
                {"$group": {"_id": "$source_service", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            source_services = self.events_collection.aggregate(pipeline)
            stats["events_by_service"] = {
                item["_id"]: item["count"] for item in source_services
            }
            
            # Storage size
            stats["storage_stats"] = {
                "collection_size_bytes": self.db.command("collstats", self.collection_name).get("size"),
                "index_size_bytes": self.db.command("collstats", self.collection_name).get("totalIndexSize")
            }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get event stats: {e}")
            return {"error": str(e)}
            
    def close(self) -> None:
        """Close connections to the database and cache."""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
            
        if self.redis_client:
            self.redis_client.close()


class EventPersistenceListener:
    """
    Listener that subscribes to events and persists them.
    
    This class connects to the event bus, subscribes to events,
    and stores them in the event store based on configured policies.
    """
    
    def __init__(
        self,
        event_bus: KafkaEventBus,
        event_store: EventStore,
        event_types_to_persist: Optional[List[EventType]] = None
    ):
        """
        Initialize the event persistence listener.
        
        Args:
            event_bus: The event bus to subscribe to
            event_store: The event store to persist events to
            event_types_to_persist: Types of events to persist (default: all)
        """
        self.event_bus = event_bus
        self.event_store = event_store
        
        # If no specific types provided, persist all event types
        if event_types_to_persist is None:
            self.event_types_to_persist = list(EventType)
        else:
            self.event_types_to_persist = event_types_to_persist
            
        # Statistics
        self.events_received = 0
        self.events_persisted = 0
        self.start_time = datetime.datetime.utcnow()
        
    def handle_event(self, event: Event) -> None:
        """
        Handle an event by persisting it to the event store.
        
        Args:
            event: The event to handle
        """
        self.events_received += 1
        
        # Check if this is a replayed event to avoid duplicate storage
        is_replayed = event.metadata.get("replayed", False)
        if not is_replayed and self.event_store.store_event(event):
            self.events_persisted += 1
            
    def start(self) -> None:
        """Start listening for events to persist."""
        # Subscribe to all configured event types
        self.event_bus.subscribe(
            self.event_types_to_persist,
            self.handle_event
        )
        
        # Start consuming events
        self.event_bus.start_consuming()
        
    def stop(self) -> None:
        """Stop listening for events."""
        self.event_bus.stop_consuming()
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about persisted events.
        
        Returns:
            Dictionary with persistence statistics
        """
        now = datetime.datetime.utcnow()
        running_time = (now - self.start_time).total_seconds()
        
        return {
            "events_received": self.events_received,
            "events_persisted": self.events_persisted,
            "persistence_ratio": (self.events_persisted / self.events_received 
                                  if self.events_received > 0 else 0),
            "running_time_seconds": running_time,
            "events_per_second": self.events_received / running_time if running_time > 0 else 0,
            "start_time": self.start_time.isoformat(),
            "current_time": now.isoformat()
        }
