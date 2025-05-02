"""
Event topic definitions and schema configurations for the Forex Trading Platform.
"""

from enum import Enum
from typing import Dict, Any

class EventTopics:
    # Core trading events
    TRADING_ORDERS = "forex.trading.orders"
    TRADING_EXECUTIONS = "forex.trading.executions"
    MARKET_DATA = "forex.market.data"
    
    # Feedback system events
    FEEDBACK_COLLECTION = "forex.feedback.collection"
    FEEDBACK_PROCESSING = "forex.feedback.processing"
    FEEDBACK_BATCH = "forex.feedback.batch"
    PARAMETER_UPDATES = "forex.feedback.parameters"
    MODEL_UPDATES = "forex.feedback.models"
    
    # System events
    SYSTEM_METRICS = "forex.system.metrics"
    SYSTEM_ALERTS = "forex.system.alerts"
    DEAD_LETTER_QUEUE = "forex.system.dlq"

class TopicConfig:
    DEFAULT_CONFIG = {
        "num_partitions": 6,
        "replication_factor": 3,
        "cleanup.policy": "delete",
        "retention.ms": 604800000,  # 7 days
        "compression.type": "lz4"
    }
    
    DLQ_CONFIG = {
        "num_partitions": 3,
        "replication_factor": 3,
        "cleanup.policy": "compact",
        "retention.ms": 2592000000,  # 30 days
        "compression.type": "lz4"
    }
    
    MARKET_DATA_CONFIG = {
        **DEFAULT_CONFIG,
        "cleanup.policy": "compact,delete",
        "retention.ms": 86400000,  # 1 day for raw market data
    }

# Event schema versions and validation rules
EVENT_SCHEMAS = {
    EventTopics.FEEDBACK_COLLECTION: {
        "version": "1.0",
        "required_fields": ["feedback_id", "source", "timestamp", "data"],
        "optional_fields": ["correlation_id", "metadata"]
    },
    EventTopics.FEEDBACK_PROCESSING: {
        "version": "1.0",
        "required_fields": ["feedback_id", "status", "timestamp"],
        "optional_fields": ["processing_details", "error_info"]
    }
}
