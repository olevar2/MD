"""
Redis configuration for the Market Analysis Service.

This module provides the configuration for Redis caching.
"""
import os
from typing import Optional

class RedisConfig:
    """
    Redis configuration.
    """
    def __init__(self):
        """
        Initialize the Redis configuration.
        """
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", "6379"))
        self.db = int(os.getenv("REDIS_DB", "0"))
        self.password = os.getenv("REDIS_PASSWORD", None)
        self.ssl = os.getenv("REDIS_SSL", "false").lower() == "true"
        self.timeout = int(os.getenv("REDIS_TIMEOUT", "5"))
        self.ttl = int(os.getenv("REDIS_TTL", "3600"))  # Default TTL: 1 hour
    
    @property
    def url(self) -> str:
        """
        Get the Redis URL.
        
        Returns:
            The Redis URL
        """
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"
    
    @property
    def enabled(self) -> bool:
        """
        Check if Redis is enabled.
        
        Returns:
            True if Redis is enabled, False otherwise
        """
        return os.getenv("REDIS_ENABLED", "false").lower() == "true"


# Create a singleton instance
redis_config = RedisConfig()