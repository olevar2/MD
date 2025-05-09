"""
Redis Setup Script for Forex Trading Platform.

This script sets up Redis for distributed caching and performance monitoring.
It configures Redis with appropriate settings for the forex trading platform.

Usage:
    python setup_redis.py --host localhost --port 6379
"""

import os
import sys
import argparse
import logging
import redis
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default Redis configuration
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_PASSWORD = None
DEFAULT_REDIS_DB = 0

# Redis configuration settings
REDIS_CONFIG = {
    # Memory management
    "maxmemory": "1gb",
    "maxmemory-policy": "volatile-lru",
    "maxmemory-samples": "10",
    
    # Persistence
    "save": "900 1 300 10 60 10000",
    "appendonly": "yes",
    "appendfsync": "everysec",
    
    # Performance
    "tcp-keepalive": "60",
    "timeout": "300",
    "tcp-backlog": "511",
    "databases": "16",
    
    # Logging
    "loglevel": "notice",
    "logfile": "redis.log",
    
    # Security
    "protected-mode": "yes"
}

def check_redis_connection(host: str, port: int, password: str = None, db: int = 0) -> bool:
    """
    Check if Redis is running and accessible.
    
    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        db: Redis database index
        
    Returns:
        True if Redis is running and accessible, False otherwise
    """
    try:
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            socket_timeout=5
        )
        
        # Test connection
        client.ping()
        
        logger.info(f"Successfully connected to Redis at {host}:{port}")
        return True
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Redis at {host}:{port}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        return False

def configure_redis(host: str, port: int, password: str = None, db: int = 0) -> bool:
    """
    Configure Redis with appropriate settings.
    
    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        db: Redis database index
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            socket_timeout=5
        )
        
        # Test connection
        client.ping()
        
        # Configure Redis
        for key, value in REDIS_CONFIG.items():
            try:
                client.config_set(key, value)
                logger.info(f"Set Redis config {key}={value}")
            except redis.exceptions.ResponseError as e:
                logger.warning(f"Failed to set Redis config {key}={value}: {e}")
        
        # Save configuration
        client.config_rewrite()
        logger.info("Redis configuration saved")
        
        return True
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Redis at {host}:{port}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error configuring Redis: {e}")
        return False

def setup_redis_monitoring(host: str, port: int, password: str = None, db: int = 0) -> bool:
    """
    Set up Redis monitoring.
    
    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        db: Redis database index
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            socket_timeout=5
        )
        
        # Test connection
        client.ping()
        
        # Enable Redis monitoring
        client.config_set("latency-monitor-threshold", "100")
        logger.info("Redis latency monitoring enabled")
        
        # Enable slow log
        client.config_set("slowlog-log-slower-than", "10000")  # 10ms
        client.config_set("slowlog-max-len", "1000")
        logger.info("Redis slow log configured")
        
        # Save configuration
        client.config_rewrite()
        logger.info("Redis monitoring configuration saved")
        
        return True
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to Redis at {host}:{port}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error setting up Redis monitoring: {e}")
        return False

def create_redis_docker_compose(output_dir: Path) -> bool:
    """
    Create a Docker Compose file for Redis.
    
    Args:
        output_dir: Output directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Docker Compose file content
        docker_compose_content = """version: '3'

services:
  redis:
    image: redis:7-alpine
    container_name: forex_redis
    ports:
      - "6379:6379"
    command: redis-server --save 60 1 --loglevel warning --maxmemory 1gb --maxmemory-policy volatile-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    restart: unless-stopped

volumes:
  redis_data:
    driver: local
"""
        
        # Write Docker Compose file
        docker_compose_file = output_dir / "docker-compose-redis.yml"
        with open(docker_compose_file, "w") as f:
            f.write(docker_compose_content)
        
        logger.info(f"Docker Compose file created at {docker_compose_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating Docker Compose file: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Set up Redis for distributed caching")
    parser.add_argument(
        "--host",
        default=DEFAULT_REDIS_HOST,
        help="Redis host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_REDIS_PORT,
        help="Redis port"
    )
    parser.add_argument(
        "--password",
        default=DEFAULT_REDIS_PASSWORD,
        help="Redis password"
    )
    parser.add_argument(
        "--db",
        type=int,
        default=DEFAULT_REDIS_DB,
        help="Redis database index"
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Create Docker Compose file for Redis"
    )
    parser.add_argument(
        "--output-dir",
        default="infrastructure/docker",
        help="Output directory for Docker Compose file"
    )
    
    args = parser.parse_args()
    
    # Check Redis connection
    if not check_redis_connection(args.host, args.port, args.password, args.db):
        if args.docker:
            logger.info("Redis not running, creating Docker Compose file")
            create_redis_docker_compose(Path(args.output_dir))
            logger.info("Run 'docker-compose -f infrastructure/docker/docker-compose-redis.yml up -d' to start Redis")
        else:
            logger.error("Redis not running, please start Redis first")
        return
    
    # Configure Redis
    if not configure_redis(args.host, args.port, args.password, args.db):
        logger.error("Failed to configure Redis")
        return
    
    # Set up Redis monitoring
    if not setup_redis_monitoring(args.host, args.port, args.password, args.db):
        logger.error("Failed to set up Redis monitoring")
        return
    
    logger.info("Redis setup completed successfully")

if __name__ == "__main__":
    main()
