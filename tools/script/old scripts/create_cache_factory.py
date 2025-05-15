
"""
Create Cache Factory

This script creates cache factory files for all services in the forex trading platform.
"""
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define the base directory
BASE_DIR = Path("D:/MD/forex_trading_platform")

# Define the services to apply caching to
SERVICES = [
    "causal-analysis-service",
    "backtesting-service",
    "market-analysis-service",
    "analysis-coordinator-service"
]

def create_cache_factory(service_dir):
    """
    Create a cache factory for a service.
    
    Args:
        service_dir: The service directory
        
    Returns:
        True if the cache factory was created, False otherwise
    """
    service_name = os.path.basename(service_dir)
    module_name = service_name.replace("-", "_")
    
    # Create the utils directory if it doesn't exist
    utils_dir = os.path.join(service_dir, module_name, "utils")
    os.makedirs(utils_dir, exist_ok=True)
    
    # Create the __init__.py file if it doesn't exist
    init_path = os.path.join(utils_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("""
Utils package for {0}.
""".format(service_name))
        logger.info(f"Created {init_path}")
    
    # Create the cache_factory.py file
    cache_factory_path = os.path.join(utils_dir, "cache_factory.py")
    
    # Check if the file already exists
    if os.path.exists(cache_factory_path):
        logger.info(f"Cache factory already exists for {service_name}")
        return False
    
    # Create the cache factory file
    cache_factory_content = """
Cache Factory for {0}

This module provides a factory for creating cache instances.
"""
import logging
from typing import Optional

from common_lib.caching.cache import Cache
from common_lib.caching.redis_cache import RedisCache
from common_lib.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class CacheFactory:
    """
    Factory for creating cache instances.
    """
    _instance: Optional[Cache] = None
    
    def get_cache(self) -> Cache:
        """
        Get a cache instance.
        
        Returns:
            A cache instance
        """
        if self._instance is None:
            config = ConfigManager().get_config()
            cache_config = config.get("cache", {{}})
            
            # Use Redis cache if configured
            if cache_config.get("type") == "redis":
                redis_config = cache_config.get("redis", {{}})
                host = redis_config.get("host", "localhost")
                port = redis_config.get("port", 6379)
                db = redis_config.get("db", 0)
                
                logger.info(f"Creating Redis cache with host={{host}}, port={{port}}, db={{db}}")
                self._instance = RedisCache(host=host, port=port, db=db)
            else:
                # Use in-memory cache by default
                logger.info("Creating in-memory cache")
                from common_lib.caching.memory_cache import MemoryCache
                self._instance = MemoryCache()
        
        return self._instance

# Create a singleton instance
cache_factory = CacheFactory()
""".format(service_name)
    
    with open(cache_factory_path, "w") as f:
        f.write(cache_factory_content)
    
    logger.info(f"Created {cache_factory_path}")
    return True

def main():
    """Main function to create cache factories for all services."""
    logger.info("Starting to create cache factories for services")
    
    for service_name in SERVICES:
        service_dir = BASE_DIR / service_name
        
        if not service_dir.exists():
            logger.warning(f"Service directory {service_dir} does not exist")
            continue
        
        logger.info(f"Processing service: {service_name}")
        
        # Create cache factory
        create_cache_factory(service_dir)
    
    logger.info("Finished creating cache factories for services")

if __name__ == "__main__":
    main()
