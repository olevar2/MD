
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

def main():
    logger.info("Starting to create cache factories for services")
    
    for service_name in SERVICES:
        service_dir = BASE_DIR / service_name
        
        if not service_dir.exists():
            logger.warning(f"Service directory {service_dir} does not exist")
            continue
        
        logger.info(f"Processing service: {service_name}")
        
        # Create utils directory
        module_name = service_name.replace("-", "_")
        utils_dir = os.path.join(service_dir, module_name, "utils")
        os.makedirs(utils_dir, exist_ok=True)
        
        # Create __init__.py
        init_path = os.path.join(utils_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write(f"""
Utils package for {service_name}.
""")
            logger.info(f"Created {init_path}")
        
        # Create cache_factory.py
        cache_factory_path = os.path.join(utils_dir, "cache_factory.py")
        if not os.path.exists(cache_factory_path):
            with open(cache_factory_path, "w") as f:
                f.write(f"""
Cache Factory for {service_name}

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
""")
            logger.info(f"Created {cache_factory_path}")
    
    logger.info("Finished creating cache factories for services")

if __name__ == "__main__":
    main()
