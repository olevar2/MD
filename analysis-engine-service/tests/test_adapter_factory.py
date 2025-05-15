"""
Test script for the common adapter factory.

This script tests if the common adapter factory and service dependencies are working correctly.
"""

import asyncio
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_lib.interfaces.trading_gateway import ITradingGateway
from common_lib.interfaces.ml_workbench import IExperimentManager
from common_lib.interfaces.risk_management import IRiskManager
from common_lib.interfaces.feature_store import IFeatureProvider

from analysis_engine.adapters.common_adapter_factory import get_common_adapter_factory
from analysis_engine.core.service_dependencies import service_dependencies


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_adapter_factory():
    """Test the common adapter factory."""
    logger.info("Testing common adapter factory...")
    
    # Get the adapter factory
    factory = get_common_adapter_factory()
    
    # Test getting adapters for different interfaces
    try:
        trading_gateway = factory.get_adapter(ITradingGateway)
        logger.info(f"Successfully got Trading Gateway adapter: {trading_gateway}")
        
        experiment_manager = factory.get_adapter(IExperimentManager)
        logger.info(f"Successfully got Experiment Manager adapter: {experiment_manager}")
        
        risk_manager = factory.get_adapter(IRiskManager)
        logger.info(f"Successfully got Risk Manager adapter: {risk_manager}")
        
        feature_provider = factory.get_adapter(IFeatureProvider)
        logger.info(f"Successfully got Feature Provider adapter: {feature_provider}")
        
        logger.info("All adapters created successfully!")
        return True
    except Exception as e:
        logger.error(f"Error creating adapters: {str(e)}")
        return False


async def test_service_dependencies():
    """Test the service dependencies."""
    logger.info("Testing service dependencies...")
    
    try:
        # Test getting adapters from service dependencies
        trading_gateway = await service_dependencies.get_trading_gateway()
        logger.info(f"Successfully got Trading Gateway from service dependencies: {trading_gateway}")
        
        experiment_manager = await service_dependencies.get_experiment_manager()
        logger.info(f"Successfully got Experiment Manager from service dependencies: {experiment_manager}")
        
        risk_manager = await service_dependencies.get_risk_manager()
        logger.info(f"Successfully got Risk Manager from service dependencies: {risk_manager}")
        
        feature_provider = await service_dependencies.get_feature_provider()
        logger.info(f"Successfully got Feature Provider from service dependencies: {feature_provider}")
        
        logger.info("All service dependencies working correctly!")
        return True
    except Exception as e:
        logger.error(f"Error in service dependencies: {str(e)}")
        return False


async def main():
    """Run all tests."""
    logger.info("Starting tests...")
    
    adapter_factory_result = await test_adapter_factory()
    service_dependencies_result = await test_service_dependencies()
    
    if adapter_factory_result and service_dependencies_result:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
