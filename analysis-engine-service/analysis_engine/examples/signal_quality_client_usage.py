"""
Example of using the Signal Quality client

This module demonstrates how to use the standardized Signal Quality client
to interact with the Analysis Engine Service API.
"""

import asyncio
from typing import Dict, Any
from datetime import datetime

from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

async def evaluate_signal_quality_example():
    """
    Example of evaluating signal quality using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the signal quality client
    client = factory.get_signal_quality_client()
    
    # Example data
    signal_id = "sig_12345"
    
    # Market context
    market_context = {
        "market_regime": "trending",
        "volatility": "medium",
        "evaluate_confluence": True
    }
    
    # Historical data
    historical_data = {
        "win_rate": 0.65,
        "average_profit": 1.2,
        "average_loss": 0.8
    }
    
    try:
        # Evaluate signal quality
        result = await client.evaluate_signal_quality(
            signal_id=signal_id,
            market_context=market_context,
            historical_data=historical_data
        )
        
        logger.info(f"Evaluated signal quality: {result}")
        return result
    except Exception as e:
        logger.error(f"Error evaluating signal quality: {str(e)}")
        raise

async def analyze_signal_quality_example():
    """
    Example of analyzing signal quality using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the signal quality client
    client = factory.get_signal_quality_client()
    
    # Example data
    tool_id = "macd_crossover_v1"
    timeframe = "1h"
    market_regime = "trending"
    days = 30
    
    try:
        # Analyze signal quality
        result = await client.analyze_signal_quality(
            tool_id=tool_id,
            timeframe=timeframe,
            market_regime=market_regime,
            days=days
        )
        
        logger.info(f"Analyzed signal quality: {result}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing signal quality: {str(e)}")
        raise

async def analyze_quality_trends_example():
    """
    Example of analyzing quality trends using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the signal quality client
    client = factory.get_signal_quality_client()
    
    # Example data
    tool_id = "macd_crossover_v1"
    window_size = 20
    days = 90
    
    try:
        # Analyze quality trends
        result = await client.analyze_quality_trends(
            tool_id=tool_id,
            window_size=window_size,
            days=days
        )
        
        logger.info(f"Analyzed quality trends: {result}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing quality trends: {str(e)}")
        raise

async def main():
    """
    Run the examples.
    """
    logger.info("Running Signal Quality client examples")
    
    # Run the examples
    await evaluate_signal_quality_example()
    await analyze_signal_quality_example()
    await analyze_quality_trends_example()
    
    logger.info("Completed Signal Quality client examples")

if __name__ == "__main__":
    asyncio.run(main())
