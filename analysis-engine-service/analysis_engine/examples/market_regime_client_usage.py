"""
Example of using the Market Regime client

This module demonstrates how to use the standardized Market Regime client
to interact with the Analysis Engine Service API.
"""

import asyncio
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

async def detect_market_regime_example():
    """
    Example of detecting market regime using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the market regime client
    client = factory.get_market_regime_client()
    
    # Example data
    symbol = "EURUSD"
    timeframe = "1h"
    
    # Create example OHLC data
    ohlc_data = [
        {"timestamp": "2025-04-01T00:00:00", "open": 1.0765, "high": 1.0780, "low": 1.0760, "close": 1.0775, "volume": 1000},
        {"timestamp": "2025-04-01T01:00:00", "open": 1.0775, "high": 1.0790, "low": 1.0770, "close": 1.0785, "volume": 1200},
        {"timestamp": "2025-04-01T02:00:00", "open": 1.0785, "high": 1.0795, "low": 1.0780, "close": 1.0790, "volume": 1100},
        {"timestamp": "2025-04-01T03:00:00", "open": 1.0790, "high": 1.0800, "low": 1.0785, "close": 1.0795, "volume": 1300},
        # Add more data points to meet the minimum requirement (20 data points)
        {"timestamp": "2025-04-01T04:00:00", "open": 1.0795, "high": 1.0805, "low": 1.0790, "close": 1.0800, "volume": 1200},
        {"timestamp": "2025-04-01T05:00:00", "open": 1.0800, "high": 1.0810, "low": 1.0795, "close": 1.0805, "volume": 1100},
        {"timestamp": "2025-04-01T06:00:00", "open": 1.0805, "high": 1.0815, "low": 1.0800, "close": 1.0810, "volume": 1000},
        {"timestamp": "2025-04-01T07:00:00", "open": 1.0810, "high": 1.0820, "low": 1.0805, "close": 1.0815, "volume": 1200},
        {"timestamp": "2025-04-01T08:00:00", "open": 1.0815, "high": 1.0825, "low": 1.0810, "close": 1.0820, "volume": 1300},
        {"timestamp": "2025-04-01T09:00:00", "open": 1.0820, "high": 1.0830, "low": 1.0815, "close": 1.0825, "volume": 1400},
        {"timestamp": "2025-04-01T10:00:00", "open": 1.0825, "high": 1.0835, "low": 1.0820, "close": 1.0830, "volume": 1500},
        {"timestamp": "2025-04-01T11:00:00", "open": 1.0830, "high": 1.0840, "low": 1.0825, "close": 1.0835, "volume": 1600},
        {"timestamp": "2025-04-01T12:00:00", "open": 1.0835, "high": 1.0845, "low": 1.0830, "close": 1.0840, "volume": 1700},
        {"timestamp": "2025-04-01T13:00:00", "open": 1.0840, "high": 1.0850, "low": 1.0835, "close": 1.0845, "volume": 1800},
        {"timestamp": "2025-04-01T14:00:00", "open": 1.0845, "high": 1.0855, "low": 1.0840, "close": 1.0850, "volume": 1900},
        {"timestamp": "2025-04-01T15:00:00", "open": 1.0850, "high": 1.0860, "low": 1.0845, "close": 1.0855, "volume": 2000},
        {"timestamp": "2025-04-01T16:00:00", "open": 1.0855, "high": 1.0865, "low": 1.0850, "close": 1.0860, "volume": 2100},
        {"timestamp": "2025-04-01T17:00:00", "open": 1.0860, "high": 1.0870, "low": 1.0855, "close": 1.0865, "volume": 2200},
        {"timestamp": "2025-04-01T18:00:00", "open": 1.0865, "high": 1.0875, "low": 1.0860, "close": 1.0870, "volume": 2300},
        {"timestamp": "2025-04-01T19:00:00", "open": 1.0870, "high": 1.0880, "low": 1.0865, "close": 1.0875, "volume": 2400},
    ]
    
    try:
        # Detect market regime
        result = await client.detect_market_regime(
            symbol=symbol,
            timeframe=timeframe,
            ohlc_data=ohlc_data
        )
        
        logger.info(f"Detected market regime: {result}")
        return result
    except Exception as e:
        logger.error(f"Error detecting market regime: {str(e)}")
        raise

async def get_regime_history_example():
    """
    Example of getting regime history using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the market regime client
    client = factory.get_market_regime_client()
    
    # Example data
    symbol = "EURUSD"
    timeframe = "1h"
    limit = 5
    
    try:
        # Get regime history
        result = await client.get_regime_history(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        logger.info(f"Got regime history: {result}")
        return result
    except Exception as e:
        logger.error(f"Error getting regime history: {str(e)}")
        raise

async def analyze_tool_regime_performance_example():
    """
    Example of analyzing tool regime performance using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the market regime client
    client = factory.get_market_regime_client()
    
    # Example data
    tool_id = "macd_crossover_v1"
    timeframe = "1h"
    instrument = "EUR_USD"
    from_date = datetime.now() - timedelta(days=30)
    to_date = datetime.now()
    
    try:
        # Analyze tool regime performance
        result = await client.analyze_tool_regime_performance(
            tool_id=tool_id,
            timeframe=timeframe,
            instrument=instrument,
            from_date=from_date,
            to_date=to_date
        )
        
        logger.info(f"Analyzed tool regime performance: {result}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing tool regime performance: {str(e)}")
        raise

async def main():
    """
    Run the examples.
    """
    logger.info("Running Market Regime client examples")
    
    # Run the examples
    await detect_market_regime_example()
    await get_regime_history_example()
    await analyze_tool_regime_performance_example()
    
    logger.info("Completed Market Regime client examples")

if __name__ == "__main__":
    asyncio.run(main())
