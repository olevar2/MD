"""
Example of using standardized clients

This module demonstrates how to use the standardized clients
to interact with the Analysis Engine Service API.
"""

import asyncio
import pandas as pd
from typing import Dict, List, Any

from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

async def generate_adaptive_parameters_example():
    """
    Example of generating adaptive parameters using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the adaptive layer client
    client = factory.get_adaptive_layer_client()
    
    # Example data
    strategy_id = "macd_rsi_strategy_v1"
    symbol = "EURUSD"
    timeframe = "1h"
    
    # Create example OHLC data
    ohlc_data = [
        {"timestamp": "2025-04-01T00:00:00", "open": 1.0765, "high": 1.0780, "low": 1.0760, "close": 1.0775, "volume": 1000},
        {"timestamp": "2025-04-01T01:00:00", "open": 1.0775, "high": 1.0790, "low": 1.0770, "close": 1.0785, "volume": 1200},
        {"timestamp": "2025-04-01T02:00:00", "open": 1.0785, "high": 1.0795, "low": 1.0780, "close": 1.0790, "volume": 1100},
        {"timestamp": "2025-04-01T03:00:00", "open": 1.0790, "high": 1.0800, "low": 1.0785, "close": 1.0795, "volume": 1300},
    ]
    
    # Available tools
    available_tools = ["macd", "rsi", "bollinger_bands", "fibonacci_retracement"]
    
    try:
        # Generate adaptive parameters
        result = await client.generate_adaptive_parameters(
            strategy_id=strategy_id,
            symbol=symbol,
            timeframe=timeframe,
            ohlc_data=ohlc_data,
            available_tools=available_tools,
            adaptation_strategy="moderate"
        )
        
        logger.info(f"Generated adaptive parameters: {result}")
        return result
    except Exception as e:
        logger.error(f"Error generating adaptive parameters: {str(e)}")
        raise

async def adjust_parameters_example():
    """
    Example of adjusting parameters using the standardized client.
    """
    # Get the client factory
    factory = get_client_factory()
    
    # Get the adaptive layer client
    client = factory.get_adaptive_layer_client()
    
    # Example data
    strategy_id = "macd_rsi_strategy_v1"
    instrument = "EUR_USD"
    timeframe = "1H"
    
    # Current parameters
    current_parameters = {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30
    }
    
    # Additional context
    context = {
        "market_regime": "trending",
        "volatility": "medium",
        "recent_performance": {
            "win_rate": 0.65,
            "profit_factor": 1.8
        }
    }
    
    try:
        # Adjust parameters
        result = await client.adjust_parameters(
            strategy_id=strategy_id,
            instrument=instrument,
            timeframe=timeframe,
            current_parameters=current_parameters,
            context=context
        )
        
        logger.info(f"Adjusted parameters: {result}")
        return result
    except Exception as e:
        logger.error(f"Error adjusting parameters: {str(e)}")
        raise

async def main():
    """
    Run the examples.
    """
    logger.info("Running standardized client examples")
    
    # Run the examples
    await generate_adaptive_parameters_example()
    await adjust_parameters_example()
    
    logger.info("Completed standardized client examples")

if __name__ == "__main__":
    asyncio.run(main())
