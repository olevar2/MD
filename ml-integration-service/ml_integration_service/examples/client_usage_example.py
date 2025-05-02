"""
Client Usage Example

This module demonstrates how to use the standardized service clients.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from ml_integration_service.clients.client_factory import (
    get_analysis_engine_client,
    get_ml_workbench_client,
    initialize_clients
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("client_example")


async def analysis_engine_example():
    """Example of using the Analysis Engine client."""
    logger.info("Running Analysis Engine client example...")
    
    # Get the client
    client = get_analysis_engine_client()
    
    # Example 1: Get technical indicators
    try:
        indicators = [
            {"name": "SMA", "params": {"period": 20}},
            {"name": "RSI", "params": {"period": 14}},
            {"name": "MACD", "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}}
        ]
        
        result = await client.get_technical_indicators(
            symbol="EURUSD",
            timeframe="1h",
            indicators=indicators,
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now()
        )
        
        logger.info(f"Got {len(result.get('data', []))} data points with indicators")
    except Exception as e:
        logger.error(f"Error getting technical indicators: {str(e)}")
    
    # Example 2: Detect market regime
    try:
        regime = await client.detect_market_regime(
            symbol="EURUSD",
            timeframe="1h"
        )
        
        logger.info(f"Current market regime: {regime.get('regime', 'unknown')}")
        logger.info(f"Regime confidence: {regime.get('confidence', 0):.2f}")
    except Exception as e:
        logger.error(f"Error detecting market regime: {str(e)}")
    
    # Example 3: Multi-timeframe analysis
    try:
        analysis = await client.get_multi_timeframe_analysis(
            symbol="EURUSD",
            timeframes=["15m", "1h", "4h", "1d"],
            analysis_types=["trend", "volatility", "support_resistance"]
        )
        
        for timeframe, data in analysis.items():
            logger.info(f"Analysis for {timeframe}:")
            logger.info(f"  Trend: {data.get('trend', {}).get('direction', 'unknown')}")
            logger.info(f"  Volatility: {data.get('volatility', {}).get('level', 'unknown')}")
    except Exception as e:
        logger.error(f"Error getting multi-timeframe analysis: {str(e)}")


async def ml_workbench_example():
    """Example of using the ML Workbench client."""
    logger.info("Running ML Workbench client example...")
    
    # Get the client
    client = get_ml_workbench_client()
    
    # Example 1: List models
    try:
        models = await client.get_models(model_type="classification", limit=5)
        
        logger.info(f"Found {len(models)} classification models:")
        for model in models:
            logger.info(f"  {model.get('name')} (ID: {model.get('id')})")
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
    
    # Example 2: Get model details
    if models:
        try:
            model_id = models[0].get('id')
            model_details = await client.get_model(model_id)
            
            logger.info(f"Model details for {model_details.get('name')}:")
            logger.info(f"  Type: {model_details.get('model_type')}")
            logger.info(f"  Created: {model_details.get('created_at')}")
            logger.info(f"  Status: {model_details.get('status')}")
        except Exception as e:
            logger.error(f"Error getting model details: {str(e)}")
    
    # Example 3: Make predictions
    if models:
        try:
            model_id = models[0].get('id')
            
            # Example input data (adjust based on your model's requirements)
            inputs = {
                "features": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
            
            prediction = await client.predict(model_id, inputs)
            
            logger.info(f"Prediction result:")
            logger.info(f"  Prediction: {prediction.get('prediction')}")
            logger.info(f"  Confidence: {prediction.get('confidence', 0):.2f}")
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")


async def main():
    """Run the client examples."""
    # Initialize clients
    initialize_clients()
    
    # Run examples
    await analysis_engine_example()
    await ml_workbench_example()
    
    # Close clients
    analysis_engine_client = get_analysis_engine_client()
    ml_workbench_client = get_ml_workbench_client()
    
    await analysis_engine_client.close()
    await ml_workbench_client.close()


if __name__ == "__main__":
    asyncio.run(main())
