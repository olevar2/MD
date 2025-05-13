"""
Client Usage Example

This module demonstrates how to use the standardized service clients.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from core.client_factory import get_analysis_engine_client, get_ml_workbench_client, initialize_clients
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('client_example')


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def analysis_engine_example():
    """Example of using the Analysis Engine client."""
    logger.info('Running Analysis Engine client example...')
    client = get_analysis_engine_client()
    try:
        indicators = [{'name': 'SMA', 'params': {'period': 20}}, {'name':
            'RSI', 'params': {'period': 14}}, {'name': 'MACD', 'params': {
            'fast_period': 12, 'slow_period': 26, 'signal_period': 9}}]
        result = await client.get_technical_indicators(symbol='EURUSD',
            timeframe='1h', indicators=indicators, start_time=datetime.now(
            ) - timedelta(days=30), end_time=datetime.now())
        logger.info(
            f"Got {len(result.get('data', []))} data points with indicators")
    except Exception as e:
        logger.error(f'Error getting technical indicators: {str(e)}')
    try:
        regime = await client.detect_market_regime(symbol='EURUSD',
            timeframe='1h')
        logger.info(f"Current market regime: {regime.get('regime', 'unknown')}"
            )
        logger.info(f"Regime confidence: {regime.get('confidence', 0):.2f}")
    except Exception as e:
        logger.error(f'Error detecting market regime: {str(e)}')
    try:
        analysis = await client.get_multi_timeframe_analysis(symbol=
            'EURUSD', timeframes=['15m', '1h', '4h', '1d'], analysis_types=
            ['trend', 'volatility', 'support_resistance'])
        for timeframe, data in analysis.items():
            logger.info(f'Analysis for {timeframe}:')
            logger.info(
                f"  Trend: {data.get('trend', {}).get('direction', 'unknown')}"
                )
            logger.info(
                f"  Volatility: {data.get('volatility', {}).get('level', 'unknown')}"
                )
    except Exception as e:
        logger.error(f'Error getting multi-timeframe analysis: {str(e)}')


@async_with_exception_handling
async def ml_workbench_example():
    """Example of using the ML Workbench client."""
    logger.info('Running ML Workbench client example...')
    client = get_ml_workbench_client()
    try:
        models = await client.get_models(model_type='classification', limit=5)
        logger.info(f'Found {len(models)} classification models:')
        for model in models:
            logger.info(f"  {model.get('name')} (ID: {model.get('id')})")
    except Exception as e:
        logger.error(f'Error listing models: {str(e)}')
    if models:
        try:
            model_id = models[0].get('id')
            model_details = await client.get_model(model_id)
            logger.info(f"Model details for {model_details.get('name')}:")
            logger.info(f"  Type: {model_details.get('model_type')}")
            logger.info(f"  Created: {model_details.get('created_at')}")
            logger.info(f"  Status: {model_details.get('status')}")
        except Exception as e:
            logger.error(f'Error getting model details: {str(e)}')
    if models:
        try:
            model_id = models[0].get('id')
            inputs = {'features': [0.1, 0.2, 0.3, 0.4, 0.5]}
            prediction = await client.predict(model_id, inputs)
            logger.info(f'Prediction result:')
            logger.info(f"  Prediction: {prediction.get('prediction')}")
            logger.info(f"  Confidence: {prediction.get('confidence', 0):.2f}")
        except Exception as e:
            logger.error(f'Error making prediction: {str(e)}')


async def main():
    """Run the client examples."""
    initialize_clients()
    await analysis_engine_example()
    await ml_workbench_example()
    analysis_engine_client = get_analysis_engine_client()
    ml_workbench_client = get_ml_workbench_client()
    await analysis_engine_client.close()
    await ml_workbench_client.close()


if __name__ == '__main__':
    asyncio.run(main())
