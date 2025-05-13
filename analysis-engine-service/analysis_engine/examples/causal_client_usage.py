"""
Example of using the Causal Analysis client

This module demonstrates how to use the standardized Causal Analysis client
to interact with the Analysis Engine Service API.
"""
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta
from analysis_engine.clients.standardized import get_client_factory
from analysis_engine.monitoring.structured_logging import get_structured_logger
logger = get_structured_logger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@async_with_exception_handling
async def discover_structure_example():
    """
    Example of discovering causal structure using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_causal_client()
    data = [{'timestamp': '2023-01-01T00:00:00Z', 'EUR_USD': 1.075,
        'GBP_USD': 1.25, 'USD_JPY': 130.5}, {'timestamp':
        '2023-01-01T01:00:00Z', 'EUR_USD': 1.0755, 'GBP_USD': 1.251,
        'USD_JPY': 130.45}, {'timestamp': '2023-01-01T02:00:00Z', 'EUR_USD':
        1.076, 'GBP_USD': 1.2515, 'USD_JPY': 130.4}, {'timestamp':
        '2023-01-01T03:00:00Z', 'EUR_USD': 1.0765, 'GBP_USD': 1.252,
        'USD_JPY': 130.35}, {'timestamp': '2023-01-01T04:00:00Z', 'EUR_USD':
        1.077, 'GBP_USD': 1.2525, 'USD_JPY': 130.3}]
    try:
        result = await client.discover_structure(data=data, method=
            'granger', cache_key='example_discovery', force_refresh=True)
        logger.info(f'Discovered causal structure: {result}')
        nodes = result.get('nodes', [])
        edges = result.get('edges', [])
        logger.info(f'Nodes: {nodes}')
        logger.info(f'Edges: {edges}')
        return result
    except Exception as e:
        logger.error(f'Error discovering causal structure: {str(e)}')
        raise


@async_with_exception_handling
async def estimate_effect_example():
    """
    Example of estimating causal effect using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_causal_client()
    data = [{'timestamp': '2023-01-01T00:00:00Z', 'volatility': 0.12,
        'spread': 2.5, 'volume': 1000, 'return': 0.01}, {'timestamp':
        '2023-01-01T01:00:00Z', 'volatility': 0.15, 'spread': 3.0, 'volume':
        1200, 'return': 0.02}, {'timestamp': '2023-01-01T02:00:00Z',
        'volatility': 0.1, 'spread': 2.0, 'volume': 800, 'return': 0.005},
        {'timestamp': '2023-01-01T03:00:00Z', 'volatility': 0.18, 'spread':
        3.5, 'volume': 1500, 'return': 0.03}, {'timestamp':
        '2023-01-01T04:00:00Z', 'volatility': 0.14, 'spread': 2.8, 'volume':
        1100, 'return': 0.015}]
    try:
        result = await client.estimate_effect(data=data, treatment=
            'volatility', outcome='return', common_causes=['spread',
            'volume'], method='backdoor.linear_regression')
        logger.info(f'Estimated causal effect: {result}')
        effect_estimate = result.get('effect_estimate', 0.0)
        confidence_interval = result.get('confidence_interval')
        p_value = result.get('p_value')
        logger.info(f'Effect estimate: {effect_estimate}')
        logger.info(f'Confidence interval: {confidence_interval}')
        logger.info(f'P-value: {p_value}')
        return result
    except Exception as e:
        logger.error(f'Error estimating causal effect: {str(e)}')
        raise


@async_with_exception_handling
async def analyze_counterfactuals_example():
    """
    Example of analyzing counterfactuals using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_causal_client()
    data = [{'timestamp': '2023-01-01T00:00:00Z', 'volatility': 0.12,
        'spread': 2.5, 'volume': 1000, 'return': 0.01}, {'timestamp':
        '2023-01-01T01:00:00Z', 'volatility': 0.15, 'spread': 3.0, 'volume':
        1200, 'return': 0.02}, {'timestamp': '2023-01-01T02:00:00Z',
        'volatility': 0.1, 'spread': 2.0, 'volume': 800, 'return': 0.005},
        {'timestamp': '2023-01-01T03:00:00Z', 'volatility': 0.18, 'spread':
        3.5, 'volume': 1500, 'return': 0.03}, {'timestamp':
        '2023-01-01T04:00:00Z', 'volatility': 0.14, 'spread': 2.8, 'volume':
        1100, 'return': 0.015}]
    interventions = {'high_volatility': {'volatility': 0.25},
        'low_volatility': {'volatility': 0.05}}
    try:
        result = await client.analyze_counterfactuals(data=data, target=
            'return', interventions=interventions, features=['volatility',
            'spread', 'volume'])
        logger.info(f'Counterfactual analysis: {result}')
        baseline = result.get('baseline_prediction', 0.0)
        predictions = result.get('counterfactual_predictions', {})
        differences = result.get('differences', {})
        logger.info(f'Baseline prediction: {baseline}')
        logger.info(f'Counterfactual predictions: {predictions}')
        logger.info(f'Differences from baseline: {differences}')
        return result
    except Exception as e:
        logger.error(f'Error analyzing counterfactuals: {str(e)}')
        raise


@async_with_exception_handling
async def analyze_currency_pair_relationships_example():
    """
    Example of analyzing currency pair relationships using the standardized client.
    """
    factory = get_client_factory()
    client = factory.get_causal_client()
    price_data = {'EUR_USD': {'ohlc': [{'timestamp': '2023-01-01T00:00:00Z',
        'open': 1.075, 'high': 1.076, 'low': 1.0745, 'close': 1.0755}, {
        'timestamp': '2023-01-01T01:00:00Z', 'open': 1.0755, 'high': 1.0765,
        'low': 1.075, 'close': 1.076}, {'timestamp': '2023-01-01T02:00:00Z',
        'open': 1.076, 'high': 1.077, 'low': 1.0755, 'close': 1.0765}, {
        'timestamp': '2023-01-01T03:00:00Z', 'open': 1.0765, 'high': 1.0775,
        'low': 1.076, 'close': 1.077}, {'timestamp': '2023-01-01T04:00:00Z',
        'open': 1.077, 'high': 1.078, 'low': 1.0765, 'close': 1.0775}]},
        'GBP_USD': {'ohlc': [{'timestamp': '2023-01-01T00:00:00Z', 'open': 
        1.25, 'high': 1.251, 'low': 1.249, 'close': 1.2505}, {'timestamp':
        '2023-01-01T01:00:00Z', 'open': 1.2505, 'high': 1.2515, 'low': 1.25,
        'close': 1.251}, {'timestamp': '2023-01-01T02:00:00Z', 'open': 
        1.251, 'high': 1.252, 'low': 1.2505, 'close': 1.2515}, {'timestamp':
        '2023-01-01T03:00:00Z', 'open': 1.2515, 'high': 1.2525, 'low': 
        1.251, 'close': 1.252}, {'timestamp': '2023-01-01T04:00:00Z',
        'open': 1.252, 'high': 1.253, 'low': 1.2515, 'close': 1.2525}]}}
    try:
        result = await client.analyze_currency_pair_relationships(price_data
            =price_data, max_lag=5, config={'significance_level': 0.05})
        logger.info(f'Currency pair relationships: {result}')
        relationships = result.get('relationships', [])
        logger.info(f'Relationships: {relationships}')
        return result
    except Exception as e:
        logger.error(f'Error analyzing currency pair relationships: {str(e)}')
        raise


async def main():
    """
    Run the examples.
    """
    logger.info('Running Causal Analysis client examples')
    await discover_structure_example()
    await estimate_effect_example()
    await analyze_counterfactuals_example()
    await analyze_currency_pair_relationships_example()
    logger.info('Completed Causal Analysis client examples')


if __name__ == '__main__':
    asyncio.run(main())
