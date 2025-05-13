"""
Client/Adapter for interacting with the Feature Store Service.
"""
import logging
from typing import Dict, Any, List
import pandas as pd
import httpx
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class FeatureStoreClient:
    """Handles communication with the Feature Store Service API."""

    def __init__(self, base_url: str='http://localhost:8002'):
        """Initializes the FeatureStoreClient.

        Args:
            base_url: The base URL of the Feature Store Service.
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        logger.info(f'FeatureStoreClient initialized for URL: {self.base_url}')

    @with_analysis_resilience('calculate_indicators')
    @async_with_exception_handling
    async def calculate_indicators(self, symbol: str, data: pd.DataFrame,
        indicator_configs: List[Dict[str, Any]]) ->Dict[str, Any]:
        """Requests indicator calculation from the Feature Store Service.

        Args:
            symbol: The asset symbol.
            data: The market data (e.g., OHLCV pandas DataFrame).
            indicator_configs: A list of configurations for the indicators to calculate.
                               Example: [{'type': 'sma', 'name': 'sma_14', 'window': 14}, ...]

        Returns:
            A dictionary containing the calculated indicator results.
            Example: {'sma_14': [ ... values ... ], ...}

        Raises:
            Exception: If the request fails or the service returns an error.
        """
        endpoint = f'/features/calculate/{symbol}'
        payload = {'data': data.to_dict(orient='list'), 'indicators':
            indicator_configs}
        try:
            logger.debug(f'Sending request to Feature Store: POST {endpoint}')
            response = await self.client.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            logger.debug(
                f'Received response from Feature Store: {result.keys()}')
            return result
        except httpx.RequestError as exc:
            logger.error(
                f'HTTP Request Error calling Feature Store at {exc.request.url!r}: {exc}'
                )
            raise Exception(f'Failed to connect to Feature Store: {exc}'
                ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                f'HTTP Status Error calling Feature Store: {exc.response.status_code} - {exc.response.text}'
                )
            raise Exception(
                f'Feature Store returned error: {exc.response.status_code}'
                ) from exc
        except Exception as e:
            logger.error(
                f'Unexpected error during Feature Store indicator calculation: {e}'
                )
            raise

    async def close(self):
        """Closes the underlying HTTP client."""
        await self.client.aclose()
        logger.info('FeatureStoreClient closed.')


@async_with_exception_handling
async def main():
    """
    Main.
    
    """

    import asyncio
    data = pd.DataFrame({'timestamp': pd.to_datetime(['2023-01-01 10:00',
        '2023-01-01 10:01', '2023-01-01 10:02']), 'open': [1.1, 1.2, 1.15],
        'high': [1.12, 1.22, 1.18], 'low': [1.08, 1.18, 1.12], 'close': [
        1.11, 1.21, 1.16], 'volume': [100, 150, 120]})
    indicator_configs = [{'type': 'sma', 'name': 'sma_2', 'window': 2}, {
        'type': 'ema', 'name': 'ema_2', 'window': 2}]
    client = FeatureStoreClient()
    try:
        results = await client.calculate_indicators('EURUSD', data,
            indicator_configs)
        print('Calculated Indicators:', results)
    except Exception as e:
        print(f'Error: {e}')
    finally:
        await client.close()


if __name__ == '__main__':
    print(
        'FeatureStoreClient defined. Run main() to test against a running service.'
        )
