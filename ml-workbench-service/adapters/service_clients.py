"""
Standardized Service Clients Module for ML Workbench Service

This module provides standardized service clients for interacting with other services
in the platform. It follows the common-lib pattern for service clients.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, cast
import httpx
import asyncio
from urllib.parse import urljoin
from common_lib.errors import ServiceError, NotFoundError, ValidationError, AuthenticationError, AuthorizationError, RateLimitError, TimeoutError, ConnectionError
from common_lib.resilience import with_resilience
from config.standardized_config_1 import settings, get_feature_store_service_url, get_analysis_engine_service_url, get_data_pipeline_service_url, get_trading_gateway_service_url
from core.logging_setup_1 import get_logger, get_correlation_id
T = TypeVar('T')
logger = get_logger(__name__)


class BaseServiceClient:
    """Base class for service clients."""

    def __init__(self, base_url: str, service_name: str, timeout: float=
        30.0, max_retries: int=3, backoff_factor: float=0.5):
        """
        Initialize the service client.

        Args:
            base_url: Base URL for the service
            service_name: Name of the service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
        """
        self.base_url = base_url
        self.service_name = service_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True
            )

    async def close(self) ->None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) ->'BaseServiceClient':
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) ->None:
        """Exit async context manager."""
        await self.close()

    def _get_headers(self, additional_headers: Optional[Dict[str, str]]=None
        ) ->Dict[str, str]:
        """
        Get request headers.

        Args:
            additional_headers: Additional headers to include

        Returns:
            Headers dictionary
        """
        headers = {'Content-Type': 'application/json', 'Accept':
            'application/json'}
        correlation_id = get_correlation_id()
        if correlation_id:
            headers['X-Correlation-ID'] = correlation_id
        if additional_headers:
            headers.update(additional_headers)
        return headers

    def _build_url(self, path: str) ->str:
        """
        Build full URL from path.

        Args:
            path: URL path

        Returns:
            Full URL
        """
        if not path.startswith('/'):
            path = f'/{path}'
        return urljoin(self.base_url, path)

    @with_exception_handling
    def _handle_error_response(self, response: httpx.Response, url: str
        ) ->None:
        """
        Handle error response.

        Args:
            response: HTTP response
            url: Request URL

        Raises:
            ServiceError: If the response indicates an error
        """
        status_code = response.status_code
        error_message = f'Error calling {self.service_name} at {url}'
        try:
            error_data = response.json()
            if isinstance(error_data, dict) and 'message' in error_data:
                error_message = error_data['message']
        except (json.JSONDecodeError, ValueError):
            error_message = response.text or error_message
        logger.error(f'Service error: {error_message}', extra={'service':
            self.service_name, 'url': url, 'status_code': status_code,
            'response_text': response.text})
        if status_code == 404:
            raise NotFoundError(message=error_message, service_name=self.
                service_name, operation=url)
        elif status_code == 400:
            raise ValidationError(message=error_message, service_name=self.
                service_name, operation=url)
        elif status_code == 401:
            raise AuthenticationError(message=error_message, service_name=
                self.service_name, operation=url)
        elif status_code == 403:
            raise AuthorizationError(message=error_message, service_name=
                self.service_name, operation=url)
        elif status_code == 429:
            raise RateLimitError(message=error_message, service_name=self.
                service_name, operation=url)
        else:
            raise ServiceError(message=error_message, service_name=self.
                service_name, operation=url)

    @with_resilience(enable_circuit_breaker=True, failure_threshold=5,
        recovery_timeout=30.0, enable_retry=True, max_retries=3, base_delay
        =1.0, max_delay=60.0, backoff_factor=2.0, jitter=True,
        enable_bulkhead=True, max_concurrent=10, max_queue=10,
        enable_timeout=True, timeout_seconds=30.0, expected_exceptions=[
        ConnectionError, TimeoutError, ServiceError])
    @async_with_exception_handling
    async def _request(self, method: str, path: str, params: Optional[Dict[
        str, Any]]=None, data: Optional[Dict[str, Any]]=None, headers:
        Optional[Dict[str, str]]=None, timeout: Optional[float]=None
        ) ->httpx.Response:
        """
        Make HTTP request to the service.

        Args:
            method: HTTP method
            path: URL path
            params: Query parameters
            data: Request data
            headers: Request headers
            timeout: Request timeout in seconds

        Returns:
            HTTP response
        """
        url = self._build_url(path)
        request_headers = self._get_headers(headers)
        request_timeout = timeout or self.timeout
        try:
            logger.debug(f'Making {method} request to {self.service_name}',
                extra={'service': self.service_name, 'method': method,
                'url': url, 'params': params})
            response = await self._client.request(method=method, url=url,
                params=params, json=data, headers=request_headers, timeout=
                request_timeout)
            if response.status_code >= 400:
                self._handle_error_response(response, url)
            return response
        except httpx.TimeoutException as e:
            logger.error(f'Timeout error calling {self.service_name}',
                extra={'service': self.service_name, 'method': method,
                'url': url, 'timeout': request_timeout})
            raise TimeoutError(message=
                f'Timeout error calling {self.service_name}: {str(e)}',
                service_name=self.service_name, operation=url) from e
        except httpx.ConnectError as e:
            logger.error(f'Connection error calling {self.service_name}',
                extra={'service': self.service_name, 'method': method,
                'url': url})
            raise ConnectionError(message=
                f'Connection error calling {self.service_name}: {str(e)}',
                service_name=self.service_name, operation=url) from e
        except httpx.RequestError as e:
            logger.error(f'Request error calling {self.service_name}',
                extra={'service': self.service_name, 'method': method,
                'url': url, 'error': str(e)})
            raise ServiceError(message=
                f'Request error calling {self.service_name}: {str(e)}',
                service_name=self.service_name, operation=url) from e

    async def get(self, path: str, params: Optional[Dict[str, Any]]=None,
        headers: Optional[Dict[str, str]]=None, timeout: Optional[float]=None
        ) ->Dict[str, Any]:
        """
        Make GET request to the service.

        Args:
            path: URL path
            params: Query parameters
            headers: Request headers
            timeout: Request timeout in seconds

        Returns:
            Response data
        """
        response = await self._request(method='GET', path=path, params=
            params, headers=headers, timeout=timeout)
        return response.json()

    async def post(self, path: str, data: Optional[Dict[str, Any]]=None,
        params: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str,
        str]]=None, timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Make POST request to the service.

        Args:
            path: URL path
            data: Request data
            params: Query parameters
            headers: Request headers
            timeout: Request timeout in seconds

        Returns:
            Response data
        """
        response = await self._request(method='POST', path=path, data=data,
            params=params, headers=headers, timeout=timeout)
        return response.json()

    async def put(self, path: str, data: Optional[Dict[str, Any]]=None,
        params: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str,
        str]]=None, timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Make PUT request to the service.

        Args:
            path: URL path
            data: Request data
            params: Query parameters
            headers: Request headers
            timeout: Request timeout in seconds

        Returns:
            Response data
        """
        response = await self._request(method='PUT', path=path, data=data,
            params=params, headers=headers, timeout=timeout)
        return response.json()

    async def delete(self, path: str, params: Optional[Dict[str, Any]]=None,
        headers: Optional[Dict[str, str]]=None, timeout: Optional[float]=None
        ) ->Dict[str, Any]:
        """
        Make DELETE request to the service.

        Args:
            path: URL path
            params: Query parameters
            headers: Request headers
            timeout: Request timeout in seconds

        Returns:
            Response data
        """
        response = await self._request(method='DELETE', path=path, params=
            params, headers=headers, timeout=timeout)
        return response.json()

    async def patch(self, path: str, data: Optional[Dict[str, Any]]=None,
        params: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str,
        str]]=None, timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Make PATCH request to the service.

        Args:
            path: URL path
            data: Request data
            params: Query parameters
            headers: Request headers
            timeout: Request timeout in seconds

        Returns:
            Response data
        """
        response = await self._request(method='PATCH', path=path, data=data,
            params=params, headers=headers, timeout=timeout)
        return response.json()


class FeatureStoreServiceClient(BaseServiceClient):
    """Client for the Feature Store Service."""

    def __init__(self, base_url: Optional[str]=None, timeout: float=30.0,
        max_retries: int=3, backoff_factor: float=0.5):
        """
        Initialize the Feature Store Service client.

        Args:
            base_url: Base URL for the service (defaults to the one from settings)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
        """
        super().__init__(base_url=base_url or get_feature_store_service_url
            (), service_name='feature-store-service', timeout=timeout,
            max_retries=max_retries, backoff_factor=backoff_factor)

    async def get_feature_set(self, feature_set_id: str) ->Dict[str, Any]:
        """
        Get feature set by ID.

        Args:
            feature_set_id: Feature set ID

        Returns:
            Feature set data
        """
        return await self.get(f'/feature-sets/{feature_set_id}')

    async def list_feature_sets(self, page: int=1, page_size: int=10,
        filter_by: Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        List feature sets.

        Args:
            page: Page number
            page_size: Page size
            filter_by: Filter criteria

        Returns:
            List of feature sets
        """
        params = {'page': page, 'page_size': page_size}
        if filter_by:
            params.update(filter_by)
        return await self.get('/feature-sets', params=params)

    async def create_feature_set(self, feature_set_data: Dict[str, Any]
        ) ->Dict[str, Any]:
        """
        Create a new feature set.

        Args:
            feature_set_data: Feature set data

        Returns:
            Created feature set
        """
        return await self.post('/feature-sets', data=feature_set_data)

    async def update_feature_set(self, feature_set_id: str,
        feature_set_data: Dict[str, Any]) ->Dict[str, Any]:
        """
        Update feature set.

        Args:
            feature_set_id: Feature set ID
            feature_set_data: Feature set data

        Returns:
            Updated feature set
        """
        return await self.put(f'/feature-sets/{feature_set_id}', data=
            feature_set_data)

    async def delete_feature_set(self, feature_set_id: str) ->Dict[str, Any]:
        """
        Delete feature set.

        Args:
            feature_set_id: Feature set ID

        Returns:
            Deletion status
        """
        return await self.delete(f'/feature-sets/{feature_set_id}')

    async def get_feature_values(self, feature_set_id: str, entity_ids:
        List[str], features: Optional[List[str]]=None) ->Dict[str, Any]:
        """
        Get feature values.

        Args:
            feature_set_id: Feature set ID
            entity_ids: Entity IDs
            features: Feature names (optional)

        Returns:
            Feature values
        """
        data = {'entity_ids': entity_ids}
        if features:
            data['features'] = features
        return await self.post(f'/feature-sets/{feature_set_id}/values',
            data=data)


class AnalysisEngineServiceClient(BaseServiceClient):
    """Client for the Analysis Engine Service."""

    def __init__(self, base_url: Optional[str]=None, timeout: float=30.0,
        max_retries: int=3, backoff_factor: float=0.5):
        """
        Initialize the Analysis Engine Service client.

        Args:
            base_url: Base URL for the service (defaults to the one from settings)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
        """
        super().__init__(base_url=base_url or
            get_analysis_engine_service_url(), service_name=
            'analysis-engine-service', timeout=timeout, max_retries=
            max_retries, backoff_factor=backoff_factor)

    async def get_analysis(self, analysis_id: str) ->Dict[str, Any]:
        """
        Get analysis by ID.

        Args:
            analysis_id: Analysis ID

        Returns:
            Analysis data
        """
        return await self.get(f'/analyses/{analysis_id}')

    async def list_analyses(self, page: int=1, page_size: int=10, filter_by:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        List analyses.

        Args:
            page: Page number
            page_size: Page size
            filter_by: Filter criteria

        Returns:
            List of analyses
        """
        params = {'page': page, 'page_size': page_size}
        if filter_by:
            params.update(filter_by)
        return await self.get('/analyses', params=params)

    async def create_analysis(self, analysis_data: Dict[str, Any]) ->Dict[
        str, Any]:
        """
        Create a new analysis.

        Args:
            analysis_data: Analysis data

        Returns:
            Created analysis
        """
        return await self.post('/analyses', data=analysis_data)

    async def update_analysis(self, analysis_id: str, analysis_data: Dict[
        str, Any]) ->Dict[str, Any]:
        """
        Update analysis.

        Args:
            analysis_id: Analysis ID
            analysis_data: Analysis data

        Returns:
            Updated analysis
        """
        return await self.put(f'/analyses/{analysis_id}', data=analysis_data)

    async def delete_analysis(self, analysis_id: str) ->Dict[str, Any]:
        """
        Delete analysis.

        Args:
            analysis_id: Analysis ID

        Returns:
            Deletion status
        """
        return await self.delete(f'/analyses/{analysis_id}')

    async def run_analysis(self, analysis_id: str, parameters: Optional[
        Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Run analysis.

        Args:
            analysis_id: Analysis ID
            parameters: Analysis parameters

        Returns:
            Analysis results
        """
        return await self.post(f'/analyses/{analysis_id}/run', data=
            parameters or {})


class DataPipelineServiceClient(BaseServiceClient):
    """Client for the Data Pipeline Service."""

    def __init__(self, base_url: Optional[str]=None, timeout: float=30.0,
        max_retries: int=3, backoff_factor: float=0.5):
        """
        Initialize the Data Pipeline Service client.

        Args:
            base_url: Base URL for the service (defaults to the one from settings)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
        """
        super().__init__(base_url=base_url or get_data_pipeline_service_url
            (), service_name='data-pipeline-service', timeout=timeout,
            max_retries=max_retries, backoff_factor=backoff_factor)

    async def get_pipeline(self, pipeline_id: str) ->Dict[str, Any]:
        """
        Get pipeline by ID.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Pipeline data
        """
        return await self.get(f'/pipelines/{pipeline_id}')

    async def list_pipelines(self, page: int=1, page_size: int=10,
        filter_by: Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        List pipelines.

        Args:
            page: Page number
            page_size: Page size
            filter_by: Filter criteria

        Returns:
            List of pipelines
        """
        params = {'page': page, 'page_size': page_size}
        if filter_by:
            params.update(filter_by)
        return await self.get('/pipelines', params=params)

    async def create_pipeline(self, pipeline_data: Dict[str, Any]) ->Dict[
        str, Any]:
        """
        Create a new pipeline.

        Args:
            pipeline_data: Pipeline data

        Returns:
            Created pipeline
        """
        return await self.post('/pipelines', data=pipeline_data)

    async def update_pipeline(self, pipeline_id: str, pipeline_data: Dict[
        str, Any]) ->Dict[str, Any]:
        """
        Update pipeline.

        Args:
            pipeline_id: Pipeline ID
            pipeline_data: Pipeline data

        Returns:
            Updated pipeline
        """
        return await self.put(f'/pipelines/{pipeline_id}', data=pipeline_data)

    async def delete_pipeline(self, pipeline_id: str) ->Dict[str, Any]:
        """
        Delete pipeline.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Deletion status
        """
        return await self.delete(f'/pipelines/{pipeline_id}')

    async def run_pipeline(self, pipeline_id: str, parameters: Optional[
        Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Run pipeline.

        Args:
            pipeline_id: Pipeline ID
            parameters: Pipeline parameters

        Returns:
            Pipeline execution status
        """
        return await self.post(f'/pipelines/{pipeline_id}/run', data=
            parameters or {})

    async def get_pipeline_status(self, pipeline_id: str, execution_id: str
        ) ->Dict[str, Any]:
        """
        Get pipeline execution status.

        Args:
            pipeline_id: Pipeline ID
            execution_id: Execution ID

        Returns:
            Pipeline execution status
        """
        return await self.get(
            f'/pipelines/{pipeline_id}/executions/{execution_id}')


class TradingGatewayServiceClient(BaseServiceClient):
    """Client for the Trading Gateway Service."""

    def __init__(self, base_url: Optional[str]=None, timeout: float=30.0,
        max_retries: int=3, backoff_factor: float=0.5):
        """
        Initialize the Trading Gateway Service client.

        Args:
            base_url: Base URL for the service (defaults to the one from settings)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
        """
        super().__init__(base_url=base_url or
            get_trading_gateway_service_url(), service_name=
            'trading-gateway-service', timeout=timeout, max_retries=
            max_retries, backoff_factor=backoff_factor)

    async def get_account(self, account_id: str) ->Dict[str, Any]:
        """
        Get account by ID.

        Args:
            account_id: Account ID

        Returns:
            Account data
        """
        return await self.get(f'/accounts/{account_id}')

    async def list_accounts(self, page: int=1, page_size: int=10, filter_by:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        List accounts.

        Args:
            page: Page number
            page_size: Page size
            filter_by: Filter criteria

        Returns:
            List of accounts
        """
        params = {'page': page, 'page_size': page_size}
        if filter_by:
            params.update(filter_by)
        return await self.get('/accounts', params=params)

    async def get_positions(self, account_id: str) ->Dict[str, Any]:
        """
        Get positions for an account.

        Args:
            account_id: Account ID

        Returns:
            Account positions
        """
        return await self.get(f'/accounts/{account_id}/positions')

    async def place_order(self, order_data: Dict[str, Any]) ->Dict[str, Any]:
        """
        Place a new order.

        Args:
            order_data: Order data

        Returns:
            Order status
        """
        return await self.post('/orders', data=order_data)

    async def get_order(self, order_id: str) ->Dict[str, Any]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order data
        """
        return await self.get(f'/orders/{order_id}')

    async def cancel_order(self, order_id: str) ->Dict[str, Any]:
        """
        Cancel order.

        Args:
            order_id: Order ID

        Returns:
            Cancellation status
        """
        return await self.delete(f'/orders/{order_id}')

    async def list_orders(self, account_id: Optional[str]=None, status:
        Optional[str]=None, page: int=1, page_size: int=10) ->Dict[str, Any]:
        """
        List orders.

        Args:
            account_id: Account ID
            status: Order status
            page: Page number
            page_size: Page size

        Returns:
            List of orders
        """
        params = {'page': page, 'page_size': page_size}
        if account_id:
            params['account_id'] = account_id
        if status:
            params['status'] = status
        return await self.get('/orders', params=params)


feature_store_client = FeatureStoreServiceClient()
analysis_engine_client = AnalysisEngineServiceClient()
data_pipeline_client = DataPipelineServiceClient()
trading_gateway_client = TradingGatewayServiceClient()


async def close_all_clients() ->None:
    """Close all service clients."""
    await asyncio.gather(feature_store_client.close(),
        analysis_engine_client.close(), data_pipeline_client.close(),
        trading_gateway_client.close())
