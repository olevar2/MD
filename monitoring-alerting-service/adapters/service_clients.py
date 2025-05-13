"""
Standardized Service Clients Module for Monitoring Alerting Service

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
from config.standardized_config_1 import settings, get_market_data_service_url, get_feature_store_service_url, get_analysis_engine_service_url, get_data_pipeline_service_url, get_trading_gateway_service_url, get_ml_workbench_service_url, get_prometheus_url, get_alertmanager_url, get_grafana_url
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


class PrometheusClient(BaseServiceClient):
    """Client for Prometheus API."""

    def __init__(self, base_url: Optional[str]=None, timeout: float=30.0,
        max_retries: int=3, backoff_factor: float=0.5):
        """
        Initialize the Prometheus client.

        Args:
            base_url: Base URL for the service (defaults to the one from settings)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
        """
        super().__init__(base_url=base_url or get_prometheus_url(),
            service_name='prometheus', timeout=timeout, max_retries=
            max_retries, backoff_factor=backoff_factor)

    async def query(self, query: str, time: Optional[str]=None, timeout:
        Optional[float]=None) ->Dict[str, Any]:
        """
        Execute a PromQL query.

        Args:
            query: PromQL query
            time: Evaluation timestamp
            timeout: Request timeout in seconds

        Returns:
            Query result
        """
        params = {'query': query}
        if time:
            params['time'] = time
        return await self.get('/query', params=params, timeout=timeout)

    async def query_range(self, query: str, start: str, end: str, step: str,
        timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Execute a PromQL query over a range of time.

        Args:
            query: PromQL query
            start: Start timestamp
            end: End timestamp
            step: Query resolution step width
            timeout: Request timeout in seconds

        Returns:
            Query result
        """
        params = {'query': query, 'start': start, 'end': end, 'step': step}
        return await self.get('/query_range', params=params, timeout=timeout)

    async def get_series(self, match: List[str], start: Optional[str]=None,
        end: Optional[str]=None, timeout: Optional[float]=None) ->Dict[str, Any
        ]:
        """
        Get time series that match a label set.

        Args:
            match: Series selectors
            start: Start timestamp
            end: End timestamp
            timeout: Request timeout in seconds

        Returns:
            Series result
        """
        params = {'match[]': match}
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        return await self.get('/series', params=params, timeout=timeout)

    async def get_labels(self, match: Optional[List[str]]=None, start:
        Optional[str]=None, end: Optional[str]=None, timeout: Optional[
        float]=None) ->Dict[str, Any]:
        """
        Get label names.

        Args:
            match: Series selectors
            start: Start timestamp
            end: End timestamp
            timeout: Request timeout in seconds

        Returns:
            Labels result
        """
        params = {}
        if match:
            params['match[]'] = match
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        return await self.get('/labels', params=params, timeout=timeout)

    async def get_label_values(self, label_name: str, match: Optional[List[
        str]]=None, start: Optional[str]=None, end: Optional[str]=None,
        timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Get label values.

        Args:
            label_name: Label name
            match: Series selectors
            start: Start timestamp
            end: End timestamp
            timeout: Request timeout in seconds

        Returns:
            Label values result
        """
        params = {}
        if match:
            params['match[]'] = match
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        return await self.get(f'/label/{label_name}/values', params=params,
            timeout=timeout)

    async def get_targets(self, timeout: Optional[float]=None) ->Dict[str, Any
        ]:
        """
        Get targets.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Targets result
        """
        return await self.get('/targets', timeout=timeout)

    async def get_rules(self, timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Get rules.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Rules result
        """
        return await self.get('/rules', timeout=timeout)

    async def get_alerts(self, timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Get alerts.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Alerts result
        """
        return await self.get('/alerts', timeout=timeout)

    async def get_status(self, status_type: str='config', timeout: Optional
        [float]=None) ->Dict[str, Any]:
        """
        Get status.

        Args:
            status_type: Status type (config, flags, runtime_info, build_info, TSDB)
            timeout: Request timeout in seconds

        Returns:
            Status result
        """
        return await self.get(f'/status/{status_type}', timeout=timeout)


class AlertmanagerClient(BaseServiceClient):
    """Client for Alertmanager API."""

    def __init__(self, base_url: Optional[str]=None, timeout: float=30.0,
        max_retries: int=3, backoff_factor: float=0.5):
        """
        Initialize the Alertmanager client.

        Args:
            base_url: Base URL for the service (defaults to the one from settings)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
        """
        super().__init__(base_url=base_url or get_alertmanager_url(),
            service_name='alertmanager', timeout=timeout, max_retries=
            max_retries, backoff_factor=backoff_factor)

    async def get_alerts(self, active: Optional[bool]=None, silenced:
        Optional[bool]=None, inhibited: Optional[bool]=None, filter:
        Optional[List[str]]=None, receiver: Optional[str]=None, timeout:
        Optional[float]=None) ->Dict[str, Any]:
        """
        Get alerts.

        Args:
            active: Show active alerts
            silenced: Show silenced alerts
            inhibited: Show inhibited alerts
            filter: Filter alerts by matchers
            receiver: Filter alerts by receiver
            timeout: Request timeout in seconds

        Returns:
            Alerts result
        """
        params = {}
        if active is not None:
            params['active'] = str(active).lower()
        if silenced is not None:
            params['silenced'] = str(silenced).lower()
        if inhibited is not None:
            params['inhibited'] = str(inhibited).lower()
        if filter:
            params['filter'] = filter
        if receiver:
            params['receiver'] = receiver
        return await self.get('/alerts', params=params, timeout=timeout)

    async def post_alerts(self, alerts: List[Dict[str, Any]], timeout:
        Optional[float]=None) ->Dict[str, Any]:
        """
        Post alerts.

        Args:
            alerts: Alerts to post
            timeout: Request timeout in seconds

        Returns:
            Post result
        """
        return await self.post('/alerts', data=alerts, timeout=timeout)

    async def get_silences(self, filter: Optional[List[str]]=None, timeout:
        Optional[float]=None) ->Dict[str, Any]:
        """
        Get silences.

        Args:
            filter: Filter silences by matchers
            timeout: Request timeout in seconds

        Returns:
            Silences result
        """
        params = {}
        if filter:
            params['filter'] = filter
        return await self.get('/silences', params=params, timeout=timeout)

    async def post_silence(self, silence: Dict[str, Any], timeout: Optional
        [float]=None) ->Dict[str, Any]:
        """
        Post silence.

        Args:
            silence: Silence to post
            timeout: Request timeout in seconds

        Returns:
            Post result
        """
        return await self.post('/silences', data=silence, timeout=timeout)

    async def get_silence(self, silence_id: str, timeout: Optional[float]=None
        ) ->Dict[str, Any]:
        """
        Get silence.

        Args:
            silence_id: Silence ID
            timeout: Request timeout in seconds

        Returns:
            Silence result
        """
        return await self.get(f'/silence/{silence_id}', timeout=timeout)

    async def delete_silence(self, silence_id: str, timeout: Optional[float
        ]=None) ->Dict[str, Any]:
        """
        Delete silence.

        Args:
            silence_id: Silence ID
            timeout: Request timeout in seconds

        Returns:
            Delete result
        """
        return await self.delete(f'/silence/{silence_id}', timeout=timeout)

    async def get_status(self, timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Get status.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Status result
        """
        return await self.get('/status', timeout=timeout)

    async def get_receivers(self, timeout: Optional[float]=None) ->Dict[str,
        Any]:
        """
        Get receivers.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Receivers result
        """
        return await self.get('/receivers', timeout=timeout)


class GrafanaClient(BaseServiceClient):
    """Client for Grafana API."""

    def __init__(self, base_url: Optional[str]=None, api_key: Optional[str]
        =None, timeout: float=30.0, max_retries: int=3, backoff_factor:
        float=0.5):
        """
        Initialize the Grafana client.

        Args:
            base_url: Base URL for the service (defaults to the one from settings)
            api_key: Grafana API key
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
        """
        super().__init__(base_url=base_url or get_grafana_url(),
            service_name='grafana', timeout=timeout, max_retries=
            max_retries, backoff_factor=backoff_factor)
        self.api_key = api_key or (settings.GRAFANA_API_KEY.
            get_secret_value() if settings.GRAFANA_API_KEY else None)

    def _get_headers(self, additional_headers: Optional[Dict[str, str]]=None
        ) ->Dict[str, str]:
        """
        Get request headers.

        Args:
            additional_headers: Additional headers to include

        Returns:
            Headers dictionary
        """
        headers = super()._get_headers(additional_headers)
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    async def get_dashboards(self, timeout: Optional[float]=None) ->Dict[
        str, Any]:
        """
        Get dashboards.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Dashboards result
        """
        return await self.get('/search', params={'type': 'dash-db'},
            timeout=timeout)

    async def get_dashboard(self, uid: str, timeout: Optional[float]=None
        ) ->Dict[str, Any]:
        """
        Get dashboard.

        Args:
            uid: Dashboard UID
            timeout: Request timeout in seconds

        Returns:
            Dashboard result
        """
        return await self.get(f'/dashboards/uid/{uid}', timeout=timeout)

    async def create_dashboard(self, dashboard: Dict[str, Any], overwrite:
        bool=False, timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Create dashboard.

        Args:
            dashboard: Dashboard to create
            overwrite: Whether to overwrite existing dashboard
            timeout: Request timeout in seconds

        Returns:
            Create result
        """
        data = {'dashboard': dashboard, 'overwrite': overwrite}
        return await self.post('/dashboards/db', data=data, timeout=timeout)

    async def delete_dashboard(self, uid: str, timeout: Optional[float]=None
        ) ->Dict[str, Any]:
        """
        Delete dashboard.

        Args:
            uid: Dashboard UID
            timeout: Request timeout in seconds

        Returns:
            Delete result
        """
        return await self.delete(f'/dashboards/uid/{uid}', timeout=timeout)

    async def get_datasources(self, timeout: Optional[float]=None) ->Dict[
        str, Any]:
        """
        Get datasources.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Datasources result
        """
        return await self.get('/datasources', timeout=timeout)

    async def get_datasource(self, id: Union[int, str], timeout: Optional[
        float]=None) ->Dict[str, Any]:
        """
        Get datasource.

        Args:
            id: Datasource ID or name
            timeout: Request timeout in seconds

        Returns:
            Datasource result
        """
        if isinstance(id, int):
            return await self.get(f'/datasources/{id}', timeout=timeout)
        else:
            return await self.get(f'/datasources/name/{id}', timeout=timeout)

    async def create_datasource(self, datasource: Dict[str, Any], timeout:
        Optional[float]=None) ->Dict[str, Any]:
        """
        Create datasource.

        Args:
            datasource: Datasource to create
            timeout: Request timeout in seconds

        Returns:
            Create result
        """
        return await self.post('/datasources', data=datasource, timeout=timeout
            )

    async def update_datasource(self, id: int, datasource: Dict[str, Any],
        timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Update datasource.

        Args:
            id: Datasource ID
            datasource: Datasource to update
            timeout: Request timeout in seconds

        Returns:
            Update result
        """
        return await self.put(f'/datasources/{id}', data=datasource,
            timeout=timeout)

    async def delete_datasource(self, id: Union[int, str], timeout:
        Optional[float]=None) ->Dict[str, Any]:
        """
        Delete datasource.

        Args:
            id: Datasource ID or name
            timeout: Request timeout in seconds

        Returns:
            Delete result
        """
        if isinstance(id, int):
            return await self.delete(f'/datasources/{id}', timeout=timeout)
        else:
            return await self.delete(f'/datasources/name/{id}', timeout=timeout
                )

    async def get_annotations(self, from_time: Optional[int]=None, to_time:
        Optional[int]=None, limit: Optional[int]=None, alert_id: Optional[
        int]=None, dashboard_id: Optional[int]=None, panel_id: Optional[int
        ]=None, user_id: Optional[int]=None, tags: Optional[List[str]]=None,
        type: Optional[str]=None, timeout: Optional[float]=None) ->Dict[str,
        Any]:
        """
        Get annotations.

        Args:
            from_time: From time (epoch milliseconds)
            to_time: To time (epoch milliseconds)
            limit: Limit
            alert_id: Alert ID
            dashboard_id: Dashboard ID
            panel_id: Panel ID
            user_id: User ID
            tags: Tags
            type: Type
            timeout: Request timeout in seconds

        Returns:
            Annotations result
        """
        params = {}
        if from_time is not None:
            params['from'] = from_time
        if to_time is not None:
            params['to'] = to_time
        if limit is not None:
            params['limit'] = limit
        if alert_id is not None:
            params['alertId'] = alert_id
        if dashboard_id is not None:
            params['dashboardId'] = dashboard_id
        if panel_id is not None:
            params['panelId'] = panel_id
        if user_id is not None:
            params['userId'] = user_id
        if tags:
            params['tags'] = tags
        if type:
            params['type'] = type
        return await self.get('/annotations', params=params, timeout=timeout)

    async def create_annotation(self, annotation: Dict[str, Any], timeout:
        Optional[float]=None) ->Dict[str, Any]:
        """
        Create annotation.

        Args:
            annotation: Annotation to create
            timeout: Request timeout in seconds

        Returns:
            Create result
        """
        return await self.post('/annotations', data=annotation, timeout=timeout
            )

    async def update_annotation(self, id: int, annotation: Dict[str, Any],
        timeout: Optional[float]=None) ->Dict[str, Any]:
        """
        Update annotation.

        Args:
            id: Annotation ID
            annotation: Annotation to update
            timeout: Request timeout in seconds

        Returns:
            Update result
        """
        return await self.put(f'/annotations/{id}', data=annotation,
            timeout=timeout)

    async def delete_annotation(self, id: int, timeout: Optional[float]=None
        ) ->Dict[str, Any]:
        """
        Delete annotation.

        Args:
            id: Annotation ID
            timeout: Request timeout in seconds

        Returns:
            Delete result
        """
        return await self.delete(f'/annotations/{id}', timeout=timeout)


prometheus_client = PrometheusClient()
alertmanager_client = AlertmanagerClient()
grafana_client = GrafanaClient()


async def close_all_clients() ->None:
    """Close all service clients."""
    await asyncio.gather(prometheus_client.close(), alertmanager_client.
        close(), grafana_client.close())
