"""
Test script for Monitoring Alerting Service.

This script tests the Monitoring Alerting Service to ensure it is correctly implemented and
functioning as expected.
"""
import os
import sys
import importlib
import inspect
import unittest
import asyncio
import httpx
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
import subprocess
import time
import signal
import requests
from urllib.parse import urljoin
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_monitoring_alerting_service')
SERVICE_NAME = 'Monitoring Alerting Service'
SERVICE_HOST = 'localhost'
SERVICE_PORT = 8009
SERVICE_URL = f'http://{SERVICE_HOST}:{SERVICE_PORT}'
API_PREFIX = '/api/v1'
TEST_ENDPOINTS = [{'method': 'GET', 'path': '/health', 'expected_status': 
    200}, {'method': 'GET', 'path': '/', 'expected_status': 200}, {'method':
    'GET', 'path': f'{API_PREFIX}/alerts', 'expected_status': 200}, {
    'method': 'GET', 'path': f'{API_PREFIX}/dashboards', 'expected_status':
    200}, {'method': 'GET', 'path': f'{API_PREFIX}/prometheus/targets',
    'expected_status': 200}, {'method': 'GET', 'path':
    f'{API_PREFIX}/alertmanager/status', 'expected_status': 200}, {'method':
    'GET', 'path': f'{API_PREFIX}/grafana/dashboards', 'expected_status': 
    200}, {'method': 'GET', 'path': f'{API_PREFIX}/notifications',
    'expected_status': 200}, {'method': 'GET', 'path':
    f'{API_PREFIX}/notifications/channels', 'expected_status': 200}]
service_process = None


from monitoring_alerting_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@with_exception_handling
def start_service():
    """Start the service."""
    global service_process
    logger.info(f'Starting {SERVICE_NAME}...')
    service_process = subprocess.Popen([sys.executable, '-m',
        'monitoring_alerting_service.main'], env={**os.environ, 'PORT': str
        (SERVICE_PORT), 'HOST': SERVICE_HOST, 'ENVIRONMENT': 'testing',
        'LOG_LEVEL': 'INFO'})
    logger.info(f'Waiting for {SERVICE_NAME} to start...')
    for _ in range(30):
        try:
            response = requests.get(f'{SERVICE_URL}/health')
            if response.status_code == 200:
                logger.info(f'{SERVICE_NAME} started successfully')
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    logger.error(f'Failed to start {SERVICE_NAME}')
    return False


def stop_service():
    """Stop the service."""
    global service_process
    if service_process:
        logger.info(f'Stopping {SERVICE_NAME}...')
        service_process.send_signal(signal.SIGTERM)
        service_process.wait()
        service_process = None
        logger.info(f'{SERVICE_NAME} stopped successfully')


class MonitoringAlertingServiceTestCase(unittest.TestCase):
    """Test case for Monitoring Alerting Service."""

    @classmethod
    def setUpClass(cls):
        """Set up the test case."""
        if not start_service():
            raise Exception(f'Failed to start {SERVICE_NAME}')

    @classmethod
    def tearDownClass(cls):
        """Tear down the test case."""
        stop_service()

    @with_exception_handling
    def test_endpoints(self):
        """Test all endpoints."""
        for endpoint in TEST_ENDPOINTS:
            with self.subTest(endpoint=endpoint):
                method = endpoint['method']
                path = endpoint['path']
                expected_status = endpoint['expected_status']
                logger.info(f'Testing {method} {path}...')
                url = urljoin(SERVICE_URL, path)
                response = requests.request(method, url)
                self.assertEqual(response.status_code, expected_status,
                    f'Expected status code {expected_status} but got {response.status_code} for {method} {path}'
                    )
                content_type = response.headers.get('Content-Type', '')
                self.assertTrue(content_type.startswith('application/json'),
                    f"Expected Content-Type to start with 'application/json' but got '{content_type}' for {method} {path}"
                    )
                try:
                    data = response.json()
                    if path == '/health':
                        self.assertEqual(data['status'], 'ok',
                            f"Expected status 'ok' but got '{data.get('status')}' for {method} {path}"
                            )
                    elif path.startswith(f'{API_PREFIX}/'):
                        self.assertIn('status', data,
                            f"Expected 'status' in response data for {method} {path}"
                            )
                        self.assertIn('message', data,
                            f"Expected 'message' in response data for {method} {path}"
                            )
                        self.assertIn('data', data,
                            f"Expected 'data' in response data for {method} {path}"
                            )
                except ValueError:
                    self.fail(f'Invalid JSON response for {method} {path}')

    def test_alerts(self):
        """Test alerts endpoints."""
        logger.info('Testing alert creation...')
        alert_data = {'name': 'test_alert', 'description': 'Test alert',
            'query': 'cpu_usage_percent > 90', 'severity': 'high', 'labels':
            {'service': 'trading-gateway', 'environment': 'testing'},
            'annotations': {'summary': 'High CPU usage', 'description':
            'CPU usage is above 90%'}}
        response = requests.post(urljoin(SERVICE_URL,
            f'{API_PREFIX}/alerts'), json=alert_data)
        self.assertEqual(response.status_code, 200,
            f'Expected status code 200 but got {response.status_code} for POST {API_PREFIX}/alerts'
            )
        data = response.json()
        self.assertEqual(data['status'], 'success',
            f"Expected status 'success' but got '{data.get('status')}'")
        self.assertIn('data', data, "Expected 'data' in response")
        self.assertEqual(data['data']['name'], alert_data['name'],
            f"Expected alert name '{alert_data['name']}' but got '{data['data'].get('name')}'"
            )
        self.assertEqual(data['data']['severity'], alert_data['severity'],
            f"Expected alert severity '{alert_data['severity']}' but got '{data['data'].get('severity')}'"
            )
        alert_id = data['data']['id']
        logger.info('Testing alert retrieval...')
        response = requests.get(urljoin(SERVICE_URL,
            f'{API_PREFIX}/alerts/{alert_id}'))
        self.assertEqual(response.status_code, 200,
            f'Expected status code 200 but got {response.status_code} for GET {API_PREFIX}/alerts/{alert_id}'
            )
        data = response.json()
        self.assertEqual(data['status'], 'success',
            f"Expected status 'success' but got '{data.get('status')}'")
        self.assertIn('data', data, "Expected 'data' in response")
        self.assertEqual(data['data']['id'], alert_id,
            f"Expected alert ID '{alert_id}' but got '{data['data'].get('id')}'"
            )
        self.assertEqual(data['data']['name'], alert_data['name'],
            f"Expected alert name '{alert_data['name']}' but got '{data['data'].get('name')}'"
            )
        self.assertEqual(data['data']['severity'], alert_data['severity'],
            f"Expected alert severity '{alert_data['severity']}' but got '{data['data'].get('severity')}'"
            )

    def test_dashboards(self):
        """Test dashboards endpoints."""
        logger.info('Testing dashboard creation...')
        dashboard_data = {'title': 'Test Dashboard', 'description':
            'Test dashboard', 'created_by': 'test_user', 'tags': ['test',
            'dashboard'], 'data': {'panels': [{'id': 1, 'title':
            'CPU Usage', 'type': 'graph', 'datasource': 'prometheus',
            'targets': [{'expr':
            'cpu_usage_percent{service="trading-gateway"}', 'legendFormat':
            'CPU Usage'}]}]}}
        response = requests.post(urljoin(SERVICE_URL,
            f'{API_PREFIX}/dashboards'), json=dashboard_data)
        self.assertEqual(response.status_code, 200,
            f'Expected status code 200 but got {response.status_code} for POST {API_PREFIX}/dashboards'
            )
        data = response.json()
        self.assertEqual(data['status'], 'success',
            f"Expected status 'success' but got '{data.get('status')}'")
        self.assertIn('data', data, "Expected 'data' in response")
        self.assertEqual(data['data']['title'], dashboard_data['title'],
            f"Expected dashboard title '{dashboard_data['title']}' but got '{data['data'].get('title')}'"
            )
        self.assertEqual(data['data']['created_by'], dashboard_data[
            'created_by'],
            f"Expected dashboard created_by '{dashboard_data['created_by']}' but got '{data['data'].get('created_by')}'"
            )
        dashboard_uid = data['data']['uid']
        logger.info('Testing dashboard retrieval...')
        response = requests.get(urljoin(SERVICE_URL,
            f'{API_PREFIX}/dashboards/{dashboard_uid}'))
        self.assertEqual(response.status_code, 200,
            f'Expected status code 200 but got {response.status_code} for GET {API_PREFIX}/dashboards/{dashboard_uid}'
            )
        data = response.json()
        self.assertEqual(data['status'], 'success',
            f"Expected status 'success' but got '{data.get('status')}'")
        self.assertIn('data', data, "Expected 'data' in response")
        self.assertEqual(data['data']['uid'], dashboard_uid,
            f"Expected dashboard UID '{dashboard_uid}' but got '{data['data'].get('uid')}'"
            )
        self.assertEqual(data['data']['title'], dashboard_data['title'],
            f"Expected dashboard title '{dashboard_data['title']}' but got '{data['data'].get('title')}'"
            )
        self.assertEqual(data['data']['created_by'], dashboard_data[
            'created_by'],
            f"Expected dashboard created_by '{dashboard_data['created_by']}' but got '{data['data'].get('created_by')}'"
            )

    def test_notifications(self):
        """Test notifications endpoints."""
        logger.info('Testing notification sending...')
        notification_data = {'channel': 'email', 'recipient':
            'test@example.com', 'message': 'Test notification'}
        response = requests.post(urljoin(SERVICE_URL,
            f'{API_PREFIX}/notifications/test'), json=notification_data)
        self.assertEqual(response.status_code, 200,
            f'Expected status code 200 but got {response.status_code} for POST {API_PREFIX}/notifications/test'
            )
        data = response.json()
        self.assertEqual(data['status'], 'success',
            f"Expected status 'success' but got '{data.get('status')}'")
        self.assertIn('data', data, "Expected 'data' in response")
        self.assertEqual(data['data']['channel'], notification_data[
            'channel'],
            f"Expected notification channel '{notification_data['channel']}' but got '{data['data'].get('channel')}'"
            )
        self.assertEqual(data['data']['recipient'], notification_data[
            'recipient'],
            f"Expected notification recipient '{notification_data['recipient']}' but got '{data['data'].get('recipient')}'"
            )
        self.assertEqual(data['data']['message'], notification_data[
            'message'],
            f"Expected notification message '{notification_data['message']}' but got '{data['data'].get('message')}'"
            )


def run_tests():
    """Run the tests."""
    logger.info('Running Monitoring Alerting Service tests')
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == '__main__':
    run_tests()
