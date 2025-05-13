"""
Test script for ML Workbench Service.

This script tests the ML Workbench Service to ensure it is correctly implemented and
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
logger = logging.getLogger('test_ml_workbench_service')
SERVICE_NAME = 'ML Workbench Service'
SERVICE_HOST = 'localhost'
SERVICE_PORT = 8030
SERVICE_URL = f'http://{SERVICE_HOST}:{SERVICE_PORT}'
API_PREFIX = '/api/v1'
TEST_ENDPOINTS = [{'method': 'GET', 'path': '/health', 'expected_status': 
    200}, {'method': 'GET', 'path': '/', 'expected_status': 200}, {'method':
    'GET', 'path': f'{API_PREFIX}/model-registry/models', 'expected_status':
    200}, {'method': 'GET', 'path': f'{API_PREFIX}/model-training/jobs',
    'expected_status': 200}, {'method': 'GET', 'path':
    f'{API_PREFIX}/model-serving/endpoints', 'expected_status': 200}, {
    'method': 'GET', 'path': f'{API_PREFIX}/model-monitoring/metrics',
    'expected_status': 200}, {'method': 'GET', 'path':
    f'{API_PREFIX}/transfer-learning/jobs', 'expected_status': 200}]
service_process = None


from ml_workbench_service.error.exceptions_bridge import (
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
        'ml_workbench_service.main'], env={**os.environ, 'PORT': str(
        SERVICE_PORT), 'HOST': SERVICE_HOST, 'ENVIRONMENT': 'testing',
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


class MLWorkbenchServiceTestCase(unittest.TestCase):
    """Test case for ML Workbench Service."""

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

    def test_model_registry(self):
        """Test model registry endpoints."""
        logger.info('Testing model creation...')
        model_data = {'name': 'test_model', 'version': '1.0.0',
            'description': 'Test model', 'framework': 'TensorFlow',
            'input_schema': {'features': ['price_open', 'price_high',
            'price_low', 'price_close', 'volume']}, 'output_schema': {
            'prediction': 'float'}, 'metrics': {'accuracy': 0.85,
            'precision': 0.82, 'recall': 0.79, 'f1_score': 0.8}, 'tags': [
            'test', 'tensorflow'], 'created_by': 'test_user'}
        response = requests.post(urljoin(SERVICE_URL,
            f'{API_PREFIX}/model-registry/models'), json=model_data)
        self.assertEqual(response.status_code, 200,
            f'Expected status code 200 but got {response.status_code} for POST {API_PREFIX}/model-registry/models'
            )
        data = response.json()
        self.assertEqual(data['status'], 'success',
            f"Expected status 'success' but got '{data.get('status')}'")
        self.assertIn('data', data, "Expected 'data' in response")
        self.assertEqual(data['data']['name'], model_data['name'],
            f"Expected model name '{model_data['name']}' but got '{data['data'].get('name')}'"
            )
        self.assertEqual(data['data']['version'], model_data['version'],
            f"Expected model version '{model_data['version']}' but got '{data['data'].get('version')}'"
            )
        logger.info('Testing model retrieval...')
        response = requests.get(urljoin(SERVICE_URL,
            f"{API_PREFIX}/model-registry/models/{model_data['name']}/{model_data['version']}"
            ))
        self.assertEqual(response.status_code, 200,
            f"Expected status code 200 but got {response.status_code} for GET {API_PREFIX}/model-registry/models/{model_data['name']}/{model_data['version']}"
            )
        data = response.json()
        self.assertEqual(data['status'], 'success',
            f"Expected status 'success' but got '{data.get('status')}'")
        self.assertIn('data', data, "Expected 'data' in response")
        self.assertEqual(data['data']['name'], model_data['name'],
            f"Expected model name '{model_data['name']}' but got '{data['data'].get('name')}'"
            )
        self.assertEqual(data['data']['version'], model_data['version'],
            f"Expected model version '{model_data['version']}' but got '{data['data'].get('version')}'"
            )

    def test_model_training(self):
        """Test model training endpoints."""
        logger.info('Testing training job creation...')
        job_data = {'model_name': 'test_model', 'model_version': '1.0.0',
            'dataset_id': 'test_dataset', 'hyperparameters': {
            'learning_rate': 0.001, 'batch_size': 32, 'epochs': 100,
            'optimizer': 'adam'}, 'created_by': 'test_user'}
        response = requests.post(urljoin(SERVICE_URL,
            f'{API_PREFIX}/model-training/jobs'), json=job_data)
        self.assertEqual(response.status_code, 200,
            f'Expected status code 200 but got {response.status_code} for POST {API_PREFIX}/model-training/jobs'
            )
        data = response.json()
        self.assertEqual(data['status'], 'success',
            f"Expected status 'success' but got '{data.get('status')}'")
        self.assertIn('data', data, "Expected 'data' in response")
        self.assertEqual(data['data']['model_name'], job_data['model_name'],
            f"Expected model name '{job_data['model_name']}' but got '{data['data'].get('model_name')}'"
            )
        self.assertEqual(data['data']['model_version'], job_data[
            'model_version'],
            f"Expected model version '{job_data['model_version']}' but got '{data['data'].get('model_version')}'"
            )
        self.assertEqual(data['data']['dataset_id'], job_data['dataset_id'],
            f"Expected dataset ID '{job_data['dataset_id']}' but got '{data['data'].get('dataset_id')}'"
            )
        job_id = data['data']['job_id']
        logger.info('Testing training job retrieval...')
        response = requests.get(urljoin(SERVICE_URL,
            f'{API_PREFIX}/model-training/jobs/{job_id}'))
        self.assertEqual(response.status_code, 200,
            f'Expected status code 200 but got {response.status_code} for GET {API_PREFIX}/model-training/jobs/{job_id}'
            )
        data = response.json()
        self.assertEqual(data['status'], 'success',
            f"Expected status 'success' but got '{data.get('status')}'")
        self.assertIn('data', data, "Expected 'data' in response")
        self.assertEqual(data['data']['job_id'], job_id,
            f"Expected job ID '{job_id}' but got '{data['data'].get('job_id')}'"
            )
        self.assertEqual(data['data']['model_name'], job_data['model_name'],
            f"Expected model name '{job_data['model_name']}' but got '{data['data'].get('model_name')}'"
            )
        self.assertEqual(data['data']['model_version'], job_data[
            'model_version'],
            f"Expected model version '{job_data['model_version']}' but got '{data['data'].get('model_version')}'"
            )
        self.assertEqual(data['data']['dataset_id'], job_data['dataset_id'],
            f"Expected dataset ID '{job_data['dataset_id']}' but got '{data['data'].get('dataset_id')}'"
            )


def run_tests():
    """Run the tests."""
    logger.info('Running ML Workbench Service tests')
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == '__main__':
    run_tests()
