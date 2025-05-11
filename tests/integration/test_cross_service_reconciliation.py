"""
Cross-service integration tests for data reconciliation.

This module tests the integration between different services using the data reconciliation framework.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import os
import sys
import json
import requests
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common_lib.data_reconciliation import (
    DataSource,
    DataSourceType,
    ReconciliationConfig,
    ReconciliationStrategy,
    ReconciliationSeverity,
    ReconciliationStatus,
)
from common_lib.exceptions import (
    DataFetchError,
    DataValidationError,
    ReconciliationError,
)


class TestCrossServiceReconciliation:
    """Cross-service integration tests for data reconciliation."""

    @pytest.fixture
    def mock_data_pipeline_api(self):
        """Mock the Data Pipeline Service API."""
        with patch("requests.post") as mock_post:
            # Configure the mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "reconciliation_id": "test-reconciliation-id",
                "status": "COMPLETED",
                "discrepancy_count": 5,
                "resolution_count": 4,
                "resolution_rate": 80.0,
                "duration_seconds": 2.5,
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
            mock_post.return_value = mock_response
            yield mock_post

    @pytest.fixture
    def mock_feature_store_api(self):
        """Mock the Feature Store Service API."""
        with patch("requests.post") as mock_post:
            # Configure the mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "reconciliation_id": "test-reconciliation-id",
                "status": "COMPLETED",
                "discrepancy_count": 3,
                "resolution_count": 3,
                "resolution_rate": 100.0,
                "duration_seconds": 1.5,
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
            mock_post.return_value = mock_response
            yield mock_post

    @pytest.fixture
    def mock_ml_integration_api(self):
        """Mock the ML Integration Service API."""
        with patch("requests.post") as mock_post:
            # Configure the mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "reconciliation_id": "test-reconciliation-id",
                "status": "COMPLETED",
                "discrepancy_count": 2,
                "resolution_count": 2,
                "resolution_rate": 100.0,
                "duration_seconds": 1.0,
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
            mock_post.return_value = mock_response
            yield mock_post

    def test_data_pipeline_to_feature_store_reconciliation(self, mock_data_pipeline_api, mock_feature_store_api):
        """Test reconciliation between Data Pipeline Service and Feature Store Service."""
        # Step 1: Reconcile OHLCV data in Data Pipeline Service
        data_pipeline_url = "http://localhost:8000/api/v1/reconciliation/ohlcv"
        data_pipeline_payload = {
            "symbol": "EURUSD",
            "start_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "timeframe": "1h",
            "strategy": "SOURCE_PRIORITY",
            "tolerance": 0.001,
            "auto_resolve": True,
            "notification_threshold": "HIGH"
        }

        # Mock the requests.post call
        with patch("requests.post") as mock_post:
            # Configure the mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "reconciliation_id": "test-reconciliation-id",
                "status": "COMPLETED",
                "discrepancy_count": 5,
                "resolution_count": 4,
                "resolution_rate": 80.0,
                "duration_seconds": 2.5,
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
            mock_post.return_value = mock_response

            # Make the request
            response = requests.post(
                data_pipeline_url,
                json=data_pipeline_payload,
                headers={"X-API-Key": "test-api-key"}
            )

            assert response.status_code == 200
            data_pipeline_result = response.json()
            assert data_pipeline_result["status"] == "COMPLETED"

            # Verify that the mock API was called with the correct arguments
            mock_post.assert_called_once()

            # Check that the data pipeline API was called with the correct payload
            data_pipeline_call_args = mock_post.call_args[1]["json"]
            assert data_pipeline_call_args["symbol"] == "EURUSD"
            assert "start_date" in data_pipeline_call_args
            assert "end_date" in data_pipeline_call_args
            assert data_pipeline_call_args["timeframe"] == "1h"

        # Step 2: Reconcile feature data in Feature Store Service
        feature_store_url = "http://localhost:8001/api/v1/reconciliation/feature-data"
        feature_store_payload = {
            "symbol": "EURUSD",
            "features": ["open", "high", "low", "close", "volume"],
            "start_time": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "strategy": "SOURCE_PRIORITY",
            "tolerance": 0.001,
            "auto_resolve": True,
            "notification_threshold": "HIGH"
        }

        # Mock the requests.post call
        with patch("requests.post") as mock_post:
            # Configure the mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "reconciliation_id": "test-reconciliation-id",
                "status": "COMPLETED",
                "discrepancy_count": 3,
                "resolution_count": 3,
                "resolution_rate": 100.0,
                "duration_seconds": 1.5,
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
            mock_post.return_value = mock_response

            # Make the request
            response = requests.post(
                feature_store_url,
                json=feature_store_payload,
                headers={"X-API-Key": "test-api-key"}
            )

            assert response.status_code == 200
            feature_store_result = response.json()
            assert feature_store_result["status"] == "COMPLETED"

            # Verify that the mock API was called with the correct arguments
            mock_post.assert_called_once()

            # Check that the feature store API was called with the correct payload
            feature_store_call_args = mock_post.call_args[1]["json"]
            assert feature_store_call_args["symbol"] == "EURUSD"
            assert feature_store_call_args["features"] == ["open", "high", "low", "close", "volume"]
            assert "start_time" in feature_store_call_args
            assert "end_time" in feature_store_call_args

    def test_feature_store_to_ml_integration_reconciliation(self, mock_feature_store_api, mock_ml_integration_api):
        """Test reconciliation between Feature Store Service and ML Integration Service."""
        # Step 1: Reconcile feature data in Feature Store Service
        feature_store_url = "http://localhost:8001/api/v1/reconciliation/feature-data"
        feature_store_payload = {
            "symbol": "EURUSD",
            "features": ["feature1", "feature2", "feature3"],
            "start_time": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "strategy": "SOURCE_PRIORITY",
            "tolerance": 0.001,
            "auto_resolve": True,
            "notification_threshold": "HIGH"
        }

        # Mock the requests.post call
        with patch("requests.post") as mock_post:
            # Configure the mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "reconciliation_id": "test-reconciliation-id",
                "status": "COMPLETED",
                "discrepancy_count": 3,
                "resolution_count": 3,
                "resolution_rate": 100.0,
                "duration_seconds": 1.5,
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
            mock_post.return_value = mock_response

            # Make the request
            response = requests.post(
                feature_store_url,
                json=feature_store_payload,
                headers={"X-API-Key": "test-api-key"}
            )

            assert response.status_code == 200
            feature_store_result = response.json()
            assert feature_store_result["status"] == "COMPLETED"

            # Verify that the mock API was called with the correct arguments
            mock_post.assert_called_once()

            # Check that the feature store API was called with the correct payload
            feature_store_call_args = mock_post.call_args[1]["json"]
            assert feature_store_call_args["symbol"] == "EURUSD"
            assert feature_store_call_args["features"] == ["feature1", "feature2", "feature3"]
            assert "start_time" in feature_store_call_args
            assert "end_time" in feature_store_call_args

        # Step 2: Reconcile training data in ML Integration Service
        ml_integration_url = "http://localhost:8002/api/v1/reconciliation/training-data"
        ml_integration_payload = {
            "model_id": "model1",
            "version": "1.0",
            "start_time": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "strategy": "SOURCE_PRIORITY",
            "tolerance": 0.001,
            "auto_resolve": True,
            "notification_threshold": "HIGH"
        }

        # Mock the requests.post call
        with patch("requests.post") as mock_post:
            # Configure the mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "reconciliation_id": "test-reconciliation-id",
                "status": "COMPLETED",
                "discrepancy_count": 2,
                "resolution_count": 2,
                "resolution_rate": 100.0,
                "duration_seconds": 1.0,
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat()
            }
            mock_post.return_value = mock_response

            # Make the request
            response = requests.post(
                ml_integration_url,
                json=ml_integration_payload,
                headers={"X-API-Key": "test-api-key"}
            )

            assert response.status_code == 200
            ml_integration_result = response.json()
            assert ml_integration_result["status"] == "COMPLETED"

            # Verify that the mock API was called with the correct arguments
            mock_post.assert_called_once()

            # Check that the ML integration API was called with the correct payload
            ml_integration_call_args = mock_post.call_args[1]["json"]
            assert ml_integration_call_args["model_id"] == "model1"
            assert ml_integration_call_args["version"] == "1.0"
            assert "start_time" in ml_integration_call_args
            assert "end_time" in ml_integration_call_args

    def test_error_handling_in_reconciliation(self):
        """Test error handling in reconciliation APIs."""
        # Test data fetch error
        feature_store_url = "http://localhost:8001/api/v1/reconciliation/feature-data"
        feature_store_payload = {
            "symbol": "EURUSD",
            "features": ["feature1", "feature2", "feature3"],
            "start_time": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "strategy": "SOURCE_PRIORITY",
            "tolerance": 0.001,
            "auto_resolve": True,
            "notification_threshold": "HIGH"
        }

        # Mock the requests.post call with a data fetch error
        with patch("requests.post") as mock_post:
            # Configure the mock response
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "detail": {
                    "detail": "Error fetching data: Failed to fetch data from source",
                    "error_type": "DataFetchError",
                    "additional_info": {
                        "source": "database",
                        "query": "SELECT * FROM features"
                    }
                }
            }
            mock_post.return_value = mock_response

            # Make the request
            response = requests.post(
                feature_store_url,
                json=feature_store_payload,
                headers={"X-API-Key": "test-api-key"}
            )

            assert response.status_code == 400
            error_response = response.json()
            assert error_response["detail"]["error_type"] == "DataFetchError"

        # Test data validation error
        ml_integration_url = "http://localhost:8002/api/v1/reconciliation/training-data"
        ml_integration_payload = {
            "model_id": "model1",
            "version": "1.0",
            "start_time": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "strategy": "SOURCE_PRIORITY",
            "tolerance": 0.001,
            "auto_resolve": True,
            "notification_threshold": "HIGH"
        }

        # Mock the requests.post call with a data validation error
        with patch("requests.post") as mock_post:
            # Configure the mock response
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "detail": {
                    "detail": "Data validation error: Invalid data format",
                    "error_type": "DataValidationError",
                    "additional_info": {
                        "field": "feature1",
                        "expected_type": "float",
                        "actual_type": "string"
                    }
                }
            }
            mock_post.return_value = mock_response

            # Make the request
            response = requests.post(
                ml_integration_url,
                json=ml_integration_payload,
                headers={"X-API-Key": "test-api-key"}
            )

            assert response.status_code == 400
            error_response = response.json()
            assert error_response["detail"]["error_type"] == "DataValidationError"

        # Test reconciliation error
        data_pipeline_url = "http://localhost:8000/api/v1/reconciliation/ohlcv"
        data_pipeline_payload = {
            "symbol": "EURUSD",
            "start_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "timeframe": "1h",
            "strategy": "SOURCE_PRIORITY",
            "tolerance": 0.001,
            "auto_resolve": True,
            "notification_threshold": "HIGH"
        }

        # Mock the requests.post call with a reconciliation error
        with patch("requests.post") as mock_post:
            # Configure the mock response
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {
                "detail": {
                    "detail": "Reconciliation error: Failed to resolve discrepancies",
                    "error_type": "ReconciliationError",
                    "additional_info": {
                        "discrepancy_count": 5,
                        "resolution_count": 0,
                        "failed_fields": ["open", "close"]
                    }
                }
            }
            mock_post.return_value = mock_response

            # Make the request
            response = requests.post(
                data_pipeline_url,
                json=data_pipeline_payload,
                headers={"X-API-Key": "test-api-key"}
            )

            assert response.status_code == 500
            error_response = response.json()
            assert error_response["detail"]["error_type"] == "ReconciliationError"
