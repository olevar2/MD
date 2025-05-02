"""
Tests for ML Pipeline Client

This module contains tests for the MLPipelineClient class to verify its functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import Response
import json

from analysis_engine.clients.ml_pipeline_client import (
    MLPipelineClient, MLClientError, MLJobSubmissionError, MLClientConnectionError
)


class TestMLPipelineClient:
    """Test suite for MLPipelineClient class"""
    
    @pytest.fixture
    def mock_config_manager(self):
        config_manager = MagicMock()
        config_manager.get_section.return_value = {
            "base_url": "http://test-ml-service:9000",
            "timeout": 10.0,
            "max_retries": 2,
        }
        return config_manager

    @pytest.fixture
    def client(self, mock_config_manager):
        return MLPipelineClient(config_manager=mock_config_manager)

    @pytest.mark.asyncio
    async def test_init_configuration(self, mock_config_manager, client):
        """Test client initialization and configuration loading"""
        assert client.base_url == "http://test-ml-service:9000"
        assert client.timeout == 10.0
        assert client.max_attempts == 2
        assert mock_config_manager.get_section.called_with("ml_pipeline_client")

    @pytest.mark.asyncio
    async def test_start_retraining_job_success(self, client):
        """Test successful retraining job submission"""
        # Mock the _make_request method to return a successful response
        client._make_request = AsyncMock(return_value={"job_id": "job-123"})
        
        # Call the method with test parameters
        job_id = await client.start_retraining_job(
            model_id="test-model",
            params={"epochs": 10}
        )
        
        # Verify the result and method call
        assert job_id == "job-123"
        client._make_request.assert_called_once_with(
            "POST",
            "/jobs/retrain/test-model",
            data={"epochs": 10}
        )

    @pytest.mark.asyncio
    async def test_start_retraining_job_error(self, client):
        """Test error handling in retraining job submission"""
        # Mock the _make_request method to raise an exception
        client._make_request = AsyncMock(side_effect=MLJobSubmissionError("Submission failed"))
        
        # Verify the exception is propagated
        with pytest.raises(MLJobSubmissionError, match="Submission failed"):
            await client.start_retraining_job(model_id="test-model")

    @pytest.mark.asyncio
    async def test_get_job_status(self, client):
        """Test fetching job status"""
        # Mock the _make_request method
        mock_status = {"status": "running", "progress": 75}
        client._make_request = AsyncMock(return_value=mock_status)
        
        # Call the method
        status = await client.get_job_status("job-123")
        
        # Verify the result and method call
        assert status == mock_status
        client._make_request.assert_called_once_with("GET", "/jobs/status/job-123")

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        """Test listing available models"""
        # Mock the _make_request method
        mock_models = [
            {"id": "model-1", "name": "Forex Prediction Model", "version": "1.0"},
            {"id": "model-2", "name": "Sentiment Analysis Model", "version": "2.1"}
        ]
        client._make_request = AsyncMock(return_value=mock_models)
        
        # Call the method
        models = await client.list_models()
        
        # Verify the result and method call
        assert len(models) == 2
        assert models[0]["id"] == "model-1"
        assert models[1]["name"] == "Sentiment Analysis Model"
        client._make_request.assert_called_once_with("GET", "/models")

    @pytest.mark.asyncio
    async def test_list_models_invalid_response(self, client):
        """Test handling of invalid response format when listing models"""
        # Mock the _make_request method to return a non-list response
        client._make_request = AsyncMock(return_value={"error": "Invalid format"})
        
        # Verify the exception is raised for invalid response format
        with pytest.raises(MLClientError, match="Invalid response format"):
            await client.list_models()

    @pytest.mark.asyncio
    async def test_get_prediction(self, client):
        """Test model prediction request"""
        # Mock the _make_request method
        mock_prediction = {"prediction": 0.75, "confidence": 0.92}
        client._make_request = AsyncMock(return_value=mock_prediction)
        
        # Prepare test data
        features = {"feature1": 0.5, "feature2": 1.0}
        
        # Call the method
        result = await client.get_prediction("model-1", features, version="2.0")
        
        # Verify metadata was added to the response
        assert "metadata" in result
        assert result["metadata"]["model_id"] == "model-1"
        assert result["metadata"]["version"] == "2.0"
        assert "latency_ms" in result["metadata"]
        
        # Verify the correct endpoint and payload
        client._make_request.assert_called_once()
        args = client._make_request.call_args
        assert args[0][0] == "POST" # Method
        assert args[0][1] == "/models/model-1/versions/2.0/predict" # Endpoint
        assert args[1]["data"] == {"features": features} # Payload

    @pytest.mark.asyncio
    async def test_make_request_handling(self, client):
        """Test the _make_request method directly"""
        # Create a patch for httpx.AsyncClient
        with patch("httpx.AsyncClient") as mock_client:
            # Configure the mock client's response
            mock_response = Response(200, content=json.dumps({"result": "success"}).encode())
            mock_response.json = MagicMock(return_value={"result": "success"})
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            # Call the method
            result = await client._make_request("GET", "/test/endpoint")
            
            # Verify the result
            assert result == {"result": "success"}
            
            # Verify circuit breaker record_success was called
            assert client.circuit_breaker.record_success.called


# To run these tests, use:
# pytest -xvs tests/test_ml_pipeline_client.py
