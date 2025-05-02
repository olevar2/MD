"""
Test suite for the automated model retraining system.

These tests verify that the model retraining system correctly processes feedback,
determines when retraining is needed, and properly executes the retraining workflow.
They also test the MLPipelineClient for proper error handling and job submission.
"""

import unittest
import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json
from datetime import datetime, timedelta
import os
import tempfile
import uuid

from core_foundations.models.feedback import (
    TradeFeedback, FeedbackCategory, FeedbackSource, FeedbackStatus
)
from analysis_engine.adaptive_layer.model_retraining_service import (
    FeedbackClassifier, ModelPerformanceEvaluator, TrainingPipelineIntegrator
)
from analysis_engine.clients.ml_pipeline_client import MLPipelineClient
from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from core_foundations.exceptions.client_exceptions import (
    MLClientConnectionError, MLJobSubmissionError
)
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.events.event_schema import Event, EventType


class TestModelRetrainingService(unittest.TestCase):
    """Test cases for the ModelRetrainingService."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock dependencies
        self.model_trainer = Mock()
        self.feedback_repository = Mock()
        
        # Configure the mock trainer
        self.model_trainer.retrain_model.return_value = {
            "status": "success",
            "model_version": "1.1",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.92
            }
        }
        self.model_trainer.evaluate_feedback_impact.return_value = {
            "estimated_improvement": 0.03,
            "confidence": 0.85,
            "recommendation": "retrain"
        }
        
        # Configure the mock repository
        self.feedback_repository.get_prioritized_feedback_since.return_value = self._create_test_feedback_items()
        self.feedback_repository.create_feedback_batch.return_value = "batch-123"
        
        # Service configuration
        self.config = {
            'retraining_threshold': 10,
            'feedback_priority_trigger': 'HIGH',
            'statistical_significance_threshold': 0.7,
            'lookback_days': 7,
            'evaluate_impact_before_retraining': True,
            'min_improvement_threshold': 0.01
        }
        
        # Create the service under test
        self.service = ModelRetrainingService(
            model_trainer=self.model_trainer,
            feedback_repository=self.feedback_repository,
            config=self.config
        )
    
    def _create_test_feedback_items(self):
        """Helper to create test feedback items."""
        items = []
        
        # Regular feedback item
        items.append(ClassifiedFeedback(
            feedback_id="feedback-1",
            category=FeedbackCategory.INCORRECT_PREDICTION,
            priority=FeedbackPriority.MEDIUM,
            source=FeedbackSource.SYSTEM_AUTO,
            status=FeedbackStatus.VALIDATED,
            model_id="model-123",
            statistical_significance=0.75,
            content={"prediction_error": 0.15}
        ))
        
        # High priority feedback
        items.append(ClassifiedFeedback(
            feedback_id="feedback-2",
            category=FeedbackCategory.MARKET_SHIFT,
            priority=FeedbackPriority.HIGH,
            source=FeedbackSource.PERFORMANCE_METRICS,
            status=FeedbackStatus.VALIDATED,
            model_id="model-123",
            statistical_significance=0.85,
            content={"prediction_error": 0.25}
        ))
        
        # Timeframe feedback
        items.append(TimeframeFeedback(
            feedback_id="feedback-3",
            timeframe="1h",
            category=FeedbackCategory.TIMEFRAME_ADJUSTMENT,
            priority=FeedbackPriority.MEDIUM,
            source=FeedbackSource.PERFORMANCE_METRICS,
            status=FeedbackStatus.VALIDATED,
            model_id="model-123",
            statistical_significance=0.65,
            content={"prediction_error": 0.2}
        ))
        
        return items
    
    def test_check_and_trigger_retraining_with_high_priority_feedback(self):
        """Test that high priority feedback triggers retraining."""
        # Create special high priority feedback
        high_priority_items = [
            ClassifiedFeedback(
                feedback_id="feedback-hp",
                category=FeedbackCategory.INCORRECT_PREDICTION,
                priority=FeedbackPriority.HIGH,
                source=FeedbackSource.SYSTEM_AUTO,
                status=FeedbackStatus.VALIDATED,
                model_id="model-123",
                statistical_significance=0.9
            )
        ]
        
        # Configure repository to return high priority feedback
        self.feedback_repository.get_prioritized_feedback_since.return_value = high_priority_items
        
        # Execute the method under test
        result = self.service.check_and_trigger_retraining("model-123")
        
        # Verify results
        self.assertTrue(result)
        self.model_trainer.retrain_model.assert_called_once()
        self.feedback_repository.mark_batch_processed.assert_called_once()
    
    def test_check_and_trigger_retraining_with_volume_threshold(self):
        """Test that volume threshold triggers retraining."""
        # Create many feedback items to exceed volume threshold
        many_items = []
        for i in range(20):  # Above our threshold of 10
            many_items.append(
                ClassifiedFeedback(
                    feedback_id=f"feedback-vol-{i}",
                    category=FeedbackCategory.INCORRECT_PREDICTION,
                    priority=FeedbackPriority.LOW,
                    status=FeedbackStatus.VALIDATED,
                    model_id="model-123",
                    statistical_significance=0.5
                )
            )
        
        # Configure repository to return volume feedback
        self.feedback_repository.get_prioritized_feedback_since.return_value = many_items
        
        # Execute the method under test
        result = self.service.check_and_trigger_retraining("model-123")
        
        # Verify results
        self.assertTrue(result)
        self.model_trainer.retrain_model.assert_called_once()
    
    def test_check_and_trigger_retraining_with_significance_trigger(self):
        """Test that statistical significance triggers retraining."""
        # Create feedback with high statistical significance
        significant_items = [
            ClassifiedFeedback(
                feedback_id="feedback-sig",
                category=FeedbackCategory.INCORRECT_PREDICTION,
                priority=FeedbackPriority.MEDIUM,
                status=FeedbackStatus.VALIDATED,
                model_id="model-123",
                statistical_significance=0.95  # Very significant
            )
        ]
        
        # Configure repository
        self.feedback_repository.get_prioritized_feedback_since.return_value = significant_items
        
        # Execute the method under test
        result = self.service.check_and_trigger_retraining("model-123")
        
        # Verify results
        self.assertTrue(result)
        self.model_trainer.retrain_model.assert_called_once()
    
    def test_no_retraining_when_no_feedback(self):
        """Test that no retraining is triggered when no feedback is found."""
        # Configure repository to return no feedback
        self.feedback_repository.get_prioritized_feedback_since.return_value = []
        
        # Execute the method under test
        result = self.service.check_and_trigger_retraining("model-123")
        
        # Verify results
        self.assertFalse(result)
        self.model_trainer.retrain_model.assert_not_called()
    
    def test_no_retraining_when_impact_below_threshold(self):
        """Test that no retraining is triggered when expected impact is low."""
        # Configure trainer to predict low impact
        self.model_trainer.evaluate_feedback_impact.return_value = {
            "estimated_improvement": 0.001,  # Below threshold of 0.01
            "confidence": 0.85
        }
        
        # Execute the method under test
        result = self.service.check_and_trigger_retraining("model-123")
        
        # Verify results
        self.assertFalse(result)
        self.model_trainer.retrain_model.assert_not_called()
    
    def test_timeframe_feedback_integration(self):
        """Test that timeframe feedback is properly prepared for training."""
        # Only provide timeframe feedback
        timeframe_items = [
            TimeframeFeedback(
                feedback_id="tf-feedback-1",
                timeframe="1h",
                related_timeframes=["4h", "1d"],
                temporal_correlation_data={"1h_4h": 0.8, "1h_1d": -0.3},
                category=FeedbackCategory.TIMEFRAME_ADJUSTMENT,
                priority=FeedbackPriority.HIGH,
                model_id="model-123",
                statistical_significance=0.85
            ),
            TimeframeFeedback(
                feedback_id="tf-feedback-2",
                timeframe="4h",
                category=FeedbackCategory.TIMEFRAME_ADJUSTMENT,
                priority=FeedbackPriority.MEDIUM,
                model_id="model-123",
                statistical_significance=0.75
            )
        ]
        
        # Configure repository
        self.feedback_repository.get_prioritized_feedback_since.return_value = timeframe_items
        
        # Execute the method under test
        result = self.service.check_and_trigger_retraining("model-123")
        
        # Verify results
        self.assertTrue(result)
        
        # Capture the training data that was prepared
        call_kwargs = self.model_trainer.retrain_model.call_args.kwargs
        training_data = call_kwargs.get('feedback_data')
        
        # Verify timeframe-specific data was included
        self.assertIn('timeframe_analysis', training_data)
        self.assertEqual(2, len(training_data['timeframe_analysis']['feedback_items']))
        self.assertIn('1h', training_data['timeframe_analysis']['timeframes'])
        self.assertIn('4h', training_data['timeframe_analysis']['timeframes'])
    
    def test_error_handling_during_retraining(self):
        """Test that errors during retraining are properly handled."""
        # Configure trainer to raise an exception
        self.model_trainer.retrain_model.side_effect = Exception("Training failure")
        
        # Execute the method under test
        result = self.service.check_and_trigger_retraining("model-123")
        
        # Verify results
        self.assertFalse(result)
        self.model_trainer.retrain_model.assert_called_once()
        self.feedback_repository.mark_batch_processed.assert_not_called()


@pytest.mark.asyncio
class TestMLPipelineClient:
    """
    Unit tests for the MLPipelineClient class to verify:
    - Proper job submission
    - Error handling
    - Retries and circuit breaker functionality
    - Status polling
    """

    @pytest.fixture
    async def ml_client(self):
        """Create an MLPipelineClient with mocked dependencies"""
        config_manager = MagicMock()
        config_manager.get_config.return_value = {
            "base_url": "http://test-ml-service:8000",
            "timeout": 5.0,
            "max_retries": 2
        }
        client = MLPipelineClient(config_manager=config_manager)
        # Replace _make_request with a mock to avoid actual HTTP calls
        client._make_request = AsyncMock()
        return client
        
    @pytest.mark.asyncio
    async def test_start_retraining_job_success(self, ml_client):
        """Test successful job submission"""
        # Arrange
        ml_client._make_request.return_value = {"job_id": "test-job-123"}
        
        # Act
        job_id = await ml_client.start_retraining_job("test-model-1", {"param1": "value1"})
        
        # Assert
        assert job_id == "test-job-123"
        ml_client._make_request.assert_called_once()
        # Verify the request had correct method, endpoint and data structure
        args, kwargs = ml_client._make_request.call_args
        assert args[0] == "POST"
        assert args[1] == "/api/v1/ml/jobs/retrain"
        assert "model_id" in kwargs["data"]
        assert kwargs["data"]["model_id"] == "test-model-1"
        assert "parameters" in kwargs["data"]
        assert kwargs["data"]["parameters"] == {"param1": "value1"}

    @pytest.mark.asyncio
    async def test_start_retraining_job_connection_error(self, ml_client):
        """Test handling of connection errors"""
        # Arrange
        ml_client._make_request.side_effect = MLClientConnectionError("Connection failed")
        
        # Act & Assert
        with pytest.raises(MLJobSubmissionError):
            await ml_client.start_retraining_job("test-model-1", {"param1": "value1"})
        
        # Verify retry attempts based on config (2 retries = 3 total attempts)
        assert ml_client._make_request.call_count == 3
        
    @pytest.mark.asyncio
    async def test_get_job_status(self, ml_client):
        """Test fetching job status"""
        # Arrange
        expected_status = {
            "job_id": "test-job-123",
            "status": "running",
            "progress": 45,
            "timestamp": datetime.utcnow().isoformat()
        }
        ml_client._make_request.return_value = expected_status
        
        # Act
        status = await ml_client.get_job_status("test-job-123")
        
        # Assert
        assert status == expected_status
        ml_client._make_request.assert_called_once_with(
            "GET", "/api/v1/ml/jobs/test-job-123"
        )
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, ml_client):
        """Test that circuit breaker opens after multiple failures"""
        # Arrange
        ml_client.circuit_breaker.failure_threshold = 3
        ml_client._make_request.side_effect = MLClientConnectionError("Connection error")
        
        # Act - trigger failures to open circuit breaker
        for _ in range(3):
            try:
                await ml_client.start_retraining_job("test-model", {})
            except:
                pass
        
        # Now attempt another call that should be blocked by open circuit breaker
        with pytest.raises(MLClientConnectionError) as exc_info:
            await ml_client.start_retraining_job("test-model", {})
            
        # Assert
        assert "Circuit breaker is open" in str(exc_info.value)
        # The _make_request should not have been called on the last attempt
        assert ml_client._make_request.call_count == 9  # 3 attempts × 3 failures
        
    @pytest.mark.asyncio
    async def test_wait_for_job_completion_success(self, ml_client):
        """Test waiting for job completion with successful outcome"""
        # Arrange
        running_status = {"status": "running", "progress": 50}
        completed_status = {"status": "completed", "result": "success"}
        
        # First call returns running, second call returns completed
        ml_client.get_job_status = AsyncMock()
        ml_client.get_job_status.side_effect = [running_status, completed_status]
        
        # Act
        result = await ml_client.wait_for_job_completion("test-job", poll_interval=0.01, timeout=1.0)
        
        # Assert
        assert result == completed_status
        assert ml_client.get_job_status.call_count == 2
        
    @pytest.mark.asyncio
    async def test_wait_for_job_completion_failure(self, ml_client):
        """Test waiting for job completion with failure outcome"""
        # Arrange
        running_status = {"status": "running", "progress": 50}
        failed_status = {"status": "failed", "error": "Model training failed"}
        
        ml_client.get_job_status = AsyncMock()
        ml_client.get_job_status.side_effect = [running_status, failed_status]
        
        # Act & Assert
        with pytest.raises(MLJobSubmissionError) as exc_info:
            await ml_client.wait_for_job_completion("test-job", poll_interval=0.01, timeout=1.0)
        
        assert "Model training failed" in str(exc_info.value)
        assert ml_client.get_job_status.call_count == 2


@pytest.mark.asyncio
class TestExecutionEngineClient:
    """
    Unit tests for the ExecutionEngineClient class to verify:
    - Parameter updates
    - Strategy deployment
    - Error handling
    - Status checking
    """
    
    @pytest.fixture
    async def exec_client(self):
        """Create an ExecutionEngineClient with mocked dependencies"""
        from analysis_engine.clients.execution_engine_client import ExecutionEngineClient
        from core_foundations.exceptions.client_exceptions import (
            ExecutionEngineConnectionError,
            StrategyDeploymentError,
            StrategyParameterUpdateError
        )
        
        config_manager = MagicMock()
        config_manager.get_config.return_value = {
            "base_url": "http://test-execution-engine:8080",
            "timeout": 5.0,
            "max_retries": 2
        }
        client = ExecutionEngineClient(config_manager=config_manager)
        # Replace _make_request with a mock to avoid actual HTTP calls
        client._make_request = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_set_strategy_parameter_success(self, exec_client):
        """Test successful parameter update"""
        # Arrange
        exec_client._make_request.return_value = {"success": True, "message": "Parameter updated"}
        
        # Act
        result = await exec_client.set_strategy_parameter("strategy-123", "risk_level", 0.5)
        
        # Assert
        assert result is True
        exec_client._make_request.assert_called_once()
        args, kwargs = exec_client._make_request.call_args
        assert args[0] == "PUT"
        assert args[1] == "/api/v1/strategies/strategy-123/parameters"
        assert kwargs["data"]["parameter_name"] == "risk_level"
        assert kwargs["data"]["value"] == 0.5
    
    @pytest.mark.asyncio
    async def test_set_strategy_parameter_failure(self, exec_client):
        """Test parameter update failure"""
        # Arrange
        from analysis_engine.clients.execution_engine_client import StrategyParameterUpdateError
        exec_client._make_request.side_effect = StrategyParameterUpdateError("Invalid parameter")
        
        # Act & Assert
        with pytest.raises(StrategyParameterUpdateError) as exc_info:
            await exec_client.set_strategy_parameter("strategy-123", "invalid_param", 999)
        
        assert "Invalid parameter" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_deploy_strategy_success(self, exec_client):
        """Test successful strategy deployment"""
        # Arrange
        exec_client._make_request.return_value = {
            "success": True,
            "deployment_id": "deploy-456",
            "message": "Strategy deployed"
        }
        
        strategy_config = {
            "id": "strategy-123",
            "version": "1.2",
            "parameters": {
                "entry_threshold": 1.5,
                "exit_threshold": 0.8
            }
        }
        
        # Act
        result = await exec_client.deploy_strategy("strategy-123", strategy_config)
        
        # Assert
        assert result["success"] is True
        assert "deployment_id" in result
        exec_client._make_request.assert_called_once()
        args, kwargs = exec_client._make_request.call_args
        assert args[0] == "POST"
        assert args[1] == "/api/v1/strategies/deploy"
        assert kwargs["data"]["strategy_id"] == "strategy-123"
        assert kwargs["data"]["config"] == strategy_config
    
    @pytest.mark.asyncio
    async def test_deploy_strategy_failure(self, exec_client):
        """Test strategy deployment failure"""
        # Arrange
        from analysis_engine.clients.execution_engine_client import StrategyDeploymentError
        exec_client._make_request.side_effect = StrategyDeploymentError("Invalid configuration")
        
        strategy_config = {
            "id": "strategy-123",
            "version": "invalid",
        }
        
        # Act & Assert
        with pytest.raises(StrategyDeploymentError) as exc_info:
            await exec_client.deploy_strategy("strategy-123", strategy_config)
            
        assert "Invalid configuration" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_strategy_status(self, exec_client):
        """Test getting strategy status"""
        # Arrange
        expected_status = {
            "strategy_id": "strategy-123",
            "status": "active",
            "last_updated": datetime.utcnow().isoformat(),
            "performance_metrics": {
                "sharpe_ratio": 1.5,
                "win_rate": 0.6
            }
        }
        exec_client._make_request.return_value = expected_status
        
        # Act
        status = await exec_client.get_strategy_status("strategy-123")
        
        # Assert
        assert status == expected_status
        exec_client._make_request.assert_called_once_with(
            "GET", "/api/v1/strategies/strategy-123/status"
        )
    
    @pytest.mark.asyncio
    async def test_list_strategies(self, exec_client):
        """Test listing strategies with filters"""
        # Arrange
        expected_strategies = {
            "strategies": [
                {"id": "strategy-1", "status": "active", "asset_class": "forex"},
                {"id": "strategy-2", "status": "active", "asset_class": "forex"}
            ]
        }
        exec_client._make_request.return_value = expected_strategies
        
        # Act
        strategies = await exec_client.list_strategies(status="active", asset_class="forex")
        
        # Assert
        assert len(strategies) == 2
        assert strategies[0]["id"] == "strategy-1"
        exec_client._make_request.assert_called_once_with(
            "GET", "/api/v1/strategies?status=active&asset_class=forex"
        )


@pytest.mark.asyncio
class TestTrainingPipelineIntegrator:
    """
    Tests for TrainingPipelineIntegrator that coordinates model retraining based on feedback
    """
    
    @pytest.fixture
    async def training_pipeline(self):
        """Create TrainingPipelineIntegrator with mocked dependencies"""
        # Mock dependencies
        adaptation_engine = AsyncMock()
        event_publisher = AsyncMock()
        
        # Create the training pipeline integrator
        integrator = TrainingPipelineIntegrator(
            event_publisher=event_publisher,
            adaptation_engine=adaptation_engine,
            config={
                "job_poll_interval_seconds": 0.01,  # Use small interval for tests
                "max_job_polls": 5
            }
        )
        
        # Mock the prepare_feedback_for_training method
        integrator.prepare_feedback_for_training = AsyncMock(return_value={
            "feedback_summary": {"priority": "high", "confidence": 0.9},
            "training_data_hints": {"suggested_data": "latest_forex_data"}
        })
        
        return integrator
    
    @pytest.mark.asyncio
    async def test_trigger_model_retraining_success(self, training_pipeline):
        """Test successful model retraining workflow"""
        # Arrange
        model_id = "forex-prediction-model-1"
        job_id = str(uuid.uuid4())
        
        # Mock the adaptation engine's trigger_model_retraining method
        training_pipeline.adaptation_engine.trigger_model_retraining = AsyncMock(return_value=job_id)
        
        # Mock feedback items
        feedback_items = [
            MagicMock(spec=TradeFeedback, 
                     id="feedback-1",
                     model_id=model_id,
                     source=FeedbackSource.STRATEGY_EXECUTION,
                     category=FeedbackCategory.PERFORMANCE)
        ]
        
        # Act
        result = await training_pipeline.trigger_model_retraining(model_id, feedback_items)
        
        # Assert
        assert result["job_id"] == job_id
        assert result["model_id"] == model_id
        assert result["status"] == "triggered"
        
        # Verify adaptation engine was called
        training_pipeline.adaptation_engine.trigger_model_retraining.assert_called_once()
        args, kwargs = training_pipeline.adaptation_engine.trigger_model_retraining.call_args
        assert args[0] == model_id
        # Verify the parameters include feedback information
        assert "feedback_summary" in args[1]
        assert "feedback_count" in args[1]
        assert args[1]["feedback_count"] == 1
    
    @pytest.mark.asyncio
    async def test_monitor_retraining_job_completion(self, training_pipeline):
        """Test monitoring a retraining job through completion"""
        # Arrange
        model_id = "forex-prediction-model-1"
        job_id = str(uuid.uuid4())
        
        # Set up the ml_client mock on the adaptation engine
        ml_client = AsyncMock()
        training_pipeline.adaptation_engine.ml_client = ml_client
        
        # Configure ml_client to return "running" status first, then "completed"
        ml_client.get_job_status.side_effect = [
            {"status": "running", "progress": 50},
            {"status": "completed", "metrics": {"accuracy": 0.92}}
        ]
        
        # Setup job tracking info
        training_pipeline.active_retraining_jobs[job_id] = {
            "model_id": model_id,
            "start_time": datetime.utcnow(),
            "status": "running",
            "feedback_ids": ["feedback-1", "feedback-2"]
        }
        
        # Act - kick off the monitoring task directly
        await training_pipeline._monitor_retraining_job(job_id, model_id)
        
        # Assert
        # Verify the job was removed from active jobs after completion
        assert job_id not in training_pipeline.active_retraining_jobs
        
        # Verify that the event publisher was called with a success event
        training_pipeline.event_publisher.publish.assert_called_once()
        args, kwargs = training_pipeline.event_publisher.publish.call_args
        assert args[0] == EventType.MODEL_TRAINING_COMPLETED
        assert args[1]["job_id"] == job_id
        assert args[1]["model_id"] == model_id
        assert args[1]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_monitor_retraining_job_failure(self, training_pipeline):
        """Test monitoring a retraining job that fails"""
        # Arrange
        model_id = "forex-prediction-model-1"
        job_id = str(uuid.uuid4())
        
        # Set up the ml_client mock
        ml_client = AsyncMock()
        training_pipeline.adaptation_engine.ml_client = ml_client
        
        # Configure ml_client to return "running" status first, then "failed"
        ml_client.get_job_status.side_effect = [
            {"status": "running", "progress": 30},
            {"status": "failed", "error": "Insufficient training data"}
        ]
        
        # Setup job tracking info
        training_pipeline.active_retraining_jobs[job_id] = {
            "model_id": model_id,
            "start_time": datetime.utcnow(),
            "status": "running",
            "feedback_ids": ["feedback-1"]
        }
        
        # Act
        await training_pipeline._monitor_retraining_job(job_id, model_id)
        
        # Assert
        # Verify that the event publisher was called with a failure event
        training_pipeline.event_publisher.publish.assert_called_once()
        args, kwargs = training_pipeline.event_publisher.publish.call_args
        assert args[0] == EventType.MODEL_TRAINING_FAILED
        assert args[1]["status"] == "failed"
        assert "Insufficient training data" in args[1]["error"]


if __name__ == '__main__':
    unittest.main()
