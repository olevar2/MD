"""
Unit tests for the Feedback Loop Connector.

This module tests the bidirectional connector between strategy execution and
the adaptive layer, ensuring feedback flows correctly in both directions.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import httpx
from core_foundations.models.feedback import TradeFeedback, FeedbackStatus, FeedbackSource, FeedbackCategory

from analysis_engine.adaptive_layer.feedback_loop_connector import FeedbackLoopConnector
from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
from analysis_engine.adaptive_layer.trading_feedback_collector import TradingFeedbackCollector

class TestFeedbackLoopConnector:
    """Test suite for the Feedback Loop Connector"""

    @pytest.fixture
    def mock_feedback_loop(self):
        """Create mock feedback loop"""
        loop = MagicMock(spec=FeedbackLoop)
        return loop

    @pytest.fixture
    def mock_trading_feedback_collector(self):
        """Create mock trading feedback collector"""
        collector = MagicMock(spec=TradingFeedbackCollector)
        collect_future = asyncio.Future()
        collect_future.set_result("test_feedback_id")
        collector.collect_feedback = MagicMock(return_value=collect_future)
        return collector

    @pytest.fixture
    def mock_event_publisher(self):
        """Create mock event publisher"""
        publisher = MagicMock()
        publish_future = asyncio.Future()
        publish_future.set_result(True)
        publisher.publish = MagicMock(return_value=publish_future)
        return publisher

    @pytest.fixture
    async def connector(self, mock_feedback_loop, mock_trading_feedback_collector, mock_event_publisher):
        """Set up feedback loop connector for testing"""
        connector = FeedbackLoopConnector(
            feedback_loop=mock_feedback_loop,
            trading_feedback_collector=mock_trading_feedback_collector,
            strategy_execution_api_url="http://test-execution-api:8000",
            event_publisher=mock_event_publisher,
            config={
                "http_timeout": 5.0,
                "max_connections": 5,
                "monitoring_interval_seconds": 1,
                "feedback_alert_threshold_seconds": 300
            }
        )
        
        # Start the connector (without actually setting up health monitoring)
        connector._is_running = True
        connector.http_client = MagicMock(spec=httpx.AsyncClient)
        
        # Mock the response for HTTP requests
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock()
        response_future = asyncio.Future()
        response_future.set_result(mock_response)
        connector.http_client.post = MagicMock(return_value=response_future)
        
        yield connector
        
        # Clean up
        connector._is_running = False

    @pytest.mark.asyncio
    async def test_process_execution_feedback(self, connector, mock_trading_feedback_collector):
        """Test processing feedback from strategy execution"""
        # Prepare test data
        feedback_data = {
            "strategy_id": "test_strategy",
            "model_id": "test_model",
            "instrument": "EURUSD",
            "timeframe": "1h",
            "source": FeedbackSource.STRATEGY_EXECUTION.value,
            "category": FeedbackCategory.SUCCESS.value,
            "metrics": {"profit_loss": 1.5, "win_rate": 0.6},
            "metadata": {"version_id": "test_version", "market_regime": "trending"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Process feedback
        feedback_id = await connector.process_execution_feedback(feedback_data)
        
        # Verify results
        assert feedback_id == "test_feedback_id"
        assert connector.feedback_count == 1
        assert connector.last_feedback_time is not None
        
        # Verify collector was called correctly
        mock_trading_feedback_collector.collect_feedback.assert_called_once()
        call_arg = mock_trading_feedback_collector.collect_feedback.call_args[0][0]
        assert isinstance(call_arg, TradeFeedback)
        assert call_arg.strategy_id == "test_strategy"
        assert call_arg.instrument == "EURUSD"
        assert call_arg.status == FeedbackStatus.NEW

    @pytest.mark.asyncio
    async def test_send_adaptation_to_strategy_execution(self, connector, mock_event_publisher):
        """Test sending adaptation to strategy execution"""
        # Prepare adaptation data
        adaptation = {
            "adaptation_id": "test_adaptation",
            "strategy_id": "test_strategy",
            "parameters": {"param1": 15, "param2": 0.7},
            "reason": "Market regime change",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send adaptation
        success = await connector.send_adaptation_to_strategy_execution(adaptation)
        
        # Verify results
        assert success == True
        assert connector.adaptation_count == 1
        assert connector.last_adaptation_time is not None
        
        # Verify HTTP request was made
        connector.http_client.post.assert_called_once()
        call_args = connector.http_client.post.call_args
        assert call_args[0][0] == "http://test-execution-api:8000/api/v1/strategies/test_strategy/adapt"
        assert call_args[1]["json"] == adaptation
        
        # Verify event was published
        mock_event_publisher.publish.assert_called_once()
        pub_args = mock_event_publisher.publish.call_args[0]
        assert pub_args[0] == "feedback.adaptation.sent"
        assert pub_args[1]["strategy_id"] == "test_strategy"
        assert pub_args[1]["adaptation_id"] == "test_adaptation"

    @pytest.mark.asyncio
    async def test_send_adaptation_with_error(self, connector, mock_event_publisher):
        """Test handling errors when sending adaptation"""
        # Set up HTTP error
        error_response = asyncio.Future()
        error_response.set_exception(httpx.RequestError("Connection error"))
        connector.http_client.post = MagicMock(return_value=error_response)
        
        # Prepare adaptation data
        adaptation = {
            "adaptation_id": "test_adaptation",
            "strategy_id": "test_strategy",
            "parameters": {"param1": 15, "param2": 0.7}
        }
        
        # Send adaptation
        success = await connector.send_adaptation_to_strategy_execution(adaptation)
        
        # Verify results
        assert success == False
        assert connector.adaptation_count == 0
        assert connector.connection_errors == 1
        
        # Verify error event was published
        mock_event_publisher.publish.assert_called_once()
        pub_args = mock_event_publisher.publish.call_args[0]
        assert pub_args[0] == "feedback.adaptation.error"
        assert pub_args[1]["strategy_id"] == "test_strategy"
        assert pub_args[1]["adaptation_id"] == "test_adaptation"
        assert "error" in pub_args[1]

    @pytest.mark.asyncio
    async def test_get_loop_health(self, connector):
        """Test getting health metrics"""
        # Set up some activity
        connector.feedback_count = 10
        connector.adaptation_count = 5
        connector.connection_errors = 1
        connector.last_feedback_time = datetime.utcnow() - timedelta(minutes=30)
        connector.last_adaptation_time = datetime.utcnow() - timedelta(minutes=60)
        
        # Get health metrics
        health = await connector.get_loop_health()
        
        # Verify health metrics
        assert health["is_running"] == True
        assert health["feedback_count"] == 10
        assert health["adaptation_count"] == 5
        assert health["connection_errors"] == 1
        assert health["last_feedback_time"] is not None
        assert health["last_adaptation_time"] is not None
        assert health["seconds_since_last_feedback"] is not None
        assert health["seconds_since_last_adaptation"] is not None
        assert health["strategy_execution_api"] == "http://test-execution-api:8000"

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping the connector"""
        # Create connector
        mock_feedback_loop = MagicMock(spec=FeedbackLoop)
        mock_trading_feedback_collector = MagicMock(spec=TradingFeedbackCollector)
        mock_event_publisher = MagicMock()
        
        connector = FeedbackLoopConnector(
            feedback_loop=mock_feedback_loop,
            trading_feedback_collector=mock_trading_feedback_collector,
            strategy_execution_api_url="http://test-execution-api:8000",
            event_publisher=mock_event_publisher,
            config={"enable_monitoring": False}
        )
        
        # Start connector
        await connector.start()
        assert connector._is_running == True
        assert connector.http_client is not None
        
        # Stop connector
        await connector.stop()
        assert connector._is_running == False
        assert connector.http_client is None
