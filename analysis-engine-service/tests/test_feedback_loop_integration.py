"""
Integration tests for the Feedback Loop system.

This module tests the integration of feedback loop components working together,
ensuring data flows correctly through the entire system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory, FeedbackStatus
from core_foundations.events.event_publisher import EventPublisher

from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
from analysis_engine.adaptive_layer.trading_feedback_collector import TradingFeedbackCollector
from analysis_engine.adaptive_layer.strategy_mutation_service import StrategyMutationService
from analysis_engine.adaptive_layer.parameter_tracking_service import ParameterTrackingService
from analysis_engine.adaptive_layer.parameter_statistical_validator import ParameterStatisticalValidator
from analysis_engine.adaptive_layer.feedback_loop_connector import FeedbackLoopConnector

class TestFeedbackLoopIntegration:
    """Integration test suite for the feedback loop system"""

    @pytest.fixture
    async def mock_event_publisher(self):
        """Create a mock event publisher"""
        publisher = MagicMock(spec=EventPublisher)
        # Make the publish method return a completed future
        publish_future = asyncio.Future()
        publish_future.set_result(True)
        publisher.publish = MagicMock(return_value=publish_future)
        return publisher

    @pytest.fixture
    async def parameter_validator(self):
        """Create parameter validator"""
        return ParameterStatisticalValidator(
            config={
                "significance_level": 0.05,
                "min_sample_size": 5,  # Small for testing
                "confidence_threshold": 0.7
            }
        )

    @pytest.fixture
    async def parameter_tracking(self, mock_event_publisher):
        """Create parameter tracking service"""
        return ParameterTrackingService(
            event_publisher=mock_event_publisher,
            config={
                "min_sample_size": 5  # Small for testing
            }
        )

    @pytest.fixture
    async def mutation_service(self, mock_event_publisher, parameter_validator):
        """Create mutation service"""
        service = StrategyMutationService(
            event_publisher=mock_event_publisher,
            parameter_validator=parameter_validator,
            config={
                "mutation_rate": 0.3,
                "mutation_magnitude": 0.2,
                "min_samples_for_evaluation": 5,  # Small for testing
                "performance_window_days": 30
            }
        )
        await service.start()
        yield service
        await service.stop()

    @pytest.fixture
    async def feedback_loop(self, parameter_tracking, mutation_service, mock_event_publisher):
        """Create feedback loop"""
        return FeedbackLoop(
            parameter_tracking=parameter_tracking,
            mutation_service=mutation_service,
            event_publisher=mock_event_publisher
        )

    @pytest.fixture
    async def feedback_collector(self, feedback_loop, mock_event_publisher):
        """Create trading feedback collector"""
        return TradingFeedbackCollector(
            feedback_loop=feedback_loop,
            event_publisher=mock_event_publisher,
            config={
                "batch_size": 5,
                "max_batch_wait_seconds": 1
            }
        )

    @pytest.fixture
    async def feedback_connector(self, feedback_loop, feedback_collector, mock_event_publisher):
        """Create feedback connector"""
        connector = FeedbackLoopConnector(
            feedback_loop=feedback_loop,
            trading_feedback_collector=feedback_collector,
            strategy_execution_api_url="http://test-execution-api:8000",
            event_publisher=mock_event_publisher,
            config={
                "enable_monitoring": False  # Disable for testing
            }
        )
        await connector.start()
        yield connector
        await connector.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_feedback_flow(
        self, 
        feedback_loop, 
        feedback_collector, 
        mutation_service, 
        parameter_tracking,
        mock_event_publisher
    ):
        """Test end-to-end flow of feedback through the system"""
        # 1. Register a new strategy
        strategy_id = "test_strategy_1"
        parameters = {
            "ma_length": 20,
            "threshold": 0.5,
            "stop_loss": 50
        }
        version_id = await mutation_service.register_strategy(strategy_id, parameters)
        
        # Reset event publisher calls
        mock_event_publisher.publish.reset_mock()
        
        # 2. Create feedback with successful trade outcomes
        for i in range(10):
            feedback = TradeFeedback(
                strategy_id=strategy_id,
                instrument="EURUSD",
                timeframe="1h",
                source=FeedbackSource.STRATEGY_EXECUTION,
                category=FeedbackCategory.SUCCESS,
                status=FeedbackStatus.NEW,
                outcome_metrics={
                    "profit_loss": 1.5 + (i * 0.1),  # Increasing profits
                    "win_rate": 0.6,
                    "max_drawdown": 0.1
                },
                metadata={
                    "version_id": version_id,
                    "market_regime": "trending"
                },
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Submit feedback
            feedback_id = await feedback_collector.collect_feedback(feedback)
            assert feedback_id is not None
        
        # Verify the events were published
        assert mock_event_publisher.publish.call_count >= 10
        
        # 3. Force a mutation to evolve the strategy
        mutation_result = await mutation_service.mutate_strategy(strategy_id, force=True)
        assert mutation_result["success"] is True
        new_version_id = mutation_result["new_version"]
        
        # Verify parameter changes were made
        assert len(mutation_result["parameter_changes"]) > 0
        
        # Reset mock
        mock_event_publisher.publish.reset_mock()
        
        # 4. Create feedback for the new version with worse performance
        for i in range(10):
            feedback = TradeFeedback(
                strategy_id=strategy_id,
                instrument="EURUSD",
                timeframe="1h",
                source=FeedbackSource.STRATEGY_EXECUTION,
                category=FeedbackCategory.FAILURE,
                status=FeedbackStatus.NEW,
                outcome_metrics={
                    "profit_loss": -0.5 - (i * 0.1),  # Decreasing profits (losses)
                    "win_rate": 0.3,
                    "max_drawdown": 0.3
                },
                metadata={
                    "version_id": new_version_id,
                    "market_regime": "trending"
                },
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Submit feedback
            feedback_id = await feedback_collector.collect_feedback(feedback)
            assert feedback_id is not None
        
        # 5. Evaluate the versions - should revert to the original version
        evaluation = await mutation_service.evaluate_and_select_best_version(strategy_id)
        
        # Verify the original version was selected
        assert evaluation["success"] is True
        assert evaluation["best_version"] == version_id
        
        # Check if the active version matches
        assert mutation_service.active_versions[strategy_id] == version_id
    
    @pytest.mark.asyncio
    async def test_feedback_connector_integration(
        self,
        feedback_connector,
        feedback_collector, 
        mutation_service
    ):
        """Test integration between connector and feedback system"""
        # 1. Register a strategy
        strategy_id = "test_strategy_2"
        parameters = {"rsi_period": 14, "overbought": 70, "oversold": 30}
        version_id = await mutation_service.register_strategy(strategy_id, parameters)
        
        # 2. Send execution feedback via connector
        feedback_data = {
            "strategy_id": strategy_id,
            "instrument": "GBPUSD",
            "timeframe": "4h",
            "source": FeedbackSource.STRATEGY_EXECUTION.value,
            "category": FeedbackCategory.SUCCESS.value,
            "metrics": {
                "profit_loss": 2.5,
                "win_rate": 0.7,
                "max_drawdown": 0.1,
                "trade_duration": 120
            },
            "metadata": {
                "version_id": version_id,
                "market_regime": "ranging",
                "position_size": 0.01
            }
        }
        
        # Process feedback via connector
        feedback_id = await feedback_connector.process_execution_feedback(feedback_data)
        assert feedback_id is not None
        
        # 3. Verify connector metrics were updated
        health = await feedback_connector.get_loop_health()
        assert health["feedback_count"] == 1
        assert health["is_running"] is True
        
        # 4. Create a mock adaptation
        adaptation = {
            "adaptation_id": "test_adapt_1",
            "strategy_id": strategy_id,
            "parameters": {"rsi_period": 10, "overbought": 75, "oversold": 25},
            "reason": "Parameter optimization",
            "version_id": f"{strategy_id}_v2"
        }
        
        # 5. Patch the HTTP client to avoid actual network calls
        with patch.object(feedback_connector.http_client, 'post') as mock_post:
            # Set up mock response
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_future = asyncio.Future()
            mock_future.set_result(mock_response)
            mock_post.return_value = mock_future
            
            # Send adaptation
            result = await feedback_connector.send_adaptation_to_strategy_execution(adaptation)
            assert result is True
            
            # Verify HTTP call was made correctly
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == f"http://test-execution-api:8000/api/v1/strategies/{strategy_id}/adapt"
            assert kwargs["json"] == adaptation
        
        # 6. Verify connector metrics were updated
        health = await feedback_connector.get_loop_health()
        assert health["adaptation_count"] == 1

    @pytest.mark.asyncio
    async def test_parameter_tracking_integration(
        self, 
        feedback_collector, 
        parameter_tracking, 
        mutation_service
    ):
        """Test parameter tracking integration with feedback system"""
        # 1. Register a strategy
        strategy_id = "test_strategy_3"
        parameters = {"ema_short": 10, "ema_long": 20, "risk_percent": 2.0}
        version_id = await mutation_service.register_strategy(strategy_id, parameters)
        
        # 2. Submit feedback for original parameters
        for i in range(10):
            # Create feedback
            feedback = TradeFeedback(
                strategy_id=strategy_id,
                source=FeedbackSource.STRATEGY_EXECUTION,
                category=FeedbackCategory.PERFORMANCE_METRICS,
                instrument="USDJPY",
                timeframe="1d",
                outcome_metrics={
                    "profit_loss": 0.8 + (i * 0.05),
                    "trade_count": 5
                },
                metadata={
                    "version_id": version_id,
                    "parameters": parameters
                }
            )
            await feedback_collector.collect_feedback(feedback)
        
        # 3. Create mutation
        mutation_result = await mutation_service.mutate_strategy(strategy_id, force=True)
        new_version_id = mutation_result["new_version"]
        new_params = {}
        
        # Extract the new parameters
        for change in mutation_result["parameter_changes"]:
            new_params[change["parameter"]] = change["new_value"]
        
        # Fill in unchanged parameters
        for param, value in parameters.items():
            if param not in new_params:
                new_params[param] = value
        
        # 4. Submit feedback for new parameters
        for i in range(10):
            # Create feedback (with worse performance)
            feedback = TradeFeedback(
                strategy_id=strategy_id,
                source=FeedbackSource.STRATEGY_EXECUTION,
                category=FeedbackCategory.PERFORMANCE_METRICS,
                instrument="USDJPY",
                timeframe="1d",
                outcome_metrics={
                    "profit_loss": 0.4 + (i * 0.02), # Worse performance
                    "trade_count": 5
                },
                metadata={
                    "version_id": new_version_id,
                    "parameters": new_params
                }
            )
            await feedback_collector.collect_feedback(feedback)
            
        # 5. Get parameter performance
        param_performance = await parameter_tracking.get_parameter_performance(strategy_id)
        
        # Verify we have performance data
        assert len(param_performance["parameters"]) > 0
        
        # Check that the original parameters should show better performance
        for param in param_performance["parameters"]:
            if param["parameter"] in new_params:
                # The changed parameters should show lower effectiveness
                if parameters[param["parameter"]] != new_params[param["parameter"]]:
                    assert param["effectiveness_score"] < 1.0
