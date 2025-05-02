"""
Unit tests for the Strategy Mutation Service.

This module tests the genetic algorithm-inspired mutation framework
that allows strategies to evolve based on feedback.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from analysis_engine.adaptive_layer.strategy_mutation_service import StrategyMutationService

class TestStrategyMutationService:
    """Test suite for the Strategy Mutation Service"""

    @pytest.fixture
    async def mock_event_publisher(self):
        """Mock event publisher fixture"""
        publisher = MagicMock()
        publish_future = asyncio.Future()
        publish_future.set_result(True)
        publisher.publish = MagicMock(return_value=publish_future)
        return publisher

    @pytest.fixture
    async def mutation_service(self, mock_event_publisher):
        """Set up a mutation service for testing"""
        service = StrategyMutationService(
            event_publisher=mock_event_publisher,
            config={
                "mutation_rate": 0.3,
                "mutation_magnitude": 0.2,
                "min_samples_for_evaluation": 5,
                "performance_window_days": 30,
                "fitness_metrics": {
                    "profit_loss": 0.4,
                    "win_rate": 0.2,
                    "profit_factor": 0.2,
                    "max_drawdown": -0.1,
                    "sharpe_ratio": 0.1
                }
            }
        )
        await service.start()
        yield service
        await service.stop()

    @pytest.mark.asyncio
    async def test_register_strategy(self, mutation_service, mock_event_publisher):
        """Test strategy registration functionality"""
        strategy_id = "test_strategy"
        parameters = {"param1": 10, "param2": 0.5, "param3": 100}
        
        version_id = await mutation_service.register_strategy(strategy_id, parameters)
        
        # Verify registration results
        assert version_id == f"{strategy_id}_v1"
        assert strategy_id in mutation_service.strategy_versions
        assert version_id in mutation_service.strategy_versions[strategy_id]
        assert mutation_service.active_versions[strategy_id] == version_id
        assert mutation_service.version_generations[version_id] == 1
        assert version_id in mutation_service.version_performance
        
        # Verify event publisher was called
        mock_event_publisher.publish.assert_called_once()
        call_args = mock_event_publisher.publish.call_args[0]
        assert call_args[0] == "strategy.registered"
        assert call_args[1]["strategy_id"] == strategy_id
        assert call_args[1]["version_id"] == version_id

    @pytest.mark.asyncio
    async def test_mutate_strategy(self, mutation_service, mock_event_publisher):
        """Test strategy mutation functionality"""
        strategy_id = "test_strategy"
        parameters = {"param1": 10, "param2": 0.5, "param3": 100}
        
        # Register strategy first
        version_id = await mutation_service.register_strategy(strategy_id, parameters)
        mock_event_publisher.publish.reset_mock()  # Reset call count
        
        # Now mutate it
        mutation_result = await mutation_service.mutate_strategy(strategy_id, force=True)
        
        # Verify mutation results
        assert mutation_result["success"] == True
        assert mutation_result["strategy_id"] == strategy_id
        assert mutation_result["parent_version"] == version_id
        assert mutation_result["new_version"] == f"{strategy_id}_v2"
        assert mutation_result["generation"] == 2
        assert len(mutation_result["parameter_changes"]) > 0
        
        # Verify new version was created
        new_version_id = mutation_result["new_version"]
        assert new_version_id in mutation_service.strategy_versions[strategy_id]
        assert new_version_id in mutation_service.version_performance
        
        # Verify old version is inactive and new version is active
        assert mutation_service.strategy_versions[strategy_id][version_id]["is_active"] == False
        assert mutation_service.strategy_versions[strategy_id][new_version_id]["is_active"] == True
        assert mutation_service.active_versions[strategy_id] == new_version_id
        
        # Verify event was published
        mock_event_publisher.publish.assert_called_once()
        call_args = mock_event_publisher.publish.call_args[0]
        assert call_args[0] == "strategy.mutated"

    @pytest.mark.asyncio
    async def test_record_version_performance(self, mutation_service):
        """Test recording performance metrics for a version"""
        # Register a strategy
        strategy_id = "test_strategy"
        parameters = {"param1": 10, "param2": 0.5, "param3": 100}
        version_id = await mutation_service.register_strategy(strategy_id, parameters)
        
        # Record performance data
        performance_metrics = {
            "profit_loss": 1.5,
            "win_rate": 0.65,
            "profit_factor": 2.1,
            "max_drawdown": 0.15
        }
        
        success = await mutation_service.record_version_performance(
            version_id=version_id,
            performance_metrics=performance_metrics,
            market_regime="trending",
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Verify recording was successful
        assert success == True
        assert len(mutation_service.version_performance[version_id]) == 1
        
        # Verify recorded data
        recorded_data = mutation_service.version_performance[version_id][0]
        assert recorded_data["metrics"] == performance_metrics
        assert recorded_data["market_regime"] == "trending"
        assert "timestamp" in recorded_data

    @pytest.mark.asyncio
    async def test_evaluate_and_select_best_version(self, mutation_service):
        """Test evaluation and selection of the best strategy version"""
        # Register a strategy with two versions
        strategy_id = "test_strategy"
        parameters = {"param1": 10, "param2": 0.5, "param3": 100}
        parent_version = await mutation_service.register_strategy(strategy_id, parameters)
        
        # Create a mutation
        mutation_result = await mutation_service.mutate_strategy(strategy_id, force=True)
        child_version = mutation_result["new_version"]
        
        # Add mediocre performance data for parent version
        for _ in range(10):
            await mutation_service.record_version_performance(
                version_id=parent_version,
                performance_metrics={
                    "profit_loss": 0.5,
                    "win_rate": 0.4,
                    "profit_factor": 1.2,
                    "max_drawdown": 0.25
                }
            )
        
        # Add better performance data for child version
        for _ in range(10):
            await mutation_service.record_version_performance(
                version_id=child_version,
                performance_metrics={
                    "profit_loss": 1.2,
                    "win_rate": 0.6,
                    "profit_factor": 1.8,
                    "max_drawdown": 0.15
                }
            )
        
        # Evaluate versions
        evaluation = await mutation_service.evaluate_and_select_best_version(strategy_id)
        
        # Verify evaluation results
        assert evaluation["success"] == True
        assert evaluation["action"] == "no_change"  # Child version already active
        assert evaluation["best_version"] == child_version

        # Force parent version to be active
        mutation_service.strategy_versions[strategy_id][parent_version]["is_active"] = True
        mutation_service.strategy_versions[strategy_id][child_version]["is_active"] = False
        mutation_service.active_versions[strategy_id] = parent_version
        
        # Re-evaluate
        evaluation = await mutation_service.evaluate_and_select_best_version(strategy_id)
        
        # Should switch to child version
        assert evaluation["success"] == True
        assert evaluation["action"] == "version_changed"
        assert evaluation["previous_version"] == parent_version
        assert evaluation["best_version"] == child_version
        
        # Verify active version was updated
        assert mutation_service.active_versions[strategy_id] == child_version
        assert mutation_service.strategy_versions[strategy_id][parent_version]["is_active"] == False
        assert mutation_service.strategy_versions[strategy_id][child_version]["is_active"] == True

    @pytest.mark.asyncio
    async def test_get_version_history(self, mutation_service):
        """Test fetching version history for a strategy"""
        # Register a strategy with multiple versions
        strategy_id = "test_strategy"
        parameters = {"param1": 10, "param2": 0.5, "param3": 100}
        v1 = await mutation_service.register_strategy(strategy_id, parameters)
        
        # Create mutations
        await mutation_service.mutate_strategy(strategy_id, force=True)
        await mutation_service.mutate_strategy(strategy_id, force=True)
        
        # Get version history
        history = await mutation_service.get_version_history(strategy_id)
        
        # Verify history
        assert len(history) == 3
        assert history[0]["version_id"] == v1
        assert history[0]["generation"] == 1
        assert history[0]["parent_id"] is None
        assert history[0]["active"] == False
        
        assert history[1]["generation"] == 2
        assert history[1]["parent_id"] == v1
        assert history[1]["active"] == False
        
        assert history[2]["generation"] == 3
        assert history[2]["parent_id"] == history[1]["version_id"]
        assert history[2]["active"] == True

    @pytest.mark.asyncio
    async def test_get_mutation_effectiveness(self, mutation_service):
        """Test analyzing mutation effectiveness"""
        # Register strategy with multiple versions and performance data
        strategy_id = "test_strategy"
        parameters = {"param1": 10, "param2": 0.5, "param3": 100}
        v1 = await mutation_service.register_strategy(strategy_id, parameters)
        
        # Add performance data for v1
        for _ in range(5):
            await mutation_service.record_version_performance(
                version_id=v1,
                performance_metrics={
                    "profit_loss": 0.5,
                    "win_rate": 0.4,
                }
            )
        
        # Create mutation v2
        mutation_result = await mutation_service.mutate_strategy(strategy_id, force=True)
        v2 = mutation_result["new_version"]
        
        # Add improved performance data for v2
        for _ in range(5):
            await mutation_service.record_version_performance(
                version_id=v2,
                performance_metrics={
                    "profit_loss": 0.8,
                    "win_rate": 0.5,
                }
            )
        
        # Create mutation v3
        mutation_result = await mutation_service.mutate_strategy(strategy_id, force=True)
        v3 = mutation_result["new_version"]
        
        # Add worse performance data for v3
        for _ in range(5):
            await mutation_service.record_version_performance(
                version_id=v3,
                performance_metrics={
                    "profit_loss": 0.3,
                    "win_rate": 0.3,
                }
            )
        
        # Get effectiveness
        effectiveness = await mutation_service.get_mutation_effectiveness(strategy_id)
        
        # Verify effectiveness results
        assert effectiveness["strategy_id"] == strategy_id
        assert effectiveness["version_count"] == 3
        assert effectiveness["generations"] == 3
        assert "success_rate" in effectiveness
        assert "overall_improvement" in effectiveness
        assert effectiveness["best_version"] == v2  # V2 should be best
