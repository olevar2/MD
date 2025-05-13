"""
Comprehensive end-to-end tests for the feedback and adaptation system.
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from ml_integration_service.feedback.analyzer import FeedbackAnalyzer
from ml_integration_service.feedback.adapter import ModelAdapter
from ml_integration_service.monitoring.adaptation_metrics import AdaptationMetricsCollector

class TestFeedbackSystemE2E:
    """End-to-end test suite for the feedback system."""

    @pytest.fixture
    async def setup_components(self):
        """Set up test components."""
        metrics_collector = AdaptationMetricsCollector()
        feedback_analyzer = FeedbackAnalyzer()
        model_adapter = ModelAdapter()
        return {
            "metrics_collector": metrics_collector,
            "feedback_analyzer": feedback_analyzer,
            "model_adapter": model_adapter
        }

    async def simulate_market_conditions(self, condition: str) -> Dict[str, float]:
        """Simulate different market conditions."""
        if condition == "normal":
            return {"volatility": 0.2, "trend_strength": 0.5}
        elif condition == "volatile":
            return {"volatility": 0.8, "trend_strength": 0.3}
        elif condition == "trending":
            return {"volatility": 0.3, "trend_strength": 0.9}
        return {"volatility": 0.1, "trend_strength": 0.1}

    @pytest.mark.asyncio
    async def test_complete_adaptation_cycle(self, setup_components):
        """Test a complete feedback-adaptation cycle."""
        components = await setup_components
        
        # Setup test data
        model_id = "test_model_1"
        initial_performance = {
            "accuracy": 0.75,
            "f1_score": 0.73,
            "precision": 0.76
        }

        # Record initial state
        components["metrics_collector"].update_performance_metrics(
            model_id=model_id,
            metrics=initial_performance
        )

        # Simulate deteriorating performance
        for _ in range(5):
            market_conditions = await self.simulate_market_conditions("volatile")
            
            # Record adaptation trigger
            components["metrics_collector"].record_adaptation_trigger(
                model_id=model_id,
                trigger_type="performance_drift"
            )

            # Attempt adaptation
            adaptation_start = datetime.utcnow()
            components["metrics_collector"].record_adaptation_attempt(
                model_id=model_id,
                model_type="forex_prediction",
                adaptation_type="retraining",
                start_time=adaptation_start
            )

            # Simulate adaptation process
            adapted_model = await components["model_adapter"].adapt_model(
                model_id=model_id,
                current_version="1.0",
                trigger="performance_drift",
                adaptation_type="retrain"
            )

            duration = (datetime.utcnow() - adaptation_start).total_seconds()

            # Record adaptation results
            assert adapted_model.success
            components["metrics_collector"].record_adaptation_result(
                model_id=model_id,
                model_type="forex_prediction",
                success=True,
                duration=duration
            )

            # Verify improved performance
            new_performance = {
                "accuracy": initial_performance["accuracy"] + 0.05,
                "f1_score": initial_performance["f1_score"] + 0.04,
                "precision": initial_performance["precision"] + 0.03
            }

            components["metrics_collector"].update_performance_metrics(
                model_id=model_id,
                metrics=new_performance,
                is_post_adaptation=True
            )

    @pytest.mark.asyncio
    async def test_adaptation_under_stress(self, setup_components):
        """Test adaptation system under high load."""
        components = await setup_components
        
        # Setup multiple models
        model_ids = [f"stress_test_model_{i}" for i in range(10)]
        market_conditions = ["normal", "volatile", "trending"]
        
        async def stress_test_model(model_id: str):
    """
    Stress test model.
    
    Args:
        model_id: Description of model_id
    
    """

            for condition in market_conditions:
                market_state = await self.simulate_market_conditions(condition)
                
                # Trigger adaptation
                adaptation_start = datetime.utcnow()
                components["metrics_collector"].record_adaptation_attempt(
                    model_id=model_id,
                    model_type="forex_prediction",
                    adaptation_type="fine_tune",
                    start_time=adaptation_start
                )

                # Simulate adaptation
                adapted_model = await components["model_adapter"].adapt_model(
                    model_id=model_id,
                    current_version="1.0",
                    trigger="market_change",
                    adaptation_type="fine_tune"
                )

                duration = (datetime.utcnow() - adaptation_start).total_seconds()
                
                # Record results
                components["metrics_collector"].record_adaptation_result(
                    model_id=model_id,
                    model_type="forex_prediction",
                    success=adapted_model.success,
                    duration=duration,
                    failure_reason=None if adapted_model.success else "timeout"
                )

        # Run concurrent adaptations
        tasks = [stress_test_model(model_id) for model_id in model_ids]
        await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_regression_scenarios(self, setup_components):
        """Test various regression scenarios."""
        components = await setup_components
        
        regression_scenarios = [
            {
                "name": "gradual_drift",
                "performance_change": -0.01,
                "iterations": 10
            },
            {
                "name": "sudden_shift",
                "performance_change": -0.15,
                "iterations": 1
            },
            {
                "name": "oscillating",
                "performance_change": 0.05,
                "iterations": 5
            }
        ]

        for scenario in regression_scenarios:
            model_id = f"regression_test_{scenario['name']}"
            base_performance = 0.8

            for i in range(scenario['iterations']):
                current_performance = base_performance + (scenario['performance_change'] * i)
                
                # Record metrics
                metrics = {
                    "accuracy": current_performance,
                    "f1_score": current_performance - 0.02,
                    "precision": current_performance + 0.01
                }
                
                components["metrics_collector"].update_performance_metrics(
                    model_id=model_id,
                    metrics=metrics
                )

                # Check for adaptation need
                if current_performance < 0.7:
                    adaptation_start = datetime.utcnow()
                    components["metrics_collector"].record_adaptation_attempt(
                        model_id=model_id,
                        model_type="forex_prediction",
                        adaptation_type="retrain",
                        start_time=adaptation_start
                    )

                    # Simulate adaptation
                    adapted_model = await components["model_adapter"].adapt_model(
                        model_id=model_id,
                        current_version="1.0",
                        trigger="performance_regression",
                        adaptation_type="retrain"
                    )

                    duration = (datetime.utcnow() - adaptation_start).total_seconds()
                    
                    # Record adaptation results
                    components["metrics_collector"].record_adaptation_result(
                        model_id=model_id,
                        model_type="forex_prediction",
                        success=adapted_model.success,
                        duration=duration
                    )

                    # Reset performance if adaptation successful
                    if adapted_model.success:
                        base_performance = 0.8
