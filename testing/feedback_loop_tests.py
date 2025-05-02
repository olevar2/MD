"""
Feedback Loop System Integration Tests

This module contains integration tests for the Feedback Loop System implemented in Phase 8.
It tests the end-to-end functionality of the feedback collection, categorization, routing,
and the bidirectional connection between trading outcomes and model/strategy adaptation.
"""

import os
import sys
import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project root to path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_foundations.models.feedback import (
    FeedbackSource, FeedbackCategory, FeedbackStatus, 
    FeedbackTag, FeedbackFailureReason, FeedbackSuccessFactor
)

from analysis_engine.adaptive_layer.trading_feedback_collector import TradingFeedbackCollector
from analysis_engine.adaptive_layer.feedback_categorizer import FeedbackCategorizer
from analysis_engine.adaptive_layer.feedback_router import FeedbackRouter
from analysis_engine.adaptive_layer.model_training_feedback_integrator import ModelTrainingFeedbackIntegrator
from analysis_engine.adaptive_layer.feedback_integration_service import FeedbackIntegrationService


class MockEventBus:
    """Mock implementation of the event bus for testing"""
    
    def __init__(self):
        self.published_events = []
        self.subscribers = {}
    
    async def publish(self, topic: str, event: Dict[str, Any]) -> None:
        """Publish an event to the mock event bus"""
        self.published_events.append((topic, event))
        
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                await callback(event)
    
    async def subscribe(self, topic: str, callback) -> None:
        """Subscribe to a topic on the mock event bus"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)


class MockDatabase:
    """Mock implementation of the database for testing"""
    
    def __init__(self):
        self.feedback_entries = {}
        self.next_id = 1
    
    async def insert_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Insert a feedback entry into the mock database"""
        feedback_id = f"fb_test_{self.next_id}"
        self.next_id += 1
        
        feedback_data["id"] = feedback_id
        self.feedback_entries[feedback_id] = feedback_data
        return feedback_id
    
    async def get_feedback(self, feedback_id: str) -> Dict[str, Any]:
        """Get a feedback entry from the mock database"""
        return self.feedback_entries.get(feedback_id)
    
    async def update_feedback(self, feedback_id: str, updates: Dict[str, Any]) -> None:
        """Update a feedback entry in the mock database"""
        if feedback_id in self.feedback_entries:
            self.feedback_entries[feedback_id].update(updates)
    
    async def find_feedback(self, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Find feedback entries matching the query"""
        results = []
        for entry in self.feedback_entries.values():
            match = True
            for key, value in query.items():
                if key not in entry or entry[key] != value:
                    match = False
                    break
            
            if match:
                results.append(entry)
                if len(results) >= limit:
                    break
        
        return results


class MockModelRegistry:
    """Mock implementation of the model registry for testing"""
    
    def __init__(self):
        self.models = {
            "lstm_model_v1": {
                "id": "lstm_model_v1",
                "type": "LSTM",
                "accuracy": 0.75,
                "last_trained": datetime.utcnow() - timedelta(days=30),
                "status": "active"
            },
            "random_forest_v2": {
                "id": "random_forest_v2",
                "type": "RandomForest",
                "accuracy": 0.82,
                "last_trained": datetime.utcnow() - timedelta(days=15),
                "status": "active"
            }
        }
        self.retraining_requests = []
    
    async def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get a model from the mock registry"""
        return self.models.get(model_id)
    
    async def request_retraining(self, model_id: str, reason: str) -> Dict[str, Any]:
        """Request retraining for a model"""
        if model_id not in self.models:
            return {"success": False, "error": "Model not found"}
        
        self.retraining_requests.append({
            "model_id": model_id,
            "reason": reason,
            "timestamp": datetime.utcnow()
        })
        
        return {
            "success": True,
            "model_id": model_id,
            "request_id": f"retraining_{len(self.retraining_requests)}",
            "status": "scheduled"
        }


@pytest.fixture
async def mock_event_bus():
    """Fixture for a mock event bus"""
    return MockEventBus()


@pytest.fixture
async def mock_database():
    """Fixture for a mock database"""
    return MockDatabase()


@pytest.fixture
async def mock_model_registry():
    """Fixture for a mock model registry"""
    return MockModelRegistry()


@pytest.fixture
async def feedback_components(mock_event_bus, mock_database, mock_model_registry):
    """Fixture for feedback system components"""
    # Create the components
    collector = TradingFeedbackCollector(
        event_bus=mock_event_bus,
        database=mock_database
    )
    
    categorizer = FeedbackCategorizer()
    
    router = FeedbackRouter(
        event_bus=mock_event_bus,
        database=mock_database
    )
    
    model_integrator = ModelTrainingFeedbackIntegrator(
        model_registry=mock_model_registry,
        event_bus=mock_event_bus
    )
    
    # Create the integration service
    service = FeedbackIntegrationService(
        collector=collector,
        categorizer=categorizer,
        router=router,
        model_integrator=model_integrator,
        event_bus=mock_event_bus
    )
    
    # Return all components
    return {
        "collector": collector,
        "categorizer": categorizer,
        "router": router, 
        "model_integrator": model_integrator,
        "service": service,
        "event_bus": mock_event_bus,
        "database": mock_database,
        "model_registry": mock_model_registry
    }


@pytest.mark.asyncio
async def test_feedback_collection(feedback_components):
    """Test feedback collection and initial processing"""
    collector = feedback_components["collector"]
    database = feedback_components["database"]
    
    # Collect feedback
    feedback_id = await collector.collect(
        source=FeedbackSource.MODEL_PREDICTION,
        category=FeedbackCategory.FAILURE,
        instrument_id="EUR_USD",
        strategy_id="trend_following_v2",
        model_id="lstm_model_v1",
        description="Model predicted upward movement but price dropped",
        metadata={
            "prediction_confidence": 0.85,
            "actual_pip_movement": -25
        }
    )
    
    # Verify the feedback was stored
    assert feedback_id is not None
    assert feedback_id in database.feedback_entries
    
    # Verify feedback data
    feedback = database.feedback_entries[feedback_id]
    assert feedback["source"] == FeedbackSource.MODEL_PREDICTION
    assert feedback["category"] == FeedbackCategory.FAILURE
    assert feedback["instrument_id"] == "EUR_USD"
    assert feedback["model_id"] == "lstm_model_v1"


@pytest.mark.asyncio
async def test_feedback_categorization(feedback_components):
    """Test feedback categorization"""
    categorizer = feedback_components["categorizer"]
    
    # Create sample feedback data
    feedback_data = {
        "source": FeedbackSource.STRATEGY_EXECUTION,
        "category": FeedbackCategory.FAILURE,
        "instrument_id": "GBP_USD",
        "strategy_id": "breakout_v3",
        "description": "Strategy entered position but stop loss was hit shortly after",
        "metadata": {
            "entry_price": 1.2450,
            "stop_loss": 1.2400,
            "exit_price": 1.2395,
            "position_duration_minutes": 45,
            "market_volatility": "high"
        }
    }
    
    # Categorize the feedback
    enriched = await categorizer.categorize(feedback_data)
    
    # Verify categorization results
    assert "failure_reasons" in enriched
    assert "tags" in enriched
    
    # Should detect stop loss hit as failure reason
    assert FeedbackFailureReason.STOP_LOSS_HIT in enriched["failure_reasons"]
    
    # Should detect high volatility
    assert FeedbackTag.HIGH_VOLATILITY in enriched["tags"]


@pytest.mark.asyncio
async def test_feedback_routing(feedback_components):
    """Test feedback routing to appropriate destinations"""
    collector = feedback_components["collector"]
    router = feedback_components["router"]
    event_bus = feedback_components["event_bus"]
    
    # Collect feedback
    feedback_id = await collector.collect(
        source=FeedbackSource.MODEL_PREDICTION,
        category=FeedbackCategory.FAILURE,
        instrument_id="EUR_USD",
        strategy_id="trend_following_v2",
        model_id="lstm_model_v1",
        description="Model prediction failure during news event",
        metadata={
            "news_impact": "high",
            "prediction_confidence": 0.75
        }
    )
    
    # Route the feedback
    feedback_data = await feedback_components["database"].get_feedback(feedback_id)
    await router.route(feedback_id, feedback_data)
    
    # Verify events were published
    events = [event for topic, event in event_bus.published_events]
    
    # Should have at least the collection and routing events
    assert len(events) >= 2
    
    # Check that model-related events were published
    model_events = [e for topic, e in event_bus.published_events 
                   if "model" in topic.lower()]
    assert len(model_events) > 0


@pytest.mark.asyncio
async def test_model_training_integration(feedback_components):
    """Test the connection between feedback and model training"""
    model_integrator = feedback_components["model_integrator"]
    model_registry = feedback_components["model_registry"]
    
    # Process feedback for a model
    await model_integrator.process_model_feedback(
        model_id="lstm_model_v1",
        feedback_data={
            "id": "test_fb_1",
            "source": FeedbackSource.MODEL_PREDICTION,
            "category": FeedbackCategory.FAILURE,
            "instrument_id": "EUR_USD",
            "model_id": "lstm_model_v1",
            "description": "Significant prediction error",
            "metadata": {
                "error_magnitude": "large",
                "consecutive_errors": 5
            }
        }
    )
    
    # The model integrator should have requested retraining
    assert len(model_registry.retraining_requests) > 0
    assert model_registry.retraining_requests[0]["model_id"] == "lstm_model_v1"


@pytest.mark.asyncio
async def test_feedback_integration_service(feedback_components):
    """Test the complete feedback integration service"""
    service = feedback_components["service"]
    event_bus = feedback_components["event_bus"]
    model_registry = feedback_components["model_registry"]
    
    # Start the service
    await service.start()
    
    # Publish a trade execution event
    await event_bus.publish("trading.position.closed", {
        "position_id": "pos_12345",
        "instrument_id": "USD_JPY",
        "strategy_id": "mean_reversion_v1",
        "model_id": "random_forest_v2",
        "entry_price": 152.50,
        "exit_price": 153.25,
        "direction": "long",
        "profit_loss_pips": 75,
        "profit_loss_base_ccy": 750.0,
        "reason": "take_profit",
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": {
            "trade_duration_minutes": 120,
            "market_regime": "trending",
            "position_size": 1.0
        }
    })
    
    # Give some time for async processing
    await asyncio.sleep(0.1)
    
    # Check feedback was collected (successful trade should create feedback)
    database = feedback_components["database"]
    entries = await database.find_feedback({"instrument_id": "USD_JPY"})
    assert len(entries) > 0
    assert entries[0]["category"] == FeedbackCategory.SUCCESS
    
    # Get insights from the service
    insights = await service.get_feedback_insights()
    
    # There should be some insights generated
    assert len(insights) > 0
    
    # Stop the service
    await service.stop()


@pytest.mark.asyncio
async def test_adaptive_trading_feedback(feedback_components):
    """Test the complete adaptive trading feedback cycle"""
    service = feedback_components["service"]
    event_bus = feedback_components["event_bus"]
    database = feedback_components["database"]
    
    # Start the service
    await service.start()
    
    # Generate a series of failures from the same model
    for i in range(5):
        await event_bus.publish("ml.prediction.result", {
            "prediction_id": f"pred_{1000+i}",
            "instrument_id": "EUR_USD",
            "model_id": "lstm_model_v1",
            "strategy_id": "trend_following_v2",
            "predicted_direction": "up",
            "actual_direction": "down",
            "confidence": 0.8,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "market_regime": "trending",
                "prediction_horizon": "4h",
                "features_used": ["price", "volume", "sentiment"]
            }
        })
    
    # Give some time for async processing
    await asyncio.sleep(0.1)
    
    # Verify multiple failures were collected
    entries = await database.find_feedback({
        "instrument_id": "EUR_USD",
        "model_id": "lstm_model_v1",
        "category": FeedbackCategory.FAILURE
    })
    assert len(entries) >= 3
    
    # Verify model retraining was requested
    model_registry = feedback_components["model_registry"]
    retraining_requests = [r for r in model_registry.retraining_requests 
                          if r["model_id"] == "lstm_model_v1"]
    assert len(retraining_requests) > 0
    
    # Stop the service
    await service.stop()


if __name__ == "__main__":
    # Run tests manually if needed
    asyncio.run(pytest.main(["-v", __file__]))
