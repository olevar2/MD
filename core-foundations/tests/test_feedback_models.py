# filepath: d:\MD\forex_trading_platform\core-foundations\tests\test_feedback_models.py
"""
Unit tests for feedback Pydantic models.
"""

import pytest
from pydantic import ValidationError
from datetime import datetime
import uuid

from core_foundations.models.feedback_models import (
    BaseFeedback,
    TradeFeedbackData, TradeFeedback,
    ModelPerformanceData, ModelPerformanceFeedback,
    ParameterFeedbackData, ParameterFeedback
)

def test_base_feedback_creation():
    """Test basic creation and default values of BaseFeedback."""
    feedback = BaseFeedback(source="test_source", feedback_type="trade")
    assert isinstance(feedback.feedback_id, uuid.UUID)
    assert isinstance(feedback.timestamp, datetime)
    assert feedback.source == "test_source"
    assert feedback.feedback_type == "trade"

    with pytest.raises(ValidationError):
        BaseFeedback(source="test") # Missing feedback_type

    with pytest.raises(ValidationError):
        BaseFeedback(feedback_type="trade") # Missing source

def test_trade_feedback_creation():
    """Test creation of TradeFeedback."""
    data = TradeFeedbackData(
        trade_id="trade123",
        strategy_id="strat_A",
        symbol="EURUSD",
        outcome="profit",
        pnl=100.50
    )
    feedback = TradeFeedback(source="exec_engine", data=data)
    assert feedback.feedback_type == "trade"
    assert feedback.source == "exec_engine"
    assert feedback.data.trade_id == "trade123"
    assert feedback.data.pnl == 100.50
    assert feedback.data.slippage is None

    # Test with optional fields
    data_full = TradeFeedbackData(
        trade_id="trade124",
        strategy_id="strat_B",
        symbol="GBPUSD",
        outcome="loss",
        pnl=-50.0,
        slippage=0.5,
        execution_time_ms=150.0,
        market_conditions={"volatility": "high"}
    )
    feedback_full = TradeFeedback(source="exec_engine", data=data_full)
    assert feedback_full.data.slippage == 0.5
    assert feedback_full.data.market_conditions == {"volatility": "high"}

    with pytest.raises(ValidationError):
        TradeFeedbackData(trade_id="t1", strategy_id="s1", symbol="XAU", outcome="invalid", pnl=10) # Invalid outcome

def test_model_performance_feedback_creation():
    """Test creation of ModelPerformanceFeedback."""
    now = datetime.utcnow()
    start = now
    end = now
    data = ModelPerformanceData(
        model_id="model_X",
        metric="accuracy",
        value=0.85,
        evaluation_window_start=start,
        evaluation_window_end=end
    )
    feedback = ModelPerformanceFeedback(source="model_monitor", data=data)
    assert feedback.feedback_type == "model_performance"
    assert feedback.data.model_id == "model_X"
    assert feedback.data.value == 0.85
    assert feedback.data.strategy_id is None

    with pytest.raises(ValidationError):
        ModelPerformanceData(model_id="m1", metric="acc", value="high", evaluation_window_start=start, evaluation_window_end=end) # Invalid value type

def test_parameter_feedback_creation():
    """Test creation of ParameterFeedback."""
    data = ParameterFeedbackData(
        strategy_id="strat_C",
        parameter_name="stop_loss_pips",
        suggested_change=25,
        reasoning="Volatility increased",
        confidence=0.7
    )
    feedback = ParameterFeedback(source="optimizer", data=data)
    assert feedback.feedback_type == "parameter"
    assert feedback.data.strategy_id == "strat_C"
    assert feedback.data.suggested_change == 25
    assert feedback.data.confidence == 0.7

    # Test without optional fields
    data_minimal = ParameterFeedbackData(
        strategy_id="strat_D",
        parameter_name="ema_period"
    )
    feedback_minimal = ParameterFeedback(source="manual", data=data_minimal)
    assert feedback_minimal.data.suggested_change is None
    assert feedback_minimal.data.reasoning is None

