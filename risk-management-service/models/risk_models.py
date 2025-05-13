"""
Pydantic models for the risk management service's dynamic risk adjustment features.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime


class StrategyWeaknessRequest(BaseModel):
    """Request model for analyzing strategy weaknesses."""
    strategy_id: str = Field(..., description="Unique identifier for the strategy")
    historical_performance: Dict[str, Any] = Field(
        ..., description="Dictionary containing performance metrics across time periods"
    )
    market_regimes_history: List[Dict[str, Any]] = Field(
        ..., description="List of historical market regime classifications"
    )


class StrategyWeaknessResponse(BaseModel):
    """Response model for strategy weakness analysis."""
    strategy_id: str
    performance_by_regime: Dict[str, Any]
    underperforming_regimes: List[Dict[str, Any]]
    risk_recommendations: List[Dict[str, Any]]
    strategy_adaptations: List[Dict[str, Any]]
    timestamp: str


class RiskMetricsRequest(BaseModel):
    """Request model for generating risk metrics for ML integration."""
    account_id: str = Field(..., description="Unique identifier for the account")
    timeframe: str = Field(
        "daily", description="Timeframe for risk metrics (daily, weekly, monthly)"
    )


class RiskMetricsResponse(BaseModel):
    """Response model for risk metrics for ML integration."""
    timestamp: str
    features: Dict[str, Any]
    feature_version: str
    timeframe: str


class MLFeedbackRequest(BaseModel):
    """Request model for processing ML model feedback."""
    ml_predictions: Dict[str, Any] = Field(
        ..., description="Predictions made by ML models"
    )
    actual_outcomes: Dict[str, Any] = Field(
        ..., description="Actual observed outcomes"
    )


class MLFeedbackResponse(BaseModel):
    """Response model for ML feedback processing."""
    timestamp: str
    accuracy_metrics: Dict[str, Any]
    bias_analysis: Dict[str, Any]
    improvement_recommendations: List[Dict[str, Any]]


class RiskThresholdsRequest(BaseModel):
    """Request model for monitoring risk thresholds."""
    account_id: str = Field(..., description="Unique identifier for the account")
    current_risk_metrics: Dict[str, Any] = Field(
        ..., description="Dictionary with current risk metrics"
    )
    thresholds: Dict[str, Any] = Field(
        ..., description="Dictionary with risk thresholds"
    )


class MonitoringResponse(BaseModel):
    """Response model for risk monitoring."""
    account_id: str
    timestamp: str
    alerts: List[Dict[str, Any]]
    warning_levels: Dict[str, Any]
    automatic_actions: List[Dict[str, Any]]
    has_critical_alerts: bool


class AutomatedControlRequest(BaseModel):
    """Request model for triggering automated risk control."""
    account_id: str = Field(..., description="Unique identifier for the account")
    alert_data: Dict[str, Any] = Field(
        ..., description="Alert data from monitor_risk_thresholds"
    )


class ActionResponse(BaseModel):
    """Response model for automated control actions."""
    account_id: str
    timestamp: str
    actions_taken: List[Dict[str, Any]]
    status: str
