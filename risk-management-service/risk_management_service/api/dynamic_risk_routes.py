"""
API endpoints for the risk management service's dynamic risk adjustment features.
"""
from fastapi import APIRouter, Depends, status
from typing import Dict, Any, List

from core_foundations.utils.logger import get_logger
from risk_management_service.db.connection import get_db_session
from risk_management_service.services.dynamic_risk_adjuster import DynamicRiskAdjuster
from risk_management_service.api.auth import get_api_key
from risk_management_service.models.risk_models import (
    StrategyWeaknessRequest,
    StrategyWeaknessResponse,
    RiskMetricsRequest,
    RiskMetricsResponse,
    MLFeedbackRequest,
    MLFeedbackResponse,
    RiskThresholdsRequest,
    MonitoringResponse,
    AutomatedControlRequest,
    ActionResponse
)

# Import error handling
from risk_management_service.error import (
    async_with_exception_handling,
    DataValidationError,
    DataFetchError,
    ModelError,
    ServiceUnavailableError,
    RiskCalculationError,
    RiskParameterError
)

logger = get_logger("risk-api-dynamic")
router = APIRouter(prefix="/api/risk/dynamic", tags=["dynamic-risk"])
risk_adjuster = DynamicRiskAdjuster()

# Error message constants
ERROR_ACCOUNT_ID_REQUIRED = "Account ID is required"
ERROR_STRATEGY_ID_REQUIRED = "Strategy ID is required"
ERROR_HISTORICAL_PERFORMANCE_REQUIRED = "Historical performance data is required"
ERROR_MARKET_REGIMES_REQUIRED = "Market regimes history is required"
ERROR_TIMEFRAME_REQUIRED = "Timeframe is required"
ERROR_ML_PREDICTIONS_REQUIRED = "ML predictions are required"
ERROR_ACTUAL_OUTCOMES_REQUIRED = "Actual outcomes are required"
ERROR_PREDICTIONS_OUTCOMES_MISMATCH = "Number of predictions must match number of outcomes"
ERROR_RISK_METRICS_REQUIRED = "Current risk metrics are required"
ERROR_THRESHOLDS_REQUIRED = "Risk thresholds are required"
ERROR_ALERT_DATA_REQUIRED = "Alert data is required"
ERROR_ALERT_TYPE_REQUIRED = "Alert type is required for each alert"
ERROR_SEVERITY_REQUIRED = "Severity is required for each alert"


@router.post("/strategy/weaknesses", response_model=StrategyWeaknessResponse)
@async_with_exception_handling
async def analyze_strategy_weaknesses(
    request: StrategyWeaknessRequest,
    api_key: str = Depends(get_api_key)
):
    """Analyze a trading strategy for potential weaknesses across market regimes."""
    # Validate request data
    if not request.strategy_id:
        raise DataValidationError(ERROR_STRATEGY_ID_REQUIRED)

    if not request.historical_performance:
        raise DataValidationError(ERROR_HISTORICAL_PERFORMANCE_REQUIRED)

    if not request.market_regimes_history:
        raise DataValidationError(ERROR_MARKET_REGIMES_REQUIRED)

    # Process the request
    result = risk_adjuster.analyze_strategy_weaknesses(
        strategy_id=request.strategy_id,
        historical_performance=request.historical_performance,
        market_regimes_history=request.market_regimes_history
    )

    return result


@router.post("/ml/metrics", response_model=RiskMetricsResponse)
@async_with_exception_handling
async def get_risk_metrics_for_ml(
    request: RiskMetricsRequest,
    api_key: str = Depends(get_api_key)
):
    """Generate risk metrics in a format suitable for machine learning model integration."""
    # Validate request data
    if not request.account_id:
        raise DataValidationError(ERROR_ACCOUNT_ID_REQUIRED)

    if not request.timeframe:
        raise DataValidationError(ERROR_TIMEFRAME_REQUIRED)

    # Process the request
    result = risk_adjuster.generate_risk_metrics_for_ml(
        account_id=request.account_id,
        timeframe=request.timeframe
    )

    return result


@router.post("/ml/feedback", response_model=MLFeedbackResponse)
@async_with_exception_handling
async def process_ml_feedback(
    request: MLFeedbackRequest,
    api_key: str = Depends(get_api_key)
):
    """Process feedback from ML model predictions to improve risk assessments."""
    # Validate request data
    if not request.ml_predictions:
        raise DataValidationError(ERROR_ML_PREDICTIONS_REQUIRED)

    if not request.actual_outcomes:
        raise DataValidationError(ERROR_ACTUAL_OUTCOMES_REQUIRED)

    if len(request.ml_predictions) != len(request.actual_outcomes):
        raise DataValidationError(ERROR_PREDICTIONS_OUTCOMES_MISMATCH)

    # Process the request
    result = risk_adjuster.process_ml_risk_feedback(
        ml_predictions=request.ml_predictions,
        actual_outcomes=request.actual_outcomes
    )

    return result


@router.post("/monitor/thresholds", response_model=MonitoringResponse)
@async_with_exception_handling
async def monitor_risk_thresholds(
    request: RiskThresholdsRequest,
    api_key: str = Depends(get_api_key)
):
    """Monitor risk metrics against thresholds and generate alerts."""
    # Validate request data
    if not request.account_id:
        raise DataValidationError(ERROR_ACCOUNT_ID_REQUIRED)

    if not request.current_risk_metrics:
        raise DataValidationError(ERROR_RISK_METRICS_REQUIRED)

    if not request.thresholds:
        raise DataValidationError(ERROR_THRESHOLDS_REQUIRED)

    # Process the request
    result = risk_adjuster.monitor_risk_thresholds(
        account_id=request.account_id,
        current_risk_metrics=request.current_risk_metrics,
        thresholds=request.thresholds
    )

    return result


@router.post("/control/automated", response_model=ActionResponse)
@async_with_exception_handling
async def trigger_automated_control(
    request: AutomatedControlRequest,
    api_key: str = Depends(get_api_key)
):
    """Trigger automated risk control actions based on alerts."""
    # Validate request data
    if not request.account_id:
        raise DataValidationError(ERROR_ACCOUNT_ID_REQUIRED)

    if not request.alert_data:
        raise DataValidationError(ERROR_ALERT_DATA_REQUIRED)

    # Check for required alert data fields
    for alert in request.alert_data:
        if not alert.get("alert_type"):
            raise DataValidationError(ERROR_ALERT_TYPE_REQUIRED)

        if not alert.get("severity"):
            raise DataValidationError(ERROR_SEVERITY_REQUIRED)

    # Process the request
    result = risk_adjuster.trigger_automated_risk_control(
        account_id=request.account_id,
        alert_data=request.alert_data
    )

    return result
