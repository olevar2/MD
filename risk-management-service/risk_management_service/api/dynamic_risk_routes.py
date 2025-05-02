"""
API endpoints for the risk management service's dynamic risk adjustment features.
"""
from fastapi import APIRouter, Depends, HTTPException, status
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

# Import common-lib exceptions
from common_lib.exceptions import (
    ForexTradingPlatformError,
    DataValidationError,
    DataFetchError,
    ModelError,
    ServiceError,
    ServiceUnavailableError
)

logger = get_logger("risk-api-dynamic")
router = APIRouter(prefix="/api/risk/dynamic", tags=["dynamic-risk"])
risk_adjuster = DynamicRiskAdjuster()


@router.post("/strategy/weaknesses", response_model=StrategyWeaknessResponse)
async def analyze_strategy_weaknesses(
    request: StrategyWeaknessRequest,
    api_key: str = Depends(get_api_key)
):
    """Analyze a trading strategy for potential weaknesses across market regimes."""
    try:
        result = risk_adjuster.analyze_strategy_weaknesses(
            strategy_id=request.strategy_id,
            historical_performance=request.historical_performance,
            market_regimes_history=request.market_regimes_history
        )
        return result
    except DataValidationError as e:
        logger.error(f"Data validation error during strategy weakness analysis: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data validation error: {e.message}"
        )
    except DataFetchError as e:
        logger.error(f"Data fetch error during strategy weakness analysis: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data fetch error: {e.message}"
        )
    except ModelError as e:
        logger.error(f"Model error during strategy weakness analysis: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model error: {e.message}"
        )
    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {e.message}"
        )
    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during strategy weakness analysis: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Platform error: {e.message}"
        )
    except Exception as e:
        logger.error(f"Error analyzing strategy weaknesses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze strategy weaknesses: {str(e)}"
        )


@router.post("/ml/metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics_for_ml(
    request: RiskMetricsRequest,
    api_key: str = Depends(get_api_key)
):
    """Generate risk metrics in a format suitable for machine learning model integration."""
    try:
        result = risk_adjuster.generate_risk_metrics_for_ml(
            account_id=request.account_id,
            timeframe=request.timeframe
        )
        return result
    except DataValidationError as e:
        logger.error(f"Data validation error during ML risk metrics generation: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data validation error: {e.message}"
        )
    except DataFetchError as e:
        logger.error(f"Data fetch error during ML risk metrics generation: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data fetch error: {e.message}"
        )
    except ModelError as e:
        logger.error(f"Model error during ML risk metrics generation: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model error: {e.message}"
        )
    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {e.message}"
        )
    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during ML risk metrics generation: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Platform error: {e.message}"
        )
    except Exception as e:
        logger.error(f"Error generating ML risk metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate ML risk metrics: {str(e)}"
        )


@router.post("/ml/feedback", response_model=MLFeedbackResponse)
async def process_ml_feedback(
    request: MLFeedbackRequest,
    api_key: str = Depends(get_api_key)
):
    """Process feedback from ML model predictions to improve risk assessments."""
    try:
        result = risk_adjuster.process_ml_risk_feedback(
            ml_predictions=request.ml_predictions,
            actual_outcomes=request.actual_outcomes
        )
        return result
    except DataValidationError as e:
        logger.error(f"Data validation error during ML feedback processing: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data validation error: {e.message}"
        )
    except DataFetchError as e:
        logger.error(f"Data fetch error during ML feedback processing: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data fetch error: {e.message}"
        )
    except ModelError as e:
        logger.error(f"Model error during ML feedback processing: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model error: {e.message}"
        )
    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {e.message}"
        )
    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during ML feedback processing: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Platform error: {e.message}"
        )
    except Exception as e:
        logger.error(f"Error processing ML feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process ML feedback: {str(e)}"
        )


@router.post("/monitor/thresholds", response_model=MonitoringResponse)
async def monitor_risk_thresholds(
    request: RiskThresholdsRequest,
    api_key: str = Depends(get_api_key)
):
    """Monitor risk metrics against thresholds and generate alerts."""
    try:
        result = risk_adjuster.monitor_risk_thresholds(
            account_id=request.account_id,
            current_risk_metrics=request.current_risk_metrics,
            thresholds=request.thresholds
        )
        return result
    except DataValidationError as e:
        logger.error(f"Data validation error during risk threshold monitoring: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data validation error: {e.message}"
        )
    except DataFetchError as e:
        logger.error(f"Data fetch error during risk threshold monitoring: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data fetch error: {e.message}"
        )
    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {e.message}"
        )
    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during risk threshold monitoring: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Platform error: {e.message}"
        )
    except Exception as e:
        logger.error(f"Error monitoring risk thresholds: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to monitor risk thresholds: {str(e)}"
        )


@router.post("/control/automated", response_model=ActionResponse)
async def trigger_automated_control(
    request: AutomatedControlRequest,
    api_key: str = Depends(get_api_key)
):
    """Trigger automated risk control actions based on alerts."""
    try:
        result = risk_adjuster.trigger_automated_risk_control(
            account_id=request.account_id,
            alert_data=request.alert_data
        )
        return result
    except DataValidationError as e:
        logger.error(f"Data validation error during automated control: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data validation error: {e.message}"
        )
    except DataFetchError as e:
        logger.error(f"Data fetch error during automated control: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data fetch error: {e.message}"
        )
    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {e.message}"
        )
    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during automated control: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Platform error: {e.message}"
        )
    except Exception as e:
        logger.error(f"Error triggering automated control: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger automated control: {str(e)}"
        )
