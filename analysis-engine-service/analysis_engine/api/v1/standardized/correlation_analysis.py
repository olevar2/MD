"""
Standardized Correlation Analysis API for Analysis Engine Service.

This module provides standardized API endpoints for accessing enhanced correlation analysis
functionality, including dynamic timeframe analysis, lead-lag relationships,
and correlation breakdown detection.

All endpoints follow the platform's standardized API design patterns.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Body, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from core_foundations.models.auth import User
from analysis_engine.analysis.correlation.currency_correlation_enhanced import CurrencyCorrelationEnhanced
from analysis_engine.models.market_data import MarketData
from analysis_engine.db.connection import get_db_session
from analysis_engine.api.auth import get_current_user
from analysis_engine.core.exceptions_bridge import (
    ForexTradingPlatformError,
    AnalysisError,
    CorrelationAnalysisError,
    InsufficientDataError,
    get_correlation_id_from_request
)
from analysis_engine.monitoring.structured_logging import get_structured_logger

# --- API Request/Response Models ---

class CorrelationAnalysisRequest(BaseModel):
    """Request model for correlation analysis"""
    data: Dict[str, Dict[str, Any]] = Field(..., description="Currency pair data for correlation analysis")

    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "EUR/USD": {
                        "ohlc": [
                            {"timestamp": "2025-04-01T00:00:00", "open": 1.0765, "high": 1.0780, "low": 1.0760, "close": 1.0775, "volume": 1000},
                            {"timestamp": "2025-04-01T01:00:00", "open": 1.0775, "high": 1.0790, "low": 1.0770, "close": 1.0785, "volume": 1200}
                        ],
                        "metadata": {"timeframe": "1h"}
                    },
                    "GBP/USD": {
                        "ohlc": [
                            {"timestamp": "2025-04-01T00:00:00", "open": 1.2765, "high": 1.2780, "low": 1.2760, "close": 1.2775, "volume": 800},
                            {"timestamp": "2025-04-01T01:00:00", "open": 1.2775, "high": 1.2790, "low": 1.2770, "close": 1.2785, "volume": 900}
                        ],
                        "metadata": {"timeframe": "1h"}
                    }
                }
            }
        }

class AnalysisResponse(BaseModel):
    """Response model for correlation analysis"""
    status: str = Field(..., description="Status of the analysis")
    results: Dict[str, Any] = Field(..., description="Analysis results")
    timestamp: str = Field(..., description="ISO datetime string of when the analysis was performed")

# --- API Router Setup ---

router = APIRouter(
    prefix="/v1/analysis/correlations",
    tags=["Correlation Analysis"]
)

# Setup logging
logger = get_structured_logger(__name__)

# --- API Endpoints ---

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze currency correlations",
    description="Analyze correlations between currency pairs with enhanced features."
)
async def analyze_currency_correlations(
    request: CorrelationAnalysisRequest,
    request_obj: Request,
    window_sizes: Optional[List[int]] = Query(None, description="Correlation window sizes in days"),
    correlation_method: Optional[str] = Query("pearson", description="Correlation method (pearson or spearman)"),
    significance_threshold: Optional[float] = Query(0.7, description="Threshold for significant correlation"),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze correlations between currency pairs with enhanced features.

    Returns correlation matrices, lead-lag relationships, correlation breakdowns, and trading signals.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Configure analyzer parameters
        parameters = {}

        if window_sizes:
            parameters["correlation_windows"] = window_sizes

        if correlation_method:
            if correlation_method.lower() not in ["pearson", "spearman"]:
                raise HTTPException(
                    status_code=400,
                    detail="Correlation method must be 'pearson' or 'spearman'"
                )
            parameters["correlation_method"] = correlation_method.lower()

        if significance_threshold:
            parameters["significant_correlation_threshold"] = significance_threshold

        # Create analyzer with configured parameters
        analyzer = CurrencyCorrelationEnhanced(parameters)

        # Verify input data
        if not request.data or not isinstance(request.data, dict):
            raise InsufficientDataError(
                message="Request must contain currency pair data",
                correlation_id=correlation_id
            )

        if len(request.data) < 2:
            raise InsufficientDataError(
                message="At least two currency pairs required for correlation analysis",
                correlation_id=correlation_id
            )

        # Execute analysis
        result = analyzer.analyze(request.data)

        if not result.is_valid:
            raise CorrelationAnalysisError(
                message=result.result_data.get("error", "Analysis failed"),
                correlation_id=correlation_id
            )

        # Log successful analysis
        logger.info(
            f"Analyzed correlations for {len(request.data)} currency pairs",
            extra={
                "correlation_id": correlation_id,
                "currency_pairs": list(request.data.keys()),
                "correlation_method": correlation_method,
                "window_sizes": window_sizes
            }
        )

        # Prepare response
        response = AnalysisResponse(
            status="success",
            results=result.result_data,
            timestamp=datetime.now().isoformat()
        )

        return response

    except InsufficientDataError:
        # Re-raise the exception to be handled by the global exception handler
        raise

    except ForexTradingPlatformError:
        # Re-raise platform-specific exceptions
        raise

    except Exception as e:
        # Wrap other exceptions in CorrelationAnalysisError
        logger.error(
            f"Error analyzing currency correlations: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        raise CorrelationAnalysisError(
            message="Failed to analyze currency correlations",
            correlation_id=correlation_id
        )


@router.post(
    "/lead-lag",
    response_model=AnalysisResponse,
    summary="Analyze lead-lag relationships",
    description="Analyze lead-lag relationships between currency pairs."
)
async def analyze_lead_lag_relationships(
    request: CorrelationAnalysisRequest,
    request_obj: Request,
    max_lag: Optional[int] = Query(10, description="Maximum lag for Granger causality test"),
    significance: Optional[float] = Query(0.05, description="P-value threshold for significance"),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze lead-lag relationships between currency pairs.

    Returns detailed lead-lag relationships with statistical significance.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Configure analyzer parameters
        parameters = {
            "granger_maxlag": max_lag,
            "granger_significance": significance
        }

        # Create analyzer with configured parameters
        analyzer = CurrencyCorrelationEnhanced(parameters)

        # Verify input data
        if not request.data or not isinstance(request.data, dict):
            raise InsufficientDataError(
                message="Request must contain currency pair data",
                correlation_id=correlation_id
            )

        if len(request.data) < 2:
            raise InsufficientDataError(
                message="At least two currency pairs required for lead-lag analysis",
                correlation_id=correlation_id
            )

        # Execute analysis with focus on lead-lag relationships
        result = analyzer.analyze(request.data)

        if not result.is_valid:
            raise CorrelationAnalysisError(
                message=result.result_data.get("error", "Analysis failed"),
                correlation_id=correlation_id
            )

        # Extract only lead-lag relevant information
        lead_lag_results = {
            "lead_lag_relationships": result.result_data.get("lead_lag_relationships", []),
            "trading_signals": [signal for signal in result.result_data.get("trading_signals", [])
                              if signal.get("signal_type") == "lead_lag"]
        }

        # Log successful analysis
        logger.info(
            f"Analyzed lead-lag relationships for {len(request.data)} currency pairs",
            extra={
                "correlation_id": correlation_id,
                "currency_pairs": list(request.data.keys()),
                "max_lag": max_lag,
                "significance": significance
            }
        )

        # Prepare response
        response = AnalysisResponse(
            status="success",
            results=lead_lag_results,
            timestamp=datetime.now().isoformat()
        )

        return response

    except InsufficientDataError:
        # Re-raise the exception to be handled by the global exception handler
        raise

    except ForexTradingPlatformError:
        # Re-raise platform-specific exceptions
        raise

    except Exception as e:
        # Wrap other exceptions in CorrelationAnalysisError
        logger.error(
            f"Error analyzing lead-lag relationships: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        raise CorrelationAnalysisError(
            message="Failed to analyze lead-lag relationships",
            correlation_id=correlation_id
        )


@router.post(
    "/breakdown-detection",
    response_model=AnalysisResponse,
    summary="Detect correlation breakdowns",
    description="Detect significant breakdowns in correlation patterns between currency pairs."
)
async def detect_correlation_breakdowns(
    request: CorrelationAnalysisRequest,
    request_obj: Request,
    short_window: Optional[int] = Query(5, description="Short-term correlation window"),
    long_window: Optional[int] = Query(60, description="Long-term correlation window for comparison"),
    change_threshold: Optional[float] = Query(0.3, description="Threshold for significant correlation change"),
    current_user: User = Depends(get_current_user)
):
    """
    Detect significant breakdowns in correlation patterns between currency pairs.

    Returns detected correlation breakdowns with trading signals.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Configure analyzer parameters
        parameters = {
            "correlation_windows": [short_window, long_window],
            "correlation_change_threshold": change_threshold
        }

        # Create analyzer with configured parameters
        analyzer = CurrencyCorrelationEnhanced(parameters)

        # Verify input data
        if not request.data or not isinstance(request.data, dict):
            raise InsufficientDataError(
                message="Request must contain currency pair data",
                correlation_id=correlation_id
            )

        if len(request.data) < 2:
            raise InsufficientDataError(
                message="At least two currency pairs required for correlation analysis",
                correlation_id=correlation_id
            )

        # Execute analysis
        result = analyzer.analyze(request.data)

        if not result.is_valid:
            raise CorrelationAnalysisError(
                message=result.result_data.get("error", "Analysis failed"),
                correlation_id=correlation_id
            )

        # Extract only breakdown-relevant information
        breakdown_results = {
            "current_window": result.result_data.get("current_window"),
            "historical_window": result.result_data.get("historical_window"),
            "correlation_breakdowns": result.result_data.get("correlation_breakdowns", []),
            "trading_signals": [signal for signal in result.result_data.get("trading_signals", [])
                              if signal.get("signal_type") == "correlation_breakdown"]
        }

        # Log successful analysis
        logger.info(
            f"Detected correlation breakdowns for {len(request.data)} currency pairs",
            extra={
                "correlation_id": correlation_id,
                "currency_pairs": list(request.data.keys()),
                "short_window": short_window,
                "long_window": long_window,
                "change_threshold": change_threshold
            }
        )

        # Prepare response
        response = AnalysisResponse(
            status="success",
            results=breakdown_results,
            timestamp=datetime.now().isoformat()
        )

        return response

    except InsufficientDataError:
        # Re-raise the exception to be handled by the global exception handler
        raise

    except ForexTradingPlatformError:
        # Re-raise platform-specific exceptions
        raise

    except Exception as e:
        # Wrap other exceptions in CorrelationAnalysisError
        logger.error(
            f"Error detecting correlation breakdowns: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        raise CorrelationAnalysisError(
            message="Failed to detect correlation breakdowns",
            correlation_id=correlation_id
        )


@router.post(
    "/cointegration",
    response_model=AnalysisResponse,
    summary="Test pair cointegration",
    description="Test for cointegration between currency pairs."
)
async def test_pair_cointegration(
    request: CorrelationAnalysisRequest,
    request_obj: Request,
    significance: Optional[float] = Query(0.05, description="P-value threshold for cointegration significance"),
    current_user: User = Depends(get_current_user)
):
    """
    Test for cointegration between currency pairs.

    Returns cointegration test results and related trading signals.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Configure analyzer parameters
        parameters = {
            "cointegration_significance": significance
        }

        # Create analyzer with configured parameters
        analyzer = CurrencyCorrelationEnhanced(parameters)

        # Verify input data
        if not request.data or not isinstance(request.data, dict):
            raise InsufficientDataError(
                message="Request must contain currency pair data",
                correlation_id=correlation_id
            )

        if len(request.data) < 2:
            raise InsufficientDataError(
                message="At least two currency pairs required for cointegration analysis",
                correlation_id=correlation_id
            )

        # Execute analysis
        result = analyzer.analyze(request.data)

        if not result.is_valid:
            raise CorrelationAnalysisError(
                message=result.result_data.get("error", "Analysis failed"),
                correlation_id=correlation_id
            )

        # Extract only cointegration-relevant information
        cointegration_results = {
            "cointegration_tests": result.result_data.get("cointegration_tests", []),
            "trading_signals": [signal for signal in result.result_data.get("trading_signals", [])
                              if signal.get("signal_type") == "cointegration"]
        }

        # Log successful analysis
        logger.info(
            f"Tested cointegration for {len(request.data)} currency pairs",
            extra={
                "correlation_id": correlation_id,
                "currency_pairs": list(request.data.keys()),
                "significance": significance
            }
        )

        # Prepare response
        response = AnalysisResponse(
            status="success",
            results=cointegration_results,
            timestamp=datetime.now().isoformat()
        )

        return response

    except InsufficientDataError:
        # Re-raise the exception to be handled by the global exception handler
        raise

    except ForexTradingPlatformError:
        # Re-raise platform-specific exceptions
        raise

    except Exception as e:
        # Wrap other exceptions in CorrelationAnalysisError
        logger.error(
            f"Error testing cointegration: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        raise CorrelationAnalysisError(
            message="Failed to test cointegration",
            correlation_id=correlation_id
        )


# --- Legacy Router for Backward Compatibility ---

legacy_router = APIRouter(
    prefix="/api/v1/correlation",
    tags=["Correlation Analysis (Legacy)"]
)

# Add legacy endpoints that redirect to the standardized endpoints
# This ensures backward compatibility while encouraging migration to the new endpoints

@legacy_router.post("/analyze", response_model=AnalysisResponse)
async def legacy_analyze_currency_correlations(
    data: Dict[str, Any],
    window_sizes: Optional[List[int]] = Query(None, description="Correlation window sizes in days"),
    correlation_method: Optional[str] = Query("pearson", description="Correlation method (pearson or spearman)"),
    significance_threshold: Optional[float] = Query(0.7, description="Threshold for significant correlation"),
    request_obj: Request = None,
    current_user: User = Depends(get_current_user)
):
    """
    Legacy endpoint for analyzing currency correlations.
    Consider migrating to /api/v1/analysis/correlations/analyze
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/correlations/analyze")

    # Convert to standardized request format
    request = CorrelationAnalysisRequest(data=data)
    return await analyze_currency_correlations(
        request, request_obj, window_sizes, correlation_method, significance_threshold, current_user
    )

@legacy_router.post("/lead-lag", response_model=AnalysisResponse)
async def legacy_analyze_lead_lag_relationships(
    data: Dict[str, Any],
    max_lag: Optional[int] = Query(10, description="Maximum lag for Granger causality test"),
    significance: Optional[float] = Query(0.05, description="P-value threshold for significance"),
    request_obj: Request = None,
    current_user: User = Depends(get_current_user)
):
    """
    Legacy endpoint for analyzing lead-lag relationships.
    Consider migrating to /api/v1/analysis/correlations/lead-lag
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/correlations/lead-lag")

    # Convert to standardized request format
    request = CorrelationAnalysisRequest(data=data)
    return await analyze_lead_lag_relationships(request, request_obj, max_lag, significance, current_user)

@legacy_router.post("/breakdown-detection", response_model=AnalysisResponse)
async def legacy_detect_correlation_breakdowns(
    data: Dict[str, Any],
    short_window: Optional[int] = Query(5, description="Short-term correlation window"),
    long_window: Optional[int] = Query(60, description="Long-term correlation window for comparison"),
    change_threshold: Optional[float] = Query(0.3, description="Threshold for significant correlation change"),
    request_obj: Request = None,
    current_user: User = Depends(get_current_user)
):
    """
    Legacy endpoint for detecting correlation breakdowns.
    Consider migrating to /api/v1/analysis/correlations/breakdown-detection
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/correlations/breakdown-detection")

    # Convert to standardized request format
    request = CorrelationAnalysisRequest(data=data)
    return await detect_correlation_breakdowns(
        request, request_obj, short_window, long_window, change_threshold, current_user
    )

@legacy_router.post("/cointegration", response_model=AnalysisResponse)
async def legacy_test_pair_cointegration(
    data: Dict[str, Any],
    significance: Optional[float] = Query(0.05, description="P-value threshold for cointegration significance"),
    request_obj: Request = None,
    current_user: User = Depends(get_current_user)
):
    """
    Legacy endpoint for testing cointegration.
    Consider migrating to /api/v1/analysis/correlations/cointegration
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/correlations/cointegration")

    # Convert to standardized request format
    request = CorrelationAnalysisRequest(data=data)
    return await test_pair_cointegration(request, request_obj, significance, current_user)


# --- Setup Function for Router Registration ---

def setup_correlation_analysis_routes(app: FastAPI) -> None:
    """
    Set up correlation analysis routes.

    Args:
        app: FastAPI application
    """
    # Include standardized router
    app.include_router(router, prefix="/api")

    # Include legacy router for backward compatibility
    app.include_router(legacy_router)
