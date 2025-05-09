"""
Standardized API Routes Setup

This module configures all standardized API routes for the Analysis Engine Service.
"""

from fastapi import FastAPI, APIRouter
from datetime import datetime

# Import standardized routers
from analysis_engine.api.v1.standardized.adaptive_layer import setup_adaptive_layer_routes
from analysis_engine.api.v1.standardized.market_regime import setup_market_regime_routes
from analysis_engine.api.v1.standardized.signal_quality import setup_signal_quality_routes
from analysis_engine.api.v1.standardized.nlp_analysis import setup_nlp_analysis_routes
from analysis_engine.api.v1.standardized.correlation_analysis import setup_correlation_analysis_routes
from analysis_engine.api.v1.standardized.manipulation_detection import setup_manipulation_detection_routes
from analysis_engine.api.v1.standardized.effectiveness import setup_effectiveness_routes
from analysis_engine.api.v1.standardized.feedback import setup_feedback_routes
from analysis_engine.api.v1.standardized.monitoring import setup_monitoring_routes
from analysis_engine.api.v1.standardized.causal import setup_causal_routes
from analysis_engine.api.v1.standardized.backtesting import setup_backtesting_routes
from analysis_engine.api.v1.standardized.health import setup_health_routes
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# Create main router
main_router = APIRouter(
    prefix="/v1/analysis",
    tags=["Analysis Engine"]
)

# Root endpoint
@main_router.get(
    "",
    summary="Analysis Engine Service Root",
    description="Root endpoint for the Analysis Engine Service API."
)
async def root():
    """Root endpoint for the Analysis Engine Service API."""
    return {
        "message": "Analysis Engine Service API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

def setup_standardized_routes(app: FastAPI) -> None:
    """
    Set up all standardized API routes for the application.

    Args:
        app (FastAPI): The FastAPI application instance
    """
    # Include main router
    app.include_router(main_router, prefix="/api")

    # Setup adaptive layer routes
    setup_adaptive_layer_routes(app)

    # Setup market regime routes
    setup_market_regime_routes(app)

    # Setup signal quality routes
    setup_signal_quality_routes(app)

    # Setup NLP analysis routes
    setup_nlp_analysis_routes(app)

    # Setup correlation analysis routes
    setup_correlation_analysis_routes(app)

    # Setup manipulation detection routes
    setup_manipulation_detection_routes(app)

    # Setup effectiveness routes
    setup_effectiveness_routes(app)

    # Setup feedback routes
    setup_feedback_routes(app)

    # Setup monitoring routes
    setup_monitoring_routes(app)

    # Setup causal routes
    setup_causal_routes(app)

    # Setup backtesting routes
    setup_backtesting_routes(app)

    # Setup health check routes
    setup_health_routes(app)

    logger.info("All standardized routes configured")
