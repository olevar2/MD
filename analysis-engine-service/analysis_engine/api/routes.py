"""
API Routes Setup

This module configures all API routes for the Analysis Engine Service.
"""

from fastapi import FastAPI, APIRouter
from datetime import datetime

# Import routers
from analysis_engine.api.feedback_router import router as feedback_router
from analysis_engine.api.causal_visualization_api import router as causal_visualization_router
from analysis_engine.api.causal_analysis_api import router as causal_analysis_router
from analysis_engine.api.v1.analysis_results_api import router as analysis_results_router
from analysis_engine.api.v1.market_regime_analysis import router as market_regime_router
# from analysis_engine.api.v1.tool_effectiveness_analytics import router as tool_effectiveness_router # Deprecated
from analysis_engine.api.v1.adaptive_layer import router as adaptive_layer_router
from analysis_engine.api.v1.signal_quality import router as signal_quality_router
from analysis_engine.chat import setup_chat_routes
# from analysis_engine.api.v1.enhanced_tool_effectiveness import router as enhanced_tool_router # Deprecated
from analysis_engine.api.v1.nlp_analysis import router as nlp_analysis_router
from analysis_engine.api.v1.correlation_analysis import router as correlation_analysis_router
from analysis_engine.api.v1.manipulation_detection import router as manipulation_detection_router
# from analysis_engine.api.v1.enhanced_effectiveness_api import router as enhanced_effectiveness_api_router # Deprecated
from analysis_engine.api.v1.effectiveness_analysis_api import router as effectiveness_analysis_router # Consolidated
from analysis_engine.api.routes.feedback_endpoints import router as feedback_endpoints_router
from analysis_engine.api.v1.monitoring import router as monitoring_router
from analysis_engine.api.v1.health import router as health_router

# Import standardized routes
from analysis_engine.api.v1.standardized import setup_standardized_routes
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# Create main router
main_router = APIRouter()

# Root endpoint
@main_router.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Analysis Engine Service is running",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# Health check endpoint
@main_router.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

def setup_routes(app: FastAPI) -> None:
    """
    Set up all API routes for the application.

    Args:
        app (FastAPI): The FastAPI application instance
    """
    # Include main router
    app.include_router(main_router)

    # Include all other routers
    app.include_router(feedback_router)
    app.include_router(causal_visualization_router)
    app.include_router(causal_analysis_router)
    app.include_router(analysis_results_router, prefix="/api/v1/analysis")
    app.include_router(market_regime_router, prefix="/api/v1/market-regime")
    # app.include_router(tool_effectiveness_router, prefix="/api/v1/tool-effectiveness") # Deprecated
    app.include_router(effectiveness_analysis_router, prefix="/api/v1/effectiveness") # Consolidated
    app.include_router(adaptive_layer_router, prefix="/api/v1/adaptive")
    app.include_router(signal_quality_router, prefix="/api/v1/signal-quality")
    # app.include_router(enhanced_tool_router, prefix="/api/v1/enhanced-tool") # Deprecated
    app.include_router(nlp_analysis_router, prefix="/api/v1/nlp")
    app.include_router(correlation_analysis_router, prefix="/api/v1/correlation")
    app.include_router(manipulation_detection_router, prefix="/api/v1/manipulation-detection")
    # app.include_router(enhanced_effectiveness_api_router, prefix="/api/v1/enhanced-effectiveness") # Deprecated
    app.include_router(feedback_endpoints_router, prefix="/api/v1/feedback")
    app.include_router(monitoring_router, prefix="/api/v1")

    # Include standardized health API endpoints
    app.include_router(health_router, prefix="/api")

    # Set up chat routes
    setup_chat_routes(app)

    # Set up standardized routes
    setup_standardized_routes(app)

    logger.info("All API routes configured")
