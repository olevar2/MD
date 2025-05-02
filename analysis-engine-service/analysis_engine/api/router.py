"""
API Router Registration

This module registers all API routers from the v1 package.

DEPRECATED: This module is deprecated and will be removed in a future version.
Please use analysis_engine.api.routes instead, which provides a more comprehensive
routing setup with the setup_routes() function.

Migration Guide:
1. If you're importing api_router from this module, use setup_routes() from analysis_engine.api.routes instead
2. If you need direct access to routers, import them directly from their respective modules
"""

import warnings
import inspect
import os
import datetime
from fastapi import APIRouter

# Import routers from modules
from analysis_engine.api.v1.analysis_results_api import router as analysis_results_router
from analysis_engine.api.v1.market_regime_analysis import router as market_regime_router
from analysis_engine.api.v1.tool_effectiveness_analytics import router as tool_effectiveness_router
from analysis_engine.api.v1.adaptive_layer import router as adaptive_layer_router
from analysis_engine.api.v1.signal_quality import router as signal_quality_router
from analysis_engine.api.v1.enhanced_tool_effectiveness import router as enhanced_tool_router

# Import Phase 7 routers
from analysis_engine.api.v1.nlp_analysis import router as nlp_analysis_router
from analysis_engine.api.v1.correlation_analysis import router as correlation_analysis_router
from analysis_engine.api.v1.manipulation_detection import router as manipulation_detection_router
from analysis_engine.api.v1.enhanced_effectiveness_api import router as enhanced_effectiveness_api_router

# Import Phase 8 routers
from analysis_engine.api.routes.feedback_endpoints import router as feedback_endpoints_router

# Main v1 router
api_router = APIRouter(prefix="/api/v1")

# Register API routers
api_router.include_router(analysis_results_router, prefix="/analysis")
api_router.include_router(market_regime_router, prefix="/market-regime")
api_router.include_router(tool_effectiveness_router, prefix="/tool-effectiveness")
api_router.include_router(adaptive_layer_router, prefix="/adaptive")
api_router.include_router(signal_quality_router, prefix="/signal-quality")
api_router.include_router(enhanced_tool_router, prefix="/enhanced-tool")

# Register Phase 7 routers
api_router.include_router(nlp_analysis_router, prefix="/nlp")
api_router.include_router(correlation_analysis_router, prefix="/correlation")
api_router.include_router(manipulation_detection_router, prefix="/manipulation")
api_router.include_router(enhanced_effectiveness_api_router, prefix="/enhanced-effectiveness")

# Register Phase 8 routers
api_router.include_router(feedback_endpoints_router)

# Calculate days until removal
REMOVAL_DATE = datetime.date(2023, 12, 31)
days_until_removal = (REMOVAL_DATE - datetime.date.today()).days
days_message = f"{days_until_removal} days" if days_until_removal > 0 else "PAST DUE"

# Show deprecation warning with file and line information
def _show_deprecation_warning():
    # Get the caller's frame
    frame = inspect.currentframe().f_back
    if frame:
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        function_name = frame.f_code.co_name

        # Get relative path for better readability
        try:
            rel_path = os.path.relpath(filename)
        except ValueError:
            rel_path = filename

        warnings.warn(
            f"DEPRECATION WARNING: analysis_engine.api.router will be removed after {REMOVAL_DATE} ({days_message}).\n"
            f"Please use analysis_engine.api.routes instead.\n"
            f"Called from {rel_path}:{lineno} in function '{function_name}'\n"
            f"Migration guide: https://confluence.example.com/display/DEV/API+Router+Migration+Guide",
            DeprecationWarning,
            stacklevel=2
        )

# Show deprecation warning when module is imported
_show_deprecation_warning()

# Record usage in a more structured way for monitoring
try:
    from analysis_engine.core.deprecation_monitor import record_usage
    record_usage("analysis_engine.api.router")
except ImportError:
    pass  # Silently continue if the module is not available
