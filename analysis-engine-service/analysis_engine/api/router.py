"""
API Router Registration

This module registers all API routers from the v1 package.

from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

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
from analysis_engine.api.v1.analysis_results_api import router as analysis_results_router
from analysis_engine.api.v1.market_regime_analysis import router as market_regime_router
from analysis_engine.api.v1.tool_effectiveness_analytics import router as tool_effectiveness_router
from analysis_engine.api.v1.adaptive_layer import router as adaptive_layer_router
from analysis_engine.api.v1.signal_quality import router as signal_quality_router
from analysis_engine.api.v1.enhanced_tool_effectiveness import router as enhanced_tool_router
from analysis_engine.api.v1.nlp_analysis import router as nlp_analysis_router
from analysis_engine.api.v1.correlation_analysis import router as correlation_analysis_router
from analysis_engine.api.v1.manipulation_detection import router as manipulation_detection_router
from analysis_engine.api.v1.enhanced_effectiveness_api import router as enhanced_effectiveness_api_router
from analysis_engine.api.routes.feedback_endpoints import router as feedback_endpoints_router
api_router = APIRouter(prefix='/api/v1')
api_router.include_router(analysis_results_router, prefix='/analysis')
api_router.include_router(market_regime_router, prefix='/market-regime')
api_router.include_router(tool_effectiveness_router, prefix=
    '/tool-effectiveness')
api_router.include_router(adaptive_layer_router, prefix='/adaptive')
api_router.include_router(signal_quality_router, prefix='/signal-quality')
api_router.include_router(enhanced_tool_router, prefix='/enhanced-tool')
api_router.include_router(nlp_analysis_router, prefix='/nlp')
api_router.include_router(correlation_analysis_router, prefix='/correlation')
api_router.include_router(manipulation_detection_router, prefix='/manipulation'
    )
api_router.include_router(enhanced_effectiveness_api_router, prefix=
    '/enhanced-effectiveness')
api_router.include_router(feedback_endpoints_router)
REMOVAL_DATE = datetime.date(2023, 12, 31)
days_until_removal = (REMOVAL_DATE - datetime.date.today()).days
days_message = (f'{days_until_removal} days' if days_until_removal > 0 else
    'PAST DUE')


@with_exception_handling
def show_deprecation_warning():
    """
    Show deprecation warning.
    
    """

    frame = inspect.currentframe().f_back
    if frame:
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        function_name = frame.f_code.co_name
        try:
            rel_path = os.path.relpath(filename)
        except ValueError:
            rel_path = filename
        warnings.warn(
            f"""DEPRECATION WARNING: analysis_engine.api.router will be removed after {REMOVAL_DATE} ({days_message}).
Please use analysis_engine.api.routes instead.
Called from {rel_path}:{lineno} in function '{function_name}'
Migration guide: https://confluence.example.com/display/DEV/API+Router+Migration+Guide"""
            , DeprecationWarning, stacklevel=2)


show_deprecation_warning()
try:
    from analysis_engine.core.deprecation_monitor import record_usage
    record_usage('analysis_engine.api.router')
except ImportError:
    pass
