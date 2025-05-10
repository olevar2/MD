"""
ML Integration Service - Main module

This service integrates machine learning models with the trading strategy optimization workflow.
It provides:
- Model selection and deployment
- Strategy parameter optimization
- Performance prediction and visualization
- Feedback loop integration
- Enhanced visualization components
- Advanced optimization algorithms
- Comprehensive stress testing
"""
import logging
import os
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from prometheus_client import make_asgi_app

# Import common-lib exceptions
from common_lib.exceptions import (
    ForexTradingPlatformError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    ServiceError,
    ModelError,
    ModelTrainingError,
    ModelPredictionError
)

from ml_integration_service.api.router import api_router
from ml_integration_service.api.enhanced_routes import router as enhanced_router
from ml_integration_service.api.security import api_key_middleware
from ml_integration_service.api.metrics_integration import setup_metrics
from ml_integration_service.config.settings import settings
from ml_integration_service.visualization.model_performance_viz import ModelPerformanceVisualizer
from ml_integration_service.optimization.advanced_optimization import (
    RegimeAwareOptimizer,
    MultiObjectiveOptimizer,
    OnlineLearningOptimizer
)
from ml_integration_service.stress_testing.model_stress_tester import (
    ModelRobustnessTester,
    SensitivityAnalyzer,
    LoadTester
)
from ml_integration_service.clients.model_registry_client import ModelRegistryClient
from ml_integration_service.clients.client_factory import initialize_clients
from ml_integration_service.services.market_regime_detector import MarketRegimeDetector
from ml_integration_service.services.market_simulator import MarketSimulator

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ml_integration_service")

# Initialize FastAPI app
app = FastAPI(
    title="ML Integration Service",
    description="Machine Learning Integration Service for Strategy Optimization",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API key authentication middleware using common-lib security
app.middleware("http")(api_key_middleware)

# Set up metrics
setup_metrics(app, service_name="ml-integration-service")

# Include API routers
app.include_router(api_router, prefix="/api")
app.include_router(enhanced_router)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Import error handlers
from ml_integration_service.error_handlers import (
    forex_platform_exception_handler,
    data_validation_exception_handler,
    data_fetch_exception_handler,
    data_storage_exception_handler,
    model_exception_handler,
    model_training_exception_handler,
    model_prediction_exception_handler,
    service_exception_handler,
    validation_exception_handler,
    generic_exception_handler
)

# Register exception handlers
app.add_exception_handler(ForexTradingPlatformError, forex_platform_exception_handler)
app.add_exception_handler(DataValidationError, data_validation_exception_handler)
app.add_exception_handler(DataFetchError, data_fetch_exception_handler)
app.add_exception_handler(DataStorageError, data_storage_exception_handler)
app.add_exception_handler(ModelError, model_exception_handler)
app.add_exception_handler(ModelTrainingError, model_training_exception_handler)
app.add_exception_handler(ModelPredictionError, model_prediction_exception_handler)
app.add_exception_handler(ServiceError, service_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

@app.on_event("startup")
async def startup_event():
    """Initialize service components on startup"""
    logger.info("Starting ML Integration Service")

    # Initialize service clients
    initialize_clients()
    logger.info("Service clients initialized successfully")

    # Initialize visualization components
    app.state.performance_visualizer = ModelPerformanceVisualizer()

    # Initialize optimization components
    app.state.regime_optimizer = RegimeAwareOptimizer(
        regime_detector=MarketRegimeDetector(),
        regime_weights=settings.default_regime_weights
    )
    app.state.multi_objective_optimizer = MultiObjectiveOptimizer(
        objectives=settings.default_objectives
    )
    app.state.online_optimizer = OnlineLearningOptimizer(
        base_optimizer=app.state.regime_optimizer,
        learning_rate=settings.online_learning_rate,
        window_size=settings.optimization_window_size
    )

    # Initialize stress testing components
    app.state.model_tester = ModelRobustnessTester(
        model_client=ModelRegistryClient(settings.ml_workbench_url),
        market_simulator=MarketSimulator()
    )
    app.state.sensitivity_analyzer = SensitivityAnalyzer(
        n_samples=settings.sensitivity_samples
    )
    app.state.load_tester = LoadTester(
        model_client=ModelRegistryClient(settings.ml_workbench_url),
        max_latency_ms=settings.max_model_latency_ms
    )

    logger.info("Enhanced components initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down ML Integration Service")

    # Close service clients
    from ml_integration_service.clients.client_factory import get_analysis_engine_client, get_ml_workbench_client

    # Close the Analysis Engine client
    try:
        analysis_engine_client = get_analysis_engine_client()
        await analysis_engine_client.close()
        logger.info("Analysis Engine client closed successfully")
    except Exception as e:
        logger.error(f"Error closing Analysis Engine client: {str(e)}")

    # Close the ML Workbench client
    try:
        ml_workbench_client = get_ml_workbench_client()
        await ml_workbench_client.close()
        logger.info("ML Workbench client closed successfully")
    except Exception as e:
        logger.error(f"Error closing ML Workbench client: {str(e)}")

    # Close other clients as needed

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
