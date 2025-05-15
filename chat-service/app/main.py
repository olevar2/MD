"""
Chat Service FastAPI Application
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from common_lib.correlation import CorrelationIdMiddleware
from common_lib.events import EventBus
from common_lib.exceptions import ServiceError
import logging
from .config.settings import Settings
from .api.v1.router import api_router
from .services.chat_service import ChatService

# Load settings
settings = Settings()

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Chat Service",
    description="Service for handling chat interactions in the forex trading platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Add correlation ID middleware
app.add_middleware(CorrelationIdMiddleware)

# Create event bus instance
event_bus = EventBus()

# Create chat service instance
chat_service = ChatService(event_bus)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    """Handle service-specific errors."""
    logger.error(f"Service error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting chat service")
    # Initialize any required resources here

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down chat service")
    # Cleanup any resources here

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}