"""
Main module for Market Analysis Service.

This module provides the FastAPI application for the Market Analysis Service.
"""
import logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uuid
import time

from market_analysis_service.api.v1.market_analysis import router as market_analysis_router
from market_analysis_service.utils.dependency_injection import get_command_bus, get_query_bus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Market Analysis Service",
    description="Service for market analysis in the forex trading platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Add middleware for request ID and logging
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    """
    Add request ID to request and log request/response.
    """
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Log request
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    # Measure request processing time
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    # Log response
    logger.info(f"Response {request_id}: {response.status_code} ({process_time:.3f}s)")
    
    return response

# Add health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

# Add readiness check endpoint
@app.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.
    """
    return {"status": "ready"}

# Initialize CQRS buses
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    """
    logger.info("Starting Market Analysis Service")
    
    # Initialize command and query buses
    get_command_bus()
    get_query_bus()
    
    logger.info("CQRS buses initialized")

# Include routers
app.include_router(market_analysis_router, prefix="/api/v1")

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)