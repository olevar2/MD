"""
Main module for Analysis Coordinator Service.

This module provides the FastAPI application for the Analysis Coordinator Service.
"""
import logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uuid
import time

from analysis_coordinator_service.api.v1.coordinator import router as coordinator_router
from analysis_coordinator_service.utils.dependency_injection import get_command_bus, get_query_bus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Analysis Coordinator Service",
    description="Service for coordinating analysis tasks in the forex trading platform",
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
    logger.info("Starting Analysis Coordinator Service")
    
    # Initialize command and query buses
    get_command_bus()
    get_query_bus()
    
    logger.info("CQRS buses initialized")

# Include routers
app.include_router(coordinator_router)

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)