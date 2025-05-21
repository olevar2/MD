#!/usr/bin/env python3
"""
Main module for the Trading Gateway Service.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# gRPC specific imports
import asyncio
import grpc.aio
from concurrent import futures # Added for ThreadPoolExecutor
from trading_gateway_service.services.grpc_servicer import TradingGatewayServicer
# Assuming 'generated_protos' is in PYTHONPATH for this import to work:
from trading_gateway_service.trading_gateway_pb2_grpc import add_TradingGatewayServiceServicer_to_server
# Import for JWT Interceptor (assuming common-lib is in PYTHONPATH)
from common_lib.security.grpc_interceptors import JwtAuthServerInterceptor

GRPC_PORT = 50051
grpc_server = None

# Create FastAPI app
app = FastAPI(
    title="Trading Gateway Service",
    description="API for the Trading Gateway Service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}

# Liveness probe
@app.get("/health/liveness")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness probe for Kubernetes.
    
    Returns:
        Simple status response
    """
    return {"status": "alive"}

# Readiness probe
@app.get("/health/readiness")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness probe for Kubernetes.
    
    Returns:
        Simple status response
    """
    return {"status": "ready"}

# Metrics endpoint
@app.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.
    
    Returns:
        Metrics in Prometheus format
    """
    return Response(
        content="# Metrics will be provided by the Prometheus client",
        media_type="text/plain"
    )

@app.get("/api/v1/instruments")
async def instruments() -> Dict[str, Any]:
    """
    Instruments endpoint.
    
    Returns:
        Instruments data
    """
    # This is a mock implementation
    return {"status": "success", "data": []}

@app.get("/api/v1/accounts")
async def accounts() -> Dict[str, Any]:
    """
    Accounts endpoint.
    
    Returns:
        Accounts data
    """
    # This is a mock implementation
    return {"status": "success", "data": []}

# Test interaction endpoints
@app.get("/api/v1/test/portfolio-interaction")
async def test_portfolio_interaction() -> Dict[str, Any]:
        """
    Test interaction with the Portfolio Management Service.
    
    Returns:
        Test result
    """
    # This is a mock implementation
    return {"status": "success", "message": "Interaction with Portfolio Management Service successful"}

@app.get("/api/v1/test/risk-interaction")
async def test_risk_interaction() -> Dict[str, Any]:
        """
    Test interaction with the Risk Management Service.
    
    Returns:
        Test result
    """
    # This is a mock implementation
    return {"status": "success", "message": "Interaction with Risk Management Service successful"}

@app.get("/api/v1/test/feature-interaction")
async def test_feature_interaction() -> Dict[str, Any]:
        """
    Test interaction with the Feature Store Service.
    
    Returns:
        Test result
    """
    # This is a mock implementation
    return {"status": "success", "message": "Interaction with Feature Store Service successful"}

@app.get("/api/v1/test/ml-interaction")
async def test_ml_interaction() -> Dict[str, Any]:
        """
    Test interaction with the ML Integration Service.
    
    Returns:
        Test result
    """
    # This is a mock implementation
    return {"status": "success", "message": "Interaction with ML Integration Service successful"}

# gRPC Server setup
async def start_grpc_server():
    """Starts the gRPC server."""
    global grpc_server
    # Initialize your interceptor
    # You might fetch JWT validation parameters (secret, audience, issuer) from config here
    jwt_interceptor = JwtAuthServerInterceptor(
        # Example placeholder values; replace with actual configuration
        # secret_key="your-trading-gateway-secret",
        # required_audience="trading-gateway-service",
        # issuer="your-auth-issuer"
    )
    
    interceptors = [jwt_interceptor]
    
    grpc_server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=interceptors
    )
    # Alternatively, without ThreadPoolExecutor for a purely async server:
    # grpc_server = grpc.aio.server(interceptors=interceptors) 
    
    add_TradingGatewayServiceServicer_to_server(TradingGatewayServicer(), grpc_server)
    listen_addr = f'[::]:{GRPC_PORT}'
    grpc_server.add_insecure_port(listen_addr)
    logger.info(f"Starting gRPC server on {listen_addr}")
    await grpc_server.start()
    logger.info("gRPC server started.")
    try:
        await grpc_server.wait_for_termination()
    except asyncio.CancelledError:
        logger.info("gRPC server stopping due to task cancellation.")
        await grpc_server.stop(0) # Graceful stop
        logger.info("gRPC server stopped.")


@app.on_event("startup")
async def startup_event():
    """Run on FastAPI startup."""
    logger.info("FastAPI app startup: Initializing services...")
    # Start gRPC server in a background task
    loop = asyncio.get_event_loop()
    loop.create_task(start_grpc_server())
    logger.info("gRPC server startup task created.")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on FastAPI shutdown."""
    logger.info("FastAPI app shutdown: Cleaning up...")
    if grpc_server:
        logger.info("Attempting to stop gRPC server...")
        # This might require more sophisticated handling if wait_for_termination
        # is the main way the task is being kept alive.
        # For now, relying on task cancellation from asyncio loop shutdown or explicit stop.
        # The stop is handled in start_grpc_server's finally block on CancelledError.
        # Alternatively, one could call grpc_server.stop(grace_period_seconds) here.
        # For simplicity, we assume the task cancellation is sufficient.
        # A more robust approach might involve `grpc_server.stop(grace_period)`
        # and ensuring the asyncio task is properly cancelled and awaited.
        # Let's trigger a stop here if it's still running.
        stop_task = asyncio.create_task(grpc_server.stop(5)) # 5 second grace
        await stop_task
        logger.info("gRPC server shutdown process initiated.")


def main():
    """Main function to run the service."""
    import uvicorn
    # Note: PYTHONPATH needs to include the 'generated_protos' directory and project root.
    # Example: export PYTHONPATH=./generated_protos:.:$PYTHONPATH (if run from project root)
    # or ensure generated_protos is installed as a package or accessible.
    
    logger.info("Starting Trading Gateway Service (FastAPI & gRPC)")
    # Uvicorn will run the FastAPI app. The gRPC server is started via FastAPI's startup event.
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    # This setup ensures that PYTHONAPTH is correctly set for the imports
    # This is a common pattern if `generated_protos` is not installed as a package.
    # For production, you'd typically ensure PYTHONPATH is set in the environment
    # or use a proper package structure.
    
    # Get the absolute path to the project root (assuming this script is in trading-gateway-service/core)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    service_root = os.path.dirname(current_dir) # This should be 'trading-gateway-service'
    project_root = os.path.dirname(service_root) # This should be the repository root

    # Path to generated_protos directory
    generated_protos_path = os.path.join(project_root, "generated_protos")

    # Add project_root for `from trading_gateway_service...` and generated_protos for proto stubs
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if generated_protos_path not in sys.path:
        sys.path.insert(0, generated_protos_path)
    
    # Log the updated sys.path for debugging if needed
    # logger.info(f"Updated sys.path: {sys.path}")

    main()
