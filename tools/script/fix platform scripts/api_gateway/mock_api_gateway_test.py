#!/usr/bin/env python3
"""
Mock API Gateway Test

This script creates a mock API Gateway and backend services to test the API Gateway
implementation without requiring actual backend services to be running.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Dict, Any, Optional, List, Callable

import jwt
import uvicorn
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mock_api_gateway_test")

# Create mock backend services
mock_services = {}

class MockService:
    """
    Mock backend service.
    """

    def __init__(self, name: str, port: int):
        """
        Initialize the mock service.

        Args:
            name: Service name
            port: Port to run on
        """
        self.name = name
        self.port = port
        self.app = FastAPI(title=f"Mock {name}")
        self.setup_routes()

    def setup_routes(self):
        """
        Set up routes for the mock service.
        """
        @self.app.get("/health")
        async def health():
            return {"status": "ok"}

        @self.app.get("/api/v1/test")
        async def test():
            return {
                "service": self.name,
                "message": f"Hello from {self.name}",
                "timestamp": time.time()
            }

    def run(self):
        """
        Run the mock service.
        """
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

# Create mock services
mock_services["trading-gateway-service"] = MockService("trading-gateway-service", 8001)
mock_services["analysis-engine-service"] = MockService("analysis-engine-service", 8002)
mock_services["feature-store-service"] = MockService("feature-store-service", 8003)

# Create mock API Gateway
api_gateway = FastAPI(title="Mock API Gateway")

# Add CORS middleware
api_gateway.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT configuration
JWT_SECRET_KEY = "test_secret_key"
JWT_ALGORITHM = "HS256"

# API keys
API_KEYS = {
    "test_api_key": {
        "service_id": "test-service",
        "roles": ["internal"]
    }
}

# Role permissions
ROLE_PERMISSIONS = {
    "admin": [
        {
            "path": "/api/v1/*",
            "methods": ["GET", "POST", "PUT", "DELETE", "PATCH"]
        }
    ],
    "trader": [
        {
            "path": "/api/v1/trading/*",
            "methods": ["GET", "POST", "PUT", "DELETE"]
        },
        {
            "path": "/api/v1/market-data/*",
            "methods": ["GET"]
        }
    ],
    "internal": [
        {
            "path": "/api/v1/internal/*",
            "methods": ["GET", "POST", "PUT", "DELETE", "PATCH"]
        }
    ]
}

# Service configuration
SERVICES = {
    "trading-gateway-service": {
        "url": "http://localhost:8001/api/v1",
        "health_check_url": "http://localhost:8001/health",
        "endpoints": ["/api/v1/trading/*"]
    },
    "analysis-engine-service": {
        "url": "http://localhost:8002/api/v1",
        "health_check_url": "http://localhost:8002/health",
        "endpoints": ["/api/v1/analysis/*"]
    },
    "feature-store-service": {
        "url": "http://localhost:8003/api/v1",
        "health_check_url": "http://localhost:8003/health",
        "endpoints": ["/api/v1/features/*"]
    }
}

# Authentication middleware
@api_gateway.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    Authentication middleware.

    Args:
        request: FastAPI request
        call_next: Next middleware or route handler

    Returns:
        Response
    """
    # Get correlation ID and request ID
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Skip authentication for health check
    if request.url.path == "/health":
        return await call_next(request)

    # Check if path requires API key authentication
    if request.url.path.startswith("/api/v1/internal"):
        # Get API key from header
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "error": {
                        "code": "AUTHENTICATION_ERROR",
                        "message": "API key required"
                    },
                    "meta": {
                        "correlation_id": correlation_id,
                        "request_id": request_id,
                        "timestamp": time.time(),
                        "version": "1.0",
                        "service": "api-gateway"
                    }
                }
            )

        # Validate API key
        if api_key not in API_KEYS:
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "error": {
                        "code": "AUTHENTICATION_ERROR",
                        "message": "Invalid API key"
                    },
                    "meta": {
                        "correlation_id": correlation_id,
                        "request_id": request_id,
                        "timestamp": time.time(),
                        "version": "1.0",
                        "service": "api-gateway"
                    }
                }
            )

        # Add API key information to request state
        request.state.api_key_info = API_KEYS[api_key]

        # Continue processing
        return await call_next(request)

    # JWT authentication
    # Get token from header
    token = request.headers.get("Authorization")

    if not token:
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "error": {
                    "code": "AUTHENTICATION_ERROR",
                    "message": "Authentication required"
                },
                "meta": {
                    "correlation_id": correlation_id,
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "version": "1.0",
                    "service": "api-gateway"
                }
            }
        )

    # Remove "Bearer " prefix
    if token.startswith("Bearer "):
        token = token[7:]

    try:
        # Validate token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # Check permissions
        user_roles = payload.get("roles", [])
        has_permission = False

        for role in user_roles:
            if role not in ROLE_PERMISSIONS:
                continue

            permissions = ROLE_PERMISSIONS[role]

            for permission in permissions:
                permission_path = permission.get("path", "")
                permission_methods = permission.get("methods", [])

                if permission_path.endswith("*") and request.url.path.startswith(permission_path[:-1]):
                    if not permission_methods or request.method in permission_methods:
                        has_permission = True
                        break
                elif permission_path == request.url.path:
                    if not permission_methods or request.method in permission_methods:
                        has_permission = True
                        break

            if has_permission:
                break

        if not has_permission:
            return JSONResponse(
                status_code=403,
                content={
                    "status": "error",
                    "error": {
                        "code": "AUTHORIZATION_ERROR",
                        "message": "Insufficient permissions"
                    },
                    "meta": {
                        "correlation_id": correlation_id,
                        "request_id": request_id,
                        "timestamp": time.time(),
                        "version": "1.0",
                        "service": "api-gateway"
                    }
                }
            )

        # Add user information to request state
        request.state.user = payload

        # Continue processing
        return await call_next(request)
    except jwt.ExpiredSignatureError:
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "error": {
                    "code": "AUTHENTICATION_ERROR",
                    "message": "Token has expired"
                },
                "meta": {
                    "correlation_id": correlation_id,
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "version": "1.0",
                    "service": "api-gateway"
                }
            }
        )
    except jwt.InvalidTokenError:
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "error": {
                    "code": "AUTHENTICATION_ERROR",
                    "message": "Invalid token"
                },
                "meta": {
                    "correlation_id": correlation_id,
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "version": "1.0",
                    "service": "api-gateway"
                }
            }
        )
    except Exception as e:
        logger.error(f"Error during authentication: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Internal server error"
                },
                "meta": {
                    "correlation_id": correlation_id,
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "version": "1.0",
                    "service": "api-gateway"
                }
            }
        )

# Add correlation ID middleware
@api_gateway.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """
    Add correlation ID to request and response.

    Args:
        request: FastAPI request
        call_next: Next middleware or route handler

    Returns:
        Response
    """
    # Get correlation ID from header, or generate a new one
    correlation_id = request.headers.get("X-Correlation-ID")
    if not correlation_id:
        correlation_id = str(uuid.uuid4())

        # Add correlation ID to request headers
        request.headers.__dict__["_list"].append(
            (b"x-correlation-id", correlation_id.encode())
        )

    # Get or generate request ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())

        # Add request ID to request headers
        request.headers.__dict__["_list"].append(
            (b"x-request-id", request_id.encode())
        )

    # Process the request
    response = await call_next(request)

    # Add correlation ID and request ID to response headers
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Request-ID"] = request_id

    return response

# Add routes
@api_gateway.get("/health")
async def health():
    """
    Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "ok"}

@api_gateway.get("/api/v1/auth/token")
async def get_token():
    """
    Get a JWT token for testing.

    Returns:
        JWT token
    """
    # Create payload
    payload = {
        "sub": "test_user",
        "name": "Test User",
        "roles": ["admin"],
        "exp": int(time.time()) + 3600  # 1 hour expiration
    }

    # Generate token
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    return {
        "status": "success",
        "data": {
            "token": token,
            "expires_in": 3600
        },
        "meta": {
            "correlation_id": str(uuid.uuid4()),
            "request_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "version": "1.0",
            "service": "api-gateway"
        }
    }

@api_gateway.api_route("/api/v1/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_request(request: Request, service_name: str, path: str):
    """
    Proxy a request to a backend service.

    Args:
        request: Request
        service_name: Service name
        path: Request path

    Returns:
        Response from the backend service
    """
    # Get correlation ID and request ID
    correlation_id = request.headers.get("X-Correlation-ID", "")
    request_id = request.headers.get("X-Request-ID", "")

    # Check if service exists
    if service_name not in SERVICES:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "error": {
                    "code": "SERVICE_NOT_FOUND",
                    "message": f"Service {service_name} not found"
                },
                "meta": {
                    "correlation_id": correlation_id,
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "version": "1.0",
                    "service": "api-gateway"
                }
            }
        )

    # Mock response
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "data": {
                "service": service_name,
                "path": path,
                "method": request.method,
                "message": f"Mock response from {service_name}"
            },
            "meta": {
                "correlation_id": correlation_id,
                "request_id": request_id,
                "timestamp": time.time(),
                "version": "1.0",
                "service": "api-gateway"
            }
        }
    )

def run_api_gateway():
    """
    Run the mock API Gateway.
    """
    uvicorn.run(api_gateway, host="0.0.0.0", port=8000)

def main():
    """
    Main function.
    """
    logger.info("Starting mock API Gateway test...")
    
    # Run API Gateway
    run_api_gateway()

if __name__ == "__main__":
    main()