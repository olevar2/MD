"""
Health Check API Routes

This module defines the API routes for health checks.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Dict: Health status
    """
    return {"status": "ok"}

@router.get("/health/liveness")
async def liveness_check():
    """
    Liveness check endpoint.
    
    Returns:
        Dict: Liveness status
    """
    return {"status": "alive"}

@router.get("/health/readiness")
async def readiness_check():
    """
    Readiness check endpoint.
    
    Returns:
        Dict: Readiness status
    """
    return {"status": "ready"}