"""
API Dependencies Module

This module provides dependency functions for the API endpoints.
"""

import logging
from typing import Optional

from fastapi import Request, HTTPException, Depends, status
from analysis_engine.causal.services.causal_inference_service import CausalInferenceService
from common_lib.interfaces.analysis_engine import IAnalysisProvider, IIndicatorProvider, IPatternRecognizer

# Configure logging
logger = logging.getLogger(__name__)


def get_causal_inference_service(request: Request) -> CausalInferenceService:
    """Dependency function to get the CausalInferenceService instance."""
    try:
        # Resolve using the container stored in app state
        service = request.app.state.service_container.resolve(CausalInferenceService)
        if not service:
            raise HTTPException(status_code=503, detail="CausalInferenceService not available")
        return service
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Error resolving CausalInferenceService: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not resolve CausalInferenceService")


async def get_analysis_provider(request: Request) -> IAnalysisProvider:
    """
    Dependency function to provide the analysis provider adapter.

    Args:
        request: The FastAPI request object

    Returns:
        An instance of IAnalysisProvider

    Raises:
        HTTPException: If the adapter factory is not initialized
    """
    if not hasattr(request.app.state, "service_container"):
        logger.error("Service container not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service container not initialized"
        )

    try:
        adapter_factory = request.app.state.service_container.get_service("adapter_factory")
        return adapter_factory.get_analysis_provider()
    except Exception as e:
        logger.error(f"Error getting analysis provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Analysis provider not available: {str(e)}"
        )


async def get_indicator_provider(request: Request) -> IIndicatorProvider:
    """
    Dependency function to provide the indicator provider adapter.

    Args:
        request: The FastAPI request object

    Returns:
        An instance of IIndicatorProvider

    Raises:
        HTTPException: If the adapter factory is not initialized
    """
    if not hasattr(request.app.state, "service_container"):
        logger.error("Service container not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service container not initialized"
        )

    try:
        adapter_factory = request.app.state.service_container.get_service("adapter_factory")
        return adapter_factory.get_indicator_provider()
    except Exception as e:
        logger.error(f"Error getting indicator provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Indicator provider not available: {str(e)}"
        )


async def get_pattern_recognizer(request: Request) -> IPatternRecognizer:
    """
    Dependency function to provide the pattern recognizer adapter.

    Args:
        request: The FastAPI request object

    Returns:
        An instance of IPatternRecognizer

    Raises:
        HTTPException: If the adapter factory is not initialized
    """
    if not hasattr(request.app.state, "service_container"):
        logger.error("Service container not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service container not initialized"
        )

    try:
        adapter_factory = request.app.state.service_container.get_service("adapter_factory")
        return adapter_factory.get_pattern_recognizer()
    except Exception as e:
        logger.error(f"Error getting pattern recognizer: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pattern recognizer not available: {str(e)}"
        )
