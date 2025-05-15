"""
Service Dependencies

This module provides dependency injection for services.
"""
import logging
from typing import Optional

from causal_analysis_service.services.causal_service import CausalService
from causal_analysis_service.repositories.causal_repository import CausalRepository

logger = logging.getLogger(__name__)

# Singleton instances
_causal_repository: Optional[CausalRepository] = None
_causal_service: Optional[CausalService] = None

def get_causal_repository() -> CausalRepository:
    """
    Get the causal repository instance.
    
    Returns:
        CausalRepository: The causal repository instance
    """
    global _causal_repository
    
    if _causal_repository is None:
        logger.info("Creating causal repository")
        _causal_repository = CausalRepository()
    
    return _causal_repository

def get_causal_service() -> CausalService:
    """
    Get the causal service instance.
    
    Returns:
        CausalService: The causal service instance
    """
    global _causal_service
    
    if _causal_service is None:
        logger.info("Creating causal service")
        repository = get_causal_repository()
        _causal_service = CausalService(repository=repository)
    
    return _causal_service