"""
Causal Strategy Enhancer Adapter Module

This module provides an adapter implementation for the causal strategy enhancer interface,
helping to break circular dependencies between services.
"""
from typing import Dict, List, Any, Optional
import logging
import asyncio
import json
import os
import httpx
from datetime import datetime

from common_lib.strategy.interfaces import ICausalStrategyEnhancer
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class CausalStrategyEnhancerAdapter(ICausalStrategyEnhancer):
    """
    Adapter for causal strategy enhancer that implements the common interface.
    
    This adapter can either use a direct API connection to the analysis engine service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Get analysis engine service URL from config or environment
        analysis_engine_base_url = self.config.get(
            "analysis_engine_base_url", 
            os.environ.get("ANALYSIS_ENGINE_BASE_URL", "http://analysis-engine-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{analysis_engine_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
    
    async def enhance_strategy(
        self,
        strategy_id: str,
        enhancement_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance a strategy with additional functionality."""
        try:
            # Prepare request data
            request_data = {
                "strategy_id": strategy_id,
                "enhancement_type": enhancement_type,
                "parameters": parameters
            }
            
            # Send request
            response = await self.client.post(
                "/causal/enhance-strategy",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error enhancing strategy: {str(e)}")
            
            # Return fallback enhancement result
            return {
                "success": False,
                "error": str(e),
                "strategy_id": strategy_id
            }
    
    async def get_enhancement_types(self) -> List[Dict[str, Any]]:
        """Get available enhancement types."""
        try:
            # Send request
            response = await self.client.get(
                "/causal/enhancement-types"
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting enhancement types: {str(e)}")
            
            # Return fallback enhancement types
            return [
                {
                    "id": "causal",
                    "name": "Causal Enhancement",
                    "description": "Enhances strategy based on causal factor analysis",
                    "parameters": [
                        {
                            "name": "significance_threshold",
                            "type": "float",
                            "default": 0.05,
                            "description": "Threshold for statistical significance"
                        },
                        {
                            "name": "max_factors",
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum number of causal factors to consider"
                        }
                    ]
                }
            ]
    
    async def get_enhancement_history(
        self,
        strategy_id: str
    ) -> List[Dict[str, Any]]:
        """Get enhancement history for a strategy."""
        try:
            # Send request
            response = await self.client.get(
                f"/causal/enhancement-history/{strategy_id}"
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting enhancement history: {str(e)}")
            
            # Return empty list as fallback
            return []
    
    async def compare_enhancements(
        self,
        strategy_id: str,
        enhancement_ids: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple enhancements for a strategy."""
        try:
            # Prepare request data
            request_data = {
                "strategy_id": strategy_id,
                "enhancement_ids": enhancement_ids
            }
            
            # Send request
            response = await self.client.post(
                "/causal/compare-enhancements",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error comparing enhancements: {str(e)}")
            
            # Return fallback comparison result
            return {
                "strategy_id": strategy_id,
                "enhancements": enhancement_ids,
                "comparison": {},
                "error": str(e),
                "is_fallback": True
            }
    
    async def identify_causal_factors(
        self,
        strategy_id: str,
        data_period: Dict[str, Any],
        significance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Identify causal factors affecting strategy performance."""
        try:
            # Prepare request data
            request_data = {
                "strategy_id": strategy_id,
                "data_period": data_period,
                "significance_threshold": significance_threshold
            }
            
            # Send request
            response = await self.client.post(
                "/causal/identify-factors",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error identifying causal factors: {str(e)}")
            
            # Return fallback causal factors
            return {
                "strategy_id": strategy_id,
                "data_period": data_period,
                "significance_threshold": significance_threshold,
                "causal_factors": [],
                "error": str(e),
                "is_fallback": True
            }
    
    async def generate_causal_graph(
        self,
        strategy_id: str,
        data_period: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a causal graph for a strategy."""
        try:
            # Prepare request data
            request_data = {
                "strategy_id": strategy_id,
                "data_period": data_period
            }
            
            # Send request
            response = await self.client.post(
                "/causal/generate-graph",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error generating causal graph: {str(e)}")
            
            # Return fallback causal graph
            return {
                "strategy_id": strategy_id,
                "data_period": data_period,
                "nodes": [],
                "edges": [],
                "error": str(e),
                "is_fallback": True
            }
    
    async def apply_causal_enhancement(
        self,
        strategy_id: str,
        causal_factors: List[Dict[str, Any]],
        enhancement_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply causal enhancement to a strategy."""
        try:
            # Prepare request data
            request_data = {
                "strategy_id": strategy_id,
                "causal_factors": causal_factors,
                "enhancement_parameters": enhancement_parameters
            }
            
            # Send request
            response = await self.client.post(
                "/causal/apply-enhancement",
                json=request_data
            )
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except Exception as e:
            logger.error(f"Error applying causal enhancement: {str(e)}")
            
            # Return fallback enhancement result
            return {
                "strategy_id": strategy_id,
                "error": str(e),
                "enhancement_id": f"causal_error_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "enhancements": [],
                "is_fallback": True
            }
