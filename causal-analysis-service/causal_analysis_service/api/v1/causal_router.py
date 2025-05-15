"""
API routes for causal analysis.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from causal_analysis_service.services.causal_analysis_service import CausalAnalysisService
from causal_analysis_service.models.causal_models import (
    CausalGraphRequest,
    CausalGraphResponse,
    InterventionEffectRequest,
    InterventionEffectResponse,
    CounterfactualScenarioRequest,
    CounterfactualScenarioResponse
)

# Create router
router = APIRouter(prefix="/api/v1/causal", tags=["Causal Analysis"])

# Dependency to get the causal analysis service
async def get_causal_analysis_service():
    """
    Get the causal analysis service.
    """
    # In a real implementation, this would get the service from a dependency injection container
    return None

@router.post("/causal-graph", response_model=CausalGraphResponse)
async def generate_causal_graph(
    request: CausalGraphRequest,
    service: CausalAnalysisService = Depends(get_causal_analysis_service)
):
    """
    Generate a causal graph from data.
    """
    return await service.generate_causal_graph(request)

@router.post("/intervention-effect", response_model=InterventionEffectResponse)
async def calculate_intervention_effect(
    request: InterventionEffectRequest,
    service: CausalAnalysisService = Depends(get_causal_analysis_service)
):
    """
    Calculate the effect of an intervention.
    """
    return await service.calculate_intervention_effect(request)

@router.post("/counterfactual-scenario", response_model=CounterfactualScenarioResponse)
async def generate_counterfactual_scenario(
    request: CounterfactualScenarioRequest,
    service: CausalAnalysisService = Depends(get_causal_analysis_service)
):
    """
    Generate a counterfactual scenario.
    """
    return await service.generate_counterfactual_scenario(request)