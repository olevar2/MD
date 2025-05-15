from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any, Optional

from causal_analysis_service.core.service_dependencies import get_causal_analysis_service
from causal_analysis_service.services.causal_analysis_service import CausalAnalysisService
from causal_analysis_service.models.causal_models import (
    CausalGraphRequest,
    CausalGraphResponse,
    InterventionEffectRequest,
    InterventionEffectResponse,
    CounterfactualScenarioRequest,
    CounterfactualScenarioResponse
)

router = APIRouter(prefix="/causal", tags=["causal-analysis"])

@router.post("/causal-graph", response_model=CausalGraphResponse)
async def generate_causal_graph(
    request: CausalGraphRequest,
    causal_analysis_service: CausalAnalysisService = Depends(get_causal_analysis_service)
):
    """
    Generate a causal graph from the provided data.
    """
    try:
        return await causal_analysis_service.generate_causal_graph(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate causal graph: {str(e)}")

@router.post("/intervention-effect", response_model=InterventionEffectResponse)
async def calculate_intervention_effect(
    request: InterventionEffectRequest,
    causal_analysis_service: CausalAnalysisService = Depends(get_causal_analysis_service)
):
    """
    Calculate the effect of an intervention on the causal graph.
    """
    try:
        return await causal_analysis_service.calculate_intervention_effect(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate intervention effect: {str(e)}")

@router.post("/counterfactual-scenario", response_model=CounterfactualScenarioResponse)
async def generate_counterfactual_scenario(
    request: CounterfactualScenarioRequest,
    causal_analysis_service: CausalAnalysisService = Depends(get_causal_analysis_service)
):
    """
    Generate a counterfactual scenario based on the causal graph.
    """
    try:
        return await causal_analysis_service.generate_counterfactual_scenario(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate counterfactual scenario: {str(e)}")