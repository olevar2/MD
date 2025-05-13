"""
API Endpoints for Causal Inference Service
"""
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
import pandas as pd
import networkx as nx
from analysis_engine.causal.services.causal_inference_service import CausalInferenceService
from analysis_engine.api.dependencies import get_causal_inference_service
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/v1/causal', tags=['Causal Analysis'])


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class CausalDiscoveryRequest(BaseModel):
    """
    CausalDiscoveryRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    data: List[Dict[str, Any]]
    method: str = 'granger'
    cache_key: Optional[str] = None
    force_refresh: bool = False


class EffectEstimationRequest(BaseModel):
    """
    EffectEstimationRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    data: List[Dict[str, Any]]
    relationships: List[tuple[str, str]]


class CounterfactualRequest(BaseModel):
    """
    CounterfactualRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    data: List[Dict[str, Any]]
    target: str
    interventions: Dict[str, Dict[str, float]]
    features: Optional[List[str]] = None


class CausalGraphResponse(BaseModel):
    """
    CausalGraphResponse class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    graph_json: Dict[str, Any]


class EffectEstimationResponse(BaseModel):
    """
    EffectEstimationResponse class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    effects: Dict[str, Dict[str, Any]]


class CounterfactualResponse(BaseModel):
    counterfactuals: Dict[str, List[Dict[str, Any]]]


def dataframe_to_list(df: pd.DataFrame) ->List[Dict[str, Any]]:
    """Convert DataFrame to list of records, handling NaNs."""
    df_reset = df.reset_index()
    df_reset = df_reset.replace({pd.NA: None, np.nan: None})
    return df_reset.to_dict(orient='records')


@with_exception_handling
def list_to_dataframe(data: List[Dict[str, Any]]) ->pd.DataFrame:
    """Convert list of records to DataFrame, setting timestamp index if present."""
    df = pd.DataFrame(data)
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        except Exception as e:
            logger.warning(f'Could not set timestamp index: {e}')
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
    df = df.infer_objects()
    return df


@router.post('/discover-structure', response_model=CausalGraphResponse)
@async_with_exception_handling
async def discover_structure(request: CausalDiscoveryRequest=Body(...),
    causal_service: CausalInferenceService=Depends(
    get_causal_inference_service)):
    """Discover causal structure from the provided data."""
    try:
        data_df = list_to_dataframe(request.data)
        if data_df.empty:
            raise HTTPException(status_code=400, detail=
                'Input data is empty or invalid.')
        graph = causal_service.discover_causal_structure(data=data_df,
            method=request.method, force_refresh=request.force_refresh,
            cache_key=request.cache_key)
        graph_json = nx.node_link_data(graph)
        return CausalGraphResponse(graph_json=graph_json)
    except ValueError as ve:
        logger.error(f'Value error during causal discovery: {ve}', exc_info
            =True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f'Error during causal discovery: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=
            'Internal server error during causal discovery.')


@router.post('/estimate-effects', response_model=EffectEstimationResponse)
@async_with_exception_handling
async def estimate_effects(request: EffectEstimationRequest=Body(...),
    causal_service: CausalInferenceService=Depends(
    get_causal_inference_service)):
    """Estimate causal effects for specified relationships."""
    try:
        data_df = list_to_dataframe(request.data)
        if data_df.empty:
            raise HTTPException(status_code=400, detail=
                'Input data is empty or invalid.')
        all_effects = {}
        for treatment, outcome in request.relationships:
            try:
                effect_result = causal_service.estimate_causal_effect(data=
                    data_df, treatment=treatment, outcome=outcome)
                all_effects[f'{treatment}->{outcome}'] = effect_result
            except Exception as inner_e:
                logger.warning(
                    f'Could not estimate effect for {treatment}->{outcome}: {inner_e}'
                    )
                all_effects[f'{treatment}->{outcome}'] = {'error': str(inner_e)
                    }
        return EffectEstimationResponse(effects=all_effects)
    except Exception as e:
        logger.error(f'Error during effect estimation: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=
            'Internal server error during effect estimation.')


@router.post('/generate-counterfactuals', response_model=CounterfactualResponse
    )
@async_with_exception_handling
async def generate_counterfactuals_api(request: CounterfactualRequest=Body(
    ...), causal_service: CausalInferenceService=Depends(
    get_causal_inference_service)):
    """Generate counterfactual scenarios based on interventions."""
    try:
        data_df = list_to_dataframe(request.data)
        if data_df.empty:
            raise HTTPException(status_code=400, detail=
                'Input data is empty or invalid.')
        if request.target not in data_df.columns:
            raise HTTPException(status_code=400, detail=
                f"Target variable '{request.target}' not found in data columns."
                )
        all_intervention_features = set()
        for scenario_interventions in request.interventions.values():
            all_intervention_features.update(scenario_interventions.keys())
        missing_features = all_intervention_features - set(data_df.columns)
        if missing_features:
            raise HTTPException(status_code=400, detail=
                f'Intervention features not found in data columns: {missing_features}'
                )
        cf_results_dfs = causal_service.generate_counterfactuals(data=
            data_df, target=request.target, interventions=request.
            interventions, features=request.features)
        cf_results_list = {name: dataframe_to_list(df) for name, df in
            cf_results_dfs.items()}
        return CounterfactualResponse(counterfactuals=cf_results_list)
    except Exception as e:
        logger.error(f'Error generating counterfactuals: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=
            'Internal server error during counterfactual generation.')
