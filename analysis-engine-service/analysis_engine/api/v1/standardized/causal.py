"""
Standardized Causal Analysis API for Analysis Engine Service.

This module provides standardized API endpoints for causal analysis capabilities,
allowing other services to access causal insights, relationships, and counterfactual scenarios.

All endpoints follow the platform's standardized API design patterns.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException, Body, Request
from pydantic import BaseModel, Field

from analysis_engine.causal.services.causal_inference_service import CausalInferenceService
from analysis_engine.api.dependencies import get_causal_inference_service
from analysis_engine.core.exceptions_bridge import (
    ForexTradingPlatformError,
    AnalysisError,
    CausalAnalysisError,
    get_correlation_id_from_request
)
from analysis_engine.monitoring.structured_logging import get_structured_logger

# --- API Request/Response Models ---

class CausalDiscoveryRequest(BaseModel):
    """Request model for causal structure discovery"""
    data: List[Dict[str, Any]] = Field(..., description="Data as list of records")
    method: str = Field("granger", description="Causal discovery method to use (granger, pc)")
    cache_key: Optional[str] = Field(None, description="Key for caching results")
    force_refresh: bool = Field(False, description="Force refresh of cached results")

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {"timestamp": "2023-01-01T00:00:00Z", "EUR_USD": 1.0750, "GBP_USD": 1.2500, "USD_JPY": 130.50},
                    {"timestamp": "2023-01-01T01:00:00Z", "EUR_USD": 1.0755, "GBP_USD": 1.2510, "USD_JPY": 130.45}
                ],
                "method": "granger",
                "cache_key": "hourly_major_pairs",
                "force_refresh": false
            }
        }

class CausalGraphResponse(BaseModel):
    """Response model for causal graph"""
    graph_json: Dict[str, Any] = Field(..., description="Graph in node-link format")
    nodes: List[str] = Field(..., description="List of node names")
    edges: List[Dict[str, Any]] = Field(..., description="List of edges with metadata")
    discovery_method: str = Field(..., description="Method used for discovery")
    timestamp: datetime = Field(..., description="When the graph was generated")

class EffectEstimationRequest(BaseModel):
    """Request model for causal effect estimation"""
    data: List[Dict[str, Any]] = Field(..., description="Data as list of records")
    treatment: str = Field(..., description="Treatment variable")
    outcome: str = Field(..., description="Outcome variable")
    common_causes: Optional[List[str]] = Field(None, description="Common causes (confounders)")
    method: str = Field("backdoor.linear_regression", description="Estimation method")

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {"timestamp": "2023-01-01T00:00:00Z", "volatility": 0.12, "spread": 2.5, "volume": 1000, "return": 0.01},
                    {"timestamp": "2023-01-01T01:00:00Z", "volatility": 0.15, "spread": 3.0, "volume": 1200, "return": 0.02}
                ],
                "treatment": "volatility",
                "outcome": "return",
                "common_causes": ["spread", "volume"],
                "method": "backdoor.linear_regression"
            }
        }

class EffectEstimationResponse(BaseModel):
    """Response model for causal effect estimation"""
    treatment: str = Field(..., description="Treatment variable")
    outcome: str = Field(..., description="Outcome variable")
    effect_estimate: float = Field(..., description="Estimated causal effect")
    confidence_interval: Optional[List[float]] = Field(None, description="Confidence interval")
    p_value: Optional[float] = Field(None, description="P-value of the estimate")
    method: str = Field(..., description="Method used for estimation")
    timestamp: datetime = Field(..., description="When the estimation was performed")

class CounterfactualRequest(BaseModel):
    """Request model for counterfactual analysis"""
    data: List[Dict[str, Any]] = Field(..., description="Data as list of records")
    target: str = Field(..., description="Target variable for prediction")
    interventions: Dict[str, Dict[str, float]] = Field(..., description="Interventions by scenario")
    features: Optional[List[str]] = Field(None, description="Features to include in analysis")

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {"timestamp": "2023-01-01T00:00:00Z", "volatility": 0.12, "spread": 2.5, "volume": 1000, "return": 0.01},
                    {"timestamp": "2023-01-01T01:00:00Z", "volatility": 0.15, "spread": 3.0, "volume": 1200, "return": 0.02}
                ],
                "target": "return",
                "interventions": {
                    "high_volatility": {"volatility": 0.25},
                    "low_volatility": {"volatility": 0.05}
                },
                "features": ["volatility", "spread", "volume"]
            }
        }

class CounterfactualResponse(BaseModel):
    """Response model for counterfactual analysis"""
    target: str = Field(..., description="Target variable")
    baseline_prediction: float = Field(..., description="Baseline prediction without intervention")
    counterfactual_predictions: Dict[str, float] = Field(..., description="Predictions by scenario")
    differences: Dict[str, float] = Field(..., description="Differences from baseline by scenario")
    timestamp: datetime = Field(..., description="When the analysis was performed")

class CurrencyPairRequest(BaseModel):
    """Request model for currency pair relationship analysis"""
    price_data: Dict[str, Dict[str, Any]] = Field(..., description="Price data by currency pair")
    max_lag: int = Field(10, description="Maximum lag for analysis")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration parameters")

    class Config:
        schema_extra = {
            "example": {
                "price_data": {
                    "EUR_USD": {
                        "ohlc": [
                            {"timestamp": "2023-01-01T00:00:00Z", "open": 1.0750, "high": 1.0760, "low": 1.0745, "close": 1.0755},
                            {"timestamp": "2023-01-01T01:00:00Z", "open": 1.0755, "high": 1.0765, "low": 1.0750, "close": 1.0760}
                        ]
                    },
                    "GBP_USD": {
                        "ohlc": [
                            {"timestamp": "2023-01-01T00:00:00Z", "open": 1.2500, "high": 1.2510, "low": 1.2490, "close": 1.2505},
                            {"timestamp": "2023-01-01T01:00:00Z", "open": 1.2505, "high": 1.2515, "low": 1.2500, "close": 1.2510}
                        ]
                    }
                },
                "max_lag": 10,
                "config": {"significance_level": 0.05}
            }
        }

class SignalEnhancementRequest(BaseModel):
    """Request model for trading signal enhancement"""
    signals: List[Dict[str, Any]] = Field(..., description="Trading signals to enhance")
    market_data: Dict[str, List[Any]] = Field(..., description="Market data for enhancement")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration parameters")

    class Config:
        schema_extra = {
            "example": {
                "signals": [
                    {"timestamp": "2023-01-01T00:00:00Z", "pair": "EUR_USD", "direction": "buy", "confidence": 0.75},
                    {"timestamp": "2023-01-01T01:00:00Z", "pair": "GBP_USD", "direction": "sell", "confidence": 0.65}
                ],
                "market_data": {
                    "timestamp": ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z"],
                    "EUR_USD_volatility": [0.12, 0.15],
                    "GBP_USD_volatility": [0.10, 0.14],
                    "market_sentiment": [0.2, 0.1]
                },
                "config": {"enhancement_threshold": 0.6}
            }
        }

# --- Helper Functions ---

def list_to_dataframe(data_list: List[Dict[str, Any]]):
    """Convert a list of dictionaries to a pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame(data_list)

# --- API Router Setup ---

router = APIRouter(
    prefix="/v1/analysis/causal",
    tags=["Causal Analysis"]
)

# Setup logging
logger = get_structured_logger(__name__)

# --- API Endpoints ---

@router.post(
    "/discover-structure",
    response_model=CausalGraphResponse,
    summary="Discover causal structure",
    description="Discover causal structure from the provided data."
)
async def discover_structure(
    request: CausalDiscoveryRequest,
    request_obj: Request,
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Discover causal structure from the provided data.

    This endpoint uses causal discovery algorithms to identify causal relationships
    in the provided data, returning a graph representation of these relationships.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Convert list to DataFrame
        data_df = list_to_dataframe(request.data)
        if data_df.empty:
            logger.warning(
                "Input data is empty or invalid",
                extra={"correlation_id": correlation_id}
            )
            raise HTTPException(status_code=400, detail="Input data is empty or invalid.")

        # Discover causal structure
        graph = causal_service.discover_causal_structure(
            data=data_df,
            method=request.method,
            force_refresh=request.force_refresh,
            cache_key=request.cache_key
        )

        # Convert graph to response format
        import networkx as nx

        # Get nodes and edges
        nodes = list(graph.nodes())
        edges = [
            {
                "source": u,
                "target": v,
                "weight": data.get("weight", 1.0),
                "significance": data.get("significance", None)
            }
            for u, v, data in graph.edges(data=True)
        ]

        # Convert to node-link format for visualization
        graph_json = nx.node_link_data(graph)

        # Log successful discovery
        logger.info(
            f"Discovered causal structure with {len(nodes)} nodes and {len(edges)} edges",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        )

        return CausalGraphResponse(
            graph_json=graph_json,
            nodes=nodes,
            edges=edges,
            discovery_method=request.method,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log error
        logger.error(
            f"Error discovering causal structure: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        # Raise standardized error
        raise CausalAnalysisError(
            message=f"Error discovering causal structure: {str(e)}",
            correlation_id=correlation_id
        )

@router.post(
    "/estimate-effect",
    response_model=EffectEstimationResponse,
    summary="Estimate causal effect",
    description="Estimate the causal effect of a treatment on an outcome."
)
async def estimate_effect(
    request: EffectEstimationRequest,
    request_obj: Request,
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Estimate the causal effect of a treatment on an outcome.

    This endpoint uses causal inference techniques to estimate the effect
    of a treatment variable on an outcome variable, controlling for confounders.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Convert list to DataFrame
        data_df = list_to_dataframe(request.data)
        if data_df.empty:
            logger.warning(
                "Input data is empty or invalid",
                extra={"correlation_id": correlation_id}
            )
            raise HTTPException(status_code=400, detail="Input data is empty or invalid.")

        # Estimate causal effect
        effect_result = causal_service.estimate_causal_effect(
            data=data_df,
            treatment=request.treatment,
            outcome=request.outcome,
            common_causes=request.common_causes,
            method=request.method
        )

        if not effect_result:
            logger.warning(
                f"Failed to estimate causal effect of {request.treatment} on {request.outcome}",
                extra={
                    "correlation_id": correlation_id,
                    "treatment": request.treatment,
                    "outcome": request.outcome
                }
            )
            raise HTTPException(
                status_code=422,
                detail=f"Failed to estimate causal effect of {request.treatment} on {request.outcome}"
            )

        # Extract results
        effect_estimate = effect_result.get("effect", 0.0)
        confidence_interval = effect_result.get("confidence_interval")
        p_value = effect_result.get("p_value")

        # Log successful estimation
        logger.info(
            f"Estimated causal effect of {request.treatment} on {request.outcome}: {effect_estimate}",
            extra={
                "correlation_id": correlation_id,
                "treatment": request.treatment,
                "outcome": request.outcome,
                "effect": effect_estimate,
                "method": request.method
            }
        )

        return EffectEstimationResponse(
            treatment=request.treatment,
            outcome=request.outcome,
            effect_estimate=effect_estimate,
            confidence_interval=confidence_interval,
            p_value=p_value,
            method=request.method,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log error
        logger.error(
            f"Error estimating causal effect: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        # Raise standardized error
        raise CausalAnalysisError(
            message=f"Error estimating causal effect: {str(e)}",
            correlation_id=correlation_id
        )

@router.post(
    "/counterfactual-analysis",
    response_model=CounterfactualResponse,
    summary="Perform counterfactual analysis",
    description="Analyze counterfactual scenarios by simulating interventions."
)
async def counterfactual_analysis(
    request: CounterfactualRequest,
    request_obj: Request,
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Perform counterfactual analysis by simulating interventions.

    This endpoint uses causal models to predict outcomes under different
    hypothetical scenarios (interventions) compared to the baseline.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Convert list to DataFrame
        data_df = list_to_dataframe(request.data)
        if data_df.empty:
            logger.warning(
                "Input data is empty or invalid",
                extra={"correlation_id": correlation_id}
            )
            raise HTTPException(status_code=400, detail="Input data is empty or invalid.")

        # Perform counterfactual analysis
        counterfactual_results = causal_service.analyze_counterfactuals(
            data=data_df,
            target=request.target,
            interventions=request.interventions,
            features=request.features
        )

        if not counterfactual_results:
            logger.warning(
                f"Failed to perform counterfactual analysis for target {request.target}",
                extra={
                    "correlation_id": correlation_id,
                    "target": request.target,
                    "interventions": list(request.interventions.keys())
                }
            )
            raise HTTPException(
                status_code=422,
                detail=f"Failed to perform counterfactual analysis for target {request.target}"
            )

        # Extract results
        baseline_prediction = counterfactual_results.get("baseline", 0.0)
        counterfactual_predictions = counterfactual_results.get("predictions", {})

        # Calculate differences from baseline
        differences = {
            scenario: prediction - baseline_prediction
            for scenario, prediction in counterfactual_predictions.items()
        }

        # Log successful analysis
        logger.info(
            f"Performed counterfactual analysis for target {request.target} with {len(request.interventions)} scenarios",
            extra={
                "correlation_id": correlation_id,
                "target": request.target,
                "scenarios": list(request.interventions.keys()),
                "baseline": baseline_prediction
            }
        )

        return CounterfactualResponse(
            target=request.target,
            baseline_prediction=baseline_prediction,
            counterfactual_predictions=counterfactual_predictions,
            differences=differences,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log error
        logger.error(
            f"Error performing counterfactual analysis: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        # Raise standardized error
        raise CausalAnalysisError(
            message=f"Error performing counterfactual analysis: {str(e)}",
            correlation_id=correlation_id
        )

@router.post(
    "/currency-pair-relationships",
    summary="Analyze currency pair relationships",
    description="Discover causal relationships between currency pairs."
)
async def analyze_currency_pair_relationships(
    request: CurrencyPairRequest,
    request_obj: Request,
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Discover causal relationships between currency pairs.

    Uses Granger causality to identify which currency pairs lead or cause movements in others.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Convert dict of dicts to dict of DataFrames
        import pandas as pd
        price_data = {}
        for pair, data_dict in request.price_data.items():
            if "ohlc" in data_dict:
                price_data[pair] = pd.DataFrame(data_dict["ohlc"])
            else:
                price_data[pair] = pd.DataFrame(data_dict)

        # Initialize CausalInsightGenerator with config if provided
        from analysis_engine.causal.integration.causal_integration import CausalInsightGenerator
        insight_generator = CausalInsightGenerator(request.config)

        # Run analysis
        results = insight_generator.discover_currency_pair_relationships(
            price_data=price_data,
            max_lag=request.max_lag
        )

        # Log successful analysis
        logger.info(
            f"Analyzed relationships between {len(request.price_data)} currency pairs",
            extra={
                "correlation_id": correlation_id,
                "pairs": list(request.price_data.keys()),
                "max_lag": request.max_lag,
                "relationship_count": len(results.get("relationships", []))
            }
        )

        return results

    except Exception as e:
        # Log error
        logger.error(
            f"Error analyzing currency pair relationships: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        # Raise standardized error
        raise CausalAnalysisError(
            message=f"Error analyzing currency pair relationships: {str(e)}",
            correlation_id=correlation_id
        )

@router.post(
    "/enhance-trading-signals",
    summary="Enhance trading signals",
    description="Enhance trading signals using causal insights."
)
async def enhance_trading_signals(
    request: SignalEnhancementRequest,
    request_obj: Request,
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Enhance trading signals using causal insights.

    This endpoint uses causal models to enhance trading signals by incorporating
    causal relationships between market variables and signal performance.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Convert dict to DataFrame
        import pandas as pd
        market_data = pd.DataFrame(request.market_data)

        # Initialize CausalInsightGenerator with config if provided
        from analysis_engine.causal.integration.causal_integration import CausalInsightGenerator
        insight_generator = CausalInsightGenerator(request.config)

        # Run enhancement
        enhanced_signals = insight_generator.enhance_trading_signals(
            signals=request.signals,
            market_data=market_data
        )

        # Log successful enhancement
        logger.info(
            f"Enhanced {len(request.signals)} trading signals",
            extra={
                "correlation_id": correlation_id,
                "signal_count": len(request.signals),
                "enhanced_count": len(enhanced_signals)
            }
        )

        return {
            "enhanced_signals": enhanced_signals,
            "count": len(enhanced_signals),
            "causal_factors_considered": ["volatility", "trend", "sentiment", "correlations"]
        }

    except Exception as e:
        # Log error
        logger.error(
            f"Error enhancing trading signals: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        # Raise standardized error
        raise CausalAnalysisError(
            message=f"Error enhancing trading signals: {str(e)}",
            correlation_id=correlation_id
        )

@router.post(
    "/validate-relationship",
    summary="Validate causal relationship",
    description="Validate a hypothesized causal relationship using multiple methods."
)
async def validate_relationship(
    request_obj: Request,
    data: List[Dict[str, Any]] = Body(..., description="Data as list of records"),
    cause: str = Body(..., description="Hypothesized cause variable"),
    effect: str = Body(..., description="Hypothesized effect variable"),
    methods: Optional[List[str]] = Body(None, description="Validation methods to use"),
    confidence_threshold: float = Body(0.7, description="Threshold for confidence score"),
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Validate a hypothesized causal relationship using multiple methods.

    This endpoint uses various validation techniques to assess the credibility
    of a hypothesized causal relationship between two variables.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Convert list to DataFrame
        data_df = list_to_dataframe(data)
        if data_df.empty:
            logger.warning(
                "Input data is empty or invalid",
                extra={"correlation_id": correlation_id}
            )
            raise HTTPException(status_code=400, detail="Input data is empty or invalid.")

        # Use default methods if not provided
        if methods is None:
            methods = ['granger', 'cross_correlation', 'mutual_information']

        # Validate relationship
        validation_results = causal_service.validate_relationship(
            data=data_df,
            cause=cause,
            effect=effect,
            methods=methods,
            confidence_threshold=confidence_threshold
        )

        # Log validation results
        is_valid = validation_results.get("is_valid", False)
        confidence_score = validation_results.get("confidence_score", 0.0)
        logger.info(
            f"Validated causal relationship from {cause} to {effect}: {'valid' if is_valid else 'invalid'} (confidence: {confidence_score:.2f})",
            extra={
                "correlation_id": correlation_id,
                "cause": cause,
                "effect": effect,
                "is_valid": is_valid,
                "confidence_score": confidence_score,
                "methods": methods
            }
        )

        return validation_results

    except Exception as e:
        # Log error
        logger.error(
            f"Error validating causal relationship: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        # Raise standardized error
        raise CausalAnalysisError(
            message=f"Error validating causal relationship: {str(e)}",
            correlation_id=correlation_id
        )

# --- Legacy Router for Backward Compatibility ---

legacy_router = APIRouter(
    prefix="/api/v1/causal",
    tags=["Causal Analysis (Legacy)"]
)

# Add legacy endpoints that redirect to the standardized endpoints
# This ensures backward compatibility while encouraging migration to the new endpoints

@legacy_router.post("/discover-structure")
async def legacy_discover_structure(
    request: CausalDiscoveryRequest,
    request_obj: Request = None,
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Legacy endpoint for discovering causal structure.
    Consider migrating to /api/v1/analysis/causal/discover-structure
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/causal/discover-structure")
    return await discover_structure(request, request_obj, causal_service)

@legacy_router.post("/estimate-effect")
async def legacy_estimate_effect(
    request: EffectEstimationRequest,
    request_obj: Request = None,
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Legacy endpoint for estimating causal effect.
    Consider migrating to /api/v1/analysis/causal/estimate-effect
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/causal/estimate-effect")
    return await estimate_effect(request, request_obj, causal_service)

@legacy_router.post("/counterfactual-analysis")
async def legacy_counterfactual_analysis(
    request: CounterfactualRequest,
    request_obj: Request = None,
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Legacy endpoint for counterfactual analysis.
    Consider migrating to /api/v1/analysis/causal/counterfactual-analysis
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/causal/counterfactual-analysis")
    return await counterfactual_analysis(request, request_obj, causal_service)

@legacy_router.post("/currency-pair-relationships")
async def legacy_analyze_currency_pair_relationships(
    request: CurrencyPairRequest,
    request_obj: Request = None,
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Legacy endpoint for analyzing currency pair relationships.
    Consider migrating to /api/v1/analysis/causal/currency-pair-relationships
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/causal/currency-pair-relationships")
    return await analyze_currency_pair_relationships(request, request_obj, causal_service)

@legacy_router.post("/enhance-trading-signals")
async def legacy_enhance_trading_signals(
    request: SignalEnhancementRequest,
    request_obj: Request = None,
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Legacy endpoint for enhancing trading signals.
    Consider migrating to /api/v1/analysis/causal/enhance-trading-signals
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/causal/enhance-trading-signals")
    return await enhance_trading_signals(request, request_obj, causal_service)

@legacy_router.post("/validate-relationship")
async def legacy_validate_relationship(
    request_obj: Request = None,
    data: List[Dict[str, Any]] = Body(..., description="Data as list of records"),
    cause: str = Body(..., description="Hypothesized cause variable"),
    effect: str = Body(..., description="Hypothesized effect variable"),
    methods: Optional[List[str]] = Body(None, description="Validation methods to use"),
    confidence_threshold: float = Body(0.7, description="Threshold for confidence score"),
    causal_service: CausalInferenceService = Depends(get_causal_inference_service)
):
    """
    Legacy endpoint for validating causal relationship.
    Consider migrating to /api/v1/analysis/causal/validate-relationship
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/causal/validate-relationship")
    return await validate_relationship(
        request_obj, data, cause, effect, methods, confidence_threshold, causal_service
    )


# --- Setup Function for Router Registration ---

def setup_causal_routes(app: FastAPI) -> None:
    """
    Set up causal analysis routes.

    Args:
        app: FastAPI application
    """
    # Include standardized router
    app.include_router(router, prefix="/api")

    # Include legacy router for backward compatibility
    app.include_router(legacy_router)
