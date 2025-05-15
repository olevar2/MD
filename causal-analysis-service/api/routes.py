"""
API routes for the Causal Analysis Service.

This module defines the API routes for the Causal Analysis Service, including
causal graph generation, intervention effect analysis, and counterfactual scenario generation.
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from common_lib.cqrs.commands import CommandBus
from common_lib.cqrs.queries import QueryBus
from causal_analysis_service.cqrs.commands import (
    GenerateCausalGraphCommand,
    AnalyzeInterventionEffectCommand,
    GenerateCounterfactualScenarioCommand,
    DiscoverCurrencyPairRelationshipsCommand,
    DiscoverRegimeChangeDriversCommand,
    EnhanceTradingSignalsCommand,
    AssessCorrelationBreakdownRiskCommand
)
from causal_analysis_service.cqrs.queries import (
    GetCausalGraphQuery,
    GetInterventionEffectQuery,
    GetCounterfactualScenarioQuery,
    GetCurrencyPairRelationshipsQuery,
    GetRegimeChangeDriversQuery,
    GetCorrelationBreakdownRiskQuery
)
from causal_analysis_service.models.causal_models import (
    CausalGraphRequest,
    CausalGraphResponse,
    InterventionEffectRequest,
    InterventionEffectResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    CurrencyPairRelationshipRequest,
    CurrencyPairRelationshipResponse,
    RegimeChangeDriverRequest,
    RegimeChangeDriverResponse,
    TradingSignalEnhancementRequest,
    TradingSignalEnhancementResponse,
    CorrelationBreakdownRiskRequest,
    CorrelationBreakdownRiskResponse
)
from causal_analysis_service.utils.correlation_id import get_correlation_id
from causal_analysis_service.utils.logging import get_logger
from causal_analysis_service.utils.dependency_injection import get_command_bus, get_query_bus

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["causal-analysis"])


@router.post("/causal-graph", response_model=Dict[str, Any])
async def generate_causal_graph(
    request: CausalGraphRequest,
    req: Request,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Generate a causal graph from the provided data.
    
    This endpoint analyzes the data to discover causal relationships and
    generates a directed graph representing these relationships.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to generate causal graph", extra={"correlation_id": correlation_id})
    
    try:
        command = GenerateCausalGraphCommand(
            correlation_id=correlation_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            algorithm=request.algorithm,
            parameters=request.parameters
        )
        
        graph_id = await command_bus.dispatch(command)
        
        return {"graph_id": graph_id}
    except Exception as e:
        logger.error(f"Error generating causal graph: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error generating causal graph: {str(e)}")


@router.get("/causal-graph/{graph_id}", response_model=CausalGraphResponse)
async def get_causal_graph(
    graph_id: str,
    req: Request,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get a causal graph by ID.
    
    This endpoint retrieves a previously generated causal graph.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to get causal graph {graph_id}", extra={"correlation_id": correlation_id})
    
    try:
        query = GetCausalGraphQuery(
            correlation_id=correlation_id,
            graph_id=graph_id
        )
        
        result = await query_bus.dispatch(query)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"Causal graph {graph_id} not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting causal graph: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error getting causal graph: {str(e)}")


@router.post("/intervention-effect", response_model=Dict[str, Any])
async def analyze_intervention_effect(
    request: InterventionEffectRequest,
    req: Request,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Analyze the effect of an intervention on the system.
    
    This endpoint estimates the causal effect of an intervention on
    the system based on the provided data and intervention details.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to analyze intervention effect", extra={"correlation_id": correlation_id})
    
    try:
        command = AnalyzeInterventionEffectCommand(
            correlation_id=correlation_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            treatment=request.treatment,
            outcome=request.outcome,
            confounders=request.confounders,
            algorithm=request.algorithm,
            parameters=request.parameters
        )
        
        effect_id = await command_bus.dispatch(command)
        
        return {"effect_id": effect_id}
    except Exception as e:
        logger.error(f"Error analyzing intervention effect: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error analyzing intervention effect: {str(e)}")


@router.get("/intervention-effect/{effect_id}", response_model=InterventionEffectResponse)
async def get_intervention_effect(
    effect_id: str,
    req: Request,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get an intervention effect by ID.
    
    This endpoint retrieves a previously analyzed intervention effect.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to get intervention effect {effect_id}", extra={"correlation_id": correlation_id})
    
    try:
        query = GetInterventionEffectQuery(
            correlation_id=correlation_id,
            effect_id=effect_id
        )
        
        result = await query_bus.dispatch(query)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"Intervention effect {effect_id} not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting intervention effect: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error getting intervention effect: {str(e)}")


@router.post("/counterfactual-scenario", response_model=Dict[str, Any])
async def generate_counterfactual_scenario(
    request: CounterfactualRequest,
    req: Request,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Generate a counterfactual scenario based on the intervention.
    
    This endpoint generates a counterfactual scenario by simulating
    what would have happened if the intervention had been applied.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to generate counterfactual scenario", extra={"correlation_id": correlation_id})
    
    try:
        command = GenerateCounterfactualScenarioCommand(
            correlation_id=correlation_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            intervention=request.intervention,
            target_variables=request.target_variables,
            algorithm=request.algorithm,
            parameters=request.parameters
        )
        
        counterfactual_id = await command_bus.dispatch(command)
        
        return {"counterfactual_id": counterfactual_id}
    except Exception as e:
        logger.error(f"Error generating counterfactual scenario: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error generating counterfactual scenario: {str(e)}")


@router.get("/counterfactual-scenario/{counterfactual_id}", response_model=CounterfactualResponse)
async def get_counterfactual_scenario(
    counterfactual_id: str,
    req: Request,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get a counterfactual scenario by ID.
    
    This endpoint retrieves a previously generated counterfactual scenario.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to get counterfactual scenario {counterfactual_id}", extra={"correlation_id": correlation_id})
    
    try:
        query = GetCounterfactualScenarioQuery(
            correlation_id=correlation_id,
            counterfactual_id=counterfactual_id
        )
        
        result = await query_bus.dispatch(query)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"Counterfactual scenario {counterfactual_id} not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting counterfactual scenario: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error getting counterfactual scenario: {str(e)}")


@router.post("/currency-pair-relationships", response_model=Dict[str, Any])
async def discover_currency_pair_relationships(
    request: CurrencyPairRelationshipRequest,
    req: Request,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Discover causal relationships between currency pairs.
    
    This endpoint uses Granger causality to identify which currency pairs
    lead or cause movements in others.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to discover currency pair relationships", extra={"correlation_id": correlation_id})
    
    try:
        command = DiscoverCurrencyPairRelationshipsCommand(
            correlation_id=correlation_id,
            symbols=request.symbols,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            variables=request.variables,
            algorithm=request.algorithm,
            parameters=request.parameters
        )
        
        relationship_id = await command_bus.dispatch(command)
        
        return {"relationship_id": relationship_id}
    except Exception as e:
        logger.error(f"Error discovering currency pair relationships: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error discovering currency pair relationships: {str(e)}")


@router.get("/currency-pair-relationships/{relationship_id}", response_model=CurrencyPairRelationshipResponse)
async def get_currency_pair_relationships(
    relationship_id: str,
    req: Request,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get currency pair relationships by ID.
    
    This endpoint retrieves previously discovered currency pair relationships.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to get currency pair relationships {relationship_id}", extra={"correlation_id": correlation_id})
    
    try:
        query = GetCurrencyPairRelationshipsQuery(
            correlation_id=correlation_id,
            relationship_id=relationship_id
        )
        
        result = await query_bus.dispatch(query)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"Currency pair relationships {relationship_id} not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting currency pair relationships: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error getting currency pair relationships: {str(e)}")


@router.post("/regime-change-drivers", response_model=Dict[str, Any])
async def discover_regime_change_drivers(
    request: RegimeChangeDriverRequest,
    req: Request,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Discover causal factors that drive market regime changes.
    
    This endpoint identifies which features have the strongest causal
    influence on regime transitions.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to discover regime change drivers", extra={"correlation_id": correlation_id})
    
    try:
        command = DiscoverRegimeChangeDriversCommand(
            correlation_id=correlation_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            regime_variable=request.regime_variable,
            potential_drivers=request.potential_drivers,
            algorithm=request.algorithm,
            parameters=request.parameters
        )
        
        driver_id = await command_bus.dispatch(command)
        
        return {"driver_id": driver_id}
    except Exception as e:
        logger.error(f"Error discovering regime change drivers: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error discovering regime change drivers: {str(e)}")


@router.get("/regime-change-drivers/{driver_id}", response_model=RegimeChangeDriverResponse)
async def get_regime_change_drivers(
    driver_id: str,
    req: Request,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get regime change drivers by ID.
    
    This endpoint retrieves previously discovered regime change drivers.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to get regime change drivers {driver_id}", extra={"correlation_id": correlation_id})
    
    try:
        query = GetRegimeChangeDriversQuery(
            correlation_id=correlation_id,
            driver_id=driver_id
        )
        
        result = await query_bus.dispatch(query)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"Regime change drivers {driver_id} not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting regime change drivers: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error getting regime change drivers: {str(e)}")


@router.post("/enhance-trading-signals", response_model=TradingSignalEnhancementResponse)
async def enhance_trading_signals(
    request: TradingSignalEnhancementRequest,
    req: Request,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Enhance trading signals with causal insights.
    
    This endpoint adds confidence adjustments, explanatory factors,
    conflicting signals, and expected duration based on causal analysis.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to enhance trading signals", extra={"correlation_id": correlation_id})
    
    try:
        command = EnhanceTradingSignalsCommand(
            correlation_id=correlation_id,
            market_data=request.market_data,
            signals=request.signals,
            parameters=request.parameters
        )
        
        result = await command_bus.dispatch(command)
        
        return result
    except Exception as e:
        logger.error(f"Error enhancing trading signals: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error enhancing trading signals: {str(e)}")


@router.post("/correlation-breakdown-risk", response_model=Dict[str, Any])
async def assess_correlation_breakdown_risk(
    request: CorrelationBreakdownRiskRequest,
    req: Request,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Assess correlation breakdown risk between assets.
    
    This endpoint uses causal models to identify pairs at risk of
    correlation breakdown and potential triggers.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to assess correlation breakdown risk", extra={"correlation_id": correlation_id})
    
    try:
        command = AssessCorrelationBreakdownRiskCommand(
            correlation_id=correlation_id,
            symbols=request.symbols,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            stress_scenarios=request.stress_scenarios,
            algorithm=request.algorithm,
            parameters=request.parameters
        )
        
        risk_id = await command_bus.dispatch(command)
        
        return {"risk_id": risk_id}
    except Exception as e:
        logger.error(f"Error assessing correlation breakdown risk: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error assessing correlation breakdown risk: {str(e)}")


@router.get("/correlation-breakdown-risk/{risk_id}", response_model=CorrelationBreakdownRiskResponse)
async def get_correlation_breakdown_risk(
    risk_id: str,
    req: Request,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get correlation breakdown risk by ID.
    
    This endpoint retrieves previously assessed correlation breakdown risk.
    """
    correlation_id = get_correlation_id(req)
    logger.info(f"Received request to get correlation breakdown risk {risk_id}", extra={"correlation_id": correlation_id})
    
    try:
        query = GetCorrelationBreakdownRiskQuery(
            correlation_id=correlation_id,
            risk_id=risk_id
        )
        
        result = await query_bus.dispatch(query)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"Correlation breakdown risk {risk_id} not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting correlation breakdown risk: {str(e)}", extra={"correlation_id": correlation_id})
        raise HTTPException(status_code=500, detail=f"Error getting correlation breakdown risk: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint.
    
    This endpoint returns the health status of the service.
    """
    return {
        "status": "healthy",
        "service": "causal-analysis-service",
        "timestamp": datetime.utcnow().isoformat()
    }