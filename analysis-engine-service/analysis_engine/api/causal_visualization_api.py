"""
Causal Visualization Dashboard API

This module provides API endpoints that connect the causal visualization components
to the UI service for interactive visualization of causal relationships and insights.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import json
import asyncio

from analysis_engine.causal.services.causal_inference_service import CausalInferenceService
from analysis_engine.causal.services.causal_data_connector import CausalDataConnector 
from analysis_engine.causal.visualization.relationship_graph import (
    CausalGraphVisualizer, CausalEffectVisualizer, CounterfactualVisualizer
)
from analysis_engine.analysis.indicators import IndicatorClient

logger = logging.getLogger(__name__)

# Define API models
class CausalGraphRequest(BaseModel):
    symbols: List[str]
    timeframe: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    method: str = "granger"

class InterventionRequest(BaseModel):
    symbols: List[str]
    timeframe: str
    target_symbol: str
    interventions: Dict[str, float]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class CounterfactualRequest(BaseModel):
    scenario_name: str
    symbols: List[str]
    timeframe: str
    target_symbol: str
    interventions: Dict[str, float]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class EnhancedDataRequest(BaseModel):
    symbols: List[str]
    timeframe: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_indicators: bool = True

# Create router
router = APIRouter(
    prefix="/api/v1/causal-visualization",
    tags=["causal-visualization"],
    responses={404: {"description": "Not found"}}
)

# Dependency functions
def get_indicator_client(request: Request) -> IndicatorClient:
    client = request.app.state.service_container.resolve(IndicatorClient)
    if not client:
        raise HTTPException(status_code=503, detail="IndicatorClient not available")
    return client

def get_causal_data_connector(indicator_client: IndicatorClient = Depends(get_indicator_client)) -> CausalDataConnector:
    # Assuming CausalDataConnector only needs indicator_client now
    # If it needs config, that would need to be injected too
    return CausalDataConnector(indicator_client=indicator_client)

# Initialize services
causal_service = CausalInferenceService()

# Create visualizers
graph_visualizer = CausalGraphVisualizer()
effect_visualizer = CausalEffectVisualizer()
counterfactual_visualizer = CounterfactualVisualizer()


@router.post("/causal-graph")
async def get_causal_graph(request_data: CausalGraphRequest, 
                         data_connector: CausalDataConnector = Depends(get_causal_data_connector)):
    """
    Generate a causal graph visualization for the specified symbols and timeframe.
    
    Returns:
        JSON with the serialized causal graph and visualization data
    """
    try:
        # Set default dates if not provided
        if not request_data.start_date:
            request_data.start_date = datetime.now() - timedelta(days=30)
        if not request_data.end_date:
            request_data.end_date = datetime.now()
            
        # Get market data
        historical_data = await data_connector.get_historical_data(
            symbols=request_data.symbols,
            start_date=request_data.start_date,
            end_date=request_data.end_date,
            timeframe=request_data.timeframe,
            include_indicators=True
        )
        
        if historical_data.empty:
            raise HTTPException(status_code=404, detail="No data found for the specified parameters")

        # Prepare data for causal analysis
        prepared_data = await data_connector.prepare_data_for_causal_analysis(historical_data)
        
        # Generate causal graph
        causal_graph = causal_service.discover_causal_structure(
            prepared_data, 
            method=request_data.method
        )
        
        # Generate interactive visualization
        figure = graph_visualizer.draw_interactive_causal_graph(
            causal_graph,
            title=f"Causal Structure for {', '.join(request_data.symbols)} ({request_data.timeframe})"
        )
        
        # Convert networkx graph to serializable format
        serialized_graph = {
            "nodes": [{"id": node, "label": node} for node in causal_graph.nodes()],
            "edges": [{"source": u, "target": v} for u, v in causal_graph.edges()]
        }
        
        # Convert plotly figure to JSON
        figure_json = figure.to_json()
        
        return {
            "graph": serialized_graph,
            "visualization": figure_json,
            "timeframe": request_data.timeframe,
            "method": request_data.method,
            "node_count": len(causal_graph.nodes()),
            "edge_count": len(causal_graph.edges())
        }
        
    except Exception as e:
        logger.error(f"Error generating causal graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating causal graph: {str(e)}")


@router.post("/intervention-effect")
async def get_intervention_effect(request: InterventionRequest):
    """
    Generate an intervention effect visualization showing the impact of
    a hypothetical intervention on a target variable.
    
    Returns:
        JSON with the visualization data
    """
    try:
        # Set default dates if not provided
        if not request.start_date:
            request.start_date = datetime.now() - timedelta(days=30)
        if not request.end_date:
            request.end_date = datetime.now()
            
        # Get market data
        market_data = await data_connector.get_historical_data(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            include_indicators=True
        )
        
        if market_data.empty:
            raise HTTPException(status_code=400, detail="Failed to retrieve market data")
            
        # Prepare data for counterfactual analysis
        target_col = f"{request.target_symbol}_close"
        features = [col for col in market_data.columns if col != target_col]
        
        # Initialize counterfactual model
        counterfactual_analyzer = causal_service.counterfactual_analyzer
        counterfactual_analyzer.fit(market_data, target_col, features)
        
        # Generate counterfactual data
        counterfactual_data = counterfactual_analyzer.generate_counterfactual(
            market_data, request.interventions
        )
        
        # Create visualization of intervention effect
        figure = effect_visualizer.plot_interactive_intervention_effect(
            original_data=market_data,
            counterfactual_data=counterfactual_data,
            intervention_vars=list(request.interventions.keys()),
            outcome_var=target_col,
            title=f"Intervention Effect on {request.target_symbol}"
        )
        
        # Convert plotly figure to JSON
        figure_json = figure.to_json()
        
        # Calculate summary statistics
        original_mean = market_data[target_col].mean()
        counterfactual_mean = counterfactual_data[f"counterfactual_{target_col}"].mean()
        percent_change = (counterfactual_mean - original_mean) / original_mean * 100
        
        return {
            "visualization": figure_json,
            "target_symbol": request.target_symbol,
            "interventions": request.interventions,
            "percent_change": percent_change,
            "original_mean": original_mean,
            "counterfactual_mean": counterfactual_mean
        }
        
    except Exception as e:
        logger.error(f"Error generating intervention effect: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating intervention effect: {str(e)}")


@router.post("/counterfactual-scenario")
async def get_counterfactual_scenario(request: CounterfactualRequest):
    """
    Generate a counterfactual scenario visualization comparing actual data
    with a counterfactual scenario.
    
    Returns:
        JSON with the visualization data
    """
    try:
        # Set default dates if not provided
        if not request.start_date:
            request.start_date = datetime.now() - timedelta(days=30)
        if not request.end_date:
            request.end_date = datetime.now()
            
        # Get market data
        market_data = await data_connector.get_historical_data(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            include_indicators=True
        )
        
        if market_data.empty:
            raise HTTPException(status_code=400, detail="Failed to retrieve market data")
            
        # Prepare data for counterfactual analysis
        target_col = f"{request.target_symbol}_close"
        features = [col for col in market_data.columns if col != target_col]
        
        # Initialize counterfactual model
        counterfactual_analyzer = causal_service.counterfactual_analyzer
        counterfactual_analyzer.fit(market_data, target_col, features)
        
        # Generate counterfactual data
        counterfactual_data = counterfactual_analyzer.generate_counterfactual(
            market_data, request.interventions
        )
        
        # Create counterfactual scenario visualization
        scenarios = {request.scenario_name: counterfactual_data}
        
        # Generate radar chart comparison
        radar_figure = counterfactual_visualizer.create_radar_chart(
            actual_data=market_data,
            counterfactual_data=counterfactual_data,
            variables=[f"{symbol}_close" for symbol in request.symbols],
            title=f"Scenario Comparison: {request.scenario_name}"
        )
        
        # Generate path comparison
        path_figure = counterfactual_visualizer.plot_counterfactual_paths(
            actual_data=market_data,
            counterfactual_scenarios=scenarios,
            target_var=target_col,
            title=f"Counterfactual Paths for {request.target_symbol}"
        )
        
        # Convert figures to JSON
        radar_json = radar_figure.to_json()
        path_json = json.dumps({"figure": "matplotlib_figure"})  # placeholder for matplotlib figure
        
        return {
            "radar_chart": radar_json,
            "path_chart": path_json,
            "scenario_name": request.scenario_name,
            "target_symbol": request.target_symbol,
            "interventions": request.interventions,
            "timeframe": request.timeframe
        }
        
    except Exception as e:
        logger.error(f"Error generating counterfactual scenario: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating counterfactual scenario: {str(e)}")


@router.get("/available-symbols")
async def get_available_symbols():
    """
    Get the list of available symbols for causal analysis.
    
    Returns:
        JSON with the list of available symbols
    """
    try:
        # This would typically fetch from your market data service
        # For now, return a placeholder list of common forex pairs
        symbols = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"
        ]
        
        return {"symbols": symbols}
        
    except Exception as e:
        logger.error(f"Error getting available symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting available symbols: {str(e)}")


@router.get("/available-timeframes")
async def get_available_timeframes():
    """
    Get the list of available timeframes for causal analysis.
    
    Returns:
        JSON with the list of available timeframes
    """
    try:
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        return {"timeframes": timeframes}
        
    except Exception as e:
        logger.error(f"Error getting available timeframes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting available timeframes: {str(e)}")


@router.post("/enhanced-data")
async def get_enhanced_data(request_data: EnhancedDataRequest,
                          data_connector: CausalDataConnector = Depends(get_causal_data_connector)):
    """
    Provides enhanced historical data including indicators, fetched via CausalDataConnector.
    This endpoint is intended for consumption by other services like the strategy engine.
    """
    try:
        historical_data = await data_connector.get_historical_data(
            symbols=request_data.symbols,
            start_date=request_data.start_date,
            end_date=request_data.end_date,
            timeframe=request_data.timeframe,
            include_indicators=request_data.include_indicators
        )
        
        if historical_data.empty:
            # Return empty structure instead of 404, as strategy might handle this
            return {"data": [], "columns": []} 

        # Convert DataFrame to JSON serializable format (list of records)
        # Reset index to include timestamp in the records
        historical_data_reset = historical_data.reset_index()
        # Convert timestamp to ISO format string
        historical_data_reset['timestamp'] = historical_data_reset['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Handle potential NaN/Infinity values for JSON compatibility
        historical_data_reset = historical_data_reset.replace([np.inf, -np.inf], None) # Replace inf with None
        data_dict = historical_data_reset.to_dict(orient='records')
        
        # Replace NaN with None in the list of dictionaries
        for record in data_dict:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None

        return {"data": data_dict, "columns": list(historical_data.columns)}

    except HTTPException:
        raise # Re-raise FastAPI HTTP exceptions
    except Exception as e:
        logger.error(f"Error getting enhanced data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error fetching enhanced data: {str(e)}")
