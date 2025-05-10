"""
Machine Learning API Endpoints

This module provides API endpoints for machine learning-based analysis.
"""

import os
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, Body
from pydantic import BaseModel, Field

from analysis_engine.ml.model_manager import ModelManager
from analysis_engine.ml.ml_confluence_detector import MLConfluenceDetector
from analysis_engine.utils.distributed_tracing import DistributedTracer
from analysis_engine.services.price_data_service import PriceDataService
from analysis_engine.services.correlation_service import CorrelationService
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Create tracer
tracer = DistributedTracer(service_name="analysis-engine")

# Models
class PatternRequest(BaseModel):
    """Request model for pattern recognition."""
    
    symbol: str = Field(..., description="Currency pair (e.g., 'EURUSD')")
    timeframe: str = Field("H1", description="Timeframe for analysis")
    window_size: int = Field(30, description="Size of the window for pattern recognition")

class PatternResponse(BaseModel):
    """Response model for pattern recognition."""
    
    symbol: str = Field(..., description="Currency pair")
    timeframe: str = Field(..., description="Timeframe for analysis")
    patterns: Dict[str, float] = Field(..., description="Dictionary mapping pattern names to probabilities")
    execution_time: float = Field(..., description="Execution time in seconds")
    request_id: str = Field(..., description="Request ID")

class PredictionRequest(BaseModel):
    """Request model for price prediction."""
    
    symbol: str = Field(..., description="Currency pair (e.g., 'EURUSD')")
    timeframe: str = Field("H1", description="Timeframe for analysis")
    input_window: int = Field(60, description="Size of the input window for prediction")
    output_window: int = Field(10, description="Size of the output window (prediction horizon)")

class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    
    symbol: str = Field(..., description="Currency pair")
    timeframe: str = Field(..., description="Timeframe for analysis")
    predictions: List[float] = Field(..., description="List of predicted prices")
    lower_bound: List[float] = Field(..., description="Lower bound of prediction interval")
    upper_bound: List[float] = Field(..., description="Upper bound of prediction interval")
    percentage_changes: List[float] = Field(..., description="List of percentage changes")
    execution_time: float = Field(..., description="Execution time in seconds")
    request_id: str = Field(..., description="Request ID")

class MLConfluenceRequest(BaseModel):
    """Request model for ML-based confluence detection."""
    
    symbol: str = Field(..., description="Primary currency pair (e.g., 'EURUSD')")
    signal_type: str = Field(..., description="Type of signal ('trend', 'reversal', 'breakout')")
    signal_direction: str = Field(..., description="Direction of the signal ('bullish', 'bearish')")
    timeframe: str = Field("H1", description="Timeframe for analysis")
    use_currency_strength: bool = Field(True, description="Whether to include currency strength in analysis")
    min_confirmation_strength: float = Field(0.3, description="Minimum strength for confirmation signals")
    related_pairs: Optional[Dict[str, float]] = Field(None, description="Dictionary of related pairs and their correlations")

class MLConfluenceResponse(BaseModel):
    """Response model for ML-based confluence detection."""
    
    symbol: str = Field(..., description="Primary currency pair")
    signal_type: str = Field(..., description="Type of signal")
    signal_direction: str = Field(..., description="Direction of the signal")
    timeframe: str = Field(..., description="Timeframe for analysis")
    confirmation_count: int = Field(..., description="Number of confirming pairs")
    contradiction_count: int = Field(..., description="Number of contradicting pairs")
    neutral_count: int = Field(..., description="Number of neutral pairs")
    confluence_score: float = Field(..., description="Overall confluence score")
    pattern_score: float = Field(..., description="Pattern recognition score")
    prediction_score: float = Field(..., description="Price prediction score")
    related_pairs_score: float = Field(..., description="Related pairs score")
    currency_strength_score: float = Field(..., description="Currency strength score")
    confirmations: List[Dict[str, Any]] = Field(..., description="List of confirming pairs")
    contradictions: List[Dict[str, Any]] = Field(..., description="List of contradicting pairs")
    neutrals: List[Dict[str, Any]] = Field(..., description="List of neutral pairs")
    price_prediction: Dict[str, Any] = Field(..., description="Price prediction results")
    patterns: Dict[str, float] = Field(..., description="Pattern recognition results")
    execution_time: float = Field(..., description="Execution time in seconds")
    request_id: str = Field(..., description="Request ID")

class MLDivergenceRequest(BaseModel):
    """Request model for ML-based divergence analysis."""
    
    symbol: str = Field(..., description="Primary currency pair (e.g., 'EURUSD')")
    timeframe: str = Field("H1", description="Timeframe for analysis")
    related_pairs: Optional[Dict[str, float]] = Field(None, description="Dictionary of related pairs and their correlations")

class MLDivergenceResponse(BaseModel):
    """Response model for ML-based divergence analysis."""
    
    symbol: str = Field(..., description="Primary currency pair")
    timeframe: str = Field(..., description="Timeframe for analysis")
    divergences_found: int = Field(..., description="Number of divergences found")
    divergence_score: float = Field(..., description="Overall divergence score")
    divergences: List[Dict[str, Any]] = Field(..., description="List of divergences")
    price_prediction: Dict[str, Any] = Field(..., description="Price prediction results")
    execution_time: float = Field(..., description="Execution time in seconds")
    request_id: str = Field(..., description="Request ID")

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    path: str = Field(..., description="Model path")
    created_at: str = Field(..., description="Creation timestamp")
    description: str = Field(..., description="Model description")
    metadata: Dict[str, Any] = Field(..., description="Model metadata")

# Dependencies
def get_model_manager():
    """Get the model manager."""
    # In a real application, you would use dependency injection
    correlation_service = CorrelationService()
    currency_strength_analyzer = CurrencyStrengthAnalyzer()
    
    return ModelManager(
        model_dir="models",
        use_gpu=True,
        correlation_service=correlation_service,
        currency_strength_analyzer=currency_strength_analyzer
    )

def get_price_data_service():
    """Get the price data service."""
    # In a real application, you would use dependency injection
    return PriceDataService()

def get_ml_confluence_detector(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get the ML confluence detector."""
    return model_manager.load_ml_confluence_detector()

# Endpoints
@router.post("/patterns", response_model=PatternResponse)
async def recognize_patterns(
    request: PatternRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
    price_data_service: PriceDataService = Depends(get_price_data_service)
):
    """
    Recognize patterns in price data using machine learning.
    
    This endpoint uses a deep learning model to recognize common chart patterns
    such as double tops, head and shoulders, triangles, etc.
    """
    with tracer.start_span("recognize_patterns") as span:
        span.set_attribute("symbol", request.symbol)
        span.set_attribute("timeframe", request.timeframe)
        
        request_id = tracer.get_current_trace_id()
        
        try:
            # Get price data
            price_data = await price_data_service.get_price_data(
                symbol=request.symbol,
                timeframe=request.timeframe
            )
            
            # Load pattern recognition model
            pattern_model = model_manager.load_pattern_model(
                window_size=request.window_size
            )
            
            # Recognize patterns
            start_time = time.time()
            patterns = pattern_model.predict(price_data)
            execution_time = time.time() - start_time
            
            # Create response
            response = PatternResponse(
                symbol=request.symbol,
                timeframe=request.timeframe,
                patterns={k: v[-1] for k, v in patterns.items()},
                execution_time=execution_time,
                request_id=request_id
            )
            
            return response
        except Exception as e:
            logger.error(f"Error recognizing patterns: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/predictions", response_model=PredictionResponse)
async def predict_prices(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager),
    price_data_service: PriceDataService = Depends(get_price_data_service)
):
    """
    Predict future prices using machine learning.
    
    This endpoint uses a deep learning model to predict future price movements
    based on historical data.
    """
    with tracer.start_span("predict_prices") as span:
        span.set_attribute("symbol", request.symbol)
        span.set_attribute("timeframe", request.timeframe)
        
        request_id = tracer.get_current_trace_id()
        
        try:
            # Get price data
            price_data = await price_data_service.get_price_data(
                symbol=request.symbol,
                timeframe=request.timeframe
            )
            
            # Load price prediction model
            prediction_model = model_manager.load_prediction_model(
                input_window=request.input_window,
                output_window=request.output_window
            )
            
            # Predict prices
            start_time = time.time()
            prediction = prediction_model.predict(price_data)
            execution_time = time.time() - start_time
            
            # Create response
            response = PredictionResponse(
                symbol=request.symbol,
                timeframe=request.timeframe,
                predictions=prediction["predictions"],
                lower_bound=prediction["lower_bound"],
                upper_bound=prediction["upper_bound"],
                percentage_changes=prediction["percentage_changes"],
                execution_time=execution_time,
                request_id=request_id
            )
            
            return response
        except Exception as e:
            logger.error(f"Error predicting prices: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/confluence", response_model=MLConfluenceResponse)
async def detect_confluence_ml(
    request: MLConfluenceRequest,
    background_tasks: BackgroundTasks,
    ml_confluence_detector: MLConfluenceDetector = Depends(get_ml_confluence_detector),
    price_data_service: PriceDataService = Depends(get_price_data_service)
):
    """
    Detect confluence using machine learning.
    
    This endpoint uses machine learning models to detect confluence signals
    across multiple currency pairs with higher accuracy.
    """
    with tracer.start_span("detect_confluence_ml") as span:
        span.set_attribute("symbol", request.symbol)
        span.set_attribute("signal_type", request.signal_type)
        span.set_attribute("signal_direction", request.signal_direction)
        span.set_attribute("timeframe", request.timeframe)
        
        request_id = tracer.get_current_trace_id()
        
        try:
            # Get price data for primary symbol
            primary_data = await price_data_service.get_price_data(
                symbol=request.symbol,
                timeframe=request.timeframe
            )
            
            # Get related pairs if not provided
            related_pairs = request.related_pairs
            if related_pairs is None:
                related_pairs = await ml_confluence_detector.find_related_pairs(request.symbol)
            
            # Get price data for related pairs
            price_data = {request.symbol: primary_data}
            for pair in related_pairs.keys():
                pair_data = await price_data_service.get_price_data(
                    symbol=pair,
                    timeframe=request.timeframe
                )
                price_data[pair] = pair_data
            
            # Detect confluence
            start_time = time.time()
            result = ml_confluence_detector.detect_confluence_ml(
                symbol=request.symbol,
                price_data=price_data,
                signal_type=request.signal_type,
                signal_direction=request.signal_direction,
                related_pairs=related_pairs,
                use_currency_strength=request.use_currency_strength,
                min_confirmation_strength=request.min_confirmation_strength
            )
            execution_time = time.time() - start_time
            
            # Create response
            response = MLConfluenceResponse(
                symbol=request.symbol,
                signal_type=request.signal_type,
                signal_direction=request.signal_direction,
                timeframe=request.timeframe,
                confirmation_count=result["confirmation_count"],
                contradiction_count=result["contradiction_count"],
                neutral_count=result["neutral_count"],
                confluence_score=result["confluence_score"],
                pattern_score=result["pattern_score"],
                prediction_score=result["prediction_score"],
                related_pairs_score=result["related_pairs_score"],
                currency_strength_score=result["currency_strength_score"],
                confirmations=result["confirmations"],
                contradictions=result["contradictions"],
                neutrals=result["neutrals"],
                price_prediction=result["price_prediction"],
                patterns=result["patterns"],
                execution_time=execution_time,
                request_id=request_id
            )
            
            return response
        except Exception as e:
            logger.error(f"Error detecting confluence: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/divergence", response_model=MLDivergenceResponse)
async def analyze_divergence_ml(
    request: MLDivergenceRequest,
    background_tasks: BackgroundTasks,
    ml_confluence_detector: MLConfluenceDetector = Depends(get_ml_confluence_detector),
    price_data_service: PriceDataService = Depends(get_price_data_service)
):
    """
    Analyze divergences using machine learning.
    
    This endpoint uses machine learning models to analyze divergences between
    correlated currency pairs with higher accuracy.
    """
    with tracer.start_span("analyze_divergence_ml") as span:
        span.set_attribute("symbol", request.symbol)
        span.set_attribute("timeframe", request.timeframe)
        
        request_id = tracer.get_current_trace_id()
        
        try:
            # Get price data for primary symbol
            primary_data = await price_data_service.get_price_data(
                symbol=request.symbol,
                timeframe=request.timeframe
            )
            
            # Get related pairs if not provided
            related_pairs = request.related_pairs
            if related_pairs is None:
                related_pairs = await ml_confluence_detector.find_related_pairs(request.symbol)
            
            # Get price data for related pairs
            price_data = {request.symbol: primary_data}
            for pair in related_pairs.keys():
                pair_data = await price_data_service.get_price_data(
                    symbol=pair,
                    timeframe=request.timeframe
                )
                price_data[pair] = pair_data
            
            # Analyze divergence
            start_time = time.time()
            result = ml_confluence_detector.analyze_divergence_ml(
                symbol=request.symbol,
                price_data=price_data,
                related_pairs=related_pairs
            )
            execution_time = time.time() - start_time
            
            # Create response
            response = MLDivergenceResponse(
                symbol=request.symbol,
                timeframe=request.timeframe,
                divergences_found=result["divergences_found"],
                divergence_score=result["divergence_score"],
                divergences=result["divergences"],
                price_prediction=result["price_prediction"],
                execution_time=execution_time,
                request_id=request_id
            )
            
            return response
        except Exception as e:
            logger.error(f"Error analyzing divergence: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[ModelInfoResponse])
async def list_models(
    model_type: Optional[str] = Query(None, description="Type of models to list"),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    List all models in the registry.
    
    This endpoint returns information about all models in the registry,
    optionally filtered by model type.
    """
    with tracer.start_span("list_models") as span:
        span.set_attribute("model_type", model_type or "all")
        
        try:
            models = model_manager.list_models(model_type)
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(
    model_name: str = Path(..., description="Name of the model"),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Get information about a model.
    
    This endpoint returns detailed information about a specific model.
    """
    with tracer.start_span("get_model_info") as span:
        span.set_attribute("model_name", model_name)
        
        try:
            model_info = model_manager.get_model_info(model_name)
            if model_info is None:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            return model_info
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise HTTPException(status_code=500, detail=str(e))
