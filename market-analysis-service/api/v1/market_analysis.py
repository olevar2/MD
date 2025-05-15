"""
Market Analysis API routes.

This module provides the API routes for market analysis.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any, Optional
from datetime import datetime

from common_lib.cqrs.commands import CommandBus
from common_lib.cqrs.queries import QueryBus
from market_analysis_service.utils.dependency_injection import get_command_bus, get_query_bus
from market_analysis_service.cqrs.commands import (
    AnalyzeMarketCommand,
    RecognizePatternsCommand,
    DetectSupportResistanceCommand,
    DetectMarketRegimeCommand,
    AnalyzeCorrelationCommand,
    AnalyzeVolatilityCommand,
    AnalyzeSentimentCommand
)
from market_analysis_service.cqrs.queries import (
    GetAnalysisResultQuery,
    ListAnalysisResultsQuery,
    GetPatternRecognitionResultQuery,
    ListPatternRecognitionResultsQuery,
    GetSupportResistanceResultQuery,
    ListSupportResistanceResultsQuery,
    GetMarketRegimeResultQuery,
    ListMarketRegimeResultsQuery,
    GetCorrelationAnalysisResultQuery,
    ListCorrelationAnalysisResultsQuery,
    GetAvailableMethodsQuery
)
from market_analysis_service.models.market_analysis_models import (
    MarketAnalysisRequest,
    MarketAnalysisResponse,
    PatternRecognitionRequest,
    PatternRecognitionResponse,
    SupportResistanceRequest,
    SupportResistanceResponse,
    MarketRegimeRequest,
    MarketRegimeResponse,
    CorrelationAnalysisRequest,
    CorrelationAnalysisResponse,
    VolatilityAnalysisRequest,
    VolatilityAnalysisResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    AnalysisType,
    PatternType,
    MarketRegimeType,
    SupportResistanceMethod
)

router = APIRouter(prefix="/market-analysis", tags=["market-analysis"])


@router.post("/analyze", response_model=MarketAnalysisResponse)
async def analyze_market(
    request: MarketAnalysisRequest,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Perform comprehensive market analysis.
    """
    try:
        # Create command
        command = AnalyzeMarketCommand(
            correlation_id=None,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=datetime.fromisoformat(request.start_date),
            end_date=datetime.fromisoformat(request.end_date) if request.end_date else None,
            analysis_types=request.analysis_types,
            additional_parameters=request.additional_parameters
        )
        
        # Dispatch command
        analysis_id = await command_bus.dispatch(command)
        
        # Get analysis result
        query = GetAnalysisResultQuery(
            correlation_id=None,
            analysis_id=analysis_id
        )
        
        query_bus = get_query_bus()
        result = await query_bus.dispatch(query)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Analysis result {analysis_id} not found")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze market: {str(e)}")


@router.post("/patterns", response_model=PatternRecognitionResponse)
async def recognize_patterns(
    request: PatternRecognitionRequest,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Recognize chart patterns in market data.
    """
    try:
        # Create command
        command = RecognizePatternsCommand(
            correlation_id=None,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=datetime.fromisoformat(request.start_date),
            end_date=datetime.fromisoformat(request.end_date) if request.end_date else None,
            pattern_types=[PatternType(p) for p in request.pattern_types] if request.pattern_types else None,
            min_confidence=request.min_confidence,
            additional_parameters=request.additional_parameters
        )
        
        # Dispatch command
        result_id = await command_bus.dispatch(command)
        
        # Get pattern recognition result
        query = GetPatternRecognitionResultQuery(
            correlation_id=None,
            result_id=result_id
        )
        
        query_bus = get_query_bus()
        result = await query_bus.dispatch(query)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Pattern recognition result {result_id} not found")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recognize patterns: {str(e)}")


@router.post("/support-resistance", response_model=SupportResistanceResponse)
async def detect_support_resistance(
    request: SupportResistanceRequest,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Detect support and resistance levels in market data.
    """
    try:
        # Create command
        command = DetectSupportResistanceCommand(
            correlation_id=None,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=datetime.fromisoformat(request.start_date),
            end_date=datetime.fromisoformat(request.end_date) if request.end_date else None,
            methods=[SupportResistanceMethod(m) for m in request.methods] if request.methods else None,
            additional_parameters=request.additional_parameters
        )
        
        # Dispatch command
        result_id = await command_bus.dispatch(command)
        
        # Get support/resistance result
        query = GetSupportResistanceResultQuery(
            correlation_id=None,
            result_id=result_id
        )
        
        query_bus = get_query_bus()
        result = await query_bus.dispatch(query)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Support/resistance result {result_id} not found")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect support/resistance: {str(e)}")


@router.post("/market-regime", response_model=MarketRegimeResponse)
async def detect_market_regime(
    request: MarketRegimeRequest,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Detect market regime in market data.
    """
    try:
        # Create command
        command = DetectMarketRegimeCommand(
            correlation_id=None,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=datetime.fromisoformat(request.start_date),
            end_date=datetime.fromisoformat(request.end_date) if request.end_date else None,
            window_size=request.window_size,
            additional_parameters=request.additional_parameters
        )
        
        # Dispatch command
        result_id = await command_bus.dispatch(command)
        
        # Get market regime result
        query = GetMarketRegimeResultQuery(
            correlation_id=None,
            result_id=result_id
        )
        
        query_bus = get_query_bus()
        result = await query_bus.dispatch(query)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Market regime result {result_id} not found")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect market regime: {str(e)}")


@router.post("/correlation", response_model=CorrelationAnalysisResponse)
async def analyze_correlation(
    request: CorrelationAnalysisRequest,
    command_bus: CommandBus = Depends(get_command_bus)
):
    """
    Analyze correlations between symbols.
    """
    try:
        # Create command
        command = AnalyzeCorrelationCommand(
            correlation_id=None,
            symbols=request.symbols,
            timeframe=request.timeframe,
            start_date=datetime.fromisoformat(request.start_date),
            end_date=datetime.fromisoformat(request.end_date) if request.end_date else None,
            window_size=request.window_size,
            method=request.method,
            additional_parameters=request.additional_parameters
        )
        
        # Dispatch command
        result_id = await command_bus.dispatch(command)
        
        # Get correlation analysis result
        query = GetCorrelationAnalysisResultQuery(
            correlation_id=None,
            result_id=result_id
        )
        
        query_bus = get_query_bus()
        result = await query_bus.dispatch(query)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Correlation analysis result {result_id} not found")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze correlation: {str(e)}")


@router.get("/available-methods", response_model=Dict[str, List[str]])
async def get_available_methods(
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get available analysis methods.
    """
    try:
        # Create query
        query = GetAvailableMethodsQuery(
            correlation_id=None
        )
        
        # Dispatch query
        result = await query_bus.dispatch(query)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available methods: {str(e)}")


@router.get("/analysis/{analysis_id}", response_model=MarketAnalysisResponse)
async def get_analysis_result(
    analysis_id: str,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get an analysis result by ID.
    """
    try:
        # Create query
        query = GetAnalysisResultQuery(
            correlation_id=None,
            analysis_id=analysis_id
        )
        
        # Dispatch query
        result = await query_bus.dispatch(query)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Analysis result {analysis_id} not found")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis result: {str(e)}")


@router.get("/patterns/{result_id}", response_model=PatternRecognitionResponse)
async def get_pattern_recognition_result(
    result_id: str,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get a pattern recognition result by ID.
    """
    try:
        # Create query
        query = GetPatternRecognitionResultQuery(
            correlation_id=None,
            result_id=result_id
        )
        
        # Dispatch query
        result = await query_bus.dispatch(query)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Pattern recognition result {result_id} not found")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pattern recognition result: {str(e)}")


@router.get("/support-resistance/{result_id}", response_model=SupportResistanceResponse)
async def get_support_resistance_result(
    result_id: str,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get a support/resistance result by ID.
    """
    try:
        # Create query
        query = GetSupportResistanceResultQuery(
            correlation_id=None,
            result_id=result_id
        )
        
        # Dispatch query
        result = await query_bus.dispatch(query)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Support/resistance result {result_id} not found")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get support/resistance result: {str(e)}")


@router.get("/market-regime/{result_id}", response_model=MarketRegimeResponse)
async def get_market_regime_result(
    result_id: str,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get a market regime result by ID.
    """
    try:
        # Create query
        query = GetMarketRegimeResultQuery(
            correlation_id=None,
            result_id=result_id
        )
        
        # Dispatch query
        result = await query_bus.dispatch(query)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Market regime result {result_id} not found")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get market regime result: {str(e)}")


@router.get("/correlation/{result_id}", response_model=CorrelationAnalysisResponse)
async def get_correlation_analysis_result(
    result_id: str,
    query_bus: QueryBus = Depends(get_query_bus)
):
    """
    Get a correlation analysis result by ID.
    """
    try:
        # Create query
        query = GetCorrelationAnalysisResultQuery(
            correlation_id=None,
            result_id=result_id
        )
        
        # Dispatch query
        result = await query_bus.dispatch(query)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Correlation analysis result {result_id} not found")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get correlation analysis result: {str(e)}")