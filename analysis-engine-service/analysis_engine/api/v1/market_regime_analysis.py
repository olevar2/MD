"""
Market Regime Analysis API endpoints

This module provides API endpoints for detecting market regimes
and analyzing tool effectiveness across different market conditions.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from analysis_engine.db.connection import get_db_session
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.market_regime_detector import MarketRegimeService
from analysis_engine.services.market_regime_analysis import MarketRegimeAnalysisService


# Define API request/response models
class DetectRegimeRequest(BaseModel):
    symbol: str
    timeframe: str
    ohlc_data: List[Dict]

    class Config:
        schema_extra = {
            "example": {
                "symbol": "EURUSD",
                "timeframe": "1h",
                "ohlc_data": [
                    {"timestamp": "2025-04-01T00:00:00", "open": 1.0765, "high": 1.0780, "low": 1.0760, "close": 1.0775, "volume": 1000},
                    {"timestamp": "2025-04-01T01:00:00", "open": 1.0775, "high": 1.0790, "low": 1.0770, "close": 1.0785, "volume": 1200},
                    # Additional data points...
                ]
            }
        }

class RegimeHistoryRequest(BaseModel):
    symbol: str
    timeframe: str
    limit: Optional[int] = 10

class RegimeAnalysisRequest(BaseModel):
    tool_id: str
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None

class OptimalConditionsRequest(BaseModel):
    tool_id: str
    min_sample_size: int = 10
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None

class ToolComplementarityRequest(BaseModel):
    tool_ids: List[str]
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None

class RecommendationRequest(BaseModel):
    current_regime: str
    instrument: Optional[str] = None
    timeframe: Optional[str] = None
    min_sample_size: int = 10
    min_win_rate: float = 50.0
    top_n: int = 3

class TrendAnalysisRequest(BaseModel):
    tool_id: str
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    period_days: int = 30
    look_back_periods: int = 6

class UnderperformingToolsRequest(BaseModel):
    win_rate_threshold: float = 50.0
    min_sample_size: int = 20
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None

class DashboardRequest(BaseModel):
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


# Create router
router = APIRouter(
    prefix="/market-regime",
    tags=["market-regime"]
)

# Setup logging
logger = logging.getLogger(__name__)


@router.post("/detect/", response_model=Dict)
async def detect_market_regime(
    request: DetectRegimeRequest,
    db: Session = Depends(get_db_session)
):
    """Detect the current market regime based on price data"""
    try:
        # Convert OHLC data to pandas DataFrame
        import pandas as pd
        df = pd.DataFrame(request.ohlc_data)

        # Initialize market regime service
        service = MarketRegimeService()

        # Detect market regime
        regime_result = await service.detect_current_regime(
            symbol=request.symbol,
            timeframe=request.timeframe,
            price_data=df
        )

        return regime_result
    except Exception as e:
        logger.error(f"Error detecting market regime: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to detect market regime: {str(e)}"
        )


@router.post("/history/", response_model=List[Dict])
async def get_regime_history(
    request: RegimeHistoryRequest,
    db: Session = Depends(get_db_session)
):
    """Get historical regime data for a specific symbol and timeframe"""
    try:
        # Initialize market regime service
        service = MarketRegimeService()

        # Get regime history
        history = await service.get_regime_history(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=request.limit
        )

        return history
    except Exception as e:
        logger.error(f"Error getting regime history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get regime history: {str(e)}"
        )


@router.post("/regime-analysis/", response_model=Dict)
async def analyze_tool_regime_performance(
    request: RegimeAnalysisRequest,
    db: Session = Depends(get_db_session)
):
    """Get the performance metrics of a tool across different market regimes"""
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)

        result = await service.get_regime_performance_matrix(
            tool_id=request.tool_id,
            timeframe=request.timeframe,
            instrument=request.instrument,
            from_date=request.from_date,
            to_date=request.to_date
        )

        return result
    except Exception as e:
        logger.error(f"Error analyzing regime performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze regime performance: {str(e)}"
        )


@router.post("/optimal-conditions/", response_model=Dict)
async def find_optimal_market_conditions(
    request: OptimalConditionsRequest,
    db: Session = Depends(get_db_session)
):
    """Find the optimal market conditions for a specific tool"""
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)

        result = await service.find_optimal_market_conditions(
            tool_id=request.tool_id,
            min_sample_size=request.min_sample_size,
            timeframe=request.timeframe,
            instrument=request.instrument,
            from_date=request.from_date,
            to_date=request.to_date
        )

        return result
    except Exception as e:
        logger.error(f"Error finding optimal market conditions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find optimal market conditions: {str(e)}"
        )


@router.post("/complementarity/", response_model=Dict)
async def analyze_tool_complementarity(
    request: ToolComplementarityRequest,
    db: Session = Depends(get_db_session)
):
    """Analyze how well different tools complement each other"""
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)

        result = await service.compute_tool_complementarity(
            tool_ids=request.tool_ids,
            timeframe=request.timeframe,
            instrument=request.instrument,
            from_date=request.from_date,
            to_date=request.to_date
        )

        return result
    except Exception as e:
        logger.error(f"Error analyzing tool complementarity: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze tool complementarity: {str(e)}"
        )


@router.post("/performance-report/", response_model=Dict)
async def generate_performance_report(
    request: DashboardRequest,
    db: Session = Depends(get_db_session)
):
    """Generate a comprehensive performance report for all tools"""
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)

        result = await service.generate_performance_report(
            timeframe=request.timeframe,
            instrument=request.instrument,
            from_date=request.from_date,
            to_date=request.to_date
        )

        return result
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate performance report: {str(e)}"
        )


@router.post("/recommend-tools/", response_model=Dict)
async def recommend_tools_for_regime(
    request: RecommendationRequest,
    db: Session = Depends(get_db_session)
):
    """Recommend the best trading tools for the current market regime"""
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)

        result = await service.recommend_tools_for_current_regime(
            current_regime=request.current_regime,
            instrument=request.instrument,
            timeframe=request.timeframe,
            min_sample_size=request.min_sample_size,
            min_win_rate=request.min_win_rate,
            top_n=request.top_n
        )

        return result
    except Exception as e:
        logger.error(f"Error recommending tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to recommend tools: {str(e)}"
        )


@router.post("/effectiveness-trends/", response_model=Dict)
async def analyze_effectiveness_trends(
    request: TrendAnalysisRequest,
    db: Session = Depends(get_db_session)
):
    """Analyze how the effectiveness of a tool has changed over time"""
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)

        result = await service.analyze_effectiveness_trends(
            tool_id=request.tool_id,
            timeframe=request.timeframe,
            instrument=request.instrument,
            period_days=request.period_days,
            look_back_periods=request.look_back_periods
        )

        return result
    except Exception as e:
        logger.error(f"Error analyzing effectiveness trends: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze effectiveness trends: {str(e)}"
        )


@router.post("/underperforming-tools/", response_model=Dict)
async def get_underperforming_tools(
    request: UnderperformingToolsRequest,
    db: Session = Depends(get_db_session)
):
    """Identify underperforming trading tools that may need optimization or retirement"""
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)

        result = await service.get_underperforming_tools(
            win_rate_threshold=request.win_rate_threshold,
            min_sample_size=request.min_sample_size,
            timeframe=request.timeframe,
            instrument=request.instrument,
            from_date=request.from_date,
            to_date=request.to_date
        )

        return result
    except Exception as e:
        logger.error(f"Error getting underperforming tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get underperforming tools: {str(e)}"
        )
