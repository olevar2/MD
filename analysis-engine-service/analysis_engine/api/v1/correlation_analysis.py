"""
Enhanced Correlation Analysis API

This module provides API endpoints for accessing enhanced correlation analysis
functionality, including dynamic timeframe analysis, lead-lag relationships,
and correlation breakdown detection.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Body
from sqlalchemy.orm import Session
import logging
import pandas as pd

from core_foundations.models.auth import User
from analysis_engine.analysis.correlation.currency_correlation_enhanced import CurrencyCorrelationEnhanced
from analysis_engine.models.market_data import MarketData
from analysis_engine.db.connection import get_db_session
from analysis_engine.api.auth import get_current_user

# Create router
router = APIRouter(
    prefix="/correlation",
    tags=["correlation"]
)

# Setup logging
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_currency_correlations(
    data: Dict[str, Any],
    window_sizes: Optional[List[int]] = Query(None, description="Correlation window sizes in days"),
    correlation_method: Optional[str] = Query("pearson", description="Correlation method (pearson or spearman)"),
    significance_threshold: Optional[float] = Query(0.7, description="Threshold for significant correlation"),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze correlations between currency pairs with enhanced features
    
    Request body format:
    {
        "EUR/USD": {"ohlc": [...], "metadata": {...}},
        "GBP/USD": {"ohlc": [...], "metadata": {...}},
        ...
    }
    
    Returns correlation matrices, lead-lag relationships, correlation breakdowns, and trading signals
    """
    try:
        # Configure analyzer parameters
        parameters = {}
        
        if window_sizes:
            parameters["correlation_windows"] = window_sizes
            
        if correlation_method:
            if correlation_method.lower() not in ["pearson", "spearman"]:
                raise HTTPException(
                    status_code=400,
                    detail="Correlation method must be 'pearson' or 'spearman'"
                )
            parameters["correlation_method"] = correlation_method.lower()
            
        if significance_threshold:
            parameters["significant_correlation_threshold"] = significance_threshold
        
        # Create analyzer with configured parameters
        analyzer = CurrencyCorrelationEnhanced(parameters)
        
        # Verify input data
        if not data or not isinstance(data, dict):
            raise HTTPException(
                status_code=400,
                detail="Request must contain currency pair data"
            )
            
        if len(data) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two currency pairs required for correlation analysis"
            )
            
        # Execute analysis
        result = analyzer.analyze(data)
        
        if not result.is_valid:
            raise HTTPException(
                status_code=422,
                detail=result.result_data.get("error", "Analysis failed")
            )
        
        return {
            "status": "success",
            "results": result.result_data,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing currency correlations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze currency correlations: {str(e)}"
        )

@router.post("/lead-lag", response_model=Dict[str, Any])
async def analyze_lead_lag_relationships(
    data: Dict[str, Any],
    max_lag: Optional[int] = Query(10, description="Maximum lag for Granger causality test"),
    significance: Optional[float] = Query(0.05, description="P-value threshold for significance"),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze lead-lag relationships between currency pairs
    
    Request body format:
    {
        "EUR/USD": {"ohlc": [...], "metadata": {...}},
        "GBP/USD": {"ohlc": [...], "metadata": {...}},
        ...
    }
    
    Returns detailed lead-lag relationships with statistical significance
    """
    try:
        # Configure analyzer parameters
        parameters = {
            "granger_maxlag": max_lag,
            "granger_significance": significance
        }
        
        # Create analyzer with configured parameters
        analyzer = CurrencyCorrelationEnhanced(parameters)
        
        # Verify input data
        if not data or not isinstance(data, dict):
            raise HTTPException(
                status_code=400,
                detail="Request must contain currency pair data"
            )
            
        if len(data) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two currency pairs required for lead-lag analysis"
            )
            
        # Execute analysis with focus on lead-lag relationships
        result = analyzer.analyze(data)
        
        if not result.is_valid:
            raise HTTPException(
                status_code=422,
                detail=result.result_data.get("error", "Analysis failed")
            )
        
        # Extract only lead-lag relevant information
        lead_lag_results = {
            "lead_lag_relationships": result.result_data.get("lead_lag_relationships", []),
            "trading_signals": [signal for signal in result.result_data.get("trading_signals", [])
                              if signal.get("signal_type") == "lead_lag"]
        }
        
        return {
            "status": "success",
            "results": lead_lag_results,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing lead-lag relationships: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze lead-lag relationships: {str(e)}"
        )

@router.post("/breakdown-detection", response_model=Dict[str, Any])
async def detect_correlation_breakdowns(
    data: Dict[str, Any],
    short_window: Optional[int] = Query(5, description="Short-term correlation window"),
    long_window: Optional[int] = Query(60, description="Long-term correlation window for comparison"),
    change_threshold: Optional[float] = Query(0.3, description="Threshold for significant correlation change"),
    current_user: User = Depends(get_current_user)
):
    """
    Detect significant breakdowns in correlation patterns between currency pairs
    
    Request body format:
    {
        "EUR/USD": {"ohlc": [...], "metadata": {...}},
        "GBP/USD": {"ohlc": [...], "metadata": {...}},
        ...
    }
    
    Returns detected correlation breakdowns with trading signals
    """
    try:
        # Configure analyzer parameters
        parameters = {
            "correlation_windows": [short_window, long_window],
            "correlation_change_threshold": change_threshold
        }
        
        # Create analyzer with configured parameters
        analyzer = CurrencyCorrelationEnhanced(parameters)
        
        # Verify input data
        if not data or not isinstance(data, dict):
            raise HTTPException(
                status_code=400,
                detail="Request must contain currency pair data"
            )
            
        if len(data) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two currency pairs required for correlation analysis"
            )
            
        # Execute analysis
        result = analyzer.analyze(data)
        
        if not result.is_valid:
            raise HTTPException(
                status_code=422,
                detail=result.result_data.get("error", "Analysis failed")
            )
        
        # Extract only breakdown-relevant information
        breakdown_results = {
            "current_window": result.result_data.get("current_window"),
            "historical_window": result.result_data.get("historical_window"),
            "correlation_breakdowns": result.result_data.get("correlation_breakdowns", []),
            "trading_signals": [signal for signal in result.result_data.get("trading_signals", [])
                              if signal.get("signal_type") == "correlation_breakdown"]
        }
        
        return {
            "status": "success",
            "results": breakdown_results,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting correlation breakdowns: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to detect correlation breakdowns: {str(e)}"
        )

@router.post("/cointegration", response_model=Dict[str, Any])
async def test_pair_cointegration(
    data: Dict[str, Any],
    significance: Optional[float] = Query(0.05, description="P-value threshold for cointegration significance"),
    current_user: User = Depends(get_current_user)
):
    """
    Test for cointegration between currency pairs
    
    Request body format:
    {
        "EUR/USD": {"ohlc": [...], "metadata": {...}},
        "GBP/USD": {"ohlc": [...], "metadata": {...}},
        ...
    }
    
    Returns cointegration test results and related trading signals
    """
    try:
        # Configure analyzer parameters
        parameters = {
            "cointegration_significance": significance
        }
        
        # Create analyzer with configured parameters
        analyzer = CurrencyCorrelationEnhanced(parameters)
        
        # Verify input data
        if not data or not isinstance(data, dict):
            raise HTTPException(
                status_code=400,
                detail="Request must contain currency pair data"
            )
            
        if len(data) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two currency pairs required for cointegration analysis"
            )
            
        # Execute analysis
        result = analyzer.analyze(data)
        
        if not result.is_valid:
            raise HTTPException(
                status_code=422,
                detail=result.result_data.get("error", "Analysis failed")
            )
        
        # Extract only cointegration-relevant information
        cointegration_results = {
            "cointegration_tests": result.result_data.get("cointegration_tests", []),
            "trading_signals": [signal for signal in result.result_data.get("trading_signals", [])
                              if signal.get("signal_type") == "cointegration"]
        }
        
        return {
            "status": "success",
            "results": cointegration_results,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing cointegration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test cointegration: {str(e)}"
        )
