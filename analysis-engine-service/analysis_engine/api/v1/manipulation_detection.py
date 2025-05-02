"""
Market Manipulation Detection API

This module provides API endpoints for detecting potential market manipulation
patterns in forex data, including stop hunting, fake breakouts, and unusual
price-volume relationships.
"""

from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Body
from sqlalchemy.orm import Session
import logging
import pandas as pd
from datetime import datetime

from core_foundations.models.auth import User
from analysis_engine.analysis.manipulation.detector import MarketManipulationAnalyzer
from analysis_engine.db.connection import get_db_session
from analysis_engine.api.auth import get_current_user

# Create router
router = APIRouter(
    prefix="/manipulation",
    tags=["manipulation"]
)

# Setup logging
logger = logging.getLogger(__name__)

@router.post("/detect", response_model=Dict[str, Any])
async def detect_manipulation_patterns(
    data: Dict[str, Any],
    sensitivity: Optional[float] = Query(1.0, description="Detection sensitivity multiplier (0.5-2.0)"),
    include_protection: Optional[bool] = Query(True, description="Include protection recommendations"),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze market data for potential manipulation patterns
    
    Request body format:
    {
        "ohlcv": [
            {"timestamp": "2025-04-01T12:00:00", "open": 1.2345, "high": 1.2360, "low": 1.2340, "close": 1.2355, "volume": 1000},
            ...
        ],
        "metadata": {
            "symbol": "EUR/USD",
            "timeframe": "1h"
        }
    }
    
    Returns detected patterns, clusters, and optional protection recommendations
    """
    try:
        # Verify input data
        if not data or not isinstance(data, dict) or "ohlcv" not in data:
            raise HTTPException(
                status_code=400,
                detail="Request must contain OHLCV data"
            )
            
        ohlcv_data = data.get("ohlcv", [])
        if not isinstance(ohlcv_data, list) or len(ohlcv_data) < 100:
            raise HTTPException(
                status_code=400,
                detail="OHLCV data must contain at least 100 data points"
            )
            
        # Configure parameters based on sensitivity
        detector_params = {}
        if sensitivity != 1.0:
            if sensitivity < 0.5 or sensitivity > 2.0:
                raise HTTPException(
                    status_code=400,
                    detail="Sensitivity must be between 0.5 and 2.0"
                )
                
            # Adjust thresholds based on sensitivity
            detector_params["volume_z_threshold"] = 2.0 / sensitivity  # Lower threshold for higher sensitivity
            detector_params["price_reversal_threshold"] = 0.5 / sensitivity
            detector_params["confidence_high_threshold"] = 0.8 - (sensitivity - 1.0) * 0.1  # Lower threshold for higher sensitivity
            detector_params["confidence_medium_threshold"] = 0.6 - (sensitivity - 1.0) * 0.1
        
        # Create detector with configured parameters
        detector = MarketManipulationAnalyzer(detector_params)
        
        # Execute analysis
        result = detector.analyze(data)
        
        if not result.is_valid:
            raise HTTPException(
                status_code=422,
                detail=result.result_data.get("error", "Analysis failed")
            )
        
        # Filter out protection recommendations if not requested
        if not include_protection and "protection_recommendations" in result.result_data:
            result.result_data["protection_recommendations"] = []
        
        return {
            "status": "success",
            "symbol": data.get("metadata", {}).get("symbol", "unknown"),
            "timeframe": data.get("metadata", {}).get("timeframe", "unknown"),
            "results": result.result_data,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting manipulation patterns: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to detect manipulation patterns: {str(e)}"
        )

@router.post("/stop-hunting", response_model=Dict[str, Any])
async def detect_stop_hunting(
    data: Dict[str, Any],
    lookback: Optional[int] = Query(30, description="Lookback period for stop hunting detection"),
    recovery_threshold: Optional[float] = Query(0.5, description="Recovery percentage threshold"),
    current_user: User = Depends(get_current_user)
):
    """
    Specifically analyze for stop hunting patterns
    
    Request body format:
    {
        "ohlcv": [...],
        "metadata": {...}
    }
    
    Returns detected stop hunting patterns and related support/resistance levels
    """
    try:
        # Verify input data
        if not data or not isinstance(data, dict) or "ohlcv" not in data:
            raise HTTPException(
                status_code=400,
                detail="Request must contain OHLCV data"
            )
            
        # Configure parameters for stop hunting focus
        detector_params = {
            "stop_hunting_lookback": lookback,
            "stop_hunting_recovery": recovery_threshold
        }
        
        # Create detector with configured parameters
        detector = MarketManipulationAnalyzer(detector_params)
        
        # Execute analysis
        result = detector.analyze(data)
        
        if not result.is_valid:
            raise HTTPException(
                status_code=422,
                detail=result.result_data.get("error", "Analysis failed")
            )
        
        # Filter results to focus on stop hunting only
        filtered_results = {
            "stop_hunting_patterns": result.result_data.get("detected_patterns", {}).get("stop_hunting", []),
            "pattern_count": result.result_data.get("pattern_count", {}).get("stop_hunting", 0),
            "support_resistance": result.result_data.get("support_resistance", {}),
            "manipulation_likelihood": result.result_data.get("manipulation_likelihood", {}),
            "protection_recommendations": [
                rec for rec in result.result_data.get("protection_recommendations", [])
                if rec.get("trigger") == "stop_hunting" or rec.get("trigger") == "manipulation_cluster"
            ]
        }
        
        return {
            "status": "success",
            "symbol": data.get("metadata", {}).get("symbol", "unknown"),
            "timeframe": data.get("metadata", {}).get("timeframe", "unknown"),
            "results": filtered_results,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting stop hunting patterns: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to detect stop hunting patterns: {str(e)}"
        )

@router.post("/fake-breakouts", response_model=Dict[str, Any])
async def detect_fake_breakouts(
    data: Dict[str, Any],
    threshold: Optional[float] = Query(0.7, description="Fake breakout detection threshold"),
    current_user: User = Depends(get_current_user)
):
    """
    Specifically analyze for fake breakout patterns
    
    Request body format:
    {
        "ohlcv": [...],
        "metadata": {...}
    }
    
    Returns detected fake breakout patterns and related support/resistance levels
    """
    try:
        # Verify input data
        if not data or not isinstance(data, dict) or "ohlcv" not in data:
            raise HTTPException(
                status_code=400,
                detail="Request must contain OHLCV data"
            )
            
        # Configure parameters for fake breakout focus
        detector_params = {
            "fake_breakout_threshold": threshold
        }
        
        # Create detector with configured parameters
        detector = MarketManipulationAnalyzer(detector_params)
        
        # Execute analysis
        result = detector.analyze(data)
        
        if not result.is_valid:
            raise HTTPException(
                status_code=422,
                detail=result.result_data.get("error", "Analysis failed")
            )
        
        # Filter results to focus on fake breakouts only
        filtered_results = {
            "fake_breakout_patterns": result.result_data.get("detected_patterns", {}).get("fake_breakouts", []),
            "pattern_count": result.result_data.get("pattern_count", {}).get("fake_breakouts", 0),
            "support_resistance": result.result_data.get("support_resistance", {}),
            "manipulation_likelihood": result.result_data.get("manipulation_likelihood", {}),
            "protection_recommendations": [
                rec for rec in result.result_data.get("protection_recommendations", [])
                if rec.get("trigger") == "fake_breakout" or rec.get("trigger") == "manipulation_cluster"
            ]
        }
        
        return {
            "status": "success",
            "symbol": data.get("metadata", {}).get("symbol", "unknown"),
            "timeframe": data.get("metadata", {}).get("timeframe", "unknown"),
            "results": filtered_results,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting fake breakout patterns: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to detect fake breakout patterns: {str(e)}"
        )

@router.post("/volume-anomalies", response_model=Dict[str, Any])
async def detect_volume_anomalies(
    data: Dict[str, Any],
    z_threshold: Optional[float] = Query(2.0, description="Z-score threshold for volume anomaly detection"),
    current_user: User = Depends(get_current_user)
):
    """
    Specifically analyze for volume anomalies
    
    Request body format:
    {
        "ohlcv": [...],
        "metadata": {...}
    }
    
    Returns detected volume anomalies and potential manipulation patterns
    """
    try:
        # Verify input data
        if not data or not isinstance(data, dict) or "ohlcv" not in data:
            raise HTTPException(
                status_code=400,
                detail="Request must contain OHLCV data"
            )
            
        # Check if we have volume data
        ohlcv_sample = data.get("ohlcv", [])
        if not ohlcv_sample or len(ohlcv_sample) == 0 or "volume" not in ohlcv_sample[0]:
            raise HTTPException(
                status_code=400,
                detail="Volume data is required for volume anomaly detection"
            )
            
        # Configure parameters for volume anomaly focus
        detector_params = {
            "volume_z_threshold": z_threshold
        }
        
        # Create detector with configured parameters
        detector = MarketManipulationAnalyzer(detector_params)
        
        # Execute analysis
        result = detector.analyze(data)
        
        if not result.is_valid:
            raise HTTPException(
                status_code=422,
                detail=result.result_data.get("error", "Analysis failed")
            )
        
        # Filter results to focus on volume anomalies only
        filtered_results = {
            "volume_anomalies": result.result_data.get("detected_patterns", {}).get("volume_anomalies", []),
            "pattern_count": result.result_data.get("pattern_count", {}).get("volume_anomalies", 0),
            "manipulation_likelihood": result.result_data.get("manipulation_likelihood", {}),
            "protection_recommendations": [
                rec for rec in result.result_data.get("protection_recommendations", [])
                if rec.get("trigger") == "volume_anomaly" or rec.get("trigger") == "manipulation_cluster"
            ]
        }
        
        return {
            "status": "success",
            "symbol": data.get("metadata", {}).get("symbol", "unknown"),
            "timeframe": data.get("metadata", {}).get("timeframe", "unknown"),
            "results": filtered_results,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting volume anomalies: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to detect volume anomalies: {str(e)}"
        )
