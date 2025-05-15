"""
Validation utilities for Market Analysis Service.

This module provides utilities for validating requests to the Market Analysis Service.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)

def validate_request(request: BaseModel) -> Dict[str, Any]:
    """
    Validate a request to the Market Analysis Service.
    
    Args:
        request: Request to validate
        
    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []
    
    # Validate symbol
    if hasattr(request, "symbol"):
        if not request.symbol:
            errors.append("Symbol is required")
        elif not isinstance(request.symbol, str):
            errors.append("Symbol must be a string")
            
    # Validate timeframe
    if hasattr(request, "timeframe"):
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
        
        if not request.timeframe:
            errors.append("Timeframe is required")
        elif not isinstance(request.timeframe, str):
            errors.append("Timeframe must be a string")
        elif request.timeframe not in valid_timeframes:
            warnings.append(f"Timeframe {request.timeframe} is not one of the standard timeframes: {', '.join(valid_timeframes)}")
            
    # Validate dates
    if hasattr(request, "start_date"):
        if not request.start_date:
            errors.append("Start date is required")
        else:
            try:
                start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
            except ValueError:
                errors.append("Start date must be a valid ISO format date")
                
    if hasattr(request, "end_date") and request.end_date:
        try:
            end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
            
            if hasattr(request, "start_date") and request.start_date:
                try:
                    start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
                    
                    if end_date < start_date:
                        errors.append("End date must be after start date")
                except ValueError:
                    pass  # Start date error already handled
        except ValueError:
            errors.append("End date must be a valid ISO format date")
            
    # Validate symbols for correlation analysis
    if hasattr(request, "symbols"):
        if not request.symbols:
            errors.append("At least one symbol is required")
        elif not isinstance(request.symbols, list):
            errors.append("Symbols must be a list")
        elif len(request.symbols) < 2:
            errors.append("At least two symbols are required for correlation analysis")
            
    # Return validation results
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }