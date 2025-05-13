"""
Market Structure Analysis

This module provides functionality for analyzing market structure and identifying
key price levels and patterns.
"""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from analysis_engine.core.base.components import AdvancedAnalysisBase
from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ConfluenceAnalyzer(AdvancedAnalysisBase):
    """
    Confluence Detector
    
    Identifies areas where multiple technical factors align to create stronger
    support and resistance zones. Uses the consolidated ConfluenceAnalyzer for
    comprehensive confluence detection.
    """

    def __init__(self, name: str='ConfluenceAnalyzer', parameters: Optional
        [Dict[str, Any]]=None):
        """
        Initialize Confluence Detector
        
        Args:
            name: Name of the detector
            parameters: Dictionary of parameters for analysis
        """
        super().__init__(name=name, parameters=parameters)
        self.confluence_analyzer = ConfluenceAnalyzer()

    @async_with_exception_handling
    async def analyze(self, data: Dict[str, Any]) ->Dict[str, Any]:
        """
        Analyze market data for confluence zones
        
        Args:
            data: Dictionary containing market data and parameters
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            results = await self.confluence_analyzer.analyze(data)
            results.metadata.update({'analyzer': self.name, 'timestamp':
                datetime.now().isoformat()})
            return results
        except Exception as e:
            logger.error(f'Error in confluence detection: {str(e)}',
                exc_info=True)
            return {'error': f'Analysis failed: {str(e)}'}
