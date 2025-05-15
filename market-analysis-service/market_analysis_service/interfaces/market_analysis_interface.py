"""
Interface for the Market Analysis Service.

This module defines the interface for the Market Analysis Service, which provides
market analysis capabilities, including pattern recognition, support/resistance detection,
market regime detection, correlation analysis, volatility analysis, and sentiment analysis.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

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
    CorrelationAnalysisResponse
)


class IMarketAnalysisService(ABC):
    """
    Interface for market analysis service.
    """
    
    @abstractmethod
    async def analyze_market(
        self,
        request: MarketAnalysisRequest
    ) -> MarketAnalysisResponse:
        """
        Perform comprehensive market analysis.
        
        Args:
            request: Market analysis request
            
        Returns:
            Market analysis response
        """
        pass
        
    @abstractmethod
    async def recognize_patterns(
        self,
        request: PatternRecognitionRequest
    ) -> PatternRecognitionResponse:
        """
        Recognize chart patterns in market data.
        
        Args:
            request: Pattern recognition request
            
        Returns:
            Pattern recognition response
        """
        pass
        
    @abstractmethod
    async def identify_support_resistance(
        self,
        request: SupportResistanceRequest
    ) -> SupportResistanceResponse:
        """
        Identify support and resistance levels.
        
        Args:
            request: Support/resistance request
            
        Returns:
            Support/resistance response
        """
        pass
        
    @abstractmethod
    async def detect_market_regime(
        self,
        request: MarketRegimeRequest
    ) -> MarketRegimeResponse:
        """
        Detect market regime (trending, ranging, volatile).
        
        Args:
            request: Market regime request
            
        Returns:
            Market regime response
        """
        pass
        
    @abstractmethod
    async def analyze_correlations(
        self,
        request: CorrelationAnalysisRequest
    ) -> CorrelationAnalysisResponse:
        """
        Analyze correlations between symbols.
        
        Args:
            request: Correlation analysis request
            
        Returns:
            Correlation analysis response
        """
        pass
        
    @abstractmethod
    async def analyze_volatility(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze market volatility.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis
            
        Returns:
            Volatility analysis data
        """
        pass
        
    @abstractmethod
    async def analyze_sentiment(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze market sentiment.
        
        Args:
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis
            
        Returns:
            Sentiment analysis data
        """
        pass
        
    @abstractmethod
    async def get_available_patterns(self) -> List[Dict[str, Any]]:
        """
        Get available chart patterns for recognition.
        
        Returns:
            List of available patterns
        """
        pass
        
    @abstractmethod
    async def get_available_regimes(self) -> List[Dict[str, Any]]:
        """
        Get available market regimes for detection.
        
        Returns:
            List of available regimes
        """
        pass
        
    @abstractmethod
    async def get_available_methods(self) -> Dict[str, List[str]]:
        """
        Get available analysis methods.
        
        Returns:
            Dictionary of available methods
        """
        pass