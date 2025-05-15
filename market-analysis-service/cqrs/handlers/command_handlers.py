"""
Command handlers for the Market Analysis Service.

This module provides the command handlers for the Market Analysis Service.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from common_lib.cqrs.commands import CommandHandler
from market_analysis_service.cqrs.commands import (
    AnalyzeMarketCommand,
    RecognizePatternsCommand,
    DetectSupportResistanceCommand,
    DetectMarketRegimeCommand,
    AnalyzeCorrelationCommand,
    AnalyzeVolatilityCommand,
    AnalyzeSentimentCommand
)
from market_analysis_service.models.market_analysis_models import (
    MarketAnalysisResponse,
    PatternRecognitionResponse,
    SupportResistanceResponse,
    MarketRegimeResponse,
    CorrelationAnalysisResponse,
    AnalysisType
)
from market_analysis_service.repositories.write_repositories import AnalysisWriteRepository
from market_analysis_service.services.market_analysis_service import MarketAnalysisService

logger = logging.getLogger(__name__)


class AnalyzeMarketCommandHandler(CommandHandler):
    """Handler for AnalyzeMarketCommand."""
    
    def __init__(
        self,
        market_analysis_service: MarketAnalysisService,
        repository: AnalysisWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            market_analysis_service: Market analysis service
            repository: Analysis write repository
        """
        self.market_analysis_service = market_analysis_service
        self.repository = repository
    
    async def handle(self, command: AnalyzeMarketCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the analysis result
        """
        logger.info(f"Handling AnalyzeMarketCommand: {command}")
        
        # Convert command to request
        from market_analysis_service.models.market_analysis_models import MarketAnalysisRequest
        
        request = MarketAnalysisRequest(
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date.isoformat(),
            end_date=command.end_date.isoformat() if command.end_date else None,
            analysis_types=command.analysis_types,
            additional_parameters=command.additional_parameters
        )
        
        # Perform analysis
        result = await self.market_analysis_service.analyze_market(request)
        
        # Save result
        await self.repository.add(result)
        
        return result.request_id


class RecognizePatternsCommandHandler(CommandHandler):
    """Handler for RecognizePatternsCommand."""
    
    def __init__(
        self,
        market_analysis_service: MarketAnalysisService,
        repository: AnalysisWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            market_analysis_service: Market analysis service
            repository: Analysis write repository
        """
        self.market_analysis_service = market_analysis_service
        self.repository = repository
    
    async def handle(self, command: RecognizePatternsCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the pattern recognition result
        """
        logger.info(f"Handling RecognizePatternsCommand: {command}")
        
        # Convert command to request
        from market_analysis_service.models.market_analysis_models import PatternRecognitionRequest
        
        request = PatternRecognitionRequest(
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date.isoformat(),
            end_date=command.end_date.isoformat() if command.end_date else None,
            pattern_types=[p.value for p in command.pattern_types] if command.pattern_types else None,
            min_confidence=command.min_confidence,
            additional_parameters=command.additional_parameters
        )
        
        # Recognize patterns
        result = await self.market_analysis_service.recognize_patterns(request)
        
        # Save result
        await self.repository.add(result)
        
        return result.request_id


class DetectSupportResistanceCommandHandler(CommandHandler):
    """Handler for DetectSupportResistanceCommand."""
    
    def __init__(
        self,
        market_analysis_service: MarketAnalysisService,
        repository: AnalysisWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            market_analysis_service: Market analysis service
            repository: Analysis write repository
        """
        self.market_analysis_service = market_analysis_service
        self.repository = repository
    
    async def handle(self, command: DetectSupportResistanceCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the support/resistance result
        """
        logger.info(f"Handling DetectSupportResistanceCommand: {command}")
        
        # Convert command to request
        from market_analysis_service.models.market_analysis_models import SupportResistanceRequest
        
        request = SupportResistanceRequest(
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date.isoformat(),
            end_date=command.end_date.isoformat() if command.end_date else None,
            methods=[m.value for m in command.methods] if command.methods else None,
            additional_parameters=command.additional_parameters
        )
        
        # Detect support/resistance
        result = await self.market_analysis_service.detect_support_resistance(request)
        
        # Save result
        await self.repository.add(result)
        
        return result.request_id


class DetectMarketRegimeCommandHandler(CommandHandler):
    """Handler for DetectMarketRegimeCommand."""
    
    def __init__(
        self,
        market_analysis_service: MarketAnalysisService,
        repository: AnalysisWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            market_analysis_service: Market analysis service
            repository: Analysis write repository
        """
        self.market_analysis_service = market_analysis_service
        self.repository = repository
    
    async def handle(self, command: DetectMarketRegimeCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the market regime result
        """
        logger.info(f"Handling DetectMarketRegimeCommand: {command}")
        
        # Convert command to request
        from market_analysis_service.models.market_analysis_models import MarketRegimeRequest
        
        request = MarketRegimeRequest(
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date.isoformat(),
            end_date=command.end_date.isoformat() if command.end_date else None,
            window_size=command.window_size,
            additional_parameters=command.additional_parameters
        )
        
        # Detect market regime
        result = await self.market_analysis_service.detect_market_regime(request)
        
        # Save result
        await self.repository.add(result)
        
        return result.request_id


class AnalyzeCorrelationCommandHandler(CommandHandler):
    """Handler for AnalyzeCorrelationCommand."""
    
    def __init__(
        self,
        market_analysis_service: MarketAnalysisService,
        repository: AnalysisWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            market_analysis_service: Market analysis service
            repository: Analysis write repository
        """
        self.market_analysis_service = market_analysis_service
        self.repository = repository
    
    async def handle(self, command: AnalyzeCorrelationCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the correlation analysis result
        """
        logger.info(f"Handling AnalyzeCorrelationCommand: {command}")
        
        # Convert command to request
        from market_analysis_service.models.market_analysis_models import CorrelationAnalysisRequest
        
        request = CorrelationAnalysisRequest(
            symbols=command.symbols,
            timeframe=command.timeframe,
            start_date=command.start_date.isoformat(),
            end_date=command.end_date.isoformat() if command.end_date else None,
            window_size=command.window_size,
            method=command.method,
            additional_parameters=command.additional_parameters
        )
        
        # Analyze correlation
        result = await self.market_analysis_service.analyze_correlation(request)
        
        # Save result
        await self.repository.add(result)
        
        return result.request_id


class AnalyzeVolatilityCommandHandler(CommandHandler):
    """Handler for AnalyzeVolatilityCommand."""
    
    def __init__(
        self,
        market_analysis_service: MarketAnalysisService,
        repository: AnalysisWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            market_analysis_service: Market analysis service
            repository: Analysis write repository
        """
        self.market_analysis_service = market_analysis_service
        self.repository = repository
    
    async def handle(self, command: AnalyzeVolatilityCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the volatility analysis result
        """
        logger.info(f"Handling AnalyzeVolatilityCommand: {command}")
        
        # Convert command to request
        from market_analysis_service.models.market_analysis_models import VolatilityAnalysisRequest
        
        request = VolatilityAnalysisRequest(
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date.isoformat(),
            end_date=command.end_date.isoformat() if command.end_date else None,
            window_size=command.window_size,
            method=command.method,
            additional_parameters=command.additional_parameters
        )
        
        # Analyze volatility
        result = await self.market_analysis_service.analyze_volatility(request)
        
        # Save result
        await self.repository.add(result)
        
        return result.request_id


class AnalyzeSentimentCommandHandler(CommandHandler):
    """Handler for AnalyzeSentimentCommand."""
    
    def __init__(
        self,
        market_analysis_service: MarketAnalysisService,
        repository: AnalysisWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            market_analysis_service: Market analysis service
            repository: Analysis write repository
        """
        self.market_analysis_service = market_analysis_service
        self.repository = repository
    
    async def handle(self, command: AnalyzeSentimentCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the sentiment analysis result
        """
        logger.info(f"Handling AnalyzeSentimentCommand: {command}")
        
        # Convert command to request
        from market_analysis_service.models.market_analysis_models import SentimentAnalysisRequest
        
        request = SentimentAnalysisRequest(
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date.isoformat(),
            end_date=command.end_date.isoformat() if command.end_date else None,
            sources=command.sources,
            additional_parameters=command.additional_parameters
        )
        
        # Analyze sentiment
        result = await self.market_analysis_service.analyze_sentiment(request)
        
        # Save result
        await self.repository.add(result)
        
        return result.request_id