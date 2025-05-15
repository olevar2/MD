"""
Market Analysis Service implementation.

This module provides the implementation of the Market Analysis Service, which provides
market analysis capabilities, including pattern recognition, support/resistance detection,
market regime detection, correlation analysis, volatility analysis, and sentiment analysis.
"""
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from market_analysis_service.interfaces.market_analysis_interface import IMarketAnalysisService
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
    AnalysisType,
    PatternType,
    MarketRegimeType,
    SupportResistanceMethod
)
from market_analysis_service.adapters.data_pipeline_adapter import DataPipelineAdapter
from market_analysis_service.adapters.analysis_coordinator_adapter import AnalysisCoordinatorAdapter
from market_analysis_service.adapters.feature_store_adapter import FeatureStoreAdapter
from market_analysis_service.repositories.analysis_repository import AnalysisRepository
from market_analysis_service.core.pattern_recognition import PatternRecognizer
from market_analysis_service.core.support_resistance import SupportResistanceDetector
from market_analysis_service.core.market_regime import MarketRegimeDetector
from market_analysis_service.core.correlation_analysis import CorrelationAnalyzer
from market_analysis_service.core.volatility_analysis import VolatilityAnalyzer
from market_analysis_service.core.sentiment_analysis import SentimentAnalyzer

from common_lib.resilience.decorators import (
    retry_with_backoff,
    circuit_breaker,
    timeout
)

logger = logging.getLogger(__name__)

class MarketAnalysisService(IMarketAnalysisService):
    """
    Implementation of the Market Analysis Service.
    """

    def __init__(
        self,
        data_pipeline_adapter: DataPipelineAdapter,
        analysis_coordinator_adapter: AnalysisCoordinatorAdapter,
        feature_store_adapter: FeatureStoreAdapter,
        analysis_repository: AnalysisRepository
    ):
        """
        Initialize the Market Analysis Service.

        Args:
            data_pipeline_adapter: Adapter for the Data Pipeline Service
            analysis_coordinator_adapter: Adapter for the Analysis Coordinator Service
            feature_store_adapter: Adapter for the Feature Store Service
            analysis_repository: Repository for analysis data
        """
        self.data_pipeline_adapter = data_pipeline_adapter
        self.analysis_coordinator_adapter = analysis_coordinator_adapter
        self.feature_store_adapter = feature_store_adapter
        self.analysis_repository = analysis_repository

        # Initialize analyzers
        self.pattern_recognizer = PatternRecognizer()
        self.support_resistance_detector = SupportResistanceDetector()
        self.market_regime_detector = MarketRegimeDetector()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(timeout_seconds=30)
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
        logger.info(f"Analyzing market for {request.symbol} {request.timeframe}")

        start_time = datetime.now()
        request_id = str(uuid.uuid4())

        # Convert dates to datetime
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00')) if request.end_date else None

        # Get market data
        data = await self.data_pipeline_adapter.get_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=start_date,
            end_date=end_date,
            include_indicators=True
        )

        # Perform requested analyses
        analysis_results = []

        for analysis_type in request.analysis_types:
            result = None

            if analysis_type == AnalysisType.TECHNICAL:
                # Technical analysis is handled by the Data Pipeline Service
                # Just extract the indicators from the data
                indicators = {col: data[col].tolist() for col in data.columns if col not in ["open", "high", "low", "close", "volume", "timestamp"]}

                result = {
                    "indicators": indicators
                }

            elif analysis_type == AnalysisType.PATTERN:
                # Recognize patterns
                pattern_request = PatternRecognitionRequest(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    pattern_types=None,  # Recognize all patterns
                    min_confidence=0.7
                )

                pattern_response = await self.recognize_patterns(pattern_request)

                result = {
                    "patterns": pattern_response.patterns
                }

            elif analysis_type == AnalysisType.SUPPORT_RESISTANCE:
                # Identify support and resistance levels
                sr_request = SupportResistanceRequest(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    methods=[SupportResistanceMethod.PRICE_SWINGS, SupportResistanceMethod.MOVING_AVERAGE]
                )

                sr_response = await self.identify_support_resistance(sr_request)

                result = {
                    "levels": sr_response.levels
                }

            elif analysis_type == AnalysisType.MARKET_REGIME:
                # Detect market regime
                regime_request = MarketRegimeRequest(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date
                )

                regime_response = await self.detect_market_regime(regime_request)

                result = {
                    "regimes": regime_response.regimes,
                    "current_regime": regime_response.current_regime
                }

            elif analysis_type == AnalysisType.CORRELATION:
                # For correlation analysis, we need data for multiple symbols
                # Get the symbols from the additional parameters
                symbols = request.additional_parameters.get("symbols", []) if request.additional_parameters else []

                if symbols:
                    correlation_request = CorrelationAnalysisRequest(
                        symbols=[request.symbol] + symbols,
                        timeframe=request.timeframe,
                        start_date=request.start_date,
                        end_date=request.end_date
                    )

                    correlation_response = await self.analyze_correlations(correlation_request)

                    result = {
                        "correlation_matrix": correlation_response.correlation_matrix,
                        "correlation_pairs": correlation_response.correlation_pairs
                    }
                else:
                    result = {
                        "correlation_matrix": {},
                        "correlation_pairs": []
                    }

            elif analysis_type == AnalysisType.VOLATILITY:
                # Analyze volatility
                volatility_result = await self.analyze_volatility(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=request.additional_parameters
                )

                result = volatility_result

            elif analysis_type == AnalysisType.SENTIMENT:
                # Analyze sentiment
                sentiment_result = await self.analyze_sentiment(
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=request.additional_parameters
                )

                result = sentiment_result

            elif analysis_type == AnalysisType.COMPREHENSIVE:
                # Perform all analyses
                # This is a placeholder for a more comprehensive analysis
                result = {
                    "summary": "Comprehensive analysis not yet implemented"
                }

            if result:
                # Calculate execution time
                execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

                # Add result to analysis results
                analysis_results.append({
                    "analysis_type": analysis_type,
                    "result": result,
                    "confidence": 0.8,  # Placeholder confidence
                    "execution_time_ms": execution_time_ms
                })

        # Calculate total execution time
        total_execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Create response
        response = MarketAnalysisResponse(
            request_id=request_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            analysis_results=analysis_results,
            execution_time_ms=total_execution_time_ms,
            timestamp=datetime.now()
        )

        # Save analysis result to repository
        await self.analysis_repository.save_analysis_result(
            analysis_type="comprehensive",
            symbol=request.symbol,
            timeframe=request.timeframe,
            result=response.dict()
        )

        logger.info(f"Completed market analysis for {request.symbol} {request.timeframe} in {total_execution_time_ms}ms")

        return response

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(timeout_seconds=20)
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
        logger.info(f"Recognizing patterns for {request.symbol} {request.timeframe}")

        start_time = datetime.now()
        request_id = str(uuid.uuid4())

        # Convert dates to datetime
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00')) if request.end_date else None

        # Get market data
        data = await self.data_pipeline_adapter.get_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # Convert pattern types to enum values
        pattern_types = None

        if request.pattern_types:
            pattern_types = [PatternType(pt) for pt in request.pattern_types]

        # Recognize patterns
        patterns = self.pattern_recognizer.recognize_patterns(
            data=data,
            pattern_types=pattern_types,
            min_confidence=request.min_confidence
        )

        # Calculate execution time
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Create response
        response = PatternRecognitionResponse(
            request_id=request_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            patterns=patterns,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now()
        )

        # Save analysis result to repository
        await self.analysis_repository.save_analysis_result(
            analysis_type="pattern_recognition",
            symbol=request.symbol,
            timeframe=request.timeframe,
            result=response.dict()
        )

        logger.info(f"Recognized {len(patterns)} patterns for {request.symbol} {request.timeframe} in {execution_time_ms}ms")

        return response

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(timeout_seconds=20)
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
        logger.info(f"Identifying support and resistance for {request.symbol} {request.timeframe}")

        start_time = datetime.now()
        request_id = str(uuid.uuid4())

        # Convert dates to datetime
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00')) if request.end_date else None

        # Get market data
        data = await self.data_pipeline_adapter.get_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # Convert methods to enum values
        methods = [SupportResistanceMethod(m) for m in request.methods]

        # Identify support and resistance levels
        levels = self.support_resistance_detector.identify_support_resistance(
            data=data,
            methods=methods,
            levels_count=request.levels_count,
            additional_parameters=request.additional_parameters
        )

        # Calculate execution time
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Create response
        response = SupportResistanceResponse(
            request_id=request_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            levels=levels,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now()
        )

        # Save analysis result to repository
        await self.analysis_repository.save_analysis_result(
            analysis_type="support_resistance",
            symbol=request.symbol,
            timeframe=request.timeframe,
            result=response.dict()
        )

        logger.info(f"Identified {len(levels)} support/resistance levels for {request.symbol} {request.timeframe} in {execution_time_ms}ms")

        return response

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(timeout_seconds=20)
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
        logger.info(f"Detecting market regime for {request.symbol} {request.timeframe}")

        start_time = datetime.now()
        request_id = str(uuid.uuid4())

        # Convert dates to datetime
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00')) if request.end_date else None

        # Get market data
        data = await self.data_pipeline_adapter.get_market_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # Detect market regimes
        regimes = self.market_regime_detector.detect_market_regime(
            data=data,
            window_size=request.window_size,
            additional_parameters=request.additional_parameters
        )

        # Determine current regime
        current_regime = MarketRegimeType.RANGING.value

        if regimes:
            # Get the most recent regime
            current_regime = regimes[-1]["regime_type"]

        # Calculate execution time
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Create response
        response = MarketRegimeResponse(
            request_id=request_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            regimes=regimes,
            current_regime=current_regime,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now()
        )

        # Save analysis result to repository
        await self.analysis_repository.save_analysis_result(
            analysis_type="market_regime",
            symbol=request.symbol,
            timeframe=request.timeframe,
            result=response.dict()
        )

        logger.info(f"Detected {len(regimes)} market regimes for {request.symbol} {request.timeframe} in {execution_time_ms}ms")

        return response

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(timeout_seconds=30)
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
        logger.info(f"Analyzing correlations for {', '.join(request.symbols)}")

        start_time = datetime.now()
        request_id = str(uuid.uuid4())

        # Convert dates to datetime
        start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00')) if request.end_date else None

        # Get market data for all symbols
        data = {}

        for symbol in request.symbols:
            symbol_data = await self.data_pipeline_adapter.get_market_data(
                symbol=symbol,
                timeframe=request.timeframe,
                start_date=start_date,
                end_date=end_date
            )

            data[symbol] = symbol_data

        # Analyze correlations
        correlation_results = self.correlation_analyzer.analyze_correlations(
            data=data,
            window_size=request.window_size,
            method=request.method,
            additional_parameters=request.additional_parameters
        )

        # Calculate execution time
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Create response
        response = CorrelationAnalysisResponse(
            request_id=request_id,
            symbols=request.symbols,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            method=request.method,
            correlation_matrix=correlation_results["correlation_matrix"],
            correlation_pairs=correlation_results["correlation_pairs"],
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now()
        )

        # Save analysis result to repository
        await self.analysis_repository.save_analysis_result(
            analysis_type="correlation_analysis",
            symbol=",".join(request.symbols),
            timeframe=request.timeframe,
            result=response.dict()
        )

        logger.info(f"Analyzed correlations for {len(request.symbols)} symbols in {execution_time_ms}ms")

        return response

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(timeout_seconds=20)
    async def analyze_volatility(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Optional[Dict[str, Any]] = None
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
        logger.info(f"Analyzing volatility for {symbol} {timeframe}")

        start_time = datetime.now()

        # Get market data
        data = await self.data_pipeline_adapter.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # Analyze volatility
        volatility_results = self.volatility_analyzer.analyze_volatility(
            data=data,
            window_sizes=parameters.get("window_sizes") if parameters else None,
            additional_parameters=parameters
        )

        # Calculate execution time
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Add execution time to results
        volatility_results["execution_time_ms"] = execution_time_ms

        # Save analysis result to repository
        await self.analysis_repository.save_analysis_result(
            analysis_type="volatility_analysis",
            symbol=symbol,
            timeframe=timeframe,
            result=volatility_results
        )

        logger.info(f"Analyzed volatility for {symbol} {timeframe} in {execution_time_ms}ms")

        return volatility_results

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @timeout(timeout_seconds=20)
    async def analyze_sentiment(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Optional[Dict[str, Any]] = None
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
        logger.info(f"Analyzing sentiment for {symbol} {timeframe}")

        start_time = datetime.now()

        # Get market data
        data = await self.data_pipeline_adapter.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # Get external sentiment data if available
        sentiment_data = None

        if parameters and "sentiment_data_source" in parameters:
            # This is a placeholder for getting external sentiment data
            # In a real implementation, this would call an external service
            pass

        # Analyze sentiment
        sentiment_results = self.sentiment_analyzer.analyze_sentiment(
            data=data,
            sentiment_data=sentiment_data,
            additional_parameters=parameters
        )

        # Calculate execution time
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Add execution time to results
        sentiment_results["execution_time_ms"] = execution_time_ms

        # Save analysis result to repository
        await self.analysis_repository.save_analysis_result(
            analysis_type="sentiment_analysis",
            symbol=symbol,
            timeframe=timeframe,
            result=sentiment_results
        )

        logger.info(f"Analyzed sentiment for {symbol} {timeframe} in {execution_time_ms}ms")

        return sentiment_results

    async def get_available_patterns(self) -> List[Dict[str, Any]]:
        """
        Get available chart patterns for recognition.

        Returns:
            List of available patterns
        """
        return self.pattern_recognizer._get_available_patterns()

    async def get_available_regimes(self) -> List[Dict[str, Any]]:
        """
        Get available market regimes for detection.

        Returns:
            List of available regimes
        """
        return self.market_regime_detector._get_available_regimes()

    async def get_available_methods(self) -> Dict[str, List[str]]:
        """
        Get available analysis methods.

        Returns:
            Dictionary of available methods
        """
        methods = {
            "pattern_recognition": [p["id"] for p in self.pattern_recognizer._get_available_patterns()],
            "support_resistance": [m["id"] for m in self.support_resistance_detector._get_available_methods()],
            "market_regime": [r["id"] for r in self.market_regime_detector._get_available_regimes()],
            "correlation": ["pearson", "spearman", "kendall"],
            "volatility": ["historical", "garch", "parkinson", "garman_klass"],
            "sentiment": ["technical", "price", "external", "combined"]
        }

        return methods