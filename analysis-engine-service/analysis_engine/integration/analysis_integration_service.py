"""
Analysis Integration Service

This service coordinates integration between all analysis components
across different asset classes.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import asyncio
from datetime import datetime
import pandas as pd

from analysis_engine.services.multi_asset_service import MultiAssetService
from analysis_engine.analysis.basic_ta.technical_indicators import TechnicalIndicatorAnalyzer
from analysis_engine.analysis.advanced_ta.pattern_recognition import PatternRecognitionAnalyzer
from analysis_engine.analysis.advanced_ta.multi_timeframe import MultiTimeframeAnalyzer
from analysis_engine.analysis.ml_integration import MLPredictionIntegrator
from analysis_engine.analysis.sentiment import SentimentAnalyzer
from analysis_engine.analysis.market_regime import MarketRegimeAnalyzer
from analysis_engine.models.market_data import MarketData
from analysis_engine.models.analysis_result import AnalysisResult
from analysis_engine.multi_asset.asset_registry import AssetClass

logger = logging.getLogger(__name__)


class AnalysisIntegrationService:
    """Central service for integrating signals from analysis components across asset classes.

    Coordinates the execution of various analysis modules (technical, pattern,
    multi-timeframe, ML, sentiment, market regime) based on the asset class
    and combines their results into a unified analysis output.

    Attributes:
        logger: Logger instance for the service.
        multi_asset_service (MultiAssetService): Service for asset-specific operations
            and data normalization.
        technical_analyzer (TechnicalIndicatorAnalyzer): Component for technical analysis.
        pattern_analyzer (PatternRecognitionAnalyzer): Component for pattern recognition.
        multi_timeframe_analyzer (MultiTimeframeAnalyzer): Component for MTF analysis.
        ml_integrator (MLPredictionIntegrator): Component for integrating ML predictions.
        sentiment_analyzer (SentimentAnalyzer): Component for sentiment analysis.
        market_regime_analyzer (MarketRegimeAnalyzer): Component for market regime analysis.
    """
    """
    Central service for integrating signals from all analysis components
    across different asset classes
    """
    
    def __init__(self, multi_asset_service: Optional[MultiAssetService] = None):
        """Initializes the AnalysisIntegrationService.

        Args:
            multi_asset_service: Optional service for asset-specific operations.
                If not provided, a default instance is created.
        """
        """
        Initialize the integration service
        
        Args:
            multi_asset_service: Service for asset-specific operations
        """
        self.logger = logging.getLogger(__name__)
        self.multi_asset_service = multi_asset_service or MultiAssetService()
        
        # Initialize component analyzers
        self.technical_analyzer = TechnicalIndicatorAnalyzer()
        self.pattern_analyzer = PatternRecognitionAnalyzer()
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()
        self.ml_integrator = MLPredictionIntegrator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_regime_analyzer = MarketRegimeAnalyzer()
        
    async def analyze_asset(self, 
                     symbol: str, 
                     market_data: Dict[str, MarketData],
                     include_components: Optional[List[str]] = None) -> Dict[str, Any]:
        """Performs comprehensive analysis of an asset using relevant components.

        Determines the appropriate analysis components based on the asset class
        (or uses `include_components` if provided), runs them in parallel,
        integrates the results, and applies asset-specific adjustments.

        Args:
            symbol: The asset symbol to analyze (e.g., 'EURUSD').
            market_data: A dictionary mapping timeframes (e.g., '1h', '4h') to
                MarketData objects containing OHLCV data.
            include_components: Optional list of component names (e.g., ['technical',
                'pattern']) to explicitly run. If None, components are selected
                based on the asset class.

        Returns:
            A dictionary containing the integrated analysis results, including
            outputs from each component, asset-specific adjustments, and metadata.
            Returns an error dictionary if the asset is not found.
        """
        """
        Perform comprehensive analysis of an asset using all relevant components
        
        Args:
            symbol: Asset symbol to analyze
            market_data: Dictionary mapping timeframes to market data
            include_components: List of component names to include (None for all)
            
        Returns:
            Dictionary with integrated analysis results
        """
        # Get asset information for asset-specific analysis
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            self.logger.warning(f"Asset info not found for {symbol}")
            return {"error": f"Asset not found: {symbol}"}
        
        asset_class = asset_info.get("asset_class", "forex")
        analysis_params = self.multi_asset_service.get_analysis_parameters(symbol)
        
        # Determine which components to run based on asset class and include_components
        components_to_run = self._get_components_for_asset(asset_class, include_components)
        
        # Run all analysis components in parallel for efficiency
        tasks = []
        for component in components_to_run:
            if component == "technical":
                tasks.append(self._run_technical_analysis(symbol, market_data, analysis_params))
            elif component == "pattern":
                tasks.append(self._run_pattern_analysis(symbol, market_data, analysis_params))
            elif component == "multi_timeframe":
                tasks.append(self._run_mtf_analysis(symbol, market_data, analysis_params))
            elif component == "ml_prediction":
                tasks.append(self._run_ml_prediction(symbol, market_data, analysis_params))
            elif component == "sentiment":
                tasks.append(self._run_sentiment_analysis(symbol, analysis_params))
            elif component == "market_regime":
                tasks.append(self._run_market_regime_detection(symbol, market_data, analysis_params))
        
        # Wait for all tasks to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
        
        # Process and combine results
        integrated_results = self._integrate_results(symbol, results, asset_class)
        
        # Add asset-specific adaptations
        integrated_results["asset_specific"] = self._apply_asset_specific_adjustments(
            integrated_results, asset_class, analysis_params
        )
        
        # Add metadata
        integrated_results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "asset_class": asset_class,
            "components_included": components_to_run
        }
        
        return integrated_results
    
    def _get_components_for_asset(self, 
                                 asset_class: str, 
                                 include_components: Optional[List[str]] = None) -> List[str]:
        """Determines which analysis components are suitable for a given asset class.

        If `include_components` is provided, it overrides the asset class logic.
        Otherwise, selects a default set of components based on the asset class
        (e.g., Forex, Crypto, Stocks).

        Args:
            asset_class: The asset class identifier (e.g., 'forex', 'crypto').
            include_components: Optional list to explicitly specify components.

        Returns:
            A list of component names (strings) to be executed for the asset.
        """
        """
        Determine which analysis components to run based on asset class
        
        Args:
            asset_class: The asset class (forex, crypto, stocks, etc)
            include_components: Explicitly requested components
            
        Returns:
            List of component names to run
        """
        # All available components
        all_components = [
            "technical", "pattern", "multi_timeframe", 
            "ml_prediction", "sentiment", "market_regime"
        ]
        
        # If specific components requested, use those
        if include_components:
            return [c for c in include_components if c in all_components]
        
        # Otherwise, select based on asset class
        if asset_class == AssetClass.FOREX:
            # For forex, include all components
            return all_components
        elif asset_class == AssetClass.CRYPTO:
            # For crypto, sentiment and market regime are very important
            return all_components
        elif asset_class == AssetClass.STOCKS:
            # For stocks, sentiment is very important
            return all_components
        elif asset_class == AssetClass.COMMODITIES:
            # For commodities, seasonality analysis might replace sentiment
            return ["technical", "pattern", "multi_timeframe", "ml_prediction", "market_regime"]
        else:
            # Default to basic components
            return ["technical", "pattern", "multi_timeframe"]
    
    async def _run_technical_analysis(self, 
                               symbol: str, 
                               market_data: Dict[str, MarketData],
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the technical analysis component for the asset.

        Retrieves data for the primary timeframe, normalizes it using
        MultiAssetService, runs the TechnicalIndicatorAnalyzer, and formats the result.

        Args:
            symbol: The asset symbol.
            market_data: Market data dictionary.
            params: Asset-specific analysis parameters.

        Returns:
            A dictionary containing the component name, results (or error), and validity.
        """
        """Run technical analysis"""
        try:
            # Get primary timeframe data
            primary_tf = params.get("primary_timeframe", "1h")
            if primary_tf not in market_data:
                return {"error": f"Missing data for primary timeframe: {primary_tf}"}
                
            df = market_data[primary_tf].to_dataframe()
            
            # Normalize data for the asset type
            df = self.multi_asset_service.normalize_data(df, symbol)
            
            # Run analysis with appropriate precision
            result = self.technical_analyzer.analyze(df, precision=params.get("pattern_precision", 5))
            
            return {
                "component": "technical",
                "result": result.result_data if isinstance(result, AnalysisResult) else result,
                "is_valid": result.is_valid if isinstance(result, AnalysisResult) else True
            }
        except Exception as e:
            self.logger.error(f"Error in technical analysis for {symbol}: {str(e)}")
            return {
                "component": "technical",
                "error": str(e),
                "is_valid": False
            }
    
    async def _run_pattern_analysis(self, 
                             symbol: str, 
                             market_data: Dict[str, MarketData],
                             params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the pattern recognition analysis component.

        Retrieves data for the primary timeframe, normalizes it, runs the
        PatternRecognitionAnalyzer, and formats the result.

        Args:
            symbol: The asset symbol.
            market_data: Market data dictionary.
            params: Asset-specific analysis parameters.

        Returns:
            A dictionary containing the component name, results (or error), and validity.
        """
        """Run pattern recognition analysis"""
        try:
            # Get primary timeframe data
            primary_tf = params.get("primary_timeframe", "1h")
            if primary_tf not in market_data:
                return {"error": f"Missing data for primary timeframe: {primary_tf}"}
                
            df = market_data[primary_tf].to_dataframe()
            
            # Normalize data for the asset type
            df = self.multi_asset_service.normalize_data(df, symbol)
            
            # Run pattern analysis with asset-specific parameters
            confidence_threshold = params.get("pattern_confidence_threshold", 0.65)
            max_patterns = params.get("max_patterns_per_analysis", 5)
            
            result = self.pattern_analyzer.analyze(
                df, 
                confidence_threshold=confidence_threshold,
                max_patterns=max_patterns
            )
            
            return {
                "component": "pattern",
                "result": result.result_data if isinstance(result, AnalysisResult) else result,
                "is_valid": result.is_valid if isinstance(result, AnalysisResult) else True
            }
        except Exception as e:
            self.logger.error(f"Error in pattern analysis for {symbol}: {str(e)}")
            return {
                "component": "pattern",
                "error": str(e),
                "is_valid": False
            }
    
    async def _run_mtf_analysis(self, 
                         symbol: str, 
                         market_data: Dict[str, MarketData],
                         params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs multi-timeframe analysis for the given asset.

        This method utilizes the MultiTimeframeAnalysis component to analyze
        market data across different timeframes, looking for trend alignment
        and signal confirmations based on configured indicators and parameters.

        Args:
            symbol: The asset symbol (e.g., 'EURUSD', 'BTCUSD').
            market_data: A dictionary where keys are timeframes (e.g., '1h', '4h')
                         and values are MarketData objects containing OHLCV data.
                         Requires data for all timeframes specified in params.
            params: A dictionary of parameters specific to this analysis run,
                    including 'mtf_timeframes', 'primary_timeframe',
                    'mtf_indicators', and potentially indicator-specific settings
                    like 'ma_periods'.

        Returns:
            A dictionary containing the multi-timeframe analysis results or an error.
            Example success structure:
            {
                "component": "multi_timeframe",
                "result": {"alignment": {...}, "signal_confirmations": {...}},
                "is_valid": True
            }
            Example error structure:
            {
                "component": "multi_timeframe",
                "error": "Error message",
                "is_valid": False
            }
        """
        try:
            # Configure MTF analyzer with asset-specific parameters
            timeframes = params.get("mtf_timeframes", ["5m", "15m", "1h", "4h", "1d"])
            primary_tf = params.get("primary_timeframe", "1h")
            
            # Make sure we have all required timeframes
            missing_timeframes = [tf for tf in timeframes if tf not in market_data]
            if missing_timeframes:
                return {
                    "component": "multi_timeframe",
                    "error": f"Missing data for timeframes: {missing_timeframes}",
                    "is_valid": False
                }
                
            # Run MTF analysis
            mtf_params = {
                "timeframes": timeframes,
                "primary_timeframe": primary_tf,
                "indicators": params.get("mtf_indicators", ["rsi", "macd", "ma"]),
                "ma_periods": params.get("ma_periods", [9, 21, 50, 200])
            }
            
            result = self.multi_timeframe_analyzer.analyze_with_parameters(market_data, mtf_params)
            
            return {
                "component": "multi_timeframe",
                "result": result.result_data if isinstance(result, AnalysisResult) else result,
                "is_valid": result.is_valid if isinstance(result, AnalysisResult) else True
            }
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis for {symbol}: {str(e)}")
            return {
                "component": "multi_timeframe", 
                "error": str(e),
                "is_valid": False
            }
    
    async def _run_ml_prediction(self, 
                          symbol: str, 
                          market_data: Dict[str, MarketData],
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs machine learning predictions for the given asset.

        Integrates with the MLPredictionService (via MLIntegrator) to generate
        predictions for specified horizons based on the provided market data.

        Args:
            symbol: The asset symbol (e.g., 'EURUSD', 'BTCUSD').
            market_data: A dictionary where keys are timeframes (e.g., '1h', '4h')
                         and values are MarketData objects containing OHLCV data.
            params: A dictionary of parameters specific to this analysis run,
                    including 'prediction_horizons' to specify which future
                    periods to predict for.

        Returns:
            A dictionary containing the machine learning prediction results or an error.
            Example success structure:
            {
                "component": "ml_prediction",
                "result": {"predictions": {"1h": {...}, "4h": {...}}},
                "is_valid": True
            }
            Example error structure:
            {
                "component": "ml_prediction",
                "error": "Error message",
                "is_valid": False
            }
        """
        try:
            # Get prediction windows from parameters
            prediction_horizons = params.get(
                "prediction_horizons", 
                ["1h", "4h", "1d"]  # Default horizons
            )
            
            # Run prediction with asset-specific configuration
            result = await self.ml_integrator.get_predictions(
                symbol, 
                market_data,
                prediction_horizons=prediction_horizons
            )
            
            return {
                "component": "ml_prediction",
                "result": result,
                "is_valid": True if result and "error" not in result else False
            }
        except Exception as e:
            self.logger.error(f"Error in ML prediction for {symbol}: {str(e)}")
            return {
                "component": "ml_prediction",
                "error": str(e),
                "is_valid": False
            }
    
    async def _run_sentiment_analysis(self, 
                               symbol: str, 
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs sentiment analysis for the given asset.

        Utilizes the SentimentAnalysisService to gather and analyze sentiment
        data from configured sources (e.g., news, social media) over a
        specified lookback period.

        Args:
            symbol: The asset symbol (e.g., 'EURUSD', 'BTCUSD').
            params: A dictionary of parameters specific to this analysis run,
                    including 'sentiment_sources' and 'sentiment_lookback_hours'.

        Returns:
            A dictionary containing the sentiment analysis results or an error.
            Example success structure:
            {
                "component": "sentiment",
                "result": {"overall": {...}, "news": {...}, "social": {...}},
                "is_valid": True
            }
            Example error structure:
            {
                "component": "sentiment",
                "error": "Error message",
                "is_valid": False
            }
        """
        try:
            # Configure sentiment sources based on asset class
            sources = params.get("sentiment_sources", ["news", "social", "economic"])
            lookback_hours = params.get("sentiment_lookback_hours", 24)
            
            # Run sentiment analysis
            sentiment_results = await self.sentiment_analyzer.get_sentiment(
                symbol, 
                sources=sources,
                lookback_hours=lookback_hours
            )
            
            return {
                "component": "sentiment",
                "result": sentiment_results,
                "is_valid": True if sentiment_results else False
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis for {symbol}: {str(e)}")
            return {
                "component": "sentiment",
                "error": str(e),
                "is_valid": False
            }
    
    async def _run_market_regime_analysis(self, 
                                    symbol: str, 
                                    market_data: Dict[str, MarketData],
                                    params: Dict[str, Any]) -> Dict[str, Any]:
        """Runs market regime detection for the given asset.

        Uses the MarketRegimeAnalyzer component to identify the current market
        regime (trend and volatility) based on primary timeframe data and
        asset-specific thresholds.

        Args:
            symbol: The asset symbol (e.g., 'EURUSD', 'BTCUSD').
            market_data: A dictionary where keys are timeframes (e.g., '1h', '4h')
                         and values are MarketData objects. Requires data for the
                         'primary_timeframe' specified in params.
            params: A dictionary of parameters, including 'primary_timeframe',
                    'regime_volatility_threshold', and
                    'regime_trend_strength_threshold'.

        Returns:
            A dictionary containing the market regime analysis results or an error.
            Example success structure:
            {
                "component": "market_regime",
                "result": {"regime": {"trend": "uptrend", "volatility": "low"}},
                "is_valid": True
            }
            Example error structure:
            {
                "component": "market_regime",
                "error": "Error message",
                "is_valid": False
            }
        """
        try:
            # Get primary timeframe data
            primary_tf = params.get("primary_timeframe", "1h")
            if primary_tf not in market_data:
                return {
                    "component": "market_regime",
                    "error": f"Missing data for primary timeframe: {primary_tf}",
                    "is_valid": False
                }
                
            df = market_data[primary_tf].to_dataframe()
            
            # Normalize data for the asset type
            df = self.multi_asset_service.normalize_data(df, symbol)
            
            # Get asset-specific regime thresholds
            volatility_threshold = params.get("regime_volatility_threshold", 1.5)
            trend_strength_threshold = params.get("regime_trend_strength_threshold", 25)
            
            # Run regime detection with asset-specific parameters
            result = self.market_regime_analyzer.detect_regime(
                df,
                volatility_threshold=volatility_threshold,
                trend_strength_threshold=trend_strength_threshold
            )
            
            return {
                "component": "market_regime",
                "result": result,
                "is_valid": True if result else False
            }
        except Exception as e:
            self.logger.error(f"Error in market regime analysis for {symbol}: {str(e)}")
            return {
                "component": "market_regime",
                "error": str(e),
                "is_valid": False
            }
    
    def _integrate_results(self, 
                          symbol: str, 
                          component_results: List[Dict[str, Any]],
                          asset_class: str) -> Dict[str, Any]:
        """
        Integrate results from different analysis components
        
        Args:
            symbol: Asset symbol
            component_results: Results from individual components
            asset_class: Asset class for asset-specific integration
            
        Returns:
            Dictionary with integrated analysis
        """
        # Initialize integrated results dictionary
        integrated = {
            "symbol": symbol,
            "asset_class": asset_class,
            "components": {},
            "signals": {
                "bullish": [],
                "bearish": [],
                "neutral": []
            },
            "confidence_scores": {},
            "overall_signal": None,
            "overall_confidence": 0.0
        }
        
        # Process each component result
        for result in component_results:
            if isinstance(result, Exception):
                self.logger.warning(f"Exception in analysis component: {str(result)}")
                continue
                
            component_name = result.get("component")
            if not component_name:
                continue
                
            # Add component result to integrated results
            integrated["components"][component_name] = result.get("result", {})
            
            # Extract signals from component
            self._extract_signals_from_component(integrated, component_name, result, asset_class)
        
        # Calculate overall signal and confidence
        if integrated["signals"]["bullish"] or integrated["signals"]["bearish"]:
            integrated["overall_signal"], integrated["overall_confidence"] = self._calculate_overall_signal(integrated)
        else:
            integrated["overall_signal"] = "neutral"
            integrated["overall_confidence"] = 0.5
            
        return integrated
    
    def _extract_signals_from_component(self, 
                                       integrated: Dict[str, Any], 
                                       component_name: str,
                                       result: Dict[str, Any],
                                       asset_class: str):
        """Extracts signals from a single component's result and updates the integrated analysis.

        This method acts as a dispatcher, calling the appropriate specific signal
        extraction method based on the component name.

        Args:
            integrated: The dictionary holding the aggregated analysis results,
                        which will be updated by this method.
            component_name: The name of the analysis component (e.g., 'technical',
                            'pattern', 'sentiment').
            result: The result dictionary from the specific analysis component.
                    Expected to contain 'result' and 'is_valid' keys.
            asset_class: The asset class (e.g., 'forex', 'crypto') which might
                         influence signal interpretation.
        """
        if "error" in result or not result.get("is_valid", False):
            return
            
        component_result = result.get("result", {})
        
        # Extract signals based on component type
        if component_name == "technical":
            self._extract_technical_signals(integrated, component_result, asset_class)
        elif component_name == "pattern":
            self._extract_pattern_signals(integrated, component_result, asset_class)
        elif component_name == "multi_timeframe":
            self._extract_mtf_signals(integrated, component_result, asset_class)
        elif component_name == "ml_prediction":
            self._extract_ml_signals(integrated, component_result, asset_class)
        elif component_name == "sentiment":
            self._extract_sentiment_signals(integrated, component_result, asset_class)
        elif component_name == "market_regime":
            self._extract_regime_signals(integrated, component_result, asset_class)
    
    def _extract_technical_signals(self, 
                                  integrated: Dict[str, Any],
                                  technical_result: Dict[str, Any],
                                  asset_class: str):
        """Extracts bullish/bearish signals from technical indicator results.

        Parses results for indicators like RSI and MACD, determines signal
        direction (bullish/bearish), calculates confidence, and adds them
        to the 'signals' and 'confidence_scores' sections of the integrated
        analysis dictionary.

        Args:
            integrated: The dictionary holding the aggregated analysis results.
            technical_result: The 'result' sub-dictionary from the technical
                              analysis component.
            asset_class: The asset class (currently unused in this specific method
                         but kept for consistency).
        """
        # Extract RSI signals
        if "rsi" in technical_result:
            rsi = technical_result["rsi"]
            confidence = min(abs(rsi - 50) / 30, 1.0)  # Scale confidence by distance from neutral
            
            if rsi > 70:  # Overbought
                integrated["signals"]["bearish"].append({
                    "source": "technical.rsi",
                    "type": "overbought",
                    "value": rsi,
                    "confidence": confidence
                })
                integrated["confidence_scores"]["rsi_bearish"] = confidence
            elif rsi < 30:  # Oversold
                integrated["signals"]["bullish"].append({
                    "source": "technical.rsi",
                    "type": "oversold",
                    "value": rsi,
                    "confidence": confidence
                })
                integrated["confidence_scores"]["rsi_bullish"] = confidence
        
        # Extract MACD signals
        if "macd" in technical_result:
            macd = technical_result["macd"]
            if "histogram" in macd and "histogram_direction" in macd:
                hist = macd["histogram"]
                direction = macd["histogram_direction"]
                
                # Scale confidence by histogram size
                confidence = min(abs(hist) * 20, 1.0)
                
                if hist > 0 and direction > 0:  # Positive and increasing
                    integrated["signals"]["bullish"].append({
                        "source": "technical.macd",
                        "type": "positive_increasing",
                        "value": hist,
                        "confidence": confidence
                    })
                    integrated["confidence_scores"]["macd_bullish"] = confidence
                elif hist < 0 and direction < 0:  # Negative and decreasing
                    integrated["signals"]["bearish"].append({
                        "source": "technical.macd",
                        "type": "negative_decreasing",
                        "value": hist,
                        "confidence": confidence
                    })
                    integrated["confidence_scores"]["macd_bearish"] = confidence
        
        # More technical indicators can be added here
    
    def _extract_pattern_signals(self, 
                               integrated: Dict[str, Any],
                               pattern_result: Dict[str, Any],
                               asset_class: str):
        """Extracts bullish/bearish signals from detected chart patterns.

        Iterates through the patterns found by the pattern recognition component,
        determines signal direction based on the pattern's properties, and adds
        them with their confidence scores to the integrated analysis dictionary.

        Args:
            integrated: The dictionary holding the aggregated analysis results.
            pattern_result: The 'result' sub-dictionary from the pattern
                            recognition component, expected to contain a 'patterns' list.
            asset_class: The asset class (currently unused).
        """
        if "patterns" not in pattern_result:
            return
            
        patterns = pattern_result["patterns"]
        for pattern in patterns:
            pattern_type = pattern.get("type")
            confidence = pattern.get("confidence", 0.5)
            
            if not pattern_type:
                continue
                
            if pattern.get("direction") == "bullish":
                integrated["signals"]["bullish"].append({
                    "source": f"pattern.{pattern_type}",
                    "type": "chart_pattern",
                    "pattern": pattern_type,
                    "confidence": confidence
                })
                integrated["confidence_scores"][f"pattern_{pattern_type}_bullish"] = confidence
                
            elif pattern.get("direction") == "bearish":
                integrated["signals"]["bearish"].append({
                    "source": f"pattern.{pattern_type}",
                    "type": "chart_pattern",
                    "pattern": pattern_type,
                    "confidence": confidence
                })
                integrated["confidence_scores"][f"pattern_{pattern_type}_bearish"] = confidence
    
    def _extract_mtf_signals(self, 
                           integrated: Dict[str, Any],
                           mtf_result: Dict[str, Any],
                           asset_class: str):
        """Extracts signals based on multi-timeframe alignment and confirmation.

        Analyzes the results from the multi-timeframe component, looking for
        strong trend alignment across timeframes or high confirmation scores
        for signals, and adds corresponding bullish/bearish signals to the
        integrated analysis.

        Args:
            integrated: The dictionary holding the aggregated analysis results.
            mtf_result: The 'result' sub-dictionary from the multi-timeframe
                        analysis component, potentially containing 'alignment'
                        and 'signal_confirmations'.
            asset_class: The asset class (currently unused).
        """
        # Check for trend alignment
        if "alignment" in mtf_result:
            alignment = mtf_result["alignment"]
            if alignment.get("overall_alignment") == "strongly_bullish":
                integrated["signals"]["bullish"].append({
                    "source": "mtf.alignment",
                    "type": "strong_alignment",
                    "confidence": 0.9
                })
                integrated["confidence_scores"]["mtf_strong_bullish"] = 0.9
            elif alignment.get("overall_alignment") == "bullish":
                integrated["signals"]["bullish"].append({
                    "source": "mtf.alignment",
                    "type": "alignment",
                    "confidence": 0.7
                })
                integrated["confidence_scores"]["mtf_bullish"] = 0.7
            elif alignment.get("overall_alignment") == "strongly_bearish":
                integrated["signals"]["bearish"].append({
                    "source": "mtf.alignment",
                    "type": "strong_alignment",
                    "confidence": 0.9
                })
                integrated["confidence_scores"]["mtf_strong_bearish"] = 0.9
            elif alignment.get("overall_alignment") == "bearish":
                integrated["signals"]["bearish"].append({
                    "source": "mtf.alignment",
                    "type": "alignment",
                    "confidence": 0.7
                })
                integrated["confidence_scores"]["mtf_bearish"] = 0.7
        
        # Check for confirmation scores
        if "signal_confirmations" in mtf_result:
            confirmations = mtf_result["signal_confirmations"]
            if confirmations.get("bullish_confirmation", 0) > 0.7:
                integrated["signals"]["bullish"].append({
                    "source": "mtf.confirmation",
                    "type": "timeframe_confirmation",
                    "confidence": confirmations.get("bullish_confirmation")
                })
                integrated["confidence_scores"]["mtf_confirmation_bullish"] = confirmations.get("bullish_confirmation")
            elif confirmations.get("bearish_confirmation", 0) > 0.7:
                integrated["signals"]["bearish"].append({
                    "source": "mtf.confirmation",
                    "type": "timeframe_confirmation",
                    "confidence": confirmations.get("bearish_confirmation")
                })
                integrated["confidence_scores"]["mtf_confirmation_bearish"] = confirmations.get("bearish_confirmation")
    
    def _extract_ml_signals(self, 
                          integrated: Dict[str, Any],
                          ml_result: Dict[str, Any],
                          asset_class: str):
        """Extracts signals from machine learning prediction results.

        Parses the predictions for different horizons (e.g., price direction,
        volatility). Adds bullish/bearish signals based on direction predictions
        with sufficient probability, and neutral signals for significant expected
        volatility increases.

        Args:
            integrated: The dictionary holding the aggregated analysis results.
            ml_result: The 'result' sub-dictionary from the ML prediction component.
                       May contain asset-class specific prediction keys.
            asset_class: The asset class, used to find the correct prediction key
                         within ml_result.
        """
        # Get the right predictions based on asset class
        prediction_key = f"{asset_class}_predictions" if f"{asset_class}_predictions" in ml_result else "predictions"
        
        if prediction_key not in ml_result:
            return
            
        predictions = ml_result[prediction_key]
        
        # Process direction predictions
        if "direction" in predictions:
            for horizon, pred in predictions["direction"].items():
                if "probability" not in pred or "direction" not in pred:
                    continue
                    
                probability = pred["probability"]
                direction = pred["direction"]
                
                # Only consider predictions with reasonable confidence
                if probability < 0.55:
                    continue
                
                if direction == "up":
                    integrated["signals"]["bullish"].append({
                        "source": f"ml.direction.{horizon}",
                        "type": "price_direction",
                        "horizon": horizon,
                        "confidence": probability
                    })
                    integrated["confidence_scores"][f"ml_direction_{horizon}_bullish"] = probability
                elif direction == "down":
                    integrated["signals"]["bearish"].append({
                        "source": f"ml.direction.{horizon}",
                        "type": "price_direction",
                        "horizon": horizon,
                        "confidence": probability
                    })
                    integrated["confidence_scores"][f"ml_direction_{horizon}_bearish"] = probability
        
        # Process volatility predictions
        if "volatility" in predictions:
            for horizon, pred in predictions["volatility"].items():
                if "expected" not in pred:
                    continue
                
                expected_volatility = pred["expected"]
                if "current" in pred:
                    current_volatility = pred["current"]
                    # If expected volatility is significantly higher than current
                    if expected_volatility > current_volatility * 1.5:
                        integrated["signals"]["neutral"].append({
                            "source": f"ml.volatility.{horizon}",
                            "type": "increasing_volatility",
                            "factor": expected_volatility / current_volatility,
                            "confidence": min(expected_volatility / current_volatility - 1, 1.0)
                        })
    
    def _extract_sentiment_signals(self, 
                                 integrated: Dict[str, Any],
                                 sentiment_result: Dict[str, Any],
                                 asset_class: str):
        """Extracts signals from sentiment analysis results.

        Analyzes overall sentiment scores and news impact scores. Adds
        bullish/bearish signals if sentiment is strongly positive/negative
        and confidence (based on volume/score extremity or news impact) is
        sufficient.

        Args:
            integrated: The dictionary holding the aggregated analysis results.
            sentiment_result: The 'result' sub-dictionary from the sentiment
                              analysis component, potentially containing 'overall'
                              and 'news' sections.
            asset_class: The asset class (currently unused).
        """
        # Extract overall sentiment if available
        if "overall" in sentiment_result:
            overall = sentiment_result["overall"]
            score = overall.get("score", 0)
            volume = overall.get("volume", 0)
            
            # Calculate confidence based on volume and score extremity
            base_confidence = min(volume / 100, 1.0) if volume > 0 else 0.5
            score_confidence = abs(score) / 100  # Assume score is -100 to +100
            confidence = base_confidence * score_confidence
            
            if score > 30 and confidence > 0.4:  # Positive sentiment
                integrated["signals"]["bullish"].append({
                    "source": "sentiment.overall",
                    "type": "positive_sentiment",
                    "score": score,
                    "confidence": confidence
                })
                integrated["confidence_scores"]["sentiment_bullish"] = confidence
            elif score < -30 and confidence > 0.4:  # Negative sentiment
                integrated["signals"]["bearish"].append({
                    "source": "sentiment.overall",
                    "type": "negative_sentiment",
                    "score": score,
                    "confidence": confidence
                })
                integrated["confidence_scores"]["sentiment_bearish"] = confidence
        
        # Extract news sentiment if available
        if "news" in sentiment_result:
            news = sentiment_result["news"]
            if "impact_score" in news and "direction" in news:
                impact = news["impact_score"]
                direction = news["direction"]
                
                if impact > 7:  # High impact news
                    if direction == "bullish":
                        integrated["signals"]["bullish"].append({
                            "source": "sentiment.news",
                            "type": "high_impact_news",
                            "impact": impact,
                            "confidence": impact / 10
                        })
                        integrated["confidence_scores"]["news_bullish"] = impact / 10
                    elif direction == "bearish":
                        integrated["signals"]["bearish"].append({
                            "source": "sentiment.news",
                            "type": "high_impact_news",
                            "impact": impact,
                            "confidence": impact / 10
                        })
                        integrated["confidence_scores"]["news_bearish"] = impact / 10
    
    def _extract_regime_signals(self, 
                              integrated: Dict[str, Any],
                              regime_result: Dict[str, Any],
                              asset_class: str):
        """Extracts signals based on the detected market regime.

        Adds bullish/bearish signals based on the identified trend regime
        (e.g., strong uptrend, downtrend). Adds neutral signals for high
        volatility regimes and potentially adjusts the confidence of existing
        directional signals if volatility is extreme.

        Args:
            integrated: The dictionary holding the aggregated analysis results.
            regime_result: The 'result' sub-dictionary from the market regime
                           detection component, expected to contain a 'regime'
                           key with 'trend' and 'volatility' information.
            asset_class: The asset class (currently unused).
        """
        if "regime" not in regime_result:
            return
            
        regime = regime_result["regime"]
        
        # Extract trend regime
        if "trend" in regime:
            trend_regime = regime["trend"]
            
            if trend_regime == "strong_uptrend":
                integrated["signals"]["bullish"].append({
                    "source": "regime.trend",
                    "type": "strong_uptrend",
                    "confidence": 0.8
                })
                integrated["confidence_scores"]["regime_strong_bullish"] = 0.8
            elif trend_regime == "uptrend":
                integrated["signals"]["bullish"].append({
                    "source": "regime.trend",
                    "type": "uptrend",
                    "confidence": 0.6
                })
                integrated["confidence_scores"]["regime_bullish"] = 0.6
            elif trend_regime == "strong_downtrend":
                integrated["signals"]["bearish"].append({
                    "source": "regime.trend",
                    "type": "strong_downtrend",
                    "confidence": 0.8
                })
                integrated["confidence_scores"]["regime_strong_bearish"] = 0.8
            elif trend_regime == "downtrend":
                integrated["signals"]["bearish"].append({
                    "source": "regime.trend",
                    "type": "downtrend",
                    "confidence": 0.6
                })
                integrated["confidence_scores"]["regime_bearish"] = 0.6
        
        # Extract volatility regime
        if "volatility" in regime:
            vol_regime = regime["volatility"]
            
            if vol_regime == "high":
                integrated["signals"]["neutral"].append({
                    "source": "regime.volatility",
                    "type": "high_volatility",
                    "confidence": 0.7
                })
            elif vol_regime == "very_high":
                integrated["signals"]["neutral"].append({
                    "source": "regime.volatility",
                    "type": "extreme_volatility",
                    "confidence": 0.9
                })
                
                # In extremely volatile markets, reduce confidence of direction signals
                for signal_list in [integrated["signals"]["bullish"], integrated["signals"]["bearish"]]:
                    for signal in signal_list:
                        signal["confidence"] *= 0.7
    
    def _calculate_overall_signal(self, integrated: Dict[str, Any]) -> tuple:
        """Calculates an overall signal (bullish/bearish/neutral) and confidence score.

        Aggregates the confidence scores from all extracted bullish and bearish
        signals. It determines the overall direction based on the stronger
        aggregate confidence and calculates a final confidence score reflecting
        the strength of the dominant direction relative to the opposing one.

        Args:
            integrated: The dictionary holding the aggregated analysis results,
                        specifically the 'signals' dictionary containing lists
                        of bullish and bearish signals with their confidence scores.

        Returns:
            A tuple containing:
            - overall_signal (str): 'bullish', 'bearish', or 'neutral'.
            - overall_confidence (float): A score between 0.0 and 1.0 indicating
              the confidence in the overall_signal.
        """
        bullish_signals = integrated["signals"]["bullish"]
        bearish_signals = integrated["signals"]["bearish"]
        
        if not bullish_signals and not bearish_signals:
            return "neutral", 0.5
        
        # Calculate weighted confidence for each direction
        bullish_confidence = sum(s["confidence"] for s in bullish_signals) if bullish_signals else 0
        bearish_confidence = sum(s["confidence"] for s in bearish_signals) if bearish_signals else 0
        
        # Normalize by number of signals to avoid bias towards more signals
        if bullish_signals:
            bullish_confidence /= len(bullish_signals)
        if bearish_signals:
            bearish_confidence /= len(bearish_signals)
            
        # Determine overall direction and confidence
        if bullish_confidence > bearish_confidence:
            signal = "bullish"
            # How much stronger is bullish than bearish?
            if bearish_confidence > 0:
                confidence = bullish_confidence / (bullish_confidence + bearish_confidence)
            else:
                confidence = bullish_confidence
        elif bearish_confidence > bullish_confidence:
            signal = "bearish"
            # How much stronger is bearish than bullish?
            if bullish_confidence > 0:
                confidence = bearish_confidence / (bearish_confidence + bullish_confidence)
            else:
                confidence = bearish_confidence
        else:
            signal = "neutral"
            confidence = 0.5
            
        return signal, min(confidence, 1.0)
    
    def _apply_asset_specific_adjustments(self, 
                                        integrated: Dict[str, Any],
                                        asset_class: AssetClass,
                                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Applies adjustments and adds context specific to the asset class.

        Calls helper methods based on the asset class (Forex, Crypto, Stocks)
        to calculate and add relevant contextual information (e.g., session
        activity for Forex, Bitcoin dominance for Crypto, market hours for Stocks)
        to the final analysis result. Also performs checks like spread viability
        for Forex based on signal confidence.

        Args:
            integrated: The dictionary holding the aggregated analysis results,
                        including the 'overall_confidence'.
            asset_class: An Enum value (AssetClass.FOREX, AssetClass.CRYPTO, etc.)
                         indicating the type of asset being analyzed.
            params: The dictionary of analysis parameters (currently unused here
                    but passed for potential future use).

        Returns:
            A dictionary containing the calculated asset-specific adjustments and context.
        """
        adjustments = {}
        
        if asset_class == AssetClass.FOREX:
            # For forex, adjust based on session activity
            adjustments["session_activity"] = self._calculate_forex_session_activity()
            
            # Apply spread considerations
            if "overall_confidence" in integrated and integrated["overall_confidence"] < 0.6:
                # Low confidence signals may not be worth the spread cost
                adjustments["spread_viability"] = False
                adjustments["spread_note"] = "Signal confidence below spread viability threshold"
            else:
                adjustments["spread_viability"] = True
            
        elif asset_class == AssetClass.CRYPTO:
            # For crypto, consider Bitcoin dominance
            adjustments["bitcoin_dominance_impact"] = self._calculate_btc_dominance_impact(
                integrated["symbol"]
            )
            
            # Consider 24/7 market volatility patterns
            adjustments["volatility_adjustment"] = self._calculate_crypto_volatility_adjustment()
            
        elif asset_class == AssetClass.STOCKS:
            # For stocks, consider market hours and pre/post market
            adjustments["market_hours_context"] = self._calculate_stock_market_hours_context()
            
            # Consider index correlation
            adjustments["index_correlation"] = self._calculate_stock_index_correlation(
                integrated["symbol"]
            )
        
        return adjustments
    
    def _calculate_forex_session_activity(self) -> Dict[str, Any]:
        """Calculates information about current Forex trading sessions.

        Placeholder implementation. A real implementation would use the current
        time (UTC) to determine which major Forex sessions (e.g., London, New York,
        Tokyo, Sydney) are active, if there's an overlap, and estimate the
        current market liquidity based on session activity.

        Returns:
            A dictionary with keys like 'active_sessions', 'session_overlap',
            and 'liquidity_rating'.
        """
        # Here you would use datetime to determine current active sessions
        # This is a simplified example
        return {
            "active_sessions": ["Europe"],  # Active trading sessions
            "session_overlap": False,  # Whether we're in a session overlap period
            "liquidity_rating": "medium"  # Current liquidity assessment
        }
    
    def _calculate_btc_dominance_impact(self, symbol: str) -> Dict[str, Any]:
        """Calculates the potential impact of Bitcoin's market dominance on an altcoin.

        Placeholder implementation. A real implementation would fetch current
        Bitcoin dominance data and potentially the historical correlation between
        the given altcoin (if symbol is not BTC) and Bitcoin to estimate the impact.

        Args:
            symbol: The crypto asset symbol.

        Returns:
            A dictionary containing information like 'btc_correlation',
            'dominance_trend', and 'independence_score' if the symbol is not BTC,
            otherwise an empty dictionary.
        """
        # For non-BTC crypto, calculate how BTC movements affect this asset
        if symbol != "BTCUSD":
            return {
                "btc_correlation": 0.85,  # Example correlation
                "dominance_trend": "increasing",  # Current BTC dominance trend
                "independence_score": 0.3  # How independent this crypto is from BTC
            }
        return {}
    
    def _calculate_crypto_volatility_adjustment(self) -> Dict[str, Any]:
        """Calculates volatility adjustments specific to the crypto market.

        Placeholder implementation. A real implementation could analyze historical
        intraday volatility patterns (e.g., higher volatility during certain hours)
        and weekend effects common in the 24/7 crypto market.

        Returns:
            A dictionary with potential keys like 'hour_of_day_factor' and
            'weekend_effect'.
        """
        return {
            "hour_of_day_factor": 1.2,  # Volatility factor based on hour of day
            "weekend_effect": 0.9  # Weekend trading typically has lower volume
        }
    
    def _calculate_stock_market_hours_context(self) -> Dict[str, Any]:
        """Determines the current context related to stock market trading hours.

        Placeholder implementation. A real implementation would check the current
        time against the standard trading hours for the relevant exchange (e.g., NYSE)
        to determine if the market is open, closed, in pre-market, or after-hours.
        It could also estimate time until close/open and liquidity factors.

        Returns:
            A dictionary with keys like 'market_status', 'time_to_close', and
            'liquidity_factor'.
        """
        # Here you would check current time against market hours
        return {
            "market_status": "open",  # open, closed, pre-market, after-hours
            "time_to_close": 180,  # Minutes until market close
            "liquidity_factor": 1.0  # Liquidity factor based on time of day
        }
    
    def _calculate_stock_index_correlation(self, symbol: str) -> Dict[str, Any]:
        """Calculates the correlation of a stock to major market indices.

        Placeholder implementation. A real implementation would likely involve
        fetching historical price data for both the stock and relevant indices
        (e.g., S&P 500, NASDAQ) and calculating their correlation over a specific
        period. It might also determine the stock's sector correlation and the
        current momentum of the indices.

        Args:
            symbol: The stock symbol.

        Returns:
            A dictionary containing correlation values (e.g., 'sp500_correlation',
            'sector_correlation') and index momentum information.
        """
        return {
            "sp500_correlation": 0.65,
            "sector_correlation": 0.82,
            "index_momentum": "positive"
        }
