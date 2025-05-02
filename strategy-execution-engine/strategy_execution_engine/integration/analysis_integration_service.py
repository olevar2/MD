"""
Analysis Integration Service

This service provides a centralized integration point for all analysis components
across the platform, connecting signals from various sources and making them
available to the strategy execution engine.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Coroutine

import pandas as pd

from strategy_execution_engine.signal_aggregator import (
    SignalAggregator, Signal, SignalDirection, SignalSource, SignalTimeframe
)
from strategy_execution_engine.adaptive_layer.adaptive_service import AdaptiveLayerService


class AnalysisIntegrationService:
    """
    Integrates signals and analysis from all analysis components across the platform.
    
    This service serves as the central integration point for:
    1. Technical Analysis signals
    2. Machine Learning predictions
    3. Sentiment Analysis
    4. Market Regime detection
    5. Pattern Recognition
    6. Multi-Asset adaptations
    7. Economic calendar impacts
    8. Correlation analysis
    """
    
    def __init__(
        self,
        signal_aggregator: SignalAggregator,
        adaptive_layer_service: AdaptiveLayerService,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the Analysis Integration Service
        
        Args:
            signal_aggregator: The signal aggregator to use
            adaptive_layer_service: Service for adaptive parameters
            config: Service configuration
        """
        self.signal_aggregator = signal_aggregator
        self.adaptive_layer = adaptive_layer_service
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize component clients
        self._init_component_clients()
        
        # Cache for recently fetched signals
        self.signal_cache = {}
        
    def _init_component_clients(self):
        """Initialize clients for various analysis components"""
        # These will be populated when needed, using lazy initialization
        self._ta_client = None
        self._ml_client = None
        self._sentiment_client = None
        self._regime_client = None
        self._pattern_client = None
        self._multi_asset_client = None
        self._calendar_client = None
        self._correlation_client = None
    def _get_default_components(self) -> List[str]:
        """Get the default list of components to fetch signals from."""
        return [
            "technical_analysis", 
            "machine_learning", 
            "market_regime",
            "sentiment",
            "pattern_recognition",
            "multi_asset",
            "economic_calendar", 
            "correlation"
        ]
        
    def _create_component_tasks(self, components: List[str], symbol: str, 
                               tf_enum: SignalTimeframe, lookback_bars: int) -> List[Coroutine]:
        """Create a list of coroutines for fetching signals from components."""
        component_tasks = []
        component_map = {
            "technical_analysis": lambda: self._get_technical_signals(symbol, tf_enum, lookback_bars),
            "machine_learning": lambda: self._get_ml_predictions(symbol, tf_enum, lookback_bars),
            "market_regime": lambda: self._get_market_regime(symbol, tf_enum),
            "sentiment": lambda: self._get_sentiment_signals(),
            "pattern_recognition": lambda: self._get_pattern_signals(),
            "multi_asset": lambda: self._get_multi_asset_adaptations(symbol),
            "economic_calendar": lambda: self._get_economic_calendar_impacts(symbol),
            "correlation": lambda: self._get_correlation_signals()
        }
        
        for component in components:
            if component in component_map:
                component_tasks.append(component_map[component]())
            else:
                self.logger.warning(f"Unknown component: {component}")
                
        return component_tasks
        
    def _process_component_results(self, components: List[str], results: List[Any]) -> Tuple[List[Dict], Dict[str, Any]]:
        """Process results from component calls, extracting signals and handling errors."""
        all_signals = []
        component_data = {}
        
        for i, component in enumerate(components):
            result = results[i]
            
            # Skip failed components but log the error
            if isinstance(result, Exception):
                self.logger.error(f"Error getting signals for {component}: {str(result)}")
                continue
                
            if "signals" in result:
                all_signals.extend(result["signals"])
            
            # Store component data
            component_data[component] = result
            
        return all_signals, component_data

    async def get_integrated_signals(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100,
        include_components: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get integrated signals from all analysis components
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_bars: Number of bars to analyze
            include_components: List of components to include, or None for all
            
        Returns:
            Dictionary with all integrated signals and metadata
        """
        # Convert timeframe to enum
        tf_enum = self._convert_timeframe(timeframe)
        
        # Determine which components to include
        components_to_fetch = include_components or self._get_default_components()
        
        # Create tasks for each component
        tasks = self._create_component_tasks(components_to_fetch, symbol, tf_enum, lookback_bars)
        
        # Wait for all component signals
        component_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_signals, component_data = self._process_component_results(components_to_fetch, component_results)
        
        # Get current market regime (needed for aggregation)
        market_regime = "unknown"
        if "market_regime" in component_data:
            market_regime = component_data["market_regime"].get("regime", "unknown")
        
        # Apply multi-asset adaptations if available
        if "multi_asset" in component_data:
            all_signals = self._apply_asset_adaptations(
                all_signals, 
                component_data["multi_asset"]
            )
        
        # Get effectiveness data from adaptive layer
        tool_ids = [s.get("source_id", "unknown") for s in all_signals if isinstance(s, dict)]
        effectiveness_data = await self._get_tool_effectiveness(tool_ids, market_regime)
        
        # Convert raw signals to Signal objects
        signal_objects = self._convert_to_signal_objects(all_signals)
        
        # Aggregate signals
        aggregated_signal = await self.signal_aggregator.aggregate_signals(
            signals=signal_objects,
            symbol=symbol,
            target_timeframe=tf_enum,
            current_market_regime=market_regime,
            effectiveness_data=effectiveness_data,
            market_context={
                "regime": market_regime,
                "economic_events": component_data.get("economic_calendar", {}).get("events", []),
                "correlations": component_data.get("correlation", {}).get("correlations", {}),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Return complete result
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "aggregated_signal": {
                "direction": aggregated_signal.direction.value if aggregated_signal else "neutral",
                "strength": aggregated_signal.strength if aggregated_signal else 0.0,
                "confidence": aggregated_signal.confidence if aggregated_signal else 0.0,
                "timestamp": aggregated_signal.timestamp.isoformat() if aggregated_signal else datetime.now().isoformat(),
            },
            "components": component_data,
            "explanation": self.signal_aggregator.generate_explanation(aggregated_signal) if aggregated_signal else "No signal generated"
        }
    
    async def _get_technical_signals(
        self, 
        symbol: str, 
        timeframe: SignalTimeframe, 
        lookback_bars: int
    ) -> Dict[str, Any]:
        """Get technical analysis signals"""
        # Lazily initialize client if needed
        if not self._ta_client:
            from analysis_engine.clients.technical_analysis_client import TechnicalAnalysisClient
            self._ta_client = TechnicalAnalysisClient()
        
        try:
            # Get analysis results from technical analysis service
            analysis_results = await self._ta_client.get_full_analysis(
                symbol=symbol,
                timeframe=timeframe.value,
                lookback_bars=lookback_bars
            )
            
            # Convert to signal format
            signals = []
            for indicator, result in analysis_results.get("indicators", {}).items():
                if "signal" in result:
                    signals.append({
                        "source_id": f"ta_{indicator}",
                        "source_type": "technical_analysis",
                        "direction": result["signal"],
                        "symbol": symbol,
                        "timeframe": timeframe.value,
                        "strength": result.get("strength", 0.5),
                        "timestamp": datetime.now(),
                        "metadata": {
                            "indicator_value": result.get("value"),
                            "indicator_type": indicator
                        }
                    })
            
            return {
                "analysis_result": analysis_results,
                "signals": signals
            }
        except Exception as e:
            self.logger.error(f"Error getting technical signals: {str(e)}", exc_info=True)
            return {"signals": []}
    
    async def _get_ml_predictions(
        self, 
        symbol: str, 
        timeframe: SignalTimeframe,
        lookback_bars: int
    ) -> Dict[str, Any]:
        """Get machine learning predictions"""
        # Lazily initialize client if needed
        if not self._ml_client:
            from ml_integration_service.client import MLIntegrationClient
            self._ml_client = MLIntegrationClient()
        
        try:
            # Get predictions from ML service
            predictions = await self._ml_client.get_predictions(
                symbol=symbol,
                timeframe=timeframe.value,
                lookback_bars=lookback_bars
            )
            
            # Convert to signal format
            signals = []
            for model_id, prediction in predictions.get("models", {}).items():
                direction = "neutral"
                if prediction["prediction_value"] > prediction.get("upper_threshold", 0.6):
                    direction = "long"
                elif prediction["prediction_value"] < prediction.get("lower_threshold", 0.4):
                    direction = "short"
                
                signals.append({
                    "source_id": f"ml_{model_id}",
                    "source_type": "machine_learning",
                    "direction": direction,
                    "symbol": symbol,
                    "timeframe": timeframe.value,
                    "strength": prediction.get("confidence", 0.5),
                    "timestamp": datetime.now(),
                    "metadata": {
                        "prediction_value": prediction["prediction_value"],
                        "model_type": prediction.get("model_type", "unknown"),
                        "model_version": prediction.get("model_version", "unknown")
                    }                })
            
            return {
                "predictions": predictions,
                "signals": signals
            }
        except Exception as e:
            self.logger.error(f"Error getting ML predictions: {str(e)}", exc_info=True)
            return {"signals": []}
            
    async def _get_sentiment_signals(self) -> Dict[str, Any]:
        """Get sentiment analysis signals"""
        # Implementation will connect to sentiment analysis service
        # For now, return placeholder data
        return {"signals": []}
    
    async def _get_market_regime(
        self, 
        symbol: str, 
        timeframe: SignalTimeframe
    ) -> Dict[str, Any]:
        """Get current market regime"""
        # Lazily initialize client if needed
        if not self._regime_client:
            from analysis_engine.clients.market_regime_client import MarketRegimeClient
            self._regime_client = MarketRegimeClient()
        
        try:
            # Get market regime from the service
            regime_info = await self._regime_client.get_current_regime(
                symbol=symbol,
                timeframe=timeframe.value
            )
            
            return {
                "regime": regime_info.get("regime", "unknown"),
                "confidence": regime_info.get("confidence", 0.5),
                "metrics": regime_info.get("metrics", {})
            }
        except Exception as e:
            self.logger.error(f"Error getting market regime: {str(e)}", exc_info=True)
            return {"regime": "unknown", "confidence": 0.0}

    async def _get_pattern_signals(self) -> Dict[str, Any]:
        """Get pattern recognition signals"""
        # Implementation will connect to pattern recognition service
        # For now, return placeholder data
        return {"signals": []}
    
    async def _get_multi_asset_adaptations(self, symbol: str) -> Dict[str, Any]:
        """Get multi-asset adaptations for the given symbol"""
        # Lazily initialize client if needed
        if not self._multi_asset_client:
            from analysis_engine.clients.multi_asset_client import MultiAssetClient
            self._multi_asset_client = MultiAssetClient()
        
        try:
            # Get asset-specific adaptations
            adaptations = await self._multi_asset_client.get_asset_adaptations(symbol)
            
            return {
                "asset_class": adaptations.get("asset_class", "forex"),
                "adaptations": adaptations.get("adaptations", {}),
                "parameters": adaptations.get("parameters", {})            }
        except Exception as e:
            self.logger.error(f"Error getting multi-asset adaptations: {str(e)}", exc_info=True)
            return {"asset_class": "unknown", "adaptations": {}}
    
    async def _get_economic_calendar_impacts(self, symbol: str) -> Dict[str, Any]:
        """Get economic calendar impacts for the given symbol"""
        # Implementation will connect to economic calendar service
        # For now, return placeholder data
        return {"events": []}
    
    async def _get_correlation_signals(self) -> Dict[str, Any]:
        """Get correlation signals"""
        # Implementation will connect to correlation analysis service
        # For now, return placeholder data
        return {"correlations": {}, "signals": []}
    
    def _convert_timeframe(self, timeframe: str) -> SignalTimeframe:
        """Convert string timeframe to SignalTimeframe enum"""
        tf_map = {
            "1m": SignalTimeframe.M1,
            "5m": SignalTimeframe.M5,
            "15m": SignalTimeframe.M15,
            "30m": SignalTimeframe.M30,
            "1h": SignalTimeframe.H1,
            "4h": SignalTimeframe.H4,
            "1d": SignalTimeframe.D1,
            "1w": SignalTimeframe.W1
        }
        return tf_map.get(timeframe, SignalTimeframe.H1)  # Default to H1 if not found
    
    def _convert_to_signal_objects(self, raw_signals: List[Dict[str, Any]]) -> List[Signal]:
        """Convert raw signal dictionaries to Signal objects"""
        signal_objects = []
        
        for signal_dict in raw_signals:
            try:
                # Map direction string to enum
                direction_str = signal_dict.get("direction", "neutral").lower()
                direction = SignalDirection.NEUTRAL
                if direction_str == "long" or direction_str == "buy":
                    direction = SignalDirection.LONG
                elif direction_str == "short" or direction_str == "sell":
                    direction = SignalDirection.SHORT
                
                # Map source type string to enum
                source_type_str = signal_dict.get("source_type", "").lower()
                source_type = SignalSource.TECHNICAL_ANALYSIS  # Default
                for enum_val in SignalSource:
                    if enum_val.value.lower() == source_type_str:
                        source_type = enum_val
                        break
                
                # Map timeframe string to enum
                timeframe_str = signal_dict.get("timeframe", "1h").lower()
                timeframe = self._convert_timeframe(timeframe_str)
                
                # Create Signal object
                signal = Signal(
                    source_id=signal_dict.get("source_id", "unknown"),
                    source_type=source_type,
                    direction=direction,
                    symbol=signal_dict.get("symbol", ""),
                    timeframe=timeframe,
                    strength=signal_dict.get("strength", 0.5),
                    timestamp=signal_dict.get("timestamp", datetime.now()),
                    metadata=signal_dict.get("metadata", {})
                )
                
                signal_objects.append(signal)
            except Exception as e:
                self.logger.error(f"Error converting signal: {str(e)}", exc_info=True)
                continue
        
        return signal_objects
    
    async def _get_tool_effectiveness(
        self, 
        tool_ids: List[str], 
        market_regime: str
    ) -> Dict[str, float]:
        """Get effectiveness scores for tools from adaptive layer"""
        try:
            return await self.adaptive_layer.get_tool_effectiveness(
                tool_ids=tool_ids,
                market_regime=market_regime
            )
        except Exception as e:
            self.logger.error(f"Error getting tool effectiveness: {str(e)}", exc_info=True)
            return {}
    
    def _apply_asset_adaptations(
        self, 
        signals: List[Dict[str, Any]], 
        asset_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply asset-specific adaptations to signals"""
        asset_class = asset_info.get("asset_class", "forex")
        adaptations = asset_info.get("adaptations", {})
        
        if not adaptations or asset_class == "unknown":
            return signals
            
        # Apply asset-specific adjustments to signals
        adapted_signals = []
        for signal in signals:
            # Create a copy to avoid modifying the original
            adapted_signal = signal.copy()
            
            # Apply strength adjustments based on asset class
            if "strength_adjustments" in adaptations:
                source_type = signal.get("source_type", "")
                if source_type in adaptations["strength_adjustments"]:
                    adjustment_factor = adaptations["strength_adjustments"][source_type]
                    adapted_signal["strength"] = min(1.0, signal.get("strength", 0.5) * adjustment_factor)
            
            # Add asset-specific metadata
            if "metadata" not in adapted_signal:
                adapted_signal["metadata"] = {}
                
            adapted_signal["metadata"]["asset_class"] = asset_class
            
            # Apply any other adaptations as needed
            if "signal_expiry_adjustment" in adaptations:
                # Adjust signal expiry if provided
                if "expiration" in signal:
                    adapted_signal["expiration"] = signal["expiration"] * adaptations["signal_expiry_adjustment"]
            
            adapted_signals.append(adapted_signal)
        
        return adapted_signals
