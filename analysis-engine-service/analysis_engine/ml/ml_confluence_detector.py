"""
Machine Learning Confluence Detector

This module provides a machine learning-based approach to detecting confluence
and divergence in forex pairs. It combines traditional technical analysis with
machine learning models to improve detection accuracy.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from analysis_engine.ml.pattern_recognition_model import PatternRecognitionModel
from analysis_engine.ml.price_prediction_model import PricePredictionModel
from analysis_engine.utils.adaptive_cache_manager import AdaptiveCacheManager

# Configure logging
logger = logging.getLogger(__name__)

class MLConfluenceDetector:
    """
    Machine learning-based confluence detector.
    
    This class combines traditional technical analysis with machine learning
    models to detect confluence and divergence in forex pairs with higher accuracy.
    """
    
    def __init__(
        self,
        correlation_service: Any,
        currency_strength_analyzer: Any,
        pattern_model_path: Optional[str] = None,
        prediction_model_path: Optional[str] = None,
        correlation_threshold: float = 0.7,
        lookback_periods: int = 20,
        cache_ttl_minutes: int = 60,
        use_gpu: bool = True
    ):
        """
        Initialize the ML confluence detector.
        
        Args:
            correlation_service: Service for getting correlations between pairs
            currency_strength_analyzer: Analyzer for calculating currency strength
            pattern_model_path: Path to saved pattern recognition model
            prediction_model_path: Path to saved price prediction model
            correlation_threshold: Minimum correlation for related pairs
            lookback_periods: Number of periods to look back for analysis
            cache_ttl_minutes: Cache time-to-live in minutes
            use_gpu: Whether to use GPU for ML models
        """
        self.correlation_service = correlation_service
        self.currency_strength_analyzer = currency_strength_analyzer
        self.correlation_threshold = correlation_threshold
        self.lookback_periods = lookback_periods
        self.use_gpu = use_gpu
        
        # Initialize cache
        self.cache_manager = AdaptiveCacheManager(
            default_ttl_seconds=cache_ttl_minutes * 60,
            max_size=1000,
            cleanup_interval_seconds=300
        )
        
        # Initialize ML models
        self.pattern_model = PatternRecognitionModel(
            model_path=pattern_model_path,
            window_size=30,
            use_gpu=use_gpu
        )
        
        self.prediction_model = PricePredictionModel(
            model_path=prediction_model_path,
            input_window=60,
            output_window=10,
            use_gpu=use_gpu
        )
        
        logger.info(f"ML confluence detector initialized with correlation_threshold={correlation_threshold}, "
                   f"lookback_periods={lookback_periods}, cache_ttl_minutes={cache_ttl_minutes}")
    
    async def find_related_pairs(self, symbol: str) -> Dict[str, float]:
        """
        Find pairs related to the given symbol based on correlation.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            
        Returns:
            Dictionary mapping related pairs to their correlations
        """
        # Check cache
        cache_key = f"related_pairs_{symbol}"
        cache_hit, cached_result = self.cache_manager.get(cache_key)
        if cache_hit:
            logger.debug(f"Cache hit for related pairs: {symbol}")
            return cached_result
        
        # Get all correlations
        all_correlations = await self.correlation_service.get_all_correlations()
        
        # Get correlations for the symbol
        symbol_correlations = all_correlations.get(symbol, {})
        
        # Filter by correlation threshold
        related_pairs = {
            pair: corr for pair, corr in symbol_correlations.items()
            if abs(corr) >= self.correlation_threshold
        }
        
        # Cache result
        self.cache_manager.set(cache_key, related_pairs)
        
        return related_pairs
    
    def detect_confluence_ml(
        self,
        symbol: str,
        price_data: Dict[str, pd.DataFrame],
        signal_type: str,
        signal_direction: str,
        related_pairs: Optional[Dict[str, float]] = None,
        use_currency_strength: bool = True,
        min_confirmation_strength: float = 0.3
    ) -> Dict[str, Any]:
        """
        Detect confluence using machine learning models.
        
        Args:
            symbol: Primary currency pair (e.g., "EURUSD")
            price_data: Dictionary mapping currency pairs to price DataFrames
            signal_type: Type of signal ("trend", "reversal", "breakout")
            signal_direction: Direction of the signal ("bullish", "bearish")
            related_pairs: Dictionary mapping related pairs to their correlations
            use_currency_strength: Whether to include currency strength in analysis
            min_confirmation_strength: Minimum strength for confirmation signals
            
        Returns:
            Dictionary with confluence analysis results
        """
        # Check cache
        cache_key = f"confluence_ml_{symbol}_{signal_type}_{signal_direction}"
        cache_hit, cached_result = self.cache_manager.get(cache_key)
        if cache_hit:
            logger.debug(f"Cache hit for confluence detection: {cache_key}")
            return cached_result
        
        # Get price data for the symbol
        symbol_data = price_data.get(symbol)
        if symbol_data is None or symbol_data.empty:
            raise ValueError(f"No price data available for {symbol}")
        
        # Get related pairs if not provided
        if related_pairs is None:
            # This is a synchronous method, so we can't use await directly
            # In a real implementation, you would need to handle this differently
            import asyncio
            related_pairs = asyncio.run(self.find_related_pairs(symbol))
        
        # Detect patterns in the primary symbol
        primary_patterns = self.pattern_model.predict(symbol_data)
        
        # Get pattern probabilities for the signal type
        pattern_probs = {}
        if signal_type == "trend":
            if signal_direction == "bullish":
                pattern_probs = {
                    "flag": primary_patterns.get("flag", [0])[-1],
                    "triangle": primary_patterns.get("triangle", [0])[-1]
                }
            else:  # bearish
                pattern_probs = {
                    "flag": primary_patterns.get("flag", [0])[-1],
                    "triangle": primary_patterns.get("triangle", [0])[-1]
                }
        elif signal_type == "reversal":
            if signal_direction == "bullish":
                pattern_probs = {
                    "double_bottom": primary_patterns.get("double_bottom", [0])[-1],
                    "inverse_head_and_shoulders": primary_patterns.get("inverse_head_and_shoulders", [0])[-1]
                }
            else:  # bearish
                pattern_probs = {
                    "double_top": primary_patterns.get("double_top", [0])[-1],
                    "head_and_shoulders": primary_patterns.get("head_and_shoulders", [0])[-1]
                }
        elif signal_type == "breakout":
            pattern_probs = {
                "rectangle": primary_patterns.get("rectangle", [0])[-1],
                "triangle": primary_patterns.get("triangle", [0])[-1],
                "wedge": primary_patterns.get("wedge", [0])[-1]
            }
        
        # Calculate pattern score
        pattern_score = sum(pattern_probs.values()) / len(pattern_probs) if pattern_probs else 0
        
        # Predict future prices
        price_prediction = self.prediction_model.predict(symbol_data)
        
        # Determine if prediction confirms the signal
        prediction_confirms = False
        if signal_direction == "bullish":
            # Check if predicted prices are higher
            prediction_confirms = price_prediction["predictions"][-1] > price_prediction["predictions"][0]
        else:  # bearish
            # Check if predicted prices are lower
            prediction_confirms = price_prediction["predictions"][-1] < price_prediction["predictions"][0]
        
        # Calculate prediction score
        prediction_score = 0
        if prediction_confirms:
            # Calculate percentage change
            pct_change = abs(price_prediction["predictions"][-1] - price_prediction["predictions"][0]) / price_prediction["predictions"][0]
            # Normalize to 0-1 range (assuming 5% is max expected change)
            prediction_score = min(pct_change / 0.05, 1.0)
        
        # Analyze related pairs
        confirmations = []
        contradictions = []
        neutrals = []
        
        for pair, correlation in related_pairs.items():
            # Skip if price data is not available
            if pair not in price_data or price_data[pair].empty:
                continue
            
            # Detect patterns in the related pair
            related_patterns = self.pattern_model.predict(price_data[pair])
            
            # Predict future prices for the related pair
            related_prediction = self.prediction_model.predict(price_data[pair])
            
            # Determine expected direction based on correlation
            expected_direction = signal_direction
            if correlation < 0:
                expected_direction = "bullish" if signal_direction == "bearish" else "bearish"
            
            # Determine actual direction from prediction
            actual_direction = "bullish" if related_prediction["predictions"][-1] > related_prediction["predictions"][0] else "bearish"
            
            # Calculate signal strength
            pattern_strength = 0
            for pattern, prob in related_patterns.items():
                pattern_strength = max(pattern_strength, prob[-1])
            
            prediction_strength = 0
            pct_change = abs(related_prediction["predictions"][-1] - related_prediction["predictions"][0]) / related_prediction["predictions"][0]
            prediction_strength = min(pct_change / 0.05, 1.0)
            
            signal_strength = (pattern_strength + prediction_strength) / 2
            
            # Determine if the pair confirms or contradicts
            if signal_strength < min_confirmation_strength:
                neutrals.append({
                    "pair": pair,
                    "correlation": correlation,
                    "signal_strength": signal_strength,
                    "expected_direction": expected_direction,
                    "actual_direction": actual_direction,
                    "message": "Signal strength below threshold"
                })
            elif expected_direction == actual_direction:
                confirmations.append({
                    "pair": pair,
                    "correlation": correlation,
                    "signal_strength": signal_strength,
                    "expected_direction": expected_direction,
                    "actual_direction": actual_direction
                })
            else:
                contradictions.append({
                    "pair": pair,
                    "correlation": correlation,
                    "signal_strength": signal_strength,
                    "expected_direction": expected_direction,
                    "actual_direction": actual_direction
                })
        
        # Include currency strength if requested
        currency_strength_score = 0
        if use_currency_strength:
            # Calculate currency strength
            currency_strength = self.currency_strength_analyzer.calculate_currency_strength(price_data)
            
            # Extract base and quote currencies
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            
            # Get strength values
            base_strength = currency_strength.get(base_currency, 0)
            quote_strength = currency_strength.get(quote_currency, 0)
            
            # Calculate relative strength
            relative_strength = base_strength - quote_strength
            
            # Determine if strength confirms the signal
            strength_confirms = False
            if signal_direction == "bullish" and relative_strength > 0:
                strength_confirms = True
            elif signal_direction == "bearish" and relative_strength < 0:
                strength_confirms = True
            
            # Calculate currency strength score
            currency_strength_score = min(abs(relative_strength), 1.0) if strength_confirms else 0
        
        # Calculate overall confluence score
        confirmation_count = len(confirmations)
        contradiction_count = len(contradictions)
        neutral_count = len(neutrals)
        
        # Weights for different components
        pattern_weight = 0.3
        prediction_weight = 0.3
        related_pairs_weight = 0.3
        currency_strength_weight = 0.1
        
        # Calculate related pairs score
        related_pairs_score = 0
        if confirmation_count + contradiction_count > 0:
            related_pairs_score = confirmation_count / (confirmation_count + contradiction_count)
        
        # Calculate overall score
        confluence_score = (
            pattern_score * pattern_weight +
            prediction_score * prediction_weight +
            related_pairs_score * related_pairs_weight +
            currency_strength_score * currency_strength_weight
        )
        
        # Create result
        result = {
            "symbol": symbol,
            "signal_type": signal_type,
            "signal_direction": signal_direction,
            "confirmation_count": confirmation_count,
            "contradiction_count": contradiction_count,
            "neutral_count": neutral_count,
            "confluence_score": confluence_score,
            "pattern_score": pattern_score,
            "prediction_score": prediction_score,
            "related_pairs_score": related_pairs_score,
            "currency_strength_score": currency_strength_score,
            "confirmations": confirmations,
            "contradictions": contradictions,
            "neutrals": neutrals,
            "price_prediction": {
                "values": price_prediction["predictions"],
                "lower_bound": price_prediction["lower_bound"],
                "upper_bound": price_prediction["upper_bound"]
            },
            "patterns": {k: v[-1] for k, v in primary_patterns.items()}
        }
        
        # Cache result
        self.cache_manager.set(cache_key, result)
        
        return result
    
    def analyze_divergence_ml(
        self,
        symbol: str,
        price_data: Dict[str, pd.DataFrame],
        related_pairs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze divergences using machine learning models.
        
        Args:
            symbol: Primary currency pair (e.g., "EURUSD")
            price_data: Dictionary mapping currency pairs to price DataFrames
            related_pairs: Dictionary mapping related pairs to their correlations
            
        Returns:
            Dictionary with divergence analysis results
        """
        # Check cache
        cache_key = f"divergence_ml_{symbol}"
        cache_hit, cached_result = self.cache_manager.get(cache_key)
        if cache_hit:
            logger.debug(f"Cache hit for divergence analysis: {cache_key}")
            return cached_result
        
        # Get price data for the symbol
        symbol_data = price_data.get(symbol)
        if symbol_data is None or symbol_data.empty:
            raise ValueError(f"No price data available for {symbol}")
        
        # Get related pairs if not provided
        if related_pairs is None:
            # This is a synchronous method, so we can't use await directly
            # In a real implementation, you would need to handle this differently
            import asyncio
            related_pairs = asyncio.run(self.find_related_pairs(symbol))
        
        # Predict future prices for the primary symbol
        primary_prediction = self.prediction_model.predict(symbol_data)
        
        # Determine primary momentum
        primary_momentum = primary_prediction["percentage_changes"][-1]
        
        # Analyze divergences
        divergences = []
        
        for pair, correlation in related_pairs.items():
            # Skip if price data is not available
            if pair not in price_data or price_data[pair].empty:
                continue
            
            # Predict future prices for the related pair
            related_prediction = self.prediction_model.predict(price_data[pair])
            
            # Determine related momentum
            related_momentum = related_prediction["percentage_changes"][-1]
            
            # Calculate expected momentum based on correlation
            expected_momentum = primary_momentum * correlation
            
            # Calculate momentum difference
            momentum_difference = related_momentum - expected_momentum
            
            # Determine if there is a divergence
            divergence_threshold = 0.5  # Percentage difference threshold
            if abs(momentum_difference) > divergence_threshold:
                # Determine divergence type
                divergence_type = "positive" if momentum_difference > 0 else "negative"
                
                # Calculate divergence strength
                divergence_strength = min(abs(momentum_difference) / 2.0, 1.0)  # Normalize to 0-1 range
                
                divergences.append({
                    "pair": pair,
                    "correlation": correlation,
                    "primary_momentum": primary_momentum,
                    "related_momentum": related_momentum,
                    "expected_momentum": expected_momentum,
                    "momentum_difference": momentum_difference,
                    "divergence_type": divergence_type,
                    "divergence_strength": divergence_strength
                })
        
        # Calculate overall divergence score
        divergence_score = 0
        if divergences:
            # Average of divergence strengths
            divergence_score = sum(d["divergence_strength"] for d in divergences) / len(divergences)
        
        # Create result
        result = {
            "symbol": symbol,
            "divergences_found": len(divergences),
            "divergence_score": divergence_score,
            "divergences": divergences,
            "price_prediction": {
                "values": primary_prediction["predictions"],
                "lower_bound": primary_prediction["lower_bound"],
                "upper_bound": primary_prediction["upper_bound"]
            }
        }
        
        # Cache result
        self.cache_manager.set(cache_key, result)
        
        return result
    
    def train_models(
        self,
        training_data: Dict[str, pd.DataFrame],
        pattern_labels: Dict[str, List[List[int]]],
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Train the ML models on historical data.
        
        Args:
            training_data: Dictionary mapping symbols to DataFrames with OHLCV data
            pattern_labels: Dictionary mapping symbols to lists of pattern labels
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training ML models on {len(training_data)} symbols")
        
        # Train pattern recognition model
        pattern_results = self.pattern_model.train(
            training_data=training_data,
            labels=pattern_labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Train price prediction model
        prediction_results = self.prediction_model.train(
            training_data=training_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return {
            "pattern_model": pattern_results,
            "prediction_model": prediction_results
        }
    
    def save_models(self, pattern_model_path: str, prediction_model_path: str) -> None:
        """
        Save the ML models to disk.
        
        Args:
            pattern_model_path: Path to save the pattern recognition model
            prediction_model_path: Path to save the price prediction model
        """
        logger.info(f"Saving ML models to {pattern_model_path} and {prediction_model_path}")
        
        # Save pattern recognition model
        self.pattern_model.save_model(pattern_model_path)
        
        # Save price prediction model
        self.prediction_model.save_model(prediction_model_path)
