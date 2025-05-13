"""
Market Regime Analyzer

This module provides the main analyzer class for market regime analysis,
coordinating detection and classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
from functools import lru_cache

from analysis_engine.analysis.market_regime.detector import RegimeDetector
from analysis_engine.analysis.market_regime.classifier import RegimeClassifier
from analysis_engine.analysis.market_regime.models import (
    RegimeClassification, RegimeType, DirectionType, VolatilityLevel
)

# Set up logging
logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """
    Main analyzer class for market regime analysis.
    
    This class coordinates the detection and classification of market regimes
    based on price data. It provides a simple interface for analyzing market
    regimes and caches results for performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MarketRegimeAnalyzer.
        
        Args:
            config: Optional configuration dictionary with parameters for
                detector and classifier components.
        """
        self.config = config or {}
        
        # Initialize components
        detector_config = self.config.get('detector', {})
        classifier_config = self.config.get('classifier', {})
        
        self.detector = RegimeDetector(detector_config)
        self.classifier = RegimeClassifier(classifier_config)
        
        # Cache settings
        self.cache_size = self.config.get('cache_size', 128)
        
        # Event subscribers
        self.regime_change_subscribers = []
    
    def analyze(
        self, 
        price_data: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ) -> RegimeClassification:
        """
        Analyze price data to determine the current market regime.
        
        Args:
            price_data: DataFrame with OHLCV data
                Required columns: 'open', 'high', 'low', 'close'
                Optional: 'volume'
            timestamp: Optional timestamp for the analysis
                
        Returns:
            RegimeClassification: Classification result with regime type,
                direction, volatility level, and confidence
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Extract features using detector
        features = self.detector.extract_features(price_data)
        
        # Classify regime using classifier
        classification = self.classifier.classify(features, timestamp)
        
        # Check for regime change and notify subscribers
        self._check_regime_change(classification)
        
        return classification
    
    @lru_cache(maxsize=128)
    def analyze_cached(
        self,
        instrument: str,
        timeframe: str,
        price_data_key: str,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cached version of analyze for repeated calls with the same data.
        
        Args:
            instrument: Instrument symbol
            timeframe: Timeframe string (e.g., 'H1', 'D1')
            price_data_key: A unique key representing the price data
                This could be a hash of the data or a timestamp range
            timestamp: Optional timestamp string for the analysis
                
        Returns:
            Dict: Classification result as a dictionary
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, you would retrieve the price data
        # based on the instrument, timeframe, and key
        
        # For demonstration purposes, we'll just return a dummy result
        return {
            'regime': RegimeType.TRENDING_BULLISH.name,
            'confidence': 0.85,
            'direction': DirectionType.BULLISH.name,
            'volatility': VolatilityLevel.MEDIUM.name,
            'timestamp': timestamp or datetime.now().isoformat(),
            'features': {
                'volatility': 0.75,
                'trend_strength': 0.65,
                'momentum': 0.45,
                'mean_reversion': -0.2,
                'range_width': 0.03
            }
        }
    
    def subscribe_to_regime_changes(self, callback: callable) -> None:
        """
        Subscribe to regime change events.
        
        Args:
            callback: Function to call when regime changes
                The callback will receive the new classification and the old one
        """
        if callback not in self.regime_change_subscribers:
            self.regime_change_subscribers.append(callback)
    
    def unsubscribe_from_regime_changes(self, callback: callable) -> None:
        """
        Unsubscribe from regime change events.
        
        Args:
            callback: Function to remove from subscribers
        """
        if callback in self.regime_change_subscribers:
            self.regime_change_subscribers.remove(callback)
    
    def _check_regime_change(self, new_classification: RegimeClassification) -> None:
        """
        Check if regime has changed and notify subscribers.
        
        Args:
            new_classification: New regime classification
        """
        # Get previous classification from classifier
        previous = self.classifier.previous_classification
        
        # If no previous classification or regime has changed
        if (previous is None or 
            previous.regime != new_classification.regime or
            previous.volatility != new_classification.volatility):
            
            # Notify subscribers
            for subscriber in self.regime_change_subscribers:
                try:
                    subscriber(new_classification, previous)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")
    
    def get_historical_regimes(
        self,
        price_data: pd.DataFrame,
        window_size: int = 1
    ) -> List[RegimeClassification]:
        """
        Analyze historical price data to determine regime changes over time.
        
        Args:
            price_data: DataFrame with OHLCV data
            window_size: Size of the rolling window for analysis
                
        Returns:
            List[RegimeClassification]: List of regime classifications over time
        """
        results = []
        
        # Reset classifier state to avoid hysteresis from previous analyses
        self.classifier.previous_classification = None
        
        # Analyze each window
        for i in range(window_size, len(price_data) + 1):
            window = price_data.iloc[i - window_size:i]
            timestamp = window.index[-1]
            
            # Extract features and classify
            features = self.detector.extract_features(window)
            classification = self.classifier.classify(features, timestamp)
            
            results.append(classification)
        
        return results