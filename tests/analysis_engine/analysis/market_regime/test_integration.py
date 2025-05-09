"""
Integration tests for the Market Regime Analysis component.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from analysis_engine.analysis.market_regime.analyzer import MarketRegimeAnalyzer
from analysis_engine.analysis.market_regime.models import (
    RegimeType, DirectionType, VolatilityLevel
)


class TestMarketRegimeIntegration(unittest.TestCase):
    """Integration tests for the Market Regime Analysis component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MarketRegimeAnalyzer()
        self.trending_data = self._create_trending_data()
        self.ranging_data = self._create_ranging_data()
        self.volatile_data = self._create_volatile_data()
    
    def _create_trending_data(self):
        """Create a sample dataset with a strong trend."""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create price data with a strong uptrend
        close = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
        high = close + np.random.uniform(0, 3, 100)
        low = close - np.random.uniform(0, 3, 100)
        open_price = close - np.random.uniform(-2, 2, 100)
        volume = np.random.uniform(1000, 5000, 100) * (1 + np.linspace(0, 0.5, 100))
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return df
    
    def _create_ranging_data(self):
        """Create a sample dataset with a ranging market."""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create price data with a range
        base = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 10
        close = base + np.random.normal(0, 1, 100)
        high = close + np.random.uniform(0, 2, 100)
        low = close - np.random.uniform(0, 2, 100)
        open_price = close - np.random.uniform(-2, 2, 100)
        volume = np.random.uniform(1000, 3000, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return df
    
    def _create_volatile_data(self):
        """Create a sample dataset with high volatility."""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create price data with high volatility
        close = 100 + np.cumsum(np.random.normal(0, 3, 100))
        high = close + np.random.uniform(1, 5, 100)
        low = close - np.random.uniform(1, 5, 100)
        open_price = close - np.random.uniform(-3, 3, 100)
        volume = np.random.uniform(3000, 10000, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return df
    
    def test_trending_market_analysis(self):
        """Test analysis of a trending market."""
        # Analyze the trending data
        result = self.analyzer.analyze(self.trending_data)
        
        # Check that the result is a trending regime
        self.assertIn(result.regime, [
            RegimeType.TRENDING_BULLISH,
            RegimeType.TRENDING_BEARISH
        ])
        
        # Check confidence
        self.assertGreater(result.confidence, 0.7)
    
    def test_ranging_market_analysis(self):
        """Test analysis of a ranging market."""
        # Analyze the ranging data
        result = self.analyzer.analyze(self.ranging_data)
        
        # Check that the result is a ranging regime
        self.assertIn(result.regime, [
            RegimeType.RANGING_NEUTRAL,
            RegimeType.RANGING_BULLISH,
            RegimeType.RANGING_BEARISH
        ])
    
    def test_volatile_market_analysis(self):
        """Test analysis of a volatile market."""
        # Analyze the volatile data
        result = self.analyzer.analyze(self.volatile_data)
        
        # Check volatility level
        self.assertIn(result.volatility, [
            VolatilityLevel.HIGH,
            VolatilityLevel.EXTREME
        ])
    
    def test_regime_transitions(self):
        """Test detection of regime transitions."""
        # Analyze trending data
        trending_result = self.analyzer.analyze(self.trending_data)
        
        # Analyze ranging data
        ranging_result = self.analyzer.analyze(self.ranging_data)
        
        # Regimes should be different
        self.assertNotEqual(trending_result.regime, ranging_result.regime)
        
        # Analyze volatile data
        volatile_result = self.analyzer.analyze(self.volatile_data)
        
        # Volatility should be different
        self.assertNotEqual(trending_result.volatility, volatile_result.volatility)
    
    def test_historical_regime_analysis(self):
        """Test analysis of historical regimes."""
        # Combine different market types
        combined_data = pd.concat([
            self.trending_data.iloc[:30],
            self.ranging_data.iloc[:30],
            self.volatile_data.iloc[:30]
        ])
        
        # Get historical regimes
        results = self.analyzer.get_historical_regimes(
            combined_data,
            window_size=20
        )
        
        # Check that we get different regimes
        regimes = [r.regime for r in results]
        self.assertTrue(len(set(regimes)) > 1)
    
    def test_full_pipeline(self):
        """Test the full analysis pipeline with different market types."""
        # Test with trending data
        trending_result = self.analyzer.analyze(self.trending_data)
        self.assertIsNotNone(trending_result.regime)
        self.assertIsNotNone(trending_result.confidence)
        self.assertIsNotNone(trending_result.direction)
        self.assertIsNotNone(trending_result.volatility)
        
        # Test with ranging data
        ranging_result = self.analyzer.analyze(self.ranging_data)
        self.assertIsNotNone(ranging_result.regime)
        self.assertIsNotNone(ranging_result.confidence)
        self.assertIsNotNone(ranging_result.direction)
        self.assertIsNotNone(ranging_result.volatility)
        
        # Test with volatile data
        volatile_result = self.analyzer.analyze(self.volatile_data)
        self.assertIsNotNone(volatile_result.regime)
        self.assertIsNotNone(volatile_result.confidence)
        self.assertIsNotNone(volatile_result.direction)
        self.assertIsNotNone(volatile_result.volatility)
    
    def test_event_publication(self):
        """Test that regime changes trigger events."""
        # Create a counter for regime changes
        regime_changes = []
        
        # Subscribe to regime changes
        def on_regime_change(new_classification, old_classification):
            regime_changes.append((new_classification, old_classification))
        
        self.analyzer.subscribe_to_regime_changes(on_regime_change)
        
        # Analyze different market types in sequence
        self.analyzer.analyze(self.trending_data)
        self.analyzer.analyze(self.ranging_data)
        self.analyzer.analyze(self.volatile_data)
        
        # Check that we got regime change events
        self.assertGreater(len(regime_changes), 0)
        
        # Check that the events contain the correct data
        for new, old in regime_changes:
            self.assertIsNotNone(new.regime)
            if old is not None:
                self.assertIsNotNone(old.regime)
                self.assertNotEqual(new.regime, old.regime)


if __name__ == '__main__':
    unittest.main()