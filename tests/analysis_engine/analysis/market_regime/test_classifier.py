"""
Unit tests for the Market Regime Classifier.
"""

import unittest
from datetime import datetime

from analysis_engine.analysis.market_regime.classifier import RegimeClassifier
from analysis_engine.analysis.market_regime.models import (
    FeatureSet, RegimeType, DirectionType, VolatilityLevel
)


class TestRegimeClassifier(unittest.TestCase):
    """Test cases for the RegimeClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = RegimeClassifier()
        self.timestamp = datetime.now()
    
    def test_trending_bullish_classification(self):
        """Test classification of a trending bullish market."""
        # Create a feature set for a trending bullish market
        features = FeatureSet(
            volatility=0.8,
            trend_strength=0.7,
            momentum=0.5,
            mean_reversion=-0.1,
            range_width=0.02,
            additional_features={
                'price_velocity': 0.02,
                'volume_trend': 0.3,
                'swing_strength': 0.01
            }
        )
        
        # Classify the regime
        classification = self.classifier.classify(features, self.timestamp)
        
        # Check the classification
        self.assertEqual(classification.regime, RegimeType.TRENDING_BULLISH)
        self.assertEqual(classification.direction, DirectionType.BULLISH)
        self.assertGreater(classification.confidence, 0.7)
    
    def test_trending_bearish_classification(self):
        """Test classification of a trending bearish market."""
        # Create a feature set for a trending bearish market
        features = FeatureSet(
            volatility=0.8,
            trend_strength=0.7,
            momentum=-0.5,
            mean_reversion=0.1,
            range_width=0.02,
            additional_features={
                'price_velocity': -0.02,
                'volume_trend': -0.3,
                'swing_strength': 0.01
            }
        )
        
        # Classify the regime
        classification = self.classifier.classify(features, self.timestamp)
        
        # Check the classification
        self.assertEqual(classification.regime, RegimeType.TRENDING_BEARISH)
        self.assertEqual(classification.direction, DirectionType.BEARISH)
        self.assertGreater(classification.confidence, 0.7)
    
    def test_ranging_neutral_classification(self):
        """Test classification of a ranging neutral market."""
        # Create a feature set for a ranging neutral market
        features = FeatureSet(
            volatility=0.5,
            trend_strength=0.2,
            momentum=0.1,
            mean_reversion=0.3,
            range_width=0.03,
            additional_features={
                'price_velocity': 0.001,
                'volume_trend': 0.1,
                'swing_strength': 0.02
            }
        )
        
        # Classify the regime
        classification = self.classifier.classify(features, self.timestamp)
        
        # Check the classification
        self.assertEqual(classification.regime, RegimeType.RANGING_NEUTRAL)
        self.assertEqual(classification.direction, DirectionType.NEUTRAL)
    
    def test_ranging_bullish_classification(self):
        """Test classification of a ranging bullish market."""
        # Create a feature set for a ranging bullish market
        features = FeatureSet(
            volatility=0.5,
            trend_strength=0.2,
            momentum=0.3,
            mean_reversion=0.2,
            range_width=0.03,
            additional_features={
                'price_velocity': 0.01,
                'volume_trend': 0.2,
                'swing_strength': 0.02
            }
        )
        
        # Classify the regime
        classification = self.classifier.classify(features, self.timestamp)
        
        # Check the classification
        self.assertEqual(classification.regime, RegimeType.RANGING_BULLISH)
        self.assertEqual(classification.direction, DirectionType.BULLISH)
    
    def test_ranging_bearish_classification(self):
        """Test classification of a ranging bearish market."""
        # Create a feature set for a ranging bearish market
        features = FeatureSet(
            volatility=0.5,
            trend_strength=0.2,
            momentum=-0.3,
            mean_reversion=-0.2,
            range_width=0.03,
            additional_features={
                'price_velocity': -0.01,
                'volume_trend': -0.2,
                'swing_strength': 0.02
            }
        )
        
        # Classify the regime
        classification = self.classifier.classify(features, self.timestamp)
        
        # Check the classification
        self.assertEqual(classification.regime, RegimeType.RANGING_BEARISH)
        self.assertEqual(classification.direction, DirectionType.BEARISH)
    
    def test_volatile_classification(self):
        """Test classification of a volatile market."""
        # Create a feature set for a volatile market
        features = FeatureSet(
            volatility=2.5,
            trend_strength=0.4,
            momentum=0.1,
            mean_reversion=0.1,
            range_width=0.05,
            additional_features={
                'price_velocity': 0.03,
                'volume_trend': 0.5,
                'swing_strength': 0.04
            }
        )
        
        # Classify the regime
        classification = self.classifier.classify(features, self.timestamp)
        
        # Check the classification
        self.assertIn(classification.regime, [
            RegimeType.VOLATILE_BULLISH,
            RegimeType.VOLATILE_BEARISH,
            RegimeType.VOLATILE_NEUTRAL
        ])
        self.assertEqual(classification.volatility, VolatilityLevel.EXTREME)
        self.assertGreater(classification.confidence, 0.8)
    
    def test_direction_classification(self):
        """Test the direction classification logic."""
        # Test bullish direction
        self.assertEqual(
            self.classifier._classify_direction(0.3),
            DirectionType.BULLISH
        )
        
        # Test bearish direction
        self.assertEqual(
            self.classifier._classify_direction(-0.3),
            DirectionType.BEARISH
        )
        
        # Test neutral direction
        self.assertEqual(
            self.classifier._classify_direction(0.1),
            DirectionType.NEUTRAL
        )
    
    def test_volatility_classification(self):
        """Test the volatility classification logic."""
        # Test low volatility
        self.assertEqual(
            self.classifier._classify_volatility(0.3),
            VolatilityLevel.LOW
        )
        
        # Test medium volatility
        self.assertEqual(
            self.classifier._classify_volatility(0.7),
            VolatilityLevel.MEDIUM
        )
        
        # Test high volatility
        self.assertEqual(
            self.classifier._classify_volatility(1.5),
            VolatilityLevel.HIGH
        )
        
        # Test extreme volatility
        self.assertEqual(
            self.classifier._classify_volatility(2.5),
            VolatilityLevel.EXTREME
        )
    
    def test_hysteresis(self):
        """Test that hysteresis prevents rapid regime switching."""
        # Create a feature set near the boundary between trending and ranging
        features1 = FeatureSet(
            volatility=0.7,
            trend_strength=0.26,  # Just above the threshold
            momentum=0.3,
            mean_reversion=0.1,
            range_width=0.02
        )
        
        # Classify the first regime
        classification1 = self.classifier.classify(features1, self.timestamp)
        
        # Create a similar feature set with slightly lower trend strength
        features2 = FeatureSet(
            volatility=0.7,
            trend_strength=0.24,  # Just below the threshold
            momentum=0.3,
            mean_reversion=0.1,
            range_width=0.02
        )
        
        # Classify the second regime
        classification2 = self.classifier.classify(features2, self.timestamp)
        
        # Due to hysteresis, the regime should not change
        self.assertEqual(classification1.regime, classification2.regime)
        
        # Now create a feature set with significantly lower trend strength
        features3 = FeatureSet(
            volatility=0.7,
            trend_strength=0.15,  # Well below the threshold
            momentum=0.3,
            mean_reversion=0.1,
            range_width=0.02
        )
        
        # Classify the third regime
        classification3 = self.classifier.classify(features3, self.timestamp)
        
        # Now the regime should change
        self.assertNotEqual(classification1.regime, classification3.regime)
    
    def test_to_dict_and_from_dict(self):
        """Test the to_dict and from_dict methods of RegimeClassification."""
        # Create a feature set
        features = FeatureSet(
            volatility=0.8,
            trend_strength=0.7,
            momentum=0.5,
            mean_reversion=-0.1,
            range_width=0.02
        )
        
        # Classify the regime
        classification = self.classifier.classify(features, self.timestamp)
        
        # Convert to dict
        classification_dict = classification.to_dict()
        
        # Check dict contents
        self.assertIn('regime', classification_dict)
        self.assertIn('confidence', classification_dict)
        self.assertIn('direction', classification_dict)
        self.assertIn('volatility', classification_dict)
        self.assertIn('timestamp', classification_dict)
        self.assertIn('features', classification_dict)
        
        # Convert back from dict
        from analysis_engine.analysis.market_regime.models import RegimeClassification
        reconstructed = RegimeClassification.from_dict(classification_dict)
        
        # Check that the reconstructed object matches the original
        self.assertEqual(reconstructed.regime, classification.regime)
        self.assertEqual(reconstructed.confidence, classification.confidence)
        self.assertEqual(reconstructed.direction, classification.direction)
        self.assertEqual(reconstructed.volatility, classification.volatility)


if __name__ == '__main__':
    unittest.main()