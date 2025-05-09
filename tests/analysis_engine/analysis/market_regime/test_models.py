"""
Unit tests for the Market Regime Analysis models.
"""

import unittest
from datetime import datetime

from analysis_engine.analysis.market_regime.models import (
    RegimeType, DirectionType, VolatilityLevel,
    RegimeClassification, FeatureSet
)


class TestRegimeModels(unittest.TestCase):
    """Test cases for the market regime models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.timestamp = datetime.now()
        self.features = {
            'volatility': 0.8,
            'trend_strength': 0.7,
            'momentum': 0.5,
            'mean_reversion': -0.1,
            'range_width': 0.02,
            'price_velocity': 0.02,
            'volume_trend': 0.3,
            'swing_strength': 0.01
        }
        
        self.classification = RegimeClassification(
            regime=RegimeType.TRENDING_BULLISH,
            confidence=0.85,
            direction=DirectionType.BULLISH,
            volatility=VolatilityLevel.MEDIUM,
            timestamp=self.timestamp,
            features=self.features,
            metadata={'source': 'test'}
        )
        
        self.feature_set = FeatureSet(
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
    
    def test_regime_type_enum(self):
        """Test the RegimeType enum."""
        # Check that all expected regime types are defined
        self.assertIn(RegimeType.TRENDING_BULLISH, RegimeType)
        self.assertIn(RegimeType.TRENDING_BEARISH, RegimeType)
        self.assertIn(RegimeType.RANGING_NEUTRAL, RegimeType)
        self.assertIn(RegimeType.RANGING_BULLISH, RegimeType)
        self.assertIn(RegimeType.RANGING_BEARISH, RegimeType)
        self.assertIn(RegimeType.VOLATILE_BULLISH, RegimeType)
        self.assertIn(RegimeType.VOLATILE_BEARISH, RegimeType)
        self.assertIn(RegimeType.VOLATILE_NEUTRAL, RegimeType)
        self.assertIn(RegimeType.UNDEFINED, RegimeType)
    
    def test_direction_type_enum(self):
        """Test the DirectionType enum."""
        # Check that all expected direction types are defined
        self.assertIn(DirectionType.BULLISH, DirectionType)
        self.assertIn(DirectionType.BEARISH, DirectionType)
        self.assertIn(DirectionType.NEUTRAL, DirectionType)
    
    def test_volatility_level_enum(self):
        """Test the VolatilityLevel enum."""
        # Check that all expected volatility levels are defined
        self.assertIn(VolatilityLevel.LOW, VolatilityLevel)
        self.assertIn(VolatilityLevel.MEDIUM, VolatilityLevel)
        self.assertIn(VolatilityLevel.HIGH, VolatilityLevel)
        self.assertIn(VolatilityLevel.EXTREME, VolatilityLevel)
    
    def test_regime_classification_to_dict(self):
        """Test the to_dict method of RegimeClassification."""
        # Convert to dict
        classification_dict = self.classification.to_dict()
        
        # Check dict contents
        self.assertEqual(classification_dict['regime'], 'TRENDING_BULLISH')
        self.assertEqual(classification_dict['confidence'], 0.85)
        self.assertEqual(classification_dict['direction'], 'BULLISH')
        self.assertEqual(classification_dict['volatility'], 'MEDIUM')
        self.assertEqual(classification_dict['timestamp'], self.timestamp.isoformat())
        self.assertEqual(classification_dict['features'], self.features)
        self.assertEqual(classification_dict['metadata'], {'source': 'test'})
    
    def test_regime_classification_from_dict(self):
        """Test the from_dict method of RegimeClassification."""
        # Create a dict
        classification_dict = {
            'regime': 'TRENDING_BULLISH',
            'confidence': 0.85,
            'direction': 'BULLISH',
            'volatility': 'MEDIUM',
            'timestamp': self.timestamp.isoformat(),
            'features': self.features,
            'metadata': {'source': 'test'}
        }
        
        # Convert from dict
        classification = RegimeClassification.from_dict(classification_dict)
        
        # Check object attributes
        self.assertEqual(classification.regime, RegimeType.TRENDING_BULLISH)
        self.assertEqual(classification.confidence, 0.85)
        self.assertEqual(classification.direction, DirectionType.BULLISH)
        self.assertEqual(classification.volatility, VolatilityLevel.MEDIUM)
        self.assertEqual(classification.timestamp.isoformat(), self.timestamp.isoformat())
        self.assertEqual(classification.features, self.features)
        self.assertEqual(classification.metadata, {'source': 'test'})
    
    def test_feature_set_to_dict(self):
        """Test the to_dict method of FeatureSet."""
        # Convert to dict
        feature_dict = self.feature_set.to_dict()
        
        # Check dict contents
        self.assertEqual(feature_dict['volatility'], 0.8)
        self.assertEqual(feature_dict['trend_strength'], 0.7)
        self.assertEqual(feature_dict['momentum'], 0.5)
        self.assertEqual(feature_dict['mean_reversion'], -0.1)
        self.assertEqual(feature_dict['range_width'], 0.02)
        self.assertEqual(feature_dict['price_velocity'], 0.02)
        self.assertEqual(feature_dict['volume_trend'], 0.3)
        self.assertEqual(feature_dict['swing_strength'], 0.01)
    
    def test_feature_set_from_dict(self):
        """Test the from_dict method of FeatureSet."""
        # Create a dict
        feature_dict = {
            'volatility': 0.8,
            'trend_strength': 0.7,
            'momentum': 0.5,
            'mean_reversion': -0.1,
            'range_width': 0.02,
            'price_velocity': 0.02,
            'volume_trend': 0.3,
            'swing_strength': 0.01
        }
        
        # Convert from dict
        feature_set = FeatureSet.from_dict(feature_dict)
        
        # Check object attributes
        self.assertEqual(feature_set.volatility, 0.8)
        self.assertEqual(feature_set.trend_strength, 0.7)
        self.assertEqual(feature_set.momentum, 0.5)
        self.assertEqual(feature_set.mean_reversion, -0.1)
        self.assertEqual(feature_set.range_width, 0.02)
        self.assertEqual(feature_set.additional_features['price_velocity'], 0.02)
        self.assertEqual(feature_set.additional_features['volume_trend'], 0.3)
        self.assertEqual(feature_set.additional_features['swing_strength'], 0.01)
    
    def test_feature_set_without_additional_features(self):
        """Test FeatureSet without additional features."""
        # Create a feature set without additional features
        feature_set = FeatureSet(
            volatility=0.8,
            trend_strength=0.7,
            momentum=0.5,
            mean_reversion=-0.1,
            range_width=0.02
        )
        
        # Convert to dict
        feature_dict = feature_set.to_dict()
        
        # Check dict contents
        self.assertEqual(feature_dict['volatility'], 0.8)
        self.assertEqual(feature_dict['trend_strength'], 0.7)
        self.assertEqual(feature_dict['momentum'], 0.5)
        self.assertEqual(feature_dict['mean_reversion'], -0.1)
        self.assertEqual(feature_dict['range_width'], 0.02)
        self.assertNotIn('price_velocity', feature_dict)
        self.assertNotIn('volume_trend', feature_dict)
        self.assertNotIn('swing_strength', feature_dict)


if __name__ == '__main__':
    unittest.main()