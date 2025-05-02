"""
Tests for the Multi-Level Verification System.
"""
import unittest
from datetime import datetime
import pandas as pd
import numpy as np
from feature_store_service.verification.multi_level_verifier import (
    MultiLevelVerifier,
    VerificationLevel,
    VerificationResult
)

class TestMultiLevelVerifier(unittest.TestCase):
    """Test suite for the MultiLevelVerifier"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.verifier = MultiLevelVerifier()
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-01-01', periods=5),
            'close': [100.0, 101.0, 102.0, 101.5, 102.5],
            'volume': [1000, 1100, 1200, 1150, 1250]
        })

    def test_input_data_verification(self):
        """Test input data verification."""
        # Test valid data
        result = self.verifier.verify(
            VerificationLevel.INPUT,
            self.test_data,
            required_columns=['timestamp', 'close', 'volume']
        )
        self.assertTrue(result.is_valid)
        
        # Test invalid data (missing columns)
        invalid_data = self.test_data.drop(columns=['volume'])
        result = self.verifier.verify(
            VerificationLevel.INPUT,
            invalid_data,
            required_columns=['timestamp', 'close', 'volume']
        )
        self.assertFalse(result.is_valid)
        self.assertIn('Missing required columns', result.message)

    def test_risk_limit_verification(self):
        """Test risk limit verification."""
        risk_data = {
            'exposure': 50000,
            'leverage': 10,
            'drawdown': 0.05
        }
        risk_limits = {
            'exposure': 100000,
            'leverage': 20,
            'drawdown': 0.10
        }
        
        # Test within limits
        result = self.verifier.verify(
            VerificationLevel.RISK,
            risk_data,
            risk_limits=risk_limits
        )
        self.assertTrue(result.is_valid)
        
        # Test exceeding limits
        risk_data['exposure'] = 150000
        result = self.verifier.verify(
            VerificationLevel.RISK,
            risk_data,
            risk_limits=risk_limits
        )
        self.assertFalse(result.is_valid)
        self.assertIn('Risk limit violations', result.message)

    def test_decision_consistency(self):
        """Test decision consistency verification."""
        decision_data = {
            'decision': 'buy',
            'context': {
                'market_condition': 'bullish',
                'volatility': 0.15,
                'trend': 0.8
            }
        }
        
        historical_decisions = [
            {
                'decision': 'buy',
                'context': {
                    'market_condition': 'bullish',
                    'volatility': 0.14,
                    'trend': 0.75
                },
                'timestamp': datetime.utcnow()
            }
        ]
        
        # Test consistent decision
        result = self.verifier.verify(
            VerificationLevel.DECISION,
            decision_data,
            historical_decisions=historical_decisions
        )
        self.assertTrue(result.is_valid)
        
        # Test inconsistent decision
        decision_data['decision'] = 'sell'
        result = self.verifier.verify(
            VerificationLevel.DECISION,
            decision_data,
            historical_decisions=historical_decisions
        )
        self.assertFalse(result.is_valid)
        self.assertIn('Decision consistency', result.message)

    def test_system_state_verification(self):
        """Test system state verification."""
        system_state = {
            'data_pipeline': 'running',
            'cache': 'active',
            'database': 'connected'
        }
        
        # Test valid state
        result = self.verifier.verify(
            VerificationLevel.SYSTEM,
            system_state,
            required_states=['data_pipeline', 'cache', 'database']
        )
        self.assertTrue(result.is_valid)
        
        # Test invalid state
        del system_state['database']
        result = self.verifier.verify(
            VerificationLevel.SYSTEM,
            system_state,
            required_states=['data_pipeline', 'cache', 'database']
        )
        self.assertFalse(result.is_valid)
        self.assertIn('Missing required system states', result.message)

    def test_cross_component_verification(self):
        """Test cross-component verification."""
        component_states = {
            'data_pipeline': {
                'status': 'running',
                'last_update': datetime.utcnow()
            },
            'feature_store': {
                'status': 'active',
                'cache_state': 'valid'
            },
            'trading_engine': {
                'status': 'running',
                'position_state': 'balanced'
            }
        }
        
        # Test valid cross-component state
        result = self.verifier.verify(
            VerificationLevel.CROSS_COMPONENT,
            component_states,
            components=['data_pipeline', 'feature_store', 'trading_engine']
        )
        self.assertTrue(result.is_valid)
        
        # Test invalid cross-component state
        component_states['feature_store'] = {'status': 'active'}  # Missing cache_state
        result = self.verifier.verify(
            VerificationLevel.CROSS_COMPONENT,
            component_states,
            components=['data_pipeline', 'feature_store', 'trading_engine']
        )
        self.assertFalse(result.is_valid)
        self.assertIn('Cross-component inconsistencies', result.message)

    def test_verification_summary(self):
        """Test verification summary generation."""
        # Perform multiple verifications
        self.verifier.verify(
            VerificationLevel.INPUT,
            self.test_data,
            required_columns=['timestamp', 'close']
        )
        self.verifier.verify(
            VerificationLevel.SYSTEM,
            {'state': 'running'},
            required_states=['state']
        )
        
        summary = self.verifier.get_verification_summary()
        self.assertIn('total_verifications', summary)
        self.assertIn('success_rate', summary)
        self.assertIn('by_level', summary)
        self.assertEqual(summary['total_verifications'], 2)
