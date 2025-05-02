"""
Tests for the Reliability Manager.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timedelta

from feature_store_service.reliability.reliability_manager import ReliabilityManager

class TestReliabilityManager(unittest.TestCase):
    """Test suite for the ReliabilityManager"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test config and state files
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / 'test_reliability.json'
        
        # Create test configuration
        test_config = {
            "verification": {
                "input_validation": {
                    "required_columns": {
                        "price_data": ["timestamp", "close", "volume"]
                    }
                },
                "risk_limits": {
                    "max_position_size": 1000,
                    "max_leverage": 5
                }
            },
            "signal_filtering": {
                "price": {
                    "outlier_std_threshold": 2.0,
                    "window_size": 3
                }
            },
            "recovery": {
                "storage": {
                    "state_dir": str(Path(self.temp_dir) / "states")
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
            
        self.reliability_manager = ReliabilityManager(str(self.config_path))
        
        # Create test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-01-01', periods=5),
            'close': [100.0, 101.0, 102.0, 101.5, 102.5],
            'volume': [1000, 1100, 1200, 1150, 1250]
        })

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)

    def test_input_verification(self):
        """Test input data verification."""
        # Test valid data
        self.assertTrue(
            self.reliability_manager.verify_input_data(
                self.test_data,
                'price_data'
            )
        )
        
        # Test invalid data (missing column)
        invalid_data = self.test_data.drop(columns=['volume'])
        self.assertFalse(
            self.reliability_manager.verify_input_data(
                invalid_data,
                'price_data'
            )
        )

    def test_risk_compliance(self):
        """Test risk compliance verification."""
        # Test compliant risk metrics
        risk_metrics = {
            'position_size': 500,
            'leverage': 2
        }
        self.assertTrue(
            self.reliability_manager.verify_risk_compliance(risk_metrics)
        )
        
        # Test non-compliant risk metrics
        risk_metrics = {
            'position_size': 1500,  # Exceeds max_position_size
            'leverage': 6  # Exceeds max_leverage
        }
        self.assertFalse(
            self.reliability_manager.verify_risk_compliance(risk_metrics)
        )

    def test_decision_verification(self):
        """Test decision verification."""
        context = {
            'market_condition': 'bullish',
            'volatility': 0.15
        }
        
        historical_decisions = [
            {
                'decision': 'buy',
                'context': {
                    'market_condition': 'bullish',
                    'volatility': 0.14
                },
                'timestamp': datetime.utcnow() - timedelta(hours=1)
            }
        ]
        
        # Test consistent decision
        self.assertTrue(
            self.reliability_manager.verify_decision(
                'buy',
                context,
                historical_decisions
            )
        )
        
        # Test inconsistent decision
        self.assertFalse(
            self.reliability_manager.verify_decision(
                'sell',  # Opposite of historical decision in similar context
                context,
                historical_decisions
            )
        )

    def test_signal_filtering(self):
        """Test signal filtering."""
        # Test price signal filtering
        price_data = pd.Series([100.0, 101.0, 150.0, 102.0])  # 150.0 is an outlier
        filtered_price = self.reliability_manager.filter_signal(
            'PRICE',
            price_data,
            {'volatility': 0.1}
        )
        self.assertIsInstance(filtered_price, pd.Series)
        self.assertNotEqual(filtered_price[2], 150.0)
        
        # Test volume signal filtering
        volume_data = pd.Series([1000, 1100, 5000, 1200])  # 5000 is an outlier
        filtered_volume = self.reliability_manager.filter_signal(
            'VOLUME',
            volume_data
        )
        self.assertIsInstance(filtered_volume, pd.Series)
        self.assertNotEqual(filtered_volume[2], 5000)

    def test_error_handling(self):
        """Test error handling and recovery."""
        # Test handling of a state corruption error
        component = "data_pipeline"
        error = Exception("State corruption detected")
        context = {'severity': 'high'}
        
        success = self.reliability_manager.handle_error(
            component=component,
            error=error,
            context=context
        )
        self.assertTrue(success)

    def test_system_health(self):
        """Test system health monitoring."""
        # Generate some activity to have data for health check
        self.reliability_manager.verify_input_data(self.test_data, 'price_data')
        self.reliability_manager.filter_signal('PRICE', pd.Series([100.0, 101.0, 102.0]))
        
        health = self.reliability_manager.get_system_health()
        
        self.assertIn('verification', health)
        self.assertIn('signal_filtering', health)
        self.assertIn('recovery', health)
        self.assertIn('overall_status', health)
        self.assertIn('timestamp', health)
        
        # Check overall status is one of the expected values
        self.assertIn(health['overall_status'], ['HEALTHY', 'WARNING', 'CRITICAL'])

if __name__ == '__main__':
    unittest.main()
