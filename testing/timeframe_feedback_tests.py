"""
Test suite for the timeframe feedback service.

These tests verify that the TimeframeFeedbackService correctly handles
feedback specific to different trading timeframes, including correlation
analysis and temporal pattern detection.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from core_foundations.models.feedback import (
    TimeframeFeedback, FeedbackPriority, FeedbackCategory,
    FeedbackSource, FeedbackStatus
)
from analysis_engine.services.timeframe_feedback_service import TimeframeFeedbackService


class TestTimeframeFeedbackService(unittest.TestCase):
    """Test cases for the TimeframeFeedbackService."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock dependencies
        self.feedback_repository = Mock()
        
        # Configure mock repository
        self.feedback_repository.store_feedback = Mock(return_value="test-feedback-id")
        
        # Service configuration
        self.config = {
            'correlation_threshold': 0.6,
            'significance_threshold': 0.7,
            'critical_error_threshold': 0.8,
            'high_error_threshold': 0.6,
            'medium_error_threshold': 0.3
        }
        
        # Create the service under test
        self.service = TimeframeFeedbackService(
            feedback_repository=self.feedback_repository,
            config=self.config
        )
    
    def test_submit_timeframe_feedback(self):
        """Test submitting a new timeframe feedback item."""
        # Test data
        model_id = "model-123"
        timeframe = "1h"
        prediction_error = 0.25
        actual_value = 1.25
        predicted_value = 1.00
        prediction_timestamp = datetime.utcnow() - timedelta(hours=1)
        metadata = {"market_conditions": "volatile"}
        
        # Execute the method under test
        feedback_id = self.service.submit_timeframe_feedback(
            model_id=model_id,
            timeframe=timeframe,
            prediction_error=prediction_error,
            actual_value=actual_value,
            predicted_value=predicted_value,
            prediction_timestamp=prediction_timestamp,
            metadata=metadata
        )
        
        # Verify results
        self.assertEqual("test-feedback-id", feedback_id)
        self.feedback_repository.store_feedback.assert_called_once()
        
        # Check that the feedback object was created correctly
        created_feedback = self.feedback_repository.store_feedback.call_args.args[0]
        self.assertIsInstance(created_feedback, TimeframeFeedback)
        self.assertEqual(model_id, created_feedback.model_id)
        self.assertEqual(timeframe, created_feedback.timeframe)
        self.assertEqual(prediction_error, created_feedback.content["prediction_error"])
        self.assertEqual(FeedbackCategory.TIMEFRAME_ADJUSTMENT, created_feedback.category)
    
    def test_timeframe_significance_calculation(self):
        """Test the calculation of significance based on timeframe and error."""
        # Test different timeframes with the same error
        error = 0.2
        
        # Execute calculations for different timeframes
        sig_1m = self.service._calculate_significance(error, "1m")
        sig_1h = self.service._calculate_significance(error, "1h")
        sig_1d = self.service._calculate_significance(error, "1d")
        
        # We expect longer timeframes to have higher significance for the same error
        self.assertLess(sig_1m, sig_1h)
        self.assertLess(sig_1h, sig_1d)
    
    def test_priority_determination(self):
        """Test the determination of feedback priority."""
        # Test cases with error and significance pairs
        test_cases = [
            (0.1, 0.2, FeedbackPriority.LOW),      # Low error, low significance
            (0.3, 0.5, FeedbackPriority.MEDIUM),   # Medium error, medium significance
            (0.5, 0.8, FeedbackPriority.HIGH),     # High error, high significance
            (0.9, 0.9, FeedbackPriority.CRITICAL), # Critical error, high significance
        ]
        
        for error, significance, expected_priority in test_cases:
            with self.subTest(error=error, significance=significance):
                priority = self.service._determine_priority(error, significance)
                self.assertEqual(expected_priority, priority)
    
    def test_correlate_timeframes(self):
        """Test the correlation of different timeframes."""
        # Test data
        model_id = "model-123"
        primary_timeframe = "1h"
        start_time = datetime.utcnow() - timedelta(days=7)
        end_time = datetime.utcnow()
        
        # Mock feedback items for different timeframes
        feedback_items = [
            # 1h timeframe feedback
            TimeframeFeedback(
                feedback_id="tf-1h-1",
                timeframe="1h",
                model_id=model_id,
                content={"prediction_error": 0.1}
            ),
            TimeframeFeedback(
                feedback_id="tf-1h-2",
                timeframe="1h",
                model_id=model_id,
                content={"prediction_error": 0.2}
            ),
            # 4h timeframe feedback (positively correlated)
            TimeframeFeedback(
                feedback_id="tf-4h-1",
                timeframe="4h",
                model_id=model_id,
                content={"prediction_error": 0.15}  # Similar pattern to 1h
            ),
            TimeframeFeedback(
                feedback_id="tf-4h-2",
                timeframe="4h",
                model_id=model_id,
                content={"prediction_error": 0.25}  # Similar pattern to 1h
            ),
            # 1d timeframe feedback (negatively correlated)
            TimeframeFeedback(
                feedback_id="tf-1d-1",
                timeframe="1d",
                model_id=model_id,
                content={"prediction_error": -0.05}  # Opposite pattern to 1h
            ),
            TimeframeFeedback(
                feedback_id="tf-1d-2",
                timeframe="1d",
                model_id=model_id,
                content={"prediction_error": -0.15}  # Opposite pattern to 1h
            ),
        ]
        
        # Mock the repository to return our test items
        with patch.object(self.service, '_fetch_timeframe_feedback', return_value=feedback_items):
            # Execute the method under test
            result = self.service.correlate_timeframes(
                model_id=model_id,
                primary_timeframe=primary_timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            # Verify the correlations are calculated correctly
            self.assertEqual("success", result["status"])
            
            # Check for positive correlation between 1h and 4h
            self.assertGreater(result["correlations"]["1h_4h"], 0)
            self.assertGreater(result["correlations"]["1h_4h"], self.config["correlation_threshold"])
            
            # Check for negative correlation between 1h and 1d
            self.assertLess(result["correlations"]["1h_1d"], 0)
            self.assertLess(result["correlations"]["1h_1d"], -self.config["correlation_threshold"])
            
            # Verify strongly correlated timeframes are identified
            self.assertEqual(2, len(result["strongly_correlated_timeframes"]))
    
    def test_generate_timeframe_adjustment_feedback(self):
        """Test generating adjustment feedback based on correlation analysis."""
        # Create mock correlation analysis results
        correlation_analysis = {
            "status": "success",
            "correlations": {
                "1h_4h": 0.85,  # Strong positive correlation
                "1h_1d": -0.75,  # Strong negative correlation
                "1h_1w": 0.3     # Weak correlation (below threshold)
            },
            "strongly_correlated_timeframes": ["1h_4h", "1h_1d"],
            "analysis_period": {
                "start": "2025-04-10T12:00:00",
                "end": "2025-04-17T12:00:00"
            }
        }
        
        # Execute the method under test
        feedback_id = self.service.generate_timeframe_adjustment_feedback(
            model_id="model-123",
            correlation_analysis=correlation_analysis
        )
        
        # Verify results
        self.assertEqual("test-feedback-id", feedback_id)
        self.feedback_repository.store_feedback.assert_called_once()
        
        # Verify the generated feedback
        generated_feedback = self.feedback_repository.store_feedback.call_args.args[0]
        self.assertIsInstance(generated_feedback, TimeframeFeedback)
        self.assertEqual("multiple", generated_feedback.timeframe)
        self.assertEqual(FeedbackPriority.HIGH, generated_feedback.priority)
        
        # Check recommendations
        adjustments = generated_feedback.content["adjustments"]
        self.assertEqual(2, len(adjustments))
        
        # Check for positive correlation recommendation
        positive_rec = next((adj for adj in adjustments if adj["correlation"] > 0), None)
        self.assertIsNotNone(positive_rec)
        self.assertEqual("1h_4h", positive_rec["timeframe_pair"])
        self.assertIn("transfer learning", positive_rec["recommendation"])
        
        # Check for negative correlation recommendation
        negative_rec = next((adj for adj in adjustments if adj["correlation"] < 0), None)
        self.assertIsNotNone(negative_rec)
        self.assertEqual("1h_1d", negative_rec["timeframe_pair"])
        self.assertIn("inverse relationship", negative_rec["recommendation"])
    
    def test_timeframe_factor_calculation(self):
        """Test the calculation of timeframe factors."""
        # Test various timeframe formats
        test_cases = [
            ("1m", 1/60),       # 1 minute = 1/60 of an hour
            ("30m", 30/60),     # 30 minutes = 0.5 hours
            ("1h", 1.0),        # 1 hour = 1 hour
            ("4h", 4.0),        # 4 hours = 4 hours
            ("1d", 24.0),       # 1 day = 24 hours
            ("1w", 24*7),       # 1 week = 168 hours
        ]
        
        for timeframe, expected_factor in test_cases:
            with self.subTest(timeframe=timeframe):
                factor = self.service._get_timeframe_factor(timeframe)
                self.assertAlmostEqual(expected_factor, factor)


if __name__ == '__main__':
    unittest.main()
