import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from analysis_engine.analysis.sequence_pattern_recognizer import SequencePatternRecognizer, PatternType
from analysis_engine.analysis.timeframe_level import TimeframeLevel

class TestSequencePatternRecognizer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock model retraining service
        self.model_retraining_service = MagicMock()

        # Create timeframe mapping
        self.timeframe_mapping = {
            "1H": TimeframeLevel.HOURLY,
            "4H": TimeframeLevel.INTRADAY,
            "1D": TimeframeLevel.DAILY
        }

        # Create the recognizer with ML validation enabled
        self.recognizer = SequencePatternRecognizer(
            min_pattern_quality=0.7,
            use_ml_validation=True,
            pattern_types=[PatternType.REVERSAL, PatternType.CONTINUATION],
            timeframe_mapping=self.timeframe_mapping,
            model_retraining_service=self.model_retraining_service
        )

        # Example patterns (replace with actual pattern definitions)
        self.recognizer.patterns = {
            'head_shoulders': lambda data: self._detect_head_shoulders(data),
            'double_top': lambda data: self._detect_double_top(data)
        }

    def _detect_head_shoulders(self, data):
        # Dummy detection logic for testing
        if len(data) > 5 and data['close'].iloc[-3] > data['close'].iloc[-1] and data['close'].iloc[-3] > data['close'].iloc[-5]:
             return {'name': 'head_shoulders', 'confidence': 0.7, 'index': data.index[-1]}
        return None

    def _detect_double_top(self, data):
        # Dummy detection logic for testing
        if len(data) > 4 and abs(data['high'].iloc[-1] - data['high'].iloc[-3]) < 0.001:
             return {'name': 'double_top', 'confidence': 0.6, 'index': data.index[-1]}
        return None

    def test_initialization(self):
        """Test service initialization"""
        self.assertIsNotNone(self.recognizer)
        self.assertEqual(self.recognizer.min_pattern_quality, 0.7)
        self.assertTrue(self.recognizer.use_ml_validation)
        self.assertEqual(len(self.recognizer.pattern_types), 2)
        self.assertIn(PatternType.REVERSAL, self.recognizer.pattern_types)
        self.assertIn(PatternType.CONTINUATION, self.recognizer.pattern_types)
        self.assertEqual(self.recognizer.timeframe_mapping, self.timeframe_mapping)
        self.assertIs(self.recognizer.model_retraining_service, self.model_retraining_service)

    def test_recognize_patterns_no_data(self):
        """Test pattern recognition with empty data."""
        data = pd.DataFrame({'close': [], 'high': []})
        patterns = self.recognizer.recognize_patterns(data)
        self.assertEqual(len(patterns), 0)

    def test_recognize_patterns_no_match(self):
        """Test pattern recognition when no patterns match."""
        data = pd.DataFrame({
            'close': np.random.rand(10),
            'high': np.random.rand(10) + 1
        }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=10)))
        patterns = self.recognizer.recognize_patterns(data)
        self.assertEqual(len(patterns), 0)

    def test_recognize_patterns_match(self):
        """Test pattern recognition when a pattern matches."""
        # Create data likely to trigger dummy double_top
        data = pd.DataFrame({
            'close': [1.1, 1.2, 1.15, 1.25, 1.18, 1.25],
            'high': [1.12, 1.22, 1.17, 1.28, 1.20, 1.28] # Highs at index -1 and -3 are equal
        }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=6)))

        patterns = self.recognizer.recognize_patterns(data)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]['name'], 'double_top')
        self.assertEqual(patterns[0]['confidence'], 0.6)
        self.assertEqual(patterns[0]['index'], data.index[-1])

    def test_recognize_patterns_multi_timeframe(self):
        """Test pattern recognition across multiple timeframes."""
        # Mock data for different timeframes
        data_1h = pd.DataFrame({
            'close': [1.1, 1.2, 1.15, 1.25, 1.18, 1.25],
            'high': [1.12, 1.22, 1.17, 1.28, 1.20, 1.28] # Double top
        }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=6, freq='H')))

        data_4h = pd.DataFrame({
            'close': [1.0, 1.3, 1.1, 1.4, 1.2, 1.0], # Head & Shoulders?
            'high': [1.1, 1.35, 1.2, 1.45, 1.3, 1.1]
        }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=6, freq='4H')))

        multi_timeframe_data = {'1H': data_1h, '4H': data_4h}
        patterns = self.recognizer.recognize_patterns_multi_timeframe(multi_timeframe_data)

        self.assertTrue('1H' in patterns)
        self.assertTrue('4H' in patterns)
        self.assertEqual(len(patterns['1H']), 1)
        self.assertEqual(patterns['1H'][0]['name'], 'double_top')
        # Assuming _detect_head_shoulders matches the 4H data
        self.assertEqual(len(patterns['4H']), 1)
        self.assertEqual(patterns['4H'][0]['name'], 'head_shoulders')


    def test_ml_feedback_loop(self):
        """Test that the ML feedback loop is triggered"""
        # Create data likely to trigger a pattern
        data = pd.DataFrame({
            'close': [1.1, 1.2, 1.15, 1.25, 1.18, 1.25],
            'high': [1.12, 1.22, 1.17, 1.28, 1.20, 1.28] # Double top
        }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=6)))

        # Create multi-timeframe data
        multi_timeframe_data = {'1H': data}

        # Call detect_patterns method
        with patch.object(self.recognizer, 'recognize_patterns_multi_timeframe') as mock_recognize:
            # Mock the recognize_patterns_multi_timeframe method to return some patterns
            mock_recognize.return_value = {
                '1H': [{
                    'name': 'double_top',
                    'confidence': 0.8,
                    'index': data.index[-1],
                    'type': PatternType.REVERSAL
                }]
            }

            # Call the method
            result = self.recognizer.detect_patterns(multi_timeframe_data)

            # Check that the model retraining service was called
            self.model_retraining_service.check_and_trigger_retraining.assert_called_once_with('sequence_pattern_model')

            # Check that the result contains the expected data
            self.assertIn('patterns', result)
            self.assertIn('1H', result['patterns'])
            self.assertEqual(len(result['patterns']['1H']), 1)
            self.assertEqual(result['patterns']['1H'][0]['name'], 'double_top')

    def test_ml_feedback_loop_disabled(self):
        """Test that the ML feedback loop is not triggered when disabled"""
        # Create a recognizer with ML validation disabled
        recognizer = SequencePatternRecognizer(
            min_pattern_quality=0.7,
            use_ml_validation=False,
            pattern_types=[PatternType.REVERSAL, PatternType.CONTINUATION],
            timeframe_mapping=self.timeframe_mapping,
            model_retraining_service=self.model_retraining_service
        )

        # Example patterns
        recognizer.patterns = {
            'head_shoulders': lambda data: self._detect_head_shoulders(data),
            'double_top': lambda data: self._detect_double_top(data)
        }

        # Create data likely to trigger a pattern
        data = pd.DataFrame({
            'close': [1.1, 1.2, 1.15, 1.25, 1.18, 1.25],
            'high': [1.12, 1.22, 1.17, 1.28, 1.20, 1.28] # Double top
        }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=6)))

        # Create multi-timeframe data
        multi_timeframe_data = {'1H': data}

        # Call detect_patterns method
        with patch.object(recognizer, 'recognize_patterns_multi_timeframe') as mock_recognize:
            # Mock the recognize_patterns_multi_timeframe method to return some patterns
            mock_recognize.return_value = {
                '1H': [{
                    'name': 'double_top',
                    'confidence': 0.8,
                    'index': data.index[-1],
                    'type': PatternType.REVERSAL
                }]
            }

            # Call the method
            result = recognizer.detect_patterns(multi_timeframe_data)

            # Check that the model retraining service was NOT called
            self.model_retraining_service.check_and_trigger_retraining.assert_not_called()

            # Check that the result contains the expected data
            self.assertIn('patterns', result)
            self.assertIn('1H', result['patterns'])
            self.assertEqual(len(result['patterns']['1H']), 1)
            self.assertEqual(result['patterns']['1H'][0]['name'], 'double_top')

    def test_detect_patterns_with_timeframes(self):
        """Test detecting patterns with specific timeframes"""
        # Create data for multiple timeframes
        data_1h = pd.DataFrame({
            'close': [1.1, 1.2, 1.15, 1.25, 1.18, 1.25],
            'high': [1.12, 1.22, 1.17, 1.28, 1.20, 1.28] # Double top
        }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=6, freq='H')))

        data_4h = pd.DataFrame({
            'close': [1.0, 1.3, 1.1, 1.4, 1.2, 1.0], # Head & Shoulders
            'high': [1.1, 1.35, 1.2, 1.45, 1.3, 1.1]
        }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=6, freq='4H')))

        data_1d = pd.DataFrame({
            'close': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6], # Uptrend
            'high': [1.15, 1.25, 1.35, 1.45, 1.55, 1.65]
        }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=6, freq='D')))

        # Create multi-timeframe data
        multi_timeframe_data = {'1H': data_1h, '4H': data_4h, '1D': data_1d}

        # Call detect_patterns method with specific timeframes
        with patch.object(self.recognizer, 'recognize_patterns_multi_timeframe') as mock_recognize:
            # Mock the recognize_patterns_multi_timeframe method to return some patterns
            mock_recognize.return_value = {
                '1H': [{
                    'name': 'double_top',
                    'confidence': 0.8,
                    'index': data_1h.index[-1],
                    'type': PatternType.REVERSAL
                }],
                '4H': [{
                    'name': 'head_shoulders',
                    'confidence': 0.9,
                    'index': data_4h.index[-1],
                    'type': PatternType.REVERSAL
                }]
            }

            # Call the method with specific timeframes
            result = self.recognizer.detect_patterns(
                price_data=multi_timeframe_data,
                timeframes=['1H', '4H']
            )

            # Check that the recognize_patterns_multi_timeframe method was called with the correct timeframes
            mock_recognize.assert_called_once()
            call_args = mock_recognize.call_args[0][0]
            self.assertEqual(set(call_args.keys()), {'1H', '4H'})

            # Check that the result contains only the specified timeframes
            self.assertIn('patterns', result)
            self.assertIn('1H', result['patterns'])
            self.assertIn('4H', result['patterns'])
            self.assertNotIn('1D', result['patterns'])

if __name__ == '__main__':
    unittest.main()
