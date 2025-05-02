import unittest
import pandas as pd
import os
import tempfile
from analysis_engine.services.timeframe_optimization_service import TimeframeOptimizationService, SignalOutcome

class TestTimeframeOptimizationService(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        self.timeframes = ['1H', '4H', '1D']
        self.optimizer = TimeframeOptimizationService(timeframes=self.timeframes, min_signals_for_weighting=2)
        # Create a temporary file for testing save/load
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.temp_dir.name, 'test_optimizer_state.json')


    def tearDown(self):
        """Clean up test fixtures, if any."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test service initialization."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(list(self.optimizer.performance_data.keys()), ['1H', '4H', '1D'])
        self.assertEqual(list(self.optimizer.timeframe_weights.keys()), ['1H', '4H', '1D'])
        # Initial weights should be equal
        self.assertAlmostEqual(self.optimizer.timeframe_weights['1H'], 1/3)
        self.assertAlmostEqual(self.optimizer.timeframe_weights['4H'], 1/3)
        self.assertAlmostEqual(self.optimizer.timeframe_weights['1D'], 1/3)

    def test_record_signal_performance(self):
        """Test recording signal performance."""
        self.optimizer.record_signal_performance('1H', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('1H', SignalOutcome.LOSS)
        self.optimizer.record_signal_performance('4H', SignalOutcome.WIN)

        self.assertEqual(len(self.optimizer.performance_data['1H']), 2)
        self.assertEqual(len(self.optimizer.performance_data['4H']), 1)
        self.assertEqual(len(self.optimizer.performance_data['1D']), 0)

    def test_calculate_optimal_weights_insufficient_data(self):
        """Test weight calculation when there isn't enough data."""
        self.optimizer.record_signal_performance('1H', SignalOutcome.WIN) # Only 1 signal
        self.optimizer.record_signal_performance('4H', SignalOutcome.LOSS) # Only 1 signal

        self.optimizer.calculate_optimal_weights()
        weights = self.optimizer.get_weights()

        # Weights should remain equal as min_signals_for_weighting (2) is not met
        expected_weight = 1 / len(self.timeframes)
        self.assertAlmostEqual(weights['1H'], expected_weight)
        self.assertAlmostEqual(weights['4H'], expected_weight)
        self.assertAlmostEqual(weights['1D'], expected_weight)


    def test_calculate_optimal_weights_sufficient_data(self):
        """Test calculating optimal weights based on performance with enough data."""
        # Add some performance data
        self.optimizer.record_signal_performance('1H', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('1H', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('4H', SignalOutcome.LOSS)
        self.optimizer.record_signal_performance('4H', SignalOutcome.LOSS)
        self.optimizer.record_signal_performance('1D', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('1D', SignalOutcome.LOSS)

        self.optimizer.calculate_optimal_weights()

        # 1H should have the highest weight (100% win rate)
        # 4H should have the lowest weight (0% win rate)
        # 1D should be in the middle (50% win rate)
        weights = self.optimizer.get_weights()
        self.assertGreater(weights['1H'], weights['1D'])
        self.assertGreater(weights['1D'], weights['4H'])
        self.assertAlmostEqual(sum(weights.values()), 1.0) # Weights should sum to 1

    def test_calculate_optimal_weights_zero_wins(self):
        """Test weight calculation when a timeframe has zero wins."""
        self.optimizer.record_signal_performance('1H', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('1H', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('4H', SignalOutcome.LOSS)
        self.optimizer.record_signal_performance('4H', SignalOutcome.LOSS)
        self.optimizer.record_signal_performance('1D', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('1D', SignalOutcome.LOSS)

        self.optimizer.calculate_optimal_weights()
        weights = self.optimizer.get_weights()

        # 4H should have a very small non-zero weight (due to smoothing/avoiding zero)
        self.assertGreater(weights['4H'], 0)
        self.assertLess(weights['4H'], weights['1D'])
        self.assertLess(weights['4H'], weights['1H'])
        self.assertAlmostEqual(sum(weights.values()), 1.0)


    def test_apply_weighted_score(self):
        """Test applying weighted scores."""
        # Set some arbitrary weights
        self.optimizer.timeframe_weights = {'1H': 0.6, '4H': 0.3, '1D': 0.1}

        raw_scores = {'1H': 0.8, '4H': 0.5, '1D': 0.9}
        weighted_score = self.optimizer.apply_weighted_score(raw_scores)

        expected_score = (0.8 * 0.6) + (0.5 * 0.3) + (0.9 * 0.1)
        self.assertAlmostEqual(weighted_score, expected_score)

    def test_apply_weighted_score_missing_timeframe(self):
        """Test applying weighted scores when a timeframe is missing from input."""
        self.optimizer.timeframe_weights = {'1H': 0.6, '4H': 0.3, '1D': 0.1}
        raw_scores = {'1H': 0.8, '1D': 0.9} # Missing '4H'
        weighted_score = self.optimizer.apply_weighted_score(raw_scores)

        # Expected score should ignore the missing timeframe and renormalize weights implicitly or explicitly
        # Current implementation implicitly gives 0 score to missing timeframe
        expected_score = (0.8 * 0.6) + (0.0 * 0.3) + (0.9 * 0.1)
        self.assertAlmostEqual(weighted_score, expected_score)

    def test_get_performance_stats(self):
        """Test retrieving performance statistics."""
        self.optimizer.record_signal_performance('1H', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('1H', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('4H', SignalOutcome.LOSS)
        self.optimizer.record_signal_performance('1D', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('1D', SignalOutcome.LOSS)

        stats = self.optimizer.get_performance_stats()

        self.assertEqual(len(stats), 3)
        self.assertIn('1H', stats)
        self.assertIn('4H', stats)
        self.assertIn('1D', stats)

        self.assertEqual(stats['1H']['total_signals'], 2)
        self.assertEqual(stats['1H']['wins'], 2)
        self.assertEqual(stats['1H']['losses'], 0)
        self.assertAlmostEqual(stats['1H']['win_rate'], 1.0)

        self.assertEqual(stats['4H']['total_signals'], 1)
        self.assertEqual(stats['4H']['wins'], 0)
        self.assertEqual(stats['4H']['losses'], 1)
        self.assertAlmostEqual(stats['4H']['win_rate'], 0.0)

        self.assertEqual(stats['1D']['total_signals'], 2)
        self.assertEqual(stats['1D']['wins'], 1)
        self.assertEqual(stats['1D']['losses'], 1)
        self.assertAlmostEqual(stats['1D']['win_rate'], 0.5)

    def test_save_and_load_state(self):
        """Test saving and loading the optimizer state."""
        # Record some data and calculate weights
        self.optimizer.record_signal_performance('1H', SignalOutcome.WIN)
        self.optimizer.record_signal_performance('4H', SignalOutcome.LOSS)
        self.optimizer.record_signal_performance('4H', SignalOutcome.LOSS)
        self.optimizer.calculate_optimal_weights()

        original_weights = self.optimizer.get_weights().copy()
        original_performance_data = {tf: list(hist) for tf, hist in self.optimizer.performance_data.items()} # Deep copy deque

        # Save state
        self.optimizer.save_to_file(self.test_file)
        self.assertTrue(os.path.exists(self.test_file))

        # Create a new instance and load state
        new_optimizer = TimeframeOptimizationService(timeframes=self.timeframes)
        new_optimizer.load_from_file(self.test_file)

        # Verify loaded state matches original state
        loaded_weights = new_optimizer.get_weights()
        loaded_performance_data = {tf: list(hist) for tf, hist in new_optimizer.performance_data.items()}

        self.assertEqual(loaded_weights, original_weights)
        # Compare performance data content (deque converted to list for comparison)
        self.assertEqual(len(loaded_performance_data['1H']), len(original_performance_data['1H']))
        self.assertListEqual(loaded_performance_data['1H'], original_performance_data['1H'])
        self.assertEqual(len(loaded_performance_data['4H']), len(original_performance_data['4H']))
        self.assertListEqual(loaded_performance_data['4H'], original_performance_data['4H'])
        self.assertEqual(len(loaded_performance_data['1D']), len(original_performance_data['1D']))
        self.assertListEqual(loaded_performance_data['1D'], original_performance_data['1D'])


    def test_load_state_file_not_found(self):
        """Test loading state when the file does not exist."""
        non_existent_file = os.path.join(self.temp_dir.name, 'non_existent.json')
        # Ensure default state remains if file not found
        initial_weights = self.optimizer.get_weights().copy()
        self.optimizer.load_from_file(non_existent_file) # Should log a warning but not raise error
        self.assertEqual(self.optimizer.get_weights(), initial_weights)


if __name__ == '__main__':
    unittest.main()
