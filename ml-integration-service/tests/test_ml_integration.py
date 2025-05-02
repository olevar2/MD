"""
Unit tests for ML integration components.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from feature_store_service.indicators.advanced_moving_averages import TEMA, DEMA, HullMA
from feature_store_service.indicators.base_indicator import BaseIndicator
from ml_integration_service.feature_extraction import (
    FeatureExtractor,
    IndicatorFeatureGenerator,
    CrossoverFeatureGenerator,
    TrendFeatureGenerator
)
from ml_integration_service.feature_selection import (
    FeatureSelector,
    ImportanceBasedSelector,
    CorrelationBasedSelector,
    RecursiveFeatureElimination
)
from ml_integration_service.indicator_feedback import (
    IndicatorMLFeedback,
    FeatureImportanceTracker,
    IndicatorPerformanceOptimizer
)
from ml_integration_service.model_preparation import (
    ModelInputPreparation,
    TabularInputPreparator,
    SequentialInputPreparator,
    MultiTimeframePreparator
)


class TestIndicatorFeatures(unittest.TestCase):
    """Base test class for indicator feature tests."""

    def setUp(self):
        """Set up test data with indicator values."""
        np.random.seed(42)
        n_samples = 500
        
        # Generate realistic price data with trend and volatility
        date_range = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Create base price with up and down trends
        trend1 = np.linspace(0, 10, n_samples//2)
        trend2 = np.linspace(10, 0, n_samples//2)
        trend = np.concatenate([trend1, trend2])
        
        # Add some cycles
        cycles = 5 * np.sin(np.linspace(0, 8*np.pi, n_samples))
        
        # Add random walk component
        random_walk = np.cumsum(np.random.normal(0, 0.5, n_samples))
        
        # Combine components
        price = 100 + trend + cycles + random_walk
        
        # Create OHLCV data
        self.data = pd.DataFrame({
            'open': price * (1 + 0.005 * np.random.randn(n_samples)),
            'high': price * (1 + 0.01 * np.random.randn(n_samples)),
            'low': price * (1 - 0.01 * np.random.randn(n_samples)),
            'close': price,
            'volume': 1000000 * (1 + 0.2 * np.random.randn(n_samples))
        }, index=date_range)
        
        # Ensure high is highest and low is lowest
        self.data['high'] = np.maximum(
            np.maximum(self.data['high'], self.data['open']), 
            self.data['close']
        )
        self.data['low'] = np.minimum(
            np.minimum(self.data['low'], self.data['open']), 
            self.data['close']
        )
        
        # Add indicators for feature extraction
        self._add_indicators()
        
        # Create target variable (future price direction)
        self.data['target'] = np.sign(self.data['close'].shift(-5) - self.data['close'])
    
    def _add_indicators(self):
        """Add technical indicators to the data."""
        # Moving averages
        tema = TEMA(period=20)
        dema = DEMA(period=20)
        hull = HullMA(period=20)
        
        # Calculate indicators
        self.data = tema.calculate(self.data)
        self.data = dema.calculate(self.data)
        self.data = hull.calculate(self.data)
        
        # RSI (simple implementation for testing)
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.data['sma_20'] = self.data['close'].rolling(window=20).mean()
        self.data['bbstd_20'] = self.data['close'].rolling(window=20).std()
        self.data['bbupper_20'] = self.data['sma_20'] + 2 * self.data['bbstd_20']
        self.data['bblower_20'] = self.data['sma_20'] - 2 * self.data['bbstd_20']
        
        # MACD
        ema12 = self.data['close'].ewm(span=12).mean()
        ema26 = self.data['close'].ewm(span=26).mean()
        self.data['macd'] = ema12 - ema26
        self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
        self.data['macd_hist'] = self.data['macd'] - self.data['macd_signal']
        
        # Add some crossover and trend indicators
        self.data['ma_cross'] = np.where(
            self.data['tema_20'] > self.data['dema_20'], 
            1, np.where(self.data['tema_20'] < self.data['dema_20'], -1, 0)
        )
        
        self.data['price_vs_ma'] = np.sign(self.data['close'] - self.data['sma_20'])
        self.data['rsi_signal'] = np.where(
            self.data['rsi_14'] > 70, -1, np.where(self.data['rsi_14'] < 30, 1, 0)
        )


class TestFeatureExtractor(TestIndicatorFeatures):
    """Test suite for FeatureExtractor class."""

    def test_basic_feature_extraction(self):
        """Test basic feature extraction from indicators."""
        # Initialize feature extractor
        feature_extractor = FeatureExtractor()
        
        # Extract features
        features = feature_extractor.extract_features(self.data)
        
        # Should have extracted features from indicators
        self.assertGreater(len(features.columns), 0)
        
        # Should exclude the target column
        self.assertNotIn('target', features.columns)
        
        # Should exclude date index as a feature
        self.assertFalse(any('date' in col.lower() for col in features.columns))
    
    def test_normalization(self):
        """Test feature normalization."""
        # Initialize feature extractor with normalization
        feature_extractor = FeatureExtractor(normalize=True)
        
        # Extract normalized features
        features = feature_extractor.extract_features(self.data)
        
        # Values should be normalized (between -1 and 1 or 0 and 1)
        for col in features.columns:
            col_min = features[col].min()
            col_max = features[col].max()
            self.assertGreaterEqual(col_min, -1.1)  # Allow slight buffer
            self.assertLessEqual(col_max, 1.1)
    
    def test_trend_feature_generation(self):
        """Test generation of trend features."""
        # Initialize trend feature generator
        trend_generator = TrendFeatureGenerator()
        
        # Generate trend features
        trend_features = trend_generator.generate(self.data)
        
        # Should have created trend features
        self.assertIn('tema_20_trend', trend_features.columns)
        self.assertIn('rsi_14_trend', trend_features.columns)
        
        # Trend features should be -1, 0, or 1
        for col in trend_features.columns:
            if '_trend' in col:
                unique_values = trend_features[col].dropna().unique()
                for val in unique_values:
                    self.assertIn(val, [-1, 0, 1])
    
    def test_crossover_feature_generation(self):
        """Test generation of crossover features."""
        # Initialize crossover feature generator
        crossover_generator = CrossoverFeatureGenerator()
        
        # Generate crossover features
        crossover_features = crossover_generator.generate(self.data)
        
        # Should have created crossover features
        expected_crossovers = [
            'tema_20_x_dema_20',
            'tema_20_x_sma_20'
        ]
        
        # At least some of these crossovers should be present
        found_crossovers = [col for col in crossover_features.columns if '_x_' in col]
        self.assertGreater(len(found_crossovers), 0)
        
        # Crossover features should be -1, 0, or 1
        for col in found_crossovers:
            unique_values = crossover_features[col].dropna().unique()
            for val in unique_values:
                self.assertIn(val, [-1, 0, 1])
    
    def test_custom_feature_extraction(self):
        """Test custom feature extraction."""
        # Define a custom feature function
        def custom_macd_features(data):
            features = pd.DataFrame(index=data.index)
            features['macd_above_signal'] = np.where(data['macd'] > data['macd_signal'], 1, 0)
            features['macd_divergence'] = (
                data['macd'].diff().apply(np.sign) != 
                data['close'].diff().apply(np.sign)
            ).astype(int)
            return features
        
        # Initialize feature extractor with custom function
        feature_extractor = FeatureExtractor(custom_feature_functions=[custom_macd_features])
        
        # Extract features
        features = feature_extractor.extract_features(self.data)
        
        # Custom features should be included
        self.assertIn('macd_above_signal', features.columns)
        self.assertIn('macd_divergence', features.columns)
    
    def test_lagged_features(self):
        """Test generation of lagged features."""
        # Initialize feature extractor with lagged features
        feature_extractor = FeatureExtractor(
            create_lagged_features=True,
            lag_periods=[1, 3, 5],
            indicators_to_lag=['tema_20', 'rsi_14']
        )
        
        # Extract features
        features = feature_extractor.extract_features(self.data)
        
        # Lagged features should be created
        self.assertIn('tema_20_lag_1', features.columns)
        self.assertIn('tema_20_lag_3', features.columns)
        self.assertIn('tema_20_lag_5', features.columns)
        self.assertIn('rsi_14_lag_1', features.columns)
    
    def test_feature_combination(self):
        """Test feature combinations."""
        # Initialize feature extractor with feature combinations
        feature_extractor = FeatureExtractor(create_feature_combinations=True)
        
        # Extract features with a limited set of indicators for testing
        limited_data = self.data[['close', 'tema_20', 'rsi_14', 'target']].copy()
        features = feature_extractor.extract_features(limited_data)
        
        # Should have created combination features
        self.assertIn('tema_20_mul_rsi_14', features.columns)
        self.assertIn('tema_20_div_close', features.columns)
        
        # Division by close should be similar to normalization
        self.assertAlmostEqual(
            features['tema_20_div_close'].mean(),
            1.0,
            delta=0.1
        )


class TestFeatureSelector(TestIndicatorFeatures):
    """Test suite for FeatureSelector class."""

    def setUp(self):
        """Set up test data and features."""
        super().setUp()
        
        # Create feature extractor
        self.feature_extractor = FeatureExtractor(normalize=True)
        
        # Extract features
        self.features = self.feature_extractor.extract_features(self.data)
        
        # Prepare target and remove NaN values
        self.X = self.features.dropna()
        self.y = self.data['target'].loc[self.X.index].dropna()
        
        # Ensure X and y have the same index
        common_idx = self.X.index.intersection(self.y.index)
        self.X = self.X.loc[common_idx]
        self.y = self.y.loc[common_idx]
    
    def test_importance_based_selection(self):
        """Test importance-based feature selection."""
        # Initialize importance-based selector
        selector = ImportanceBasedSelector(n_features=10)
        
        # Select features
        selected_features = selector.select(self.X, self.y)
        
        # Should have selected the specified number of features
        self.assertEqual(len(selected_features.columns), 10)
        
        # Get feature importances
        importances = selector.get_feature_importances()
        
        # Should have importance values for all original features
        self.assertEqual(len(importances), len(self.X.columns))
        
        # Importances should sum to approximately 1.0
        self.assertAlmostEqual(sum(importances.values()), 1.0, delta=0.01)
    
    def test_correlation_based_selection(self):
        """Test correlation-based feature selection."""
        # Initialize correlation-based selector
        selector = CorrelationBasedSelector(
            correlation_threshold=0.85,
            target_correlation_method='pearson'
        )
        
        # Select features
        selected_features = selector.select(self.X, self.y)
        
        # Should have eliminated some features
        self.assertLess(len(selected_features.columns), len(self.X.columns))
        
        # Get correlation matrix
        corr_matrix = selector.get_correlation_matrix()
        
        # Should have correlation values
        self.assertGreater(len(corr_matrix), 0)
        
        # No remaining features should be highly correlated
        high_corr_pairs = []
        for i in range(len(selected_features.columns)):
            for j in range(i+1, len(selected_features.columns)):
                col1 = selected_features.columns[i]
                col2 = selected_features.columns[j]
                corr = abs(selected_features[col1].corr(selected_features[col2]))
                if corr > 0.85:
                    high_corr_pairs.append((col1, col2, corr))
        
        self.assertEqual(len(high_corr_pairs), 0)
    
    def test_recursive_feature_elimination(self):
        """Test recursive feature elimination."""
        # Initialize RFE selector
        selector = RecursiveFeatureElimination(
            n_features=5,
            step=0.3,  # Remove 30% of features at each step
            cv=3
        )
        
        # Select features
        selected_features = selector.select(self.X, self.y)
        
        # Should have selected the specified number of features
        self.assertEqual(len(selected_features.columns), 5)
        
        # Get feature rankings
        rankings = selector.get_feature_rankings()
        
        # Should have ranking values for all original features
        self.assertEqual(len(rankings), len(self.X.columns))
    
    def test_feature_selector_chaining(self):
        """Test chaining multiple feature selection methods."""
        # Initialize feature selector with method chaining
        selector = FeatureSelector(methods=[
            ('correlation', CorrelationBasedSelector(correlation_threshold=0.85)),
            ('importance', ImportanceBasedSelector(n_features=10))
        ])
        
        # Select features
        selected_features = selector.select(self.X, self.y)
        
        # Should have selected at most 10 features
        self.assertLessEqual(len(selected_features.columns), 10)
        
        # Should have importance values for all features after correlation filtering
        importances = selector.methods[1][1].get_feature_importances()
        self.assertLessEqual(len(importances), len(self.X.columns))
    
    def test_feature_stability(self):
        """Test stability of feature selection."""
        # Initialize selector with stability assessment
        selector = ImportanceBasedSelector(n_features=10, assess_stability=True)
        
        # Run selection multiple times with bootstrap sampling
        selected_features = selector.select(self.X, self.y)
        
        # Get stability scores
        stability_scores = selector.get_stability_scores()
        
        # Should have stability scores for each feature
        self.assertEqual(len(stability_scores), len(self.X.columns))
        
        # Stability scores should be between 0 and 1
        for score in stability_scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)


class TestIndicatorMLFeedback(TestIndicatorFeatures):
    """Test suite for IndicatorMLFeedback class."""
    
    def setUp(self):
        """Set up test data and ML model."""
        super().setUp()
        
        # Create feature extractor
        self.feature_extractor = FeatureExtractor(normalize=True)
        
        # Extract features
        self.features = self.feature_extractor.extract_features(self.data)
        
        # Prepare target and remove NaN values
        self.X = self.features.dropna()
        self.y = self.data['target'].loc[self.X.index].dropna()
        
        # Ensure X and y have the same index
        common_idx = self.X.index.intersection(self.y.index)
        self.X = self.X.loc[common_idx]
        self.y = self.y.loc[common_idx]
        
        # Create a simple ML model
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(self.X, self.y)
        
        # Create test directory
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        try:
            import shutil
            shutil.rmtree(self.test_dir)
        except (OSError, IOError):
            pass
    
    def test_feature_importance_tracking(self):
        """Test tracking of feature importances from ML model."""
        # Initialize feature importance tracker
        tracker = FeatureImportanceTracker(
            storage_path=os.path.join(self.test_dir, 'importance_history.pkl')
        )
        
        # Track importances from model
        indicator_importances = tracker.track_importances(
            self.model, self.X, self.X.columns
        )
        
        # Should have importances for all features
        self.assertEqual(len(indicator_importances), len(self.X.columns))
        
        # Update with another model
        model2 = RandomForestClassifier(n_estimators=100, random_state=43)
        model2.fit(self.X, self.y)
        
        tracker.track_importances(model2, self.X, self.X.columns)
        
        # Should have history of importances
        importance_history = tracker.get_importance_history()
        self.assertEqual(len(importance_history), 2)
        
        # Save and load history
        tracker.save_history()
        
        # Create new tracker and load history
        new_tracker = FeatureImportanceTracker(
            storage_path=os.path.join(self.test_dir, 'importance_history.pkl')
        )
        new_tracker.load_history()
        
        # Should have the same history
        self.assertEqual(
            len(new_tracker.get_importance_history()),
            len(importance_history)
        )
    
    def test_indicator_performance_optimizer(self):
        """Test optimization of indicator parameters based on ML performance."""
        # Define test indicators with parameters
        class OptimizableMA(BaseIndicator):
            """Test MA indicator with optimizable parameters."""
            
            category = "moving_average"
            
            def __init__(self, period: int = 20):
                """Initialize with period parameter."""
                self.period = period
                
            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                """Calculate MA."""
                result = data.copy()
                result[f'opt_ma_{self.period}'] = data['close'].rolling(window=self.period).mean()
                return result
            
            @classmethod
            def get_parameter_ranges(cls):
                """Get parameter ranges for optimization."""
                return {
                    'period': list(range(5, 51, 5))  # 5, 10, 15, ..., 50
                }
        
        # Initialize optimizer
        optimizer = IndicatorPerformanceOptimizer(
            storage_path=os.path.join(self.test_dir, 'optimizer_state.pkl')
        )
        
        # Add indicator for optimization
        optimizer.add_indicator(OptimizableMA)
        
        # Run optimization
        best_params = optimizer.optimize(
            self.data, self.X, self.y, 
            metric=accuracy_score,
            n_trials=3  # Small number for testing
        )
        
        # Should have found best parameters
        self.assertIn('OptimizableMA', best_params)
        self.assertIn('period', best_params['OptimizableMA'])
        
        # Best period should be within the parameter range
        self.assertGreaterEqual(best_params['OptimizableMA']['period'], 5)
        self.assertLessEqual(best_params['OptimizableMA']['period'], 50)
        
        # Save and load state
        optimizer.save_state()
        
        # Create new optimizer and load state
        new_optimizer = IndicatorPerformanceOptimizer(
            storage_path=os.path.join(self.test_dir, 'optimizer_state.pkl')
        )
        new_optimizer.load_state()
        
        # Should have the same best parameters
        self.assertEqual(
            new_optimizer.best_parameters,
            best_params
        )
    
    def test_indicator_feedback_loop(self):
        """Test the indicator-ML feedback loop."""
        # Initialize feedback system
        feedback = IndicatorMLFeedback(
            storage_path=os.path.join(self.test_dir, 'feedback_state.pkl')
        )
        
        # Add indicators with performance data
        indicators_info = {
            'tema_20': {'class': 'TEMA', 'params': {'period': 20}, 'category': 'trend'},
            'dema_20': {'class': 'DEMA', 'params': {'period': 20}, 'category': 'trend'},
            'rsi_14': {'class': 'RSI', 'params': {'period': 14}, 'category': 'momentum'}
        }
        
        for name, info in indicators_info.items():
            feedback.register_indicator(name, info)
        
        # Add model performance data
        performance_data = {
            'accuracy': 0.65,
            'f1': 0.67,
            'precision': 0.70,
            'recall': 0.64
        }
        
        feedback.update_model_performance(performance_data)
        
        # Add feature importance data
        feature_importances = {
            'tema_20': 0.15,
            'dema_20': 0.05,
            'rsi_14': 0.20,
            'hullma_20': 0.10,
            'macd': 0.25,
            'price_vs_ma': 0.12,
            'ma_cross': 0.08,
            'rsi_signal': 0.05
        }
        
        feedback.update_feature_importances(feature_importances)
        
        # Get indicator performance evaluation
        evaluation = feedback.evaluate_indicators()
        
        # Should have evaluations for all registered indicators
        for name in indicators_info.keys():
            self.assertIn(name, evaluation)
            
        # Get indicator recommendations
        recommendations = feedback.get_recommendations()
        
        # Should provide recommendations for indicator adjustments
        self.assertGreater(len(recommendations), 0)
        
        # Save and load state
        feedback.save_state()
        
        # Create new feedback system and load state
        new_feedback = IndicatorMLFeedback(
            storage_path=os.path.join(self.test_dir, 'feedback_state.pkl')
        )
        new_feedback.load_state()
        
        # Should have the same indicators registered
        self.assertEqual(
            len(new_feedback.indicators),
            len(indicators_info)
        )


class TestModelInputPreparation(TestIndicatorFeatures):
    """Test suite for model input preparation classes."""
    
    def setUp(self):
        """Set up test data and features."""
        super().setUp()
        
        # Create feature extractor
        self.feature_extractor = FeatureExtractor(normalize=True)
        
        # Extract features
        self.features = self.feature_extractor.extract_features(self.data)
        
        # Prepare target and remove NaN values
        self.X = self.features.dropna()
        self.y = self.data['target'].loc[self.X.index].dropna()
        
        # Ensure X and y have the same index
        common_idx = self.X.index.intersection(self.y.index)
        self.X = self.X.loc[common_idx]
        self.y = self.y.loc[common_idx]
    
    def test_tabular_input_preparation(self):
        """Test preparation of tabular inputs."""
        # Initialize tabular input preparator
        preparator = TabularInputPreparator()
        
        # Prepare inputs
        X_prepared, y_prepared = preparator.prepare(self.X, self.y)
        
        # Should maintain same dimensions
        self.assertEqual(X_prepared.shape[0], self.X.shape[0])
        self.assertEqual(X_prepared.shape[1], self.X.shape[1])
        
        # Test with feature selection
        preparator = TabularInputPreparator(
            n_features=10, 
            selector=ImportanceBasedSelector(n_features=10)
        )
        
        X_prepared, y_prepared = preparator.prepare(self.X, self.y)
        
        # Should have reduced number of features
        self.assertEqual(X_prepared.shape[1], 10)
    
    def test_sequential_input_preparation(self):
        """Test preparation of sequential inputs for RNNs."""
        # Initialize sequential input preparator
        preparator = SequentialInputPreparator(
            sequence_length=10,
            step_size=1
        )
        
        # Prepare inputs
        X_prepared, y_prepared = preparator.prepare(self.X, self.y)
        
        # Should have 3D shape for sequences
        self.assertEqual(len(X_prepared.shape), 3)
        self.assertEqual(X_prepared.shape[1], 10)  # sequence length
        self.assertEqual(X_prepared.shape[2], self.X.shape[1])  # features
        
        # Test with overlapping sequences
        preparator = SequentialInputPreparator(
            sequence_length=10,
            step_size=5
        )
        
        X_prepared, y_prepared = preparator.prepare(self.X, self.y)
        
        # Should have more sequences with smaller step size
        self.assertGreater(
            X_prepared.shape[0],
            (self.X.shape[0] - 10) // 10  # Non-overlapping count
        )
    
    def test_multi_timeframe_preparation(self):
        """Test preparation of inputs with multiple timeframes."""
        # Create multi-timeframe data
        # Daily data
        daily_data = self.data.copy()
        
        # Weekly data (resample)
        weekly_data = daily_data.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Add some weekly indicators
        weekly_data['sma_4'] = weekly_data['close'].rolling(window=4).mean()
        weekly_data['rsi_2'] = weekly_data['close'].diff().apply(
            lambda x: 100 if x > 0 else 0
        ).rolling(window=2).mean()
        
        # Create features for each timeframe
        daily_features = self.features.copy()
        
        # Extract weekly features
        weekly_feature_extractor = FeatureExtractor(normalize=True)
        weekly_features = weekly_feature_extractor.extract_features(weekly_data)
        
        # Initialize multi-timeframe preparator
        preparator = MultiTimeframePreparator(
            timeframes=['D', 'W'],
            alignment_method='ffill'
        )
        
        # Prepare inputs
        X_prepared, y_prepared = preparator.prepare(
            {'D': daily_features, 'W': weekly_features},
            self.y
        )
        
        # Should include features from both timeframes
        self.assertGreater(X_prepared.shape[1], daily_features.shape[1])
        
        # All daily dates should be present
        self.assertEqual(X_prepared.shape[0], len(self.y))
    
    def test_model_input_preparation(self):
        """Test the unified ModelInputPreparation class."""
        # Initialize with different preparators
        model_prep = ModelInputPreparation(
            preparators={
                'tabular': TabularInputPreparator(n_features=10),
                'sequential': SequentialInputPreparator(sequence_length=5)
            }
        )
        
        # Prepare inputs for different model types
        tabular_X, tabular_y = model_prep.prepare_for_model('tabular', self.X, self.y)
        seq_X, seq_y = model_prep.prepare_for_model('sequential', self.X, self.y)
        
        # Should have produced different output formats
        self.assertEqual(len(tabular_X.shape), 2)  # 2D for tabular
        self.assertEqual(len(seq_X.shape), 3)  # 3D for sequential
        
        # Test with custom preparator
        class CustomPreparator:
            """Custom preparator for testing."""
            def prepare(self, X, y):
                """Simply return first 100 rows."""
                return X.iloc[:100], y.iloc[:100]
        
        model_prep.add_preparator('custom', CustomPreparator())
        
        custom_X, custom_y = model_prep.prepare_for_model('custom', self.X, self.y)
        
        # Should have used custom preparator
        self.assertEqual(len(custom_X), 100)


if __name__ == '__main__':
    unittest.main()
