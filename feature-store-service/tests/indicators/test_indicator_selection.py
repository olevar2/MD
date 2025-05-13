"""
Unit tests for intelligent indicator selection system.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import json
from feature_store_service.indicators.indicator_selection import MarketRegimeClassifier, IndicatorPerformanceMetrics, IndicatorSelectionEngine, MarketRegime, IndicatorCategory
from feature_store_service.indicators.base_indicator import BaseIndicator


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class SimpleMAIndicator(BaseIndicator):
    """Simple moving average indicator for testing."""
    category = 'moving_average'

    def __init__(self, period: int=20, price_source: str='close', **kwargs):
        """Initialize simple MA indicator."""
        self.period = period
        self.price_source = price_source

    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """Calculate simple MA."""
        result = data.copy()
        result[f'sma_{self.period}'] = data[self.price_source].rolling(window
            =self.period).mean()
        return result

    @classmethod
    def get_info(cls):
        """Get indicator information."""
        return {'name': 'Simple Moving Average', 'description':
            'Simple Moving Average indicator', 'category': cls.category,
            'parameters': [{'name': 'period', 'description':
            'The window size for the moving average', 'type': 'int',
            'default': 20}, {'name': 'price_source', 'description':
            'The price column to use', 'type': 'string', 'default': 'close'}]}


class SimpleRSIIndicator(BaseIndicator):
    """Simple RSI indicator for testing."""
    category = 'momentum'

    def __init__(self, period: int=14, price_source: str='close', **kwargs):
        """Initialize simple RSI indicator."""
        self.period = period
        self.price_source = price_source

    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """Calculate RSI."""
        result = data.copy()
        delta = data[self.price_source].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - 100 / (1 + rs)
        result[f'rsi_{self.period}'] = rsi
        return result

    @classmethod
    def get_info(cls):
        """Get indicator information."""
        return {'name': 'Relative Strength Index', 'description':
            'Momentum oscillator that measures speed and change of price movements'
            , 'category': cls.category, 'parameters': [{'name': 'period',
            'description': 'The window size for the calculation', 'type':
            'int', 'default': 14}, {'name': 'price_source', 'description':
            'The price column to use', 'type': 'string', 'default': 'close'}]}


class SimpleBBIndicator(BaseIndicator):
    """Simple Bollinger Bands indicator for testing."""
    category = 'volatility'

    def __init__(self, period: int=20, stddev: float=2.0, price_source: str
        ='close', **kwargs):
        """Initialize simple Bollinger Bands indicator."""
        self.period = period
        self.stddev = stddev
        self.price_source = price_source

    def calculate(self, data: pd.DataFrame) ->pd.DataFrame:
        """Calculate Bollinger Bands."""
        result = data.copy()
        result['bb_middle'] = data[self.price_source].rolling(window=self.
            period).mean()
        result['bb_stddev'] = data[self.price_source].rolling(window=self.
            period).std()
        result['bb_upper'] = result['bb_middle'] + result['bb_stddev'
            ] * self.stddev
        result['bb_lower'] = result['bb_middle'] - result['bb_stddev'
            ] * self.stddev
        return result

    @classmethod
    def get_info(cls):
        """Get indicator information."""
        return {'name': 'Bollinger Bands', 'description':
            'Volatility bands that contract and expand based on price action',
            'category': cls.category, 'parameters': [{'name': 'period',
            'description': 'The window size for the calculation', 'type':
            'int', 'default': 20}, {'name': 'stddev', 'description':
            'Number of standard deviations for the bands', 'type': 'float',
            'default': 2.0}, {'name': 'price_source', 'description':
            'The price column to use', 'type': 'string', 'default': 'close'}]}


class TestMarketRegimeClassifier(unittest.TestCase):
    """Test suite for market regime classification."""

    def setUp(self):
        """Set up test data with different market regimes."""
        np.random.seed(42)
        self._create_trending_bullish_data()
        self._create_trending_bearish_data()
        self._create_ranging_volatile_data()
        self._create_ranging_tight_data()
        self._create_breakout_data()
        self._create_reversal_data()
        self.classifier = MarketRegimeClassifier()

    def _create_trending_bullish_data(self):
        """Create data for a bullish trending market."""
        days = 30
        start_date = datetime(2023, 1, 1)
        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        price = np.linspace(100, 150, days)
        noise = np.random.normal(0, 1, days)
        self.bullish_data = pd.DataFrame({'open': price * (1 - 0.005),
            'high': price * (1 + 0.01), 'low': price * (1 - 0.01), 'close':
            price + noise, 'volume': 1000000 * (1 + 0.2 * np.random.randn(
            days))}, index=date_range)

    def _create_trending_bearish_data(self):
        """Create data for a bearish trending market."""
        days = 30
        start_date = datetime(2023, 2, 1)
        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        price = np.linspace(150, 100, days)
        noise = np.random.normal(0, 1, days)
        self.bearish_data = pd.DataFrame({'open': price * (1 + 0.005),
            'high': price * (1 + 0.01), 'low': price * (1 - 0.01), 'close':
            price + noise, 'volume': 1000000 * (1 + 0.2 * np.random.randn(
            days))}, index=date_range)

    def _create_ranging_volatile_data(self):
        """Create data for a volatile ranging market."""
        days = 30
        start_date = datetime(2023, 3, 1)
        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        base = 120
        volatility = 5.0
        price = base + volatility * np.sin(np.linspace(0, 4 * np.pi, days))
        noise = np.random.normal(0, 2, days)
        self.volatile_ranging_data = pd.DataFrame({'open': price + np.
            random.normal(0, 2, days), 'high': price + np.random.uniform(1,
            4, days), 'low': price - np.random.uniform(1, 4, days), 'close':
            price + noise, 'volume': 1000000 * (1 + 0.3 * np.random.randn(
            days))}, index=date_range)

    def _create_ranging_tight_data(self):
        """Create data for a tight ranging market."""
        days = 30
        start_date = datetime(2023, 4, 1)
        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        base = 120
        volatility = 1.0
        price = base + volatility * np.sin(np.linspace(0, 2 * np.pi, days))
        noise = np.random.normal(0, 0.5, days)
        self.tight_ranging_data = pd.DataFrame({'open': price + np.random.
            normal(0, 0.5, days), 'high': price + np.random.uniform(0.2, 1,
            days), 'low': price - np.random.uniform(0.2, 1, days), 'close':
            price + noise, 'volume': 1000000 * (1 + 0.1 * np.random.randn(
            days))}, index=date_range)

    def _create_breakout_data(self):
        """Create data for a breakout market."""
        days = 30
        start_date = datetime(2023, 5, 1)
        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        price1 = np.ones(days // 2) * 120 + np.random.normal(0, 0.5, days // 2)
        price2 = np.linspace(120, 140, days // 2)
        price = np.concatenate([price1, price2])
        volume = np.ones(days) * 1000000
        volume[days // 2:] *= 2.5
        self.breakout_data = pd.DataFrame({'open': [(p * (1 - 0.005)) for p in
            price], 'high': [(p * (1 + 0.01)) for p in price], 'low': [(p *
            (1 - 0.01)) for p in price], 'close': price, 'volume': volume *
            (1 + 0.2 * np.random.randn(days))}, index=date_range)

    def _create_reversal_data(self):
        """Create data for a market reversal."""
        days = 30
        start_date = datetime(2023, 6, 1)
        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        price1 = np.linspace(100, 130, days // 2)
        price2 = np.linspace(130, 110, days // 2)
        price = np.concatenate([price1, price2])
        volume = np.ones(days) * 1000000
        volume[days // 2 - 2:days // 2 + 3] *= 2.0
        self.reversal_data = pd.DataFrame({'open': [(p * (1 + np.random.
            uniform(-0.01, 0.01))) for p in price], 'high': [(p * (1 + 
            0.015)) for p in price], 'low': [(p * (1 - 0.015)) for p in
            price], 'close': price + np.random.normal(0, 0.5, days),
            'volume': volume * (1 + 0.2 * np.random.randn(days))}, index=
            date_range)

    def test_bullish_trend_classification(self):
        """Test classification of bullish trending market."""
        regime = self.classifier.classify(self.bullish_data)
        self.assertEqual(regime, MarketRegime.TRENDING_BULLISH)

    def test_bearish_trend_classification(self):
        """Test classification of bearish trending market."""
        regime = self.classifier.classify(self.bearish_data)
        self.assertEqual(regime, MarketRegime.TRENDING_BEARISH)

    def test_volatile_range_classification(self):
        """Test classification of volatile ranging market."""
        regime = self.classifier.classify(self.volatile_ranging_data)
        self.assertEqual(regime, MarketRegime.RANGING_VOLATILE)

    def test_tight_range_classification(self):
        """Test classification of tight ranging market."""
        regime = self.classifier.classify(self.tight_ranging_data)
        self.assertEqual(regime, MarketRegime.RANGING_TIGHT)

    def test_breakout_classification(self):
        """Test classification of breakout market."""
        breakout_part = self.breakout_data.iloc[len(self.breakout_data) // 
            2 - 5:]
        regime = self.classifier.classify(breakout_part)
        self.assertEqual(regime, MarketRegime.BREAKOUT)

    def test_reversal_classification(self):
        """Test classification of market reversal."""
        reversal_part = self.reversal_data.iloc[len(self.reversal_data) // 
            2 - 5:len(self.reversal_data) // 2 + 5]
        regime = self.classifier.classify(reversal_part)
        self.assertEqual(regime, MarketRegime.REVERSAL)


class TestIndicatorPerformanceMetrics(unittest.TestCase):
    """Test suite for indicator performance metrics calculation."""

    def setUp(self):
        """Set up test data and indicators for performance tracking."""
        np.random.seed(42)
        days = 100
        start_date = datetime(2023, 1, 1)
        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        price = 100
        prices = [price]
        for i in range(1, days):
            if i < days // 2:
                price += 0.1 + np.random.normal(0, 0.3)
            else:
                price -= 0.1 + np.random.normal(0, 0.3)
            prices.append(price)
        self.data = pd.DataFrame({'open': [(p * (1 + np.random.uniform(-
            0.005, 0.005))) for p in prices], 'high': [(p * (1 + np.random.
            uniform(0, 0.01))) for p in prices], 'low': [(p * (1 - np.
            random.uniform(0, 0.01))) for p in prices], 'close': prices,
            'volume': 1000000 * (1 + 0.2 * np.random.randn(days))}, index=
            date_range)
        self.ma_fast = SimpleMAIndicator(period=10)
        self.ma_slow = SimpleMAIndicator(period=30)
        self.rsi = SimpleRSIIndicator(period=14)
        self.data = self.ma_fast.calculate(self.data)
        self.data = self.ma_slow.calculate(self.data)
        self.data = self.rsi.calculate(self.data)
        self.data['ma_signal'] = np.where(self.data['sma_10'] > self.data[
            'sma_30'], 1, np.where(self.data['sma_10'] < self.data['sma_30'
            ], -1, 0))
        self.data['rsi_signal'] = np.where(self.data['rsi_14'] > 70, -1, np
            .where(self.data['rsi_14'] < 30, 1, 0))
        self.metrics = IndicatorPerformanceMetrics()

    def test_metrics_calculation(self):
        """Test basic calculation of performance metrics."""
        ma_metrics = self.metrics.calculate_metrics(self.data['sma_10'],
            self.data, self.data['ma_signal'])
        self.assertIn('overall', ma_metrics)
        self.assertIn(f'period_{self.metrics.lookback_periods[0]}', ma_metrics)
        rsi_metrics = self.metrics.calculate_metrics(self.data['rsi_14'],
            self.data, self.data['rsi_signal'])
        self.assertIn('overall', rsi_metrics)

    def test_prediction_power(self):
        """Test prediction power calculation."""
        ma_fast_metrics = self.metrics.calculate_metrics(self.data['sma_10'
            ], self.data)
        ma_slow_metrics = self.metrics.calculate_metrics(self.data['sma_30'
            ], self.data)
        uptrend_data = self.data.iloc[:days // 2]
        ma_fast_uptrend = self.metrics.calculate_metrics(uptrend_data[
            'sma_10'], uptrend_data)
        ma_slow_uptrend = self.metrics.calculate_metrics(uptrend_data[
            'sma_30'], uptrend_data)
        if 'prediction_power_1' in ma_fast_uptrend['overall'
            ] and 'prediction_power_1' in ma_slow_uptrend['overall']:
            self.assertGreaterEqual(ma_fast_uptrend['overall'][
                'prediction_power_1'], ma_slow_uptrend['overall'][
                'prediction_power_1'])

    def test_signal_accuracy(self):
        """Test signal accuracy metrics."""
        ma_metrics = self.metrics.calculate_metrics(self.data['sma_10'],
            self.data, self.data['ma_signal'])
        for period_key in ma_metrics.keys():
            if period_key != 'overall':
                if any('signal_accuracy' in k for k in ma_metrics[
                    period_key].keys()):
                    break
        else:
            self.fail('No signal accuracy metrics found')


class TestIndicatorSelectionEngine(unittest.TestCase):
    """Test suite for indicator selection engine."""

    def setUp(self):
        """Set up test data and indicator registry."""
        np.random.seed(42)
        days = 100
        start_date = datetime(2023, 1, 1)
        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        prices = []
        price = 100
        for _ in range(30):
            price += 0.2 + np.random.normal(0, 0.3)
            prices.append(price)
        for _ in range(30):
            price += np.random.normal(0, 0.5)
            prices.append(price)
        for _ in range(40):
            price -= 0.2 + np.random.normal(0, 0.3)
            prices.append(price)
        self.data = pd.DataFrame({'open': [(p * (1 + np.random.uniform(-
            0.005, 0.005))) for p in prices], 'high': [(p * (1 + np.random.
            uniform(0, 0.01))) for p in prices], 'low': [(p * (1 - np.
            random.uniform(0, 0.01))) for p in prices], 'close': prices,
            'volume': 1000000 * (1 + 0.2 * np.random.randn(days))}, index=
            date_range)
        self.indicators = {'ma_10': SimpleMAIndicator(period=10), 'ma_20':
            SimpleMAIndicator(period=20), 'ma_50': SimpleMAIndicator(period
            =50), 'ma_200': SimpleMAIndicator(period=200), 'rsi_7':
            SimpleRSIIndicator(period=7), 'rsi_14': SimpleRSIIndicator(
            period=14), 'bb_20_2': SimpleBBIndicator(period=20, stddev=2.0),
            'bb_20_3': SimpleBBIndicator(period=20, stddev=3.0)}
        self.temp_dir = tempfile.mkdtemp()
        self.performance_path = os.path.join(self.temp_dir,
            'performance_history.json')
        self.config_path = os.path.join(self.temp_dir, 'selection_config.json')
        self.selection_engine = IndicatorSelectionEngine(indicators_registry
            =self.indicators, performance_history_path=self.
            performance_path, config_path=self.config_path)

    @with_exception_handling
    def tearDown(self):
        """Clean up temporary files."""
        try:
            os.remove(self.performance_path)
            os.remove(self.config_path)
            os.rmdir(self.temp_dir)
        except (OSError, IOError):
            pass

    def test_indicator_categorization(self):
        """Test automatic categorization of indicators."""
        self.assertEqual(self.selection_engine.indicator_categories['ma_10'
            ], IndicatorCategory.TREND)
        self.assertEqual(self.selection_engine.indicator_categories[
            'rsi_14'], IndicatorCategory.MOMENTUM)
        self.assertEqual(self.selection_engine.indicator_categories[
            'bb_20_2'], IndicatorCategory.VOLATILITY)

    def test_regime_classification_and_selection(self):
        """Test indicator selection for different market regimes."""
        bullish_data = self.data.iloc[:30]
        bullish_indicators = self.selection_engine.select_indicators(
            bullish_data)
        self.assertGreater(len(bullish_indicators), 0)
        ranging_data = self.data.iloc[30:60]
        ranging_indicators = self.selection_engine.select_indicators(
            ranging_data)
        self.assertGreater(len(ranging_indicators), 0)
        bearish_data = self.data.iloc[60:]
        bearish_indicators = self.selection_engine.select_indicators(
            bearish_data)
        self.assertGreater(len(bearish_indicators), 0)
        self.assertNotEqual(set(bullish_indicators.keys()), set(
            ranging_indicators.keys()))

    def test_performance_tracking(self):
        """Test indicator performance tracking."""
        ma_result = self.indicators['ma_20'].calculate(self.data)
        signal = np.where(ma_result['sma_20'] > ma_result['close'].shift(1),
            1, -1)
        self.selection_engine.update_performance('ma_20', ma_result[
            'sma_20'], self.data, pd.Series(signal, index=self.data.index))
        self.assertIn('ma_20', self.selection_engine.performance_history)
        self.assertTrue(os.path.exists(self.performance_path))
        with open(self.performance_path, 'r') as f:
            saved_data = json.load(f)
            self.assertIn('ma_20', saved_data)

    def test_optimize_selections(self):
        """Test optimization of indicator selections."""
        for name, indicator in self.indicators.items():
            result = indicator.calculate(self.data)
            if name.startswith('ma'):
                col_name = f"sma_{name.split('_')[1]}"
            elif name.startswith('rsi'):
                col_name = f"rsi_{name.split('_')[1]}"
            else:
                col_name = 'bb_middle'
            if col_name in result.columns:
                signal = np.where(result[col_name] > result['close'].shift(
                    1), 1, -1)
                if 'ma_10' in name:
                    signal = np.where(result[col_name] > result[col_name].
                        shift(5), 1, -1)
                elif 'rsi' in name:
                    signal = np.where(result[col_name] > 70, -1, np.where(
                        result[col_name] < 30, 1, 0))
                self.selection_engine.update_performance(name, result[
                    col_name], self.data, pd.Series(signal, index=self.data
                    .index))
        self.selection_engine.optimize_selections()
        self.assertTrue(os.path.exists(self.config_path))
        for regime in MarketRegime:
            if regime != MarketRegime.UNKNOWN:
                self.assertIn(regime, self.selection_engine.regime_indicators)


if __name__ == '__main__':
    unittest.main()
