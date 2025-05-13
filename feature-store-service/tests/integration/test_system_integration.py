"""
Integration tests for the feature store service components.

Tests the interaction between validation, error management, optimization,
and logging components in real-world scenarios.
"""
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from feature_store_service.validation.data_validator import DataValidationService, IndicatorValidationType, ValidationSeverity
from feature_store_service.error.error_manager import IndicatorErrorManager, CalculationError, DataError, ParameterError
from feature_store_service.error.recovery_service import ErrorRecoveryService
from feature_store_service.error.monitoring_service import ErrorMonitoringService
from feature_store_service.optimization.resource_manager import AdaptiveResourceManager
from feature_store_service.optimization.performance_optimizer import PerformanceOptimizer, PerformanceMonitor
from feature_store_service.logging.indicator_logging import IndicatorLogger, IndicatorReport


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def dummy_rsi_calculation(data: pd.DataFrame, period: int=14) ->pd.Series:
    """Dummy RSI implementation for testing."""
    if period <= 0:
        raise CalculationError('Invalid period', {'period': period})
    if data.isna().any().any():
        raise DataError('Missing values in data', {'input_data': data})
    return pd.Series(np.random.random(len(data)))


class TestSystemIntegration(unittest.TestCase):
    """Integration test suite for feature store service components."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / 'logs'
        self.cache_dir = Path(self.temp_dir) / 'cache'
        self.profile_dir = Path(self.temp_dir) / 'profiles'
        self.validator = DataValidationService()
        self.error_manager = IndicatorErrorManager()
        self.recovery_service = ErrorRecoveryService()
        self.monitoring_service = ErrorMonitoringService(storage_dir=str(
            self.temp_dir / 'monitoring'))
        self.resource_manager = AdaptiveResourceManager(cache_dir=str(self.
            cache_dir))
        self.performance_optimizer = PerformanceOptimizer(profile_dir=str(
            self.profile_dir))
        self.logger = IndicatorLogger(log_dir=str(self.log_dir))
        self.reporter = IndicatorReport(log_dir=str(self.log_dir))
        self.test_data = pd.DataFrame({'timestamp': pd.date_range(start=
            '2025-01-01', periods=100), 'open': np.random.random(100) * 100,
            'high': np.random.random(100) * 100, 'low': np.random.random(
            100) * 100, 'close': np.random.random(100) * 100, 'volume': np.
            random.random(100) * 1000})
        self.test_data['high'] = self.test_data[['open', 'close']].max(axis=1
            ) + 1
        self.test_data['low'] = self.test_data[['open', 'close']].min(axis=1
            ) - 1

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @with_exception_handling
    def test_end_to_end_calculation_flow(self):
        """Test complete calculation flow with all components."""
        indicator_name = 'RSI'
        params = {'period': 14, 'price_source': 'close'}
        validation_result = self.validator.validate('ohlcv', self.test_data)
        self.assertTrue(validation_result.is_valid)
        self.logger.log_validation(indicator_name=indicator_name,
            validation_type='input_data', is_valid=validation_result.
            is_valid, details=validation_result.details)
        calc_id = f'{indicator_name}_{datetime.utcnow().timestamp()}'
        try:
            self.performance_optimizer.monitor.start_calculation(calc_id=
                calc_id, indicator_name=indicator_name, parameters=params)
            future = self.resource_manager.submit_calculation(calc_id=
                calc_id, calc_func=dummy_rsi_calculation, args=(self.
                test_data,), cache_key=
                f"{indicator_name}_{params['period']}", **params)
            result = future.result()
            profile = self.performance_optimizer.monitor.end_calculation(
                calc_id=calc_id, data_points=len(self.test_data))
            self.logger.log_performance(indicator_name=indicator_name,
                execution_time=profile.execution_time, data_points=profile.
                data_points, details=profile.parameters)
        except Exception as e:
            if isinstance(e, (CalculationError, DataError, ParameterError)):
                self.error_manager.register_error(e, indicator_name)
                self.monitoring_service.record_error(e, indicator_name)
                recovery_result = self.recovery_service.recover(e)
                if recovery_result:
                    self.logger.log_error(indicator_name=indicator_name,
                        error_type=e.__class__.__name__, message=str(e),
                        details={'recovered': True, 'recovery':
                        recovery_result})
                else:
                    self.logger.log_error(indicator_name=indicator_name,
                        error_type=e.__class__.__name__, message=str(e),
                        details={'recovered': False})
            raise
        result_validation = self.validator.validate('calculation', result,
            params=params)
        self.assertTrue(result_validation.is_valid)
        self.logger.log_validation(indicator_name=indicator_name,
            validation_type='calculation', is_valid=result_validation.
            is_valid, details=result_validation.details)
        system_status = self.resource_manager.get_system_status()
        error_patterns = self.monitoring_service.get_error_patterns(time_window
            =timedelta(hours=1))
        performance_analysis = self.performance_optimizer.analyze_performance(
            indicator_name)
        self.assertGreater(system_status['cache_stats']['hit_rate'], 0)
        self.assertEqual(len(error_patterns), 0)
        self.assertIn('execution_time', performance_analysis)

    @with_exception_handling
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery flow."""
        indicator_name = 'RSI'
        data_with_nans = self.test_data.copy()
        data_with_nans.loc[2:4, 'close'] = np.nan
        try:
            result = dummy_rsi_calculation(data_with_nans, period=14)
            self.fail('Expected DataError')
        except DataError as e:
            self.error_manager.register_error(e, indicator_name)
            self.monitoring_service.record_error(e, indicator_name)
            recovery_result = self.recovery_service.recover(e)
            self.assertIsNotNone(recovery_result)
            self.assertFalse(recovery_result.isna().any().any())
            self.logger.log_error(indicator_name=indicator_name, error_type
                ='DataError', message=str(e), details={'recovered': True})
            error_report = self.reporter.generate_error_report()
            self.assertEqual(error_report['total_errors'], 1)
            self.assertIn(indicator_name, error_report['by_indicator'])

    def test_performance_optimization(self):
        """Test performance optimization flow."""
        indicator_name = 'RSI'
        test_periods = [5, 10, 14, 20, 30]
        for period in test_periods:
            calc_id = f'{indicator_name}_{period}'
            _, profile = self.performance_optimizer.profile_calculation(
                indicator_name=indicator_name, calc_func=
                dummy_rsi_calculation, data=self.test_data, parameters={
                'period': period})
            self.logger.log_performance(indicator_name=indicator_name,
                execution_time=profile.execution_time, data_points=profile.
                data_points, details={'period': period})
        analysis = self.performance_optimizer.analyze_performance(
            indicator_name)
        optimal_params = self.performance_optimizer.optimize_parameters(
            indicator_name, {'period': 14})
        self.assertIn('execution_time', analysis)
        self.assertIn('parameter_impact', analysis)
        self.assertIn('period', optimal_params)
        perf_report = self.reporter.generate_performance_report()
        self.assertIn(indicator_name, perf_report['by_indicator'])

    def test_resource_management(self):
        """Test resource management and caching."""
        indicator_name = 'RSI'
        params = {'period': 14}
        future1 = self.resource_manager.submit_calculation(calc_id='calc_1',
            calc_func=dummy_rsi_calculation, args=(self.test_data,),
            cache_key=f"{indicator_name}_{params['period']}", **params)
        result1 = future1.result()
        future2 = self.resource_manager.submit_calculation(calc_id='calc_2',
            calc_func=dummy_rsi_calculation, args=(self.test_data,),
            cache_key=f"{indicator_name}_{params['period']}", **params)
        result2 = future2.result()
        status = self.resource_manager.get_system_status()
        self.assertGreater(status['cache_stats']['hits'], 0)
        self.assertIn('resources', status)
        self.assertIn('cpu', status['resources'])
        self.assertIn('memory', status['resources'])

    @with_exception_handling
    def test_system_monitoring(self):
        """Test system monitoring and health reporting."""
        for i in range(5):
            try:
                if i % 2 == 0:
                    dummy_rsi_calculation(self.test_data, period=14)
                else:
                    dummy_rsi_calculation(self.test_data, period=-1)
            except Exception as e:
                self.monitoring_service.record_error(e, 'RSI')
        health_report = self.monitoring_service.generate_health_report()
        self.assertIn('trends', health_report)
        self.assertIn('active_patterns', health_report)
        self.assertIn('system_status', health_report)
        patterns = self.monitoring_service.get_error_patterns()
        self.assertGreater(len(patterns), 0)
        status = self.resource_manager.get_system_status()
        self.assertIn('resources', status)
        self.assertGreater(status['resources']['cpu']['current'], 0)


if __name__ == '__main__':
    unittest.main()
