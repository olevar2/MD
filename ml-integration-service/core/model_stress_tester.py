"""
Model Stress Testing Components

This module provides comprehensive stress testing capabilities including:
- Model robustness under extreme market conditions
- Sensitivity analysis for key parameters
- Monte Carlo simulations
- Performance degradation testing
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor
import logging
logger = logging.getLogger(__name__)


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@dataclass
class StressTestResult:
    """Container for stress test results"""
    test_name: str
    metrics: Dict[str, float]
    details: Dict[str, Any]
    passed: bool
    recommendations: List[str]


class ModelRobustnessTester:
    """
    Tests model robustness under various market conditions and scenarios.
    """

    def __init__(self, model_client: Any, market_simulator: Any):
    """
      init  .
    
    Args:
        model_client: Description of model_client
        market_simulator: Description of market_simulator
    
    """

        self.model_client = model_client
        self.market_simulator = market_simulator

    def test_extreme_conditions(self, model_id: str, scenarios: List[Dict[
        str, Any]], performance_threshold: float) ->List[StressTestResult]:
        """
        Test model performance under extreme market conditions.
        
        Args:
            model_id: ID of the model to test
            scenarios: List of extreme market scenarios to test
            performance_threshold: Minimum acceptable performance
            
        Returns:
            List of test results
        """
        results = []
        for scenario in scenarios:
            scenario_data = self.market_simulator.generate_scenario(scenario
                ['conditions'], scenario.get('duration', '1D'), scenario.
                get('volatility', 'high'))
            predictions = self.model_client.get_predictions(model_id,
                scenario_data)
            metrics = self._calculate_metrics(predictions, scenario_data)
            passed = all(metric >= performance_threshold for metric in
                metrics.values())
            recommendations = self._generate_recommendations(metrics,
                scenario['conditions']) if not passed else []
            results.append(StressTestResult(test_name=
                f"Extreme Scenario: {scenario['name']}", metrics=metrics,
                details={'scenario': scenario, 'predictions': predictions},
                passed=passed, recommendations=recommendations))
        return results

    def test_recovery_time(self, model_id: str, shock_scenarios: List[Dict[
        str, Any]], max_recovery_time: str='1D') ->List[StressTestResult]:
        """
        Test how quickly model recovers from market shocks.
        
        Args:
            model_id: ID of the model to test
            shock_scenarios: List of market shock scenarios
            max_recovery_time: Maximum acceptable recovery time
            
        Returns:
            List of test results
        """
        results = []
        for scenario in shock_scenarios:
            shock_data = self.market_simulator.generate_shock_scenario(scenario
                ['shock_type'], scenario.get('magnitude', 'high'), scenario
                .get('duration', '1H'))
            predictions = self.model_client.get_predictions(model_id,
                shock_data)
            recovery_time, metrics = self._analyze_recovery(predictions,
                shock_data, scenario['shock_type'])
            passed = pd.Timedelta(recovery_time) <= pd.Timedelta(
                max_recovery_time)
            results.append(StressTestResult(test_name=
                f"Recovery from {scenario['shock_type']}", metrics={
                'recovery_time': recovery_time, **metrics}, details={
                'scenario': scenario, 'predictions': predictions}, passed=
                passed, recommendations=self._get_recovery_recommendations(
                scenario['shock_type'], recovery_time) if not passed else []))
        return results

    def _calculate_metrics(self, predictions: pd.DataFrame, actual_data: pd
        .DataFrame) ->Dict[str, float]:
        """Calculate performance metrics."""
        return {'accuracy': self._calculate_accuracy(predictions,
            actual_data), 'stability': self._calculate_stability(
            predictions), 'confidence_correlation': self.
            _calculate_confidence_correlation(predictions, actual_data)}

    def _calculate_accuracy(self, predictions: pd.DataFrame, actual_data:
        pd.DataFrame) ->float:
        """Calculate prediction accuracy."""
        return float(np.mean(np.abs(predictions['prediction'] - actual_data
            ['actual'])))

    def _calculate_stability(self, predictions: pd.DataFrame) ->float:
        """Calculate prediction stability."""
        return float(np.std(predictions['prediction']))

    def _calculate_confidence_correlation(self, predictions: pd.DataFrame,
        actual_data: pd.DataFrame) ->float:
        """Calculate correlation between confidence and accuracy."""
        errors = np.abs(predictions['prediction'] - actual_data['actual'])
        return float(np.corrcoef(predictions['confidence'], errors)[0, 1])


class SensitivityAnalyzer:
    """
    Analyzes model sensitivity to parameter changes and input variations.
    """

    def __init__(self, n_samples: int=1000):
    """
      init  .
    
    Args:
        n_samples: Description of n_samples
    
    """

        self.n_samples = n_samples

    def analyze_parameter_sensitivity(self, model_func: Callable,
        param_ranges: Dict[str, Tuple[float, float]], base_params: Dict[str,
        float]) ->Dict[str, Dict[str, float]]:
        """
        Analyze model sensitivity to parameter variations.
        
        Args:
            model_func: Function that runs the model
            param_ranges: Parameter ranges to test
            base_params: Base parameter values
            
        Returns:
            Sensitivity metrics for each parameter
        """
        results = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            param_values = np.linspace(min_val, max_val, self.n_samples)
            outputs = []
            for value in param_values:
                test_params = base_params.copy()
                test_params[param_name] = value
                outputs.append(model_func(test_params))
            sensitivity = {'gradient': np.gradient(outputs, param_values).
                mean(), 'variance': np.var(outputs), 'effect_size': (max(
                outputs) - min(outputs)) / np.std(outputs)}
            results[param_name] = sensitivity
        return results

    def analyze_input_sensitivity(self, model_func: Callable, input_ranges:
        Dict[str, Tuple[float, float]], base_inputs: Dict[str, float],
        n_montecarlo: int=1000) ->Dict[str, Dict[str, float]]:
        """
        Analyze model sensitivity to input variations using Monte Carlo.
        
        Args:
            model_func: Function that runs the model
            input_ranges: Input variable ranges
            base_inputs: Base input values
            n_montecarlo: Number of Monte Carlo simulations
            
        Returns:
            Sensitivity metrics for each input
        """
        results = {}
        samples = {name: np.random.uniform(min_val, max_val, n_montecarlo) for
            name, (min_val, max_val) in input_ranges.items()}
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(n_montecarlo):
                test_inputs = base_inputs.copy()
                for name in input_ranges:
                    test_inputs[name] = samples[name][i]
                futures.append(executor.submit(model_func, test_inputs))
            outputs = [f.result() for f in futures]
        for input_name in input_ranges:
            correlation = np.corrcoef(samples[input_name], outputs)[0, 1]
            partial_rank = stats.spearmanr(samples[input_name], outputs)[0]
            results[input_name] = {'correlation': correlation,
                'partial_rank': partial_rank, 'importance_score': abs(
                correlation * partial_rank)}
        return results


class LoadTester:
    """
    Tests model performance under various load conditions.
    """

    def __init__(self, model_client: Any, max_latency_ms: float=100.0):
    """
      init  .
    
    Args:
        model_client: Description of model_client
        max_latency_ms: Description of max_latency_ms
    
    """

        self.model_client = model_client
        self.max_latency_ms = max_latency_ms

    @with_exception_handling
    def test_throughput(self, model_id: str, max_qps: int, duration: str=
        '5M', ramp_up: str='1M') ->StressTestResult:
        """
        Test model throughput under sustained load.
        
        Args:
            model_id: ID of the model to test
            max_qps: Maximum queries per second to test
            duration: Test duration
            ramp_up: Ramp-up period
            
        Returns:
            Test results
        """
        duration_seconds = pd.Timedelta(duration).total_seconds()
        ramp_up_seconds = pd.Timedelta(ramp_up).total_seconds()
        test_data = self._generate_test_data(int(duration_seconds * max_qps))
        latencies = []
        throughputs = []
        errors = []
        current_qps = 0
        start_time = pd.Timestamp.now()
        with ThreadPoolExecutor() as executor:
            while (pd.Timestamp.now() - start_time).total_seconds(
                ) < duration_seconds:
                elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                if elapsed < ramp_up_seconds:
                    target_qps = elapsed / ramp_up_seconds * max_qps
                else:
                    target_qps = max_qps
                batch_size = int(target_qps)
                futures = []
                for _ in range(batch_size):
                    futures.append(executor.submit(self.
                        _make_prediction_request, model_id, next(test_data)))
                for future in futures:
                    try:
                        latency = future.result()
                        latencies.append(latency)
                    except Exception as e:
                        errors.append(str(e))
                current_qps = len(latencies) / elapsed
                throughputs.append(current_qps)
        metrics = {'avg_latency_ms': np.mean(latencies), 'p95_latency_ms':
            np.percentile(latencies, 95), 'p99_latency_ms': np.percentile(
            latencies, 99), 'max_throughput': max(throughputs),
            'error_rate': len(errors) / len(latencies) if latencies else 1.0}
        passed = metrics['p95_latency_ms'] <= self.max_latency_ms and metrics[
            'error_rate'] <= 0.01
        return StressTestResult(test_name=
            f'Throughput Test ({max_qps} QPS)', metrics=metrics, details={
            'errors': errors, 'throughput_timeline': throughputs}, passed=
            passed, recommendations=self._get_performance_recommendations(
            metrics) if not passed else [])

    def _generate_test_data(self, n_samples: int):
        """Generate test data samples."""
        while True:
            yield pd.DataFrame({'feature1': np.random.randn(100),
                'feature2': np.random.randn(100)})

    def _make_prediction_request(self, model_id: str, data: pd.DataFrame
        ) ->float:
        """Make a prediction request and measure latency."""
        start_time = pd.Timestamp.now()
        self.model_client.get_predictions(model_id, data)
        end_time = pd.Timestamp.now()
        return (end_time - start_time).total_seconds() * 1000

    def _get_performance_recommendations(self, metrics: Dict[str, float]
        ) ->List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        if metrics['p95_latency_ms'] > self.max_latency_ms:
            recommendations.append(
                f"High latency detected (P95: {metrics['p95_latency_ms']:.2f}ms). Consider model optimization or hardware scaling."
                )
        if metrics['error_rate'] > 0.01:
            recommendations.append(
                f"High error rate detected ({metrics['error_rate']:.2%}). Investigate error patterns and implement retry logic."
                )
        return recommendations
