"""
Indicator Tester Module

This module provides a comprehensive framework for testing technical indicators
in the feature store service. It offers utilities for:
- Unit testing individual indicators
- Comparing indicator results against known values
- Validating indicator behavior with edge cases
- Performance benchmarking for indicators
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Callable, Union, Optional, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndicatorTester:
    """
    A class for testing technical indicators accuracy, performance, and behavior.
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Initialize the indicator tester.
        
        Args:
            reference_data (pd.DataFrame, optional): Historical price data to use for testing
        """
        self.reference_data = reference_data
        self.test_results = {}
        
    def set_reference_data(self, data: pd.DataFrame) -> None:
        """
        Set the reference data used for testing indicators.
        
        Args:
            data (pd.DataFrame): Historical price data
        """
        self.reference_data = data
        
    def test_indicator_accuracy(self, 
                               indicator_func: Callable, 
                               expected_results: pd.DataFrame,
                               params: Dict[str, Any] = None,
                               tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Test an indicator's calculation accuracy against expected results.
        
        Args:
            indicator_func: Function that implements the indicator
            expected_results: DataFrame with expected indicator values
            params: Parameters to pass to the indicator function
            tolerance: Acceptable difference between actual and expected results
            
        Returns:
            Dictionary containing test results and metrics
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
            
        if params is None:
            params = {}
            
        start_time = time.time()
        actual_results = indicator_func(self.reference_data, **params)
        execution_time = time.time() - start_time
        
        # Compare results
        if not isinstance(actual_results, pd.DataFrame):
            actual_results = pd.DataFrame(actual_results)
            
        # Check shapes match
        shape_match = actual_results.shape == expected_results.shape
        
        # Calculate difference and check if within tolerance
        try:
            diff = np.abs(actual_results.values - expected_results.values)
            accuracy = np.mean(diff <= tolerance)
            max_diff = np.max(diff)
            
            passed = shape_match and accuracy > 0.999  # Allow for minimal floating point differences
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            accuracy = 0.0
            max_diff = float('inf')
            passed = False
            
        result = {
            "indicator_name": indicator_func.__name__,
            "passed": passed,
            "accuracy": accuracy,
            "execution_time": execution_time,
            "max_difference": max_diff,
            "parameters": params,
            "shape_match": shape_match
        }
        
        self.test_results[indicator_func.__name__] = result
        return result
    
    def test_edge_cases(self, indicator_func: Callable, edge_cases: List[Dict]) -> Dict[str, Any]:
        """
        Test an indicator with various edge cases.
        
        Args:
            indicator_func: Function that implements the indicator
            edge_cases: List of dictionaries with edge case data and expected behavior
            
        Returns:
            Dictionary with edge case test results
        """
        results = {}
        
        for i, case in enumerate(edge_cases):
            case_data = case.get("data")
            case_params = case.get("params", {})
            expected_behavior = case.get("expected_behavior")
            case_name = case.get("name", f"edge_case_{i}")
            
            try:
                result = indicator_func(case_data, **case_params)
                error = None
                result_matches = self._check_expected_behavior(result, expected_behavior)
            except Exception as e:
                result = None
                error = str(e)
                result_matches = expected_behavior == "error"
                
            results[case_name] = {
                "passed": result_matches,
                "error": error,
                "case_description": case.get("description", ""),
                "expected_behavior": expected_behavior
            }
            
        self.test_results[f"{indicator_func.__name__}_edge_cases"] = results
        return results
    
    def benchmark_performance(self, 
                             indicator_func: Callable, 
                             params_list: List[Dict[str, Any]], 
                             data_sizes: List[int],
                             iterations: int = 5) -> Dict[str, Any]:
        """
        Benchmark indicator performance with different parameters and data sizes.
        
        Args:
            indicator_func: Function that implements the indicator
            params_list: List of parameter dictionaries to test
            data_sizes: List of data sizes to test with
            iterations: Number of iterations to run for each configuration
            
        Returns:
            Dictionary with performance benchmarking results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
            
        results = {}
        
        for size in data_sizes:
            # Truncate data to specified size
            data = self.reference_data.iloc[:size]
            size_results = {}
            
            for param_set in params_list:
                param_key = "_".join([f"{k}_{v}" for k, v in param_set.items()])
                execution_times = []
                
                for _ in range(iterations):
                    start_time = time.time()
                    _ = indicator_func(data, **param_set)
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                
                size_results[param_key] = {
                    "mean_time": np.mean(execution_times),
                    "min_time": np.min(execution_times),
                    "max_time": np.max(execution_times),
                    "std_time": np.std(execution_times),
                    "parameters": param_set
                }
                
            results[f"size_{size}"] = size_results
            
        benchmark_id = f"{indicator_func.__name__}_benchmark"
        self.test_results[benchmark_id] = results
        return results
    
    def parallel_benchmark(self, 
                         indicator_func: Callable, 
                         params: Dict[str, Any], 
                         num_processes: List[int],
                         data_size: int) -> Dict[str, Any]:
        """
        Benchmark indicator performance when run in parallel.
        
        Args:
            indicator_func: Function that implements the indicator
            params: Parameters to pass to the indicator function
            num_processes: List of process counts to benchmark with
            data_size: Size of data to use for testing
            
        Returns:
            Dictionary with parallel benchmarking results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data first.")
            
        data = self.reference_data.iloc[:data_size]
        results = {}
        
        def _run_indicator(_):
            return indicator_func(data, **params)
        
        for processes in num_processes:
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=processes) as executor:
                # Run the indicator calculation using specified number of processes
                _ = list(executor.map(_run_indicator, range(processes)))
                
            execution_time = time.time() - start_time
            
            results[f"processes_{processes}"] = {
                "total_time": execution_time,
                "time_per_process": execution_time / processes,
                "num_processes": processes
            }
            
        benchmark_id = f"{indicator_func.__name__}_parallel_benchmark"
        self.test_results[benchmark_id] = results
        return results
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive test report.
        
        Args:
            output_path: Path to save the report to
            
        Returns:
            Report as a string
        """
        report = ["# Indicator Test Report\n"]
        report.append(f"Generated on: {pd.Timestamp.now()}\n")
        
        # Accuracy tests
        report.append("## Accuracy Tests\n")
        accuracy_tests = {k: v for k, v in self.test_results.items() 
                          if not (k.endswith("_edge_cases") or k.endswith("_benchmark"))}
        
        for name, result in accuracy_tests.items():
            status = "PASSED" if result.get("passed", False) else "FAILED"
            report.append(f"### {name}: {status}\n")
            report.append(f"- Accuracy: {result.get('accuracy', 0):.4f}\n")
            report.append(f"- Execution time: {result.get('execution_time', 0):.6f} seconds\n")
            report.append(f"- Max difference: {result.get('max_difference', 0):.8f}\n")
            report.append(f"- Parameters: {result.get('parameters', {})}\n")
            
        # Edge case tests
        edge_case_tests = {k: v for k, v in self.test_results.items() if k.endswith("_edge_cases")}
        if edge_case_tests:
            report.append("\n## Edge Case Tests\n")
            
            for test_name, cases in edge_case_tests.items():
                indicator_name = test_name.replace("_edge_cases", "")
                report.append(f"### {indicator_name}\n")
                
                for case_name, case_result in cases.items():
                    status = "PASSED" if case_result.get("passed", False) else "FAILED"
                    report.append(f"- {case_name}: {status}\n")
                    report.append(f"  - Description: {case_result.get('case_description', '')}\n")
                    if case_result.get("error"):
                        report.append(f"  - Error: {case_result.get('error')}\n")
                        
        # Performance benchmarks
        benchmark_tests = {k: v for k, v in self.test_results.items() if k.endswith("_benchmark")}
        if benchmark_tests:
            report.append("\n## Performance Benchmarks\n")
            
            for benchmark_name, results in benchmark_tests.items():
                indicator_name = benchmark_name.replace("_benchmark", "")
                report.append(f"### {indicator_name}\n")
                
                is_parallel = benchmark_name.endswith("_parallel_benchmark")
                
                if is_parallel:
                    report.append("#### Parallel Execution Results\n")
                    report.append("| Processes | Total Time (s) | Time per Process (s) |\n")
                    report.append("|-----------|---------------|---------------------|\n")
                    
                    for key, data in results.items():
                        processes = data.get("num_processes", 0)
                        total_time = data.get("total_time", 0)
                        per_process = data.get("time_per_process", 0)
                        report.append(f"| {processes} | {total_time:.6f} | {per_process:.6f} |\n")
                else:
                    for size_key, size_data in results.items():
                        size = size_key.replace("size_", "")
                        report.append(f"#### Data Size: {size}\n")
                        report.append("| Parameters | Mean Time (s) | Min Time (s) | Max Time (s) | Std Dev |\n")
                        report.append("|-----------|--------------|-------------|-------------|--------|\n")
                        
                        for param_key, data in size_data.items():
                            mean = data.get("mean_time", 0)
                            min_t = data.get("min_time", 0)
                            max_t = data.get("max_time", 0)
                            std = data.get("std_time", 0)
                            report.append(f"| {param_key} | {mean:.6f} | {min_t:.6f} | {max_t:.6f} | {std:.6f} |\n")
        
        report_text = "".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
                
        return report_text
    
    def _check_expected_behavior(self, result: Any, expected_behavior: Any) -> bool:
        """
        Check if a result matches the expected behavior.
        
        Args:
            result: The actual result from the indicator
            expected_behavior: The expected behavior or value
            
        Returns:
            Boolean indicating if the result matches the expected behavior
        """
        if expected_behavior == "error":
            return False  # No error was raised
        elif expected_behavior == "null" or expected_behavior is None:
            return result is None or (isinstance(result, pd.DataFrame) and result.empty)
        elif expected_behavior == "not_null":
            return result is not None and not (isinstance(result, pd.DataFrame) and result.empty)
        elif isinstance(expected_behavior, dict) and "type" in expected_behavior:
            # Check specific type behavior
            if expected_behavior["type"] == "nan_count":
                # Check if NaN count is within expected range
                if not isinstance(result, pd.DataFrame):
                    return False
                actual_nan_count = result.isna().sum().sum()
                min_nan = expected_behavior.get("min", 0)
                max_nan = expected_behavior.get("max", float('inf'))
                return min_nan <= actual_nan_count <= max_nan
        else:
            # Try direct comparison
            try:
                if isinstance(result, pd.DataFrame) and isinstance(expected_behavior, pd.DataFrame):
                    return result.equals(expected_behavior)
                else:
                    return result == expected_behavior
            except:
                return False
