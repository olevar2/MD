"""
Integration Tests for Enhanced ML Integration Components

This module provides comprehensive integration tests for the enhanced
visualization, optimization, and stress testing capabilities.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from fastapi.testclient import TestClient

from ml_integration_service.visualization.model_performance_viz import ModelPerformanceVisualizer
from ml_integration_service.optimization.advanced_optimization import (
    RegimeAwareOptimizer,
    MultiObjectiveOptimizer
)
from ml_integration_service.stress_testing.model_stress_tester import (
    ModelRobustnessTester,
    SensitivityAnalyzer,
    LoadTester
)
from ml_integration_service.services.data_service import DataService
from ml_integration_service.monitoring.metrics_collector import MetricsCollector

# Test fixtures
@pytest.fixture
def test_client():
    """Create a test client."""
    from ml_integration_service.main import app
    return TestClient(app)

@pytest.fixture
def sample_performance_data():
    """Generate sample performance data."""
    dates = pd.date_range(start='2025-01-01', end='2025-04-01', freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'actual': np.random.normal(0, 1, len(dates)),
        'predicted': np.random.normal(0, 1, len(dates)),
        'confidence_upper': np.random.normal(1, 0.2, len(dates)),
        'confidence_lower': np.random.normal(-1, 0.2, len(dates))
    })

@pytest.fixture
def sample_feature_importance():
    """Generate sample feature importance data."""
    features = ['feature_' + str(i) for i in range(10)]
    return {f: np.random.random() for f in features}

@pytest.fixture
def sample_regime_data():
    """Generate sample regime performance data."""
    regimes = ['bull', 'bear', 'sideways']
    metrics = ['return', 'sharpe', 'sortino']
    data = np.random.rand(len(metrics), len(regimes))
    return pd.DataFrame(data, index=metrics, columns=regimes)

# Visualization Tests
class TestVisualization:
    """Test visualization components."""

    def test_performance_plot_generation(self, sample_performance_data):
        """Test performance plot generation."""
        viz = ModelPerformanceVisualizer()
        plot = viz.create_performance_plot(sample_performance_data)
        
        assert isinstance(plot, dict)
        assert 'data' in plot
        assert 'layout' in plot
        assert len(plot['data']) >= 2  # At least actual and predicted traces

    def test_feature_importance_chart(self, sample_feature_importance):
        """Test feature importance visualization."""
        viz = ModelPerformanceVisualizer()
        plot = viz.create_feature_importance_chart(sample_feature_importance)
        
        assert isinstance(plot, dict)
        assert 'data' in plot
        assert len(plot['data']) == 1  # Single bar trace

    def test_regime_performance_heatmap(self, sample_regime_data):
        """Test regime performance visualization."""
        viz = ModelPerformanceVisualizer()
        plot = viz.create_regime_performance_heatmap(sample_regime_data)
        
        assert isinstance(plot, dict)
        assert 'data' in plot
        assert plot['data'][0]['type'] == 'heatmap'

# Optimization Tests
class TestOptimization:
    """Test optimization components."""

    def test_regime_aware_optimization(self):
        """Test regime-aware parameter optimization."""
        optimizer = RegimeAwareOptimizer()
        
        # Sample data
        parameters = pd.DataFrame({
            'param1': np.random.rand(100),
            'param2': np.random.rand(100)
        })
        performance = pd.Series(np.random.rand(100))
        regimes = pd.Series(['bull'] * 50 + ['bear'] * 50)
        
        optimizer.fit(parameters, performance, regimes)
        result = optimizer.optimize(
            param_bounds={'param1': (0, 1), 'param2': (0, 1)},
            current_regime='bull'
        )
        
        assert isinstance(result, dict)
        assert all(0 <= v <= 1 for v in result.values())

    def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        optimizer = MultiObjectiveOptimizer()
        
        def obj1(x):
    """
    Obj1.
    
    Args:
        x: Description of x
    
    """
 return -((x['param1'] - 0.5) ** 2)
        def obj2(x):
    """
    Obj2.
    
    Args:
        x: Description of x
    
    """
 return -((x['param2'] - 0.5) ** 2)
        
        results = optimizer.optimize(
            param_bounds={'param1': (0, 1), 'param2': (0, 1)},
            objectives=[obj1, obj2]
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

# Stress Testing Tests
class TestStressTesting:
    """Test stress testing components."""

    def test_model_robustness(self, sample_performance_data):
        """Test model robustness testing."""
        tester = ModelRobustnessTester()
        
        def mock_predictor(data):
    """
    Mock predictor.
    
    Args:
        data: Description of data
    
    """

            return np.random.normal(0, 1, len(data))
        
        results = tester.test_model_robustness(
            mock_predictor,
            sample_performance_data,
            [lambda x, y: np.mean((x - y) ** 2)]  # MSE metric
        )
        
        assert isinstance(results, dict)
        assert 'n_scenarios_tested' in results
        assert 'avg_metrics' in results

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        analyzer = SensitivityAnalyzer()
        
        def mock_predictor(data, params):
    """
    Mock predictor.
    
    Args:
        data: Description of data
        params: Description of params
    
    """

            return np.random.normal(params['param1'], params['param2'], len(data))
        
        results = analyzer.analyze_parameter_sensitivity(
            mock_predictor,
            {'param1': 0, 'param2': 1},
            {'param1': (-1, 1), 'param2': (0.1, 2)},
            pd.DataFrame(np.random.rand(100, 2)),
            lambda x, y: np.mean((x - y) ** 2)
        )
        
        assert isinstance(results, dict)
        assert all(param in results for param in ['param1', 'param2'])

# API Integration Tests
class TestAPIIntegration:
    """Test API endpoints."""

    def test_visualization_endpoint(self, test_client, sample_performance_data):
        """Test visualization API endpoint."""
        response = test_client.post(
            "/api/v1/enhanced/visualize/performance",
            json={
                "model_id": "test_model",
                "start_date": "2025-01-01T00:00:00",
                "end_date": "2025-04-01T00:00:00"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "visualization_data" in data
        assert "metadata" in data

    def test_optimization_endpoint(self, test_client):
        """Test optimization API endpoint."""
        response = test_client.post(
            "/api/v1/enhanced/optimize/regime-aware",
            json={
                "strategy_id": "test_strategy",
                "parameter_bounds": {
                    "param1": [0, 1],
                    "param2": [0, 1]
                },
                "market_regime": "bull"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "optimal_parameters" in data

    def test_stress_test_endpoint(self, test_client):
        """Test stress testing API endpoint."""
        response = test_client.post(
            "/api/v1/enhanced/stress-test/robustness",
            json={
                "model_id": "test_model",
                "test_scenario": "robustness",
                "test_parameters": {
                    "n_scenarios": 100,
                    "volatility_multiplier": 2.0
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "test_results" in data
