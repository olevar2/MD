"""
Integration tests for multi-asset performance analytics functionality.

This module contains tests that verify the performance analytics module works
correctly across different asset classes and properly integrates with other services.
"""
import pytest
import json
from datetime import datetime, timedelta
import uuid
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from core.main_1 import app
from core.position import AssetClass
from portfolio_management_service.multi_asset.performance_analytics import PerformanceAnalytics
from portfolio_management_service.services.performance_service import PerformanceService


@pytest.fixture
def test_client():
    """Create a test client for API testing"""
    return TestClient(app)


@pytest.fixture
def mock_performance_service():
    """Create a mock performance service for testing"""
    with patch('portfolio_management_service.multi_asset.performance_analytics.PerformanceService') as mock_service:
        service_instance = MagicMock()
        mock_service.return_value = service_instance
        yield service_instance


@pytest.fixture
def mock_portfolio_service():
    """Create a mock portfolio service for testing"""
    with patch('portfolio_management_service.multi_asset.performance_analytics.PortfolioService') as mock_service:
        service_instance = MagicMock()
        mock_service.return_value = service_instance
        yield service_instance


@pytest.fixture
def performance_analytics(mock_performance_service, mock_portfolio_service):
    """Create a performance analytics instance with mocked dependencies"""
    return PerformanceAnalytics(
        performance_service=mock_performance_service,
        portfolio_service=mock_portfolio_service
    )


@pytest.fixture
def sample_account_id():
    """Generate a sample account ID for testing"""
    return f"test-account-{uuid.uuid4()}"


@pytest.fixture
def sample_performance_data():
    """Create sample performance data for testing"""
    today = datetime.now()
    return {
        "account_id": "test-account",
        "timeframe": "1d",
        "overall_metrics": {
            "total_return": 8.5,
            "sharpe_ratio": 1.2,
            "max_drawdown": 4.3,
            "volatility": 2.8,
            "win_rate": 65.0,
            "profit_factor": 1.85
        },
        "by_asset_class": {
            "forex": {
                "total_return": 5.2,
                "sharpe_ratio": 0.95,
                "max_drawdown": 3.1,
                "volatility": 2.1,
                "win_rate": 62.0,
                "profit_factor": 1.75
            },
            "crypto": {
                "total_return": 12.8,
                "sharpe_ratio": 1.4,
                "max_drawdown": 8.7,
                "volatility": 6.5,
                "win_rate": 58.0,
                "profit_factor": 1.95
            },
            "stocks": {
                "total_return": 7.3,
                "sharpe_ratio": 1.1,
                "max_drawdown": 3.5,
                "volatility": 3.2,
                "win_rate": 70.0,
                "profit_factor": 2.1
            }
        },
        "daily_returns": [
            {
                "date": today - timedelta(days=5),
                "return": 0.5,
                "by_asset_class": {"forex": 0.3, "crypto": 1.2, "stocks": 0.4}
            },
            {
                "date": today - timedelta(days=4),
                "return": -0.2,
                "by_asset_class": {"forex": -0.1, "crypto": -0.8, "stocks": 0.2}
            },
            {
                "date": today - timedelta(days=3),
                "return": 0.8,
                "by_asset_class": {"forex": 0.4, "crypto": 2.1, "stocks": 0.3}
            },
            {
                "date": today - timedelta(days=2),
                "return": 0.3,
                "by_asset_class": {"forex": 0.2, "crypto": 0.5, "stocks": 0.2}
            },
            {
                "date": today - timedelta(days=1),
                "return": -0.1,
                "by_asset_class": {"forex": 0.1, "crypto": -0.7, "stocks": 0.0}
            }
        ]
    }


class TestPerformanceAnalytics:
    """Test suite for performance analytics functionality"""

    def test_calculate_performance_metrics(self, performance_analytics, mock_portfolio_service, mock_performance_service, sample_account_id):
        """Test calculation of performance metrics across asset classes"""
        # Setup - Sample positions
        positions = [
            {
                "id": "pos1",
                "symbol": "EURUSD",
                "asset_class": AssetClass.FOREX,
                "entry_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "entry_price": 1.1000,
                "current_price": 1.1500,
                "quantity": 10000,
                "unrealized_pl": 500.0
            },
            {
                "id": "pos2",
                "symbol": "AAPL",
                "asset_class": AssetClass.STOCKS,
                "entry_date": (datetime.now() - timedelta(days=25)).isoformat(),
                "entry_price": 150.0,
                "current_price": 165.0,
                "quantity": 20,
                "unrealized_pl": 300.0
            },
            {
                "id": "pos3",
                "symbol": "BTCUSDT",
                "asset_class": AssetClass.CRYPTO,
                "entry_date": (datetime.now() - timedelta(days=20)).isoformat(),
                "entry_price": 40000.0,
                "current_price": 45000.0,
                "quantity": 0.5,
                "unrealized_pl": 2500.0
            }
        ]
        
        # Mock position history - This would normally be retrieved from a database
        position_history = []
        for pos in positions:
            position_history.append({
                **pos,
                "history": [
                    {"date": datetime.now() - timedelta(days=i), "price": pos["entry_price"] * (1 + 0.01 * i)}
                    for i in range(10)
                ]
            })
        
        mock_portfolio_service.get_positions.return_value = positions
        mock_performance_service.get_position_history.return_value = position_history
        
        # Sample historical returns
        returns_data = {
            "daily_returns": [
                {"date": (datetime.now() - timedelta(days=i)).isoformat(), "return": 0.5 * (i % 3 - 1)}
                for i in range(30)
            ]
        }
        mock_performance_service.get_historical_returns.return_value = returns_data
        
        # Execute
        timeframe = "1d"
        lookback = 30
        result = performance_analytics.calculate_performance_metrics(sample_account_id, timeframe, lookback)
        
        # Verify structure of results
        assert "overall_metrics" in result
        overall = result["overall_metrics"]
        assert "total_return" in overall
        assert "sharpe_ratio" in overall
        assert "max_drawdown" in overall
        assert "volatility" in overall
        assert "win_rate" in overall
        assert "profit_factor" in overall
        
        # Verify asset class breakdown
        assert "by_asset_class" in result
        by_asset_class = result["by_asset_class"]
        assert AssetClass.FOREX.value in by_asset_class
        assert AssetClass.STOCKS.value in by_asset_class
        assert AssetClass.CRYPTO.value in by_asset_class
        
        # Verify service calls
        mock_portfolio_service.get_positions.assert_called_once_with(sample_account_id)
        mock_performance_service.get_position_history.assert_called_once()
        mock_performance_service.get_historical_returns.assert_called_once_with(sample_account_id, lookback)

    def test_calculate_comparison_metrics(self, performance_analytics, mock_performance_service, sample_account_id):
        """Test calculation of comparison metrics against benchmarks"""
        # Setup - Sample benchmark data
        benchmarks = {
            AssetClass.FOREX: {"symbol": "USDX", "return": 1.2},
            AssetClass.STOCKS: {"symbol": "SPY", "return": 8.5},
            AssetClass.CRYPTO: {"symbol": "BTC", "return": 15.3}
        }
        
        # Mock performance data for account
        account_performance = {
            "by_asset_class": {
                AssetClass.FOREX.value: {"total_return": 3.5},
                AssetClass.STOCKS.value: {"total_return": 7.2},
                AssetClass.CRYPTO.value: {"total_return": 20.1}
            }
        }
        
        with patch.object(performance_analytics, 'calculate_performance_metrics') as mock_perf:
            mock_perf.return_value = account_performance
            
            # Mock benchmark data retrieval
            mock_performance_service.get_benchmark_performance.side_effect = lambda asset_class, *args: benchmarks[asset_class]
            
            # Execute
            timeframe = "1m"
            result = performance_analytics.calculate_comparison_metrics(sample_account_id, timeframe)
            
            # Verify
            assert "comparisons" in result
            comparisons = result["comparisons"]
            
            for asset_class in [AssetClass.FOREX, AssetClass.STOCKS, AssetClass.CRYPTO]:
                assert asset_class.value in comparisons
                asset_comp = comparisons[asset_class.value]
                assert "benchmark_symbol" in asset_comp
                assert "benchmark_return" in asset_comp
                assert "relative_performance" in asset_comp
                
                # Calculate expected relative performance
                account_return = account_performance["by_asset_class"][asset_class.value]["total_return"]
                benchmark_return = benchmarks[asset_class]["return"]
                expected_relative = account_return - benchmark_return
                
                assert abs(asset_comp["relative_performance"] - expected_relative) < 0.001
            
            # Verify service calls
            mock_perf.assert_called_once_with(sample_account_id, timeframe, None)
            assert mock_performance_service.get_benchmark_performance.call_count == 3

    def test_api_get_performance_metrics(self, test_client, sample_performance_data, sample_account_id):
        """Test API endpoint for getting performance metrics"""
        with patch('portfolio_management_service.api.v1.performance_api.performance_analytics') as mock_analytics:
            # Setup
            mock_analytics.calculate_performance_metrics.return_value = sample_performance_data
            
            # Execute
            timeframe = "1d"
            lookback = 30
            response = test_client.get(f"/api/v1/performance/{sample_account_id}?timeframe={timeframe}&lookback={lookback}")
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            
            assert "overall_metrics" in result
            assert "by_asset_class" in result
            assert "daily_returns" in result
            
            # Verify service call
            mock_analytics.calculate_performance_metrics.assert_called_once_with(sample_account_id, timeframe, lookback)

    def test_api_get_comparison_metrics(self, test_client, sample_account_id):
        """Test API endpoint for getting comparison metrics"""
        with patch('portfolio_management_service.api.v1.performance_api.performance_analytics') as mock_analytics:
            # Setup
            comparison_data = {
                "comparisons": {
                    "forex": {
                        "benchmark_symbol": "USDX",
                        "benchmark_return": 1.2,
                        "relative_performance": 2.3
                    },
                    "stocks": {
                        "benchmark_symbol": "SPY",
                        "benchmark_return": 8.5,
                        "relative_performance": -1.3
                    },
                    "crypto": {
                        "benchmark_symbol": "BTC",
                        "benchmark_return": 15.3,
                        "relative_performance": 4.8
                    }
                }
            }
            
            mock_analytics.calculate_comparison_metrics.return_value = comparison_data
            
            # Execute
            timeframe = "1m"
            response = test_client.get(f"/api/v1/performance/{sample_account_id}/comparison?timeframe={timeframe}")
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            
            assert "comparisons" in result
            assert "forex" in result["comparisons"]
            assert "stocks" in result["comparisons"]
            assert "crypto" in result["comparisons"]
            
            # Verify service call
            mock_analytics.calculate_comparison_metrics.assert_called_once_with(sample_account_id, timeframe)


class TestAssetClassComparison:
    """Test suite for asset class performance comparison functionality"""

    def test_asset_class_performance_ranking(self, performance_analytics, mock_performance_service):
        """Test ranking of asset classes by performance metrics"""
        # Setup
        accounts = ["account1", "account2", "account3"]
        
        # Mock performance data for multiple accounts
        performance_data = {}
        for account in accounts:
            performance_data[account] = {
                "by_asset_class": {
                    "forex": {"total_return": 3.5 * (accounts.index(account) + 1),
                              "sharpe_ratio": 0.8 + 0.2 * accounts.index(account)},
                    "stocks": {"total_return": 7.2 * (accounts.index(account) + 1),
                               "sharpe_ratio": 1.1 + 0.1 * accounts.index(account)},
                    "crypto": {"total_return": 10.1 * (accounts.index(account) + 1),
                               "sharpe_ratio": 1.4 + 0.3 * accounts.index(account)}
                }
            }
        
        mock_performance_service.get_all_accounts_performance.return_value = performance_data
        
        # Execute
        timeframe = "1m"
        metric = "total_return"
        result = performance_analytics.rank_asset_classes_by_performance(timeframe, metric)
        
        # Verify
        assert "rankings" in result
        rankings = result["rankings"]
        
        # Check that rankings are in descending order of the metric
        for i in range(len(rankings) - 1):
            assert rankings[i]["average_value"] >= rankings[i+1]["average_value"]
        
        # Check that all asset classes are represented
        asset_classes = [ranking["asset_class"] for ranking in rankings]
        assert "forex" in asset_classes
        assert "stocks" in asset_classes
        assert "crypto" in asset_classes
        
        # Verify service call
        mock_performance_service.get_all_accounts_performance.assert_called_once_with(timeframe)

    def test_api_get_asset_class_rankings(self, test_client):
        """Test API endpoint for getting asset class performance rankings"""
        with patch('portfolio_management_service.api.v1.performance_api.performance_analytics') as mock_analytics:
            # Setup
            ranking_data = {
                "rankings": [
                    {
                        "asset_class": "crypto",
                        "average_value": 30.3,
                        "metric": "total_return"
                    },
                    {
                        "asset_class": "stocks",
                        "average_value": 21.6,
                        "metric": "total_return"
                    },
                    {
                        "asset_class": "forex",
                        "average_value": 10.5,
                        "metric": "total_return"
                    }
                ]
            }
            
            mock_analytics.rank_asset_classes_by_performance.return_value = ranking_data
            
            # Execute
            timeframe = "1m"
            metric = "total_return"
            response = test_client.get(f"/api/v1/performance/asset-classes/ranking?timeframe={timeframe}&metric={metric}")
            
            # Verify
            assert response.status_code == 200
            result = response.json()
            
            assert "rankings" in result
            assert len(result["rankings"]) == 3
            
            # Verify service call
            mock_analytics.rank_asset_classes_by_performance.assert_called_once_with(timeframe, metric)
