"""
Portfolio Risk Calculator for Risk Management.

Implements comprehensive portfolio risk metrics calculation,
including advanced risk measures and portfolio analytics.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .risk_manager import RiskManager, Position
from .stress_testing import StressTestingEngine, StressScenario

logger = logging.getLogger(__name__)

@dataclass
class PortfolioMetrics:
    """Container for portfolio risk metrics."""
    timestamp: datetime
    total_value: float
    exposure: float
    leverage: float
    margin_usage: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown: float
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    correlation_matrix: Optional[pd.DataFrame] = None
    stress_test_results: Optional[Dict[str, Any]] = None

class PortfolioRiskCalculator:
    """
    Advanced portfolio risk metrics calculator.
    
    Calculates comprehensive risk metrics including VaR,
    Expected Shortfall, correlation analysis, and scenario impacts.
    """
    
    def __init__(
        self,
        risk_manager: RiskManager,
        stress_testing_engine: StressTestingEngine,
        historical_data_provider: Any,  # Interface to historical data
        lookback_days: int = 252,  # One trading year
        var_confidence_level: float = 0.95,
        risk_free_rate: float = 0.02  # 2% annual risk-free rate
    ):
    """
      init  .
    
    Args:
        risk_manager: Description of risk_manager
        stress_testing_engine: Description of stress_testing_engine
        historical_data_provider: Description of historical_data_provider
        # Interface to historical data
        lookback_days: Description of # Interface to historical data
        lookback_days
        # One trading year
        var_confidence_level: Description of # One trading year
        var_confidence_level
        risk_free_rate: Description of risk_free_rate
    
    """

        self.risk_manager = risk_manager
        self.stress_testing_engine = stress_testing_engine
        self.historical_data = historical_data_provider
        self.lookback_days = lookback_days
        self.var_confidence_level = var_confidence_level
        self.risk_free_rate = risk_free_rate
        self.metrics_history: List[PortfolioMetrics] = []
        
    def calculate_portfolio_metrics(
        self,
        current_prices: Dict[str, float]
    ) -> PortfolioMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        # Get basic portfolio metrics
        basic_metrics = self.risk_manager.get_portfolio_metrics()
        
        # Calculate advanced risk metrics
        var_95, var_99 = self._calculate_var_metrics(current_prices)
        expected_shortfall = self._calculate_expected_shortfall(current_prices)
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix()
        
        # Run stress tests
        stress_results = self.stress_testing_engine.run_all_stress_tests(
            current_prices
        )
        
        # Create metrics object
        metrics = PortfolioMetrics(
            timestamp=datetime.utcnow(),
            total_value=basic_metrics['current_balance'],
            exposure=basic_metrics['total_exposure'],
            leverage=basic_metrics['exposure_ratio'],
            margin_usage=self._calculate_margin_usage(),
            unrealized_pnl=basic_metrics.get('unrealized_pnl', 0.0),
            realized_pnl=0.0,  # Would need transaction history
            drawdown=basic_metrics['drawdown'],
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            correlation_matrix=correlation_matrix,
            stress_test_results=stress_results
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        return metrics
        
    def _calculate_var_metrics(
        self,
        current_prices: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate Value at Risk at 95% and 99% confidence levels."""
        portfolio_returns = self._calculate_historical_returns()
        
        if len(portfolio_returns) < 2:
            return 0.0, 0.0
            
        var_95 = np.percentile(portfolio_returns, 5)  # 95% VaR
        var_99 = np.percentile(portfolio_returns, 1)  # 99% VaR
        
        # Convert to monetary values
        current_value = self.risk_manager.current_balance
        var_95 = abs(var_95 * current_value)
        var_99 = abs(var_99 * current_value)
        
        return var_95, var_99
        
    def _calculate_expected_shortfall(
        self,
        current_prices: Dict[str, float]
    ) -> float:
        """Calculate Expected Shortfall (CVaR)."""
        portfolio_returns = self._calculate_historical_returns()
        
        if len(portfolio_returns) < 2:
            return 0.0
            
        # Sort returns
        sorted_returns = np.sort(portfolio_returns)
        
        # Calculate cutoff index for VaR
        cutoff_index = int(len(sorted_returns) * 0.05)  # 95% confidence
        
        # Calculate expected shortfall
        worst_returns = sorted_returns[:cutoff_index]
        es = np.mean(worst_returns) if len(worst_returns) > 0 else 0.0
        
        return abs(es * self.risk_manager.current_balance)
        
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe Ratio."""
        returns = self._calculate_historical_returns()
        
        if len(returns) < 2:
            return 0.0
            
        # Annualize metrics
        avg_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        if volatility == 0:
            return 0.0
            
        return (avg_return - self.risk_free_rate) / volatility
        
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino Ratio."""
        returns = self._calculate_historical_returns()
        
        if len(returns) < 2:
            return 0.0
            
        # Calculate downside returns
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside volatility
            
        # Annualize metrics
        avg_return = np.mean(returns) * 252
        downside_vol = np.std(downside_returns) * np.sqrt(252)
        
        if downside_vol == 0:
            return 0.0
            
        return (avg_return - self.risk_free_rate) / downside_vol
        
    def _calculate_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix of portfolio positions."""
        if not self.risk_manager.positions:
            return None
            
        # Get historical prices for all symbols
        symbols = list(self.risk_manager.positions.keys())
        prices = {}
        
        for symbol in symbols:
            # This would need actual historical data implementation
            prices[symbol] = self.historical_data.get_prices(
                symbol,
                lookback_days=self.lookback_days
            )
            
        if not prices:
            return None
            
        # Calculate returns
        returns_data = {}
        for symbol, price_data in prices.items():
            returns_data[symbol] = pd.Series(price_data).pct_change().dropna()
            
        # Create correlation matrix
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
        
    def _calculate_margin_usage(self) -> float:
        """Calculate current margin usage."""
        total_margin = 0.0
        
        for position in self.risk_manager.positions.values():
            # Basic margin calculation (would need actual broker margin rules)
            margin = position.size * position.entry_price / position.leverage
            total_margin += margin
            
        return total_margin / self.risk_manager.current_balance
        
    def _calculate_historical_returns(self) -> np.ndarray:
        """Calculate historical returns of the portfolio."""
        if not self.metrics_history:
            return np.array([])
            
        values = [m.total_value for m in self.metrics_history]
        returns = np.diff(values) / values[:-1]
        
        return returns
        
    def get_risk_recommendations(
        self,
        metrics: PortfolioMetrics
    ) -> List[Dict[str, Any]]:
        """Generate risk management recommendations."""
        recommendations = []
        
        # Check leverage
        if metrics.leverage > 2.0:
            recommendations.append({
                'severity': 'high',
                'issue': 'High Leverage',
                'metric': 'leverage',
                'value': metrics.leverage,
                'threshold': 2.0,
                'action': 'Consider reducing position sizes or closing some positions'
            })
            
        # Check drawdown
        if metrics.drawdown > self.risk_manager.max_drawdown * 0.8:
            recommendations.append({
                'severity': 'critical',
                'issue': 'Approaching Max Drawdown',
                'metric': 'drawdown',
                'value': metrics.drawdown,
                'threshold': self.risk_manager.max_drawdown,
                'action': 'Consider closing positions or reducing exposure'
            })
            
        # Check VaR utilization
        var_limit = self.risk_manager.current_balance * 0.1  # 10% VaR limit
        if metrics.var_95 > var_limit:
            recommendations.append({
                'severity': 'medium',
                'issue': 'High Value at Risk',
                'metric': 'var_95',
                'value': metrics.var_95,
                'threshold': var_limit,
                'action': 'Consider portfolio rebalancing or hedging'
            })
            
        # Check correlation
        if metrics.correlation_matrix is not None:
            high_corr = (metrics.correlation_matrix > 0.8).sum().sum()
            if high_corr > 2:  # More than 2 highly correlated pairs
                recommendations.append({
                    'severity': 'medium',
                    'issue': 'High Portfolio Correlation',
                    'metric': 'correlation',
                    'value': high_corr,
                    'threshold': 2,
                    'action': 'Consider diversifying into less correlated pairs'
                })
                
        # Check margin usage
        if metrics.margin_usage > 0.8:  # 80% margin usage
            recommendations.append({
                'severity': 'high',
                'issue': 'High Margin Usage',
                'metric': 'margin_usage',
                'value': metrics.margin_usage,
                'threshold': 0.8,
                'action': 'Reduce positions to free up margin'
            })
            
        # Add stress test based recommendations
        if metrics.stress_test_results:
            for scenario, results in metrics.stress_test_results.items():
                if results['value_at_risk'] > self.risk_manager.current_balance * 0.3:
                    recommendations.append({
                        'severity': 'high',
                        'issue': f'High Risk Under {scenario}',
                        'metric': 'stress_test',
                        'value': results['value_at_risk'],
                        'threshold': self.risk_manager.current_balance * 0.3,
                        'action': 'Review portfolio for scenario resilience'
                    })
                    
        return recommendations
        
    def get_historical_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get historical risk metrics as a DataFrame."""
        metrics_data = []
        
        for metrics in self.metrics_history:
            if start_time and metrics.timestamp < start_time:
                continue
            if end_time and metrics.timestamp > end_time:
                continue
                
            metrics_dict = {
                'timestamp': metrics.timestamp,
                'total_value': metrics.total_value,
                'exposure': metrics.exposure,
                'leverage': metrics.leverage,
                'margin_usage': metrics.margin_usage,
                'unrealized_pnl': metrics.unrealized_pnl,
                'realized_pnl': metrics.realized_pnl,
                'drawdown': metrics.drawdown,
                'var_95': metrics.var_95,
                'var_99': metrics.var_99,
                'expected_shortfall': metrics.expected_shortfall,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio
            }
            metrics_data.append(metrics_dict)
            
        return pd.DataFrame(metrics_data)
