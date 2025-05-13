"""
Portfolio Risk Calculator Service Module.

Provides functionality for calculating various portfolio risk metrics including
Value at Risk (VaR), Concentration Risk, and Portfolio Drawdown.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta

from core_foundations.utils.logger import get_logger

logger = get_logger("portfolio-risk-calculator")

@dataclass
class PortfolioRiskMetrics:
    """Container for portfolio risk metrics."""
    value_at_risk: float  # Value at Risk
    expected_shortfall: float  # Expected Shortfall (CVaR)
    concentration_score: float  # Herfindahl-Hirschman Index
    correlation_exposure: Dict[str, float]  # Pairwise correlation exposures
    max_drawdown: float  # Maximum historical drawdown
    sharpe_ratio: float  # Risk-adjusted return metric
    total_exposure: float  # Total portfolio exposure

class PortfolioRiskCalculator:
    """Service for calculating portfolio risk metrics."""
    
    def calculate_portfolio_var(
        self,
        portfolio_positions: Dict[str, Any],
        market_data: Dict[str, Any],
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ) -> float:
        """
        Calculate portfolio Value at Risk (VaR) using historical simulation.
        
        Args:
            portfolio_positions: Current portfolio positions
            market_data: Historical price data for positions
            confidence_level: VaR confidence level (default 95%)
            time_horizon_days: Time horizon for VaR calculation
            
        Returns:
            Portfolio VaR as a percentage of portfolio value
        """
        if not portfolio_positions:
            return 0.0
            
        # Calculate historical returns for each position
        position_returns = []
        weights = []
        
        total_value = sum(pos['current_value'] for pos in portfolio_positions.values())
        
        for symbol, position in portfolio_positions.items():
            if symbol not in market_data:
                logger.warning(f"No market data found for {symbol}, skipping in VaR calculation")
                continue
                
            # Get position weight
            weight = position['current_value'] / total_value
            weights.append(weight)
            
            # Calculate returns from market data
            prices = market_data[symbol]['prices']
            returns = np.diff(np.log(prices)) # Log returns
            position_returns.append(returns)
            
        if not position_returns:
            return 0.0
            
        # Convert to numpy arrays
        weights = np.array(weights)
        position_returns = np.array(position_returns)
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(weights, position_returns)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Scale to time horizon
        var = var * np.sqrt(time_horizon_days)
        
        return abs(var)  # Return positive value
        
    def assess_concentration_risk(self, portfolio_positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate portfolio concentration risk using Herfindahl-Hirschman Index (HHI).
        
        Args:
            portfolio_positions: Current portfolio positions
            
        Returns:
            Dict containing HHI score and concentration analysis
        """
        if not portfolio_positions:
            return {
                "HHI": 0,
                "concentration_level": "None",
                "largest_positions": []
            }
            
        # Calculate total portfolio value
        total_value = sum(pos['current_value'] for pos in portfolio_positions.values())
        
        # Calculate market shares
        market_shares = []
        positions = []
        
        for symbol, position in portfolio_positions.items():
            share = (position['current_value'] / total_value) * 100
            market_shares.append(share)
            positions.append({
                "symbol": symbol,
                "share": share
            })
            
        # Calculate HHI
        hhi = sum(share * share for share in market_shares)
        
        # Sort positions by share
        positions.sort(key=lambda x: x['share'], reverse=True)
        
        # Determine concentration level
        concentration_level = "Low"
        if hhi > 2500:
            concentration_level = "High"
        elif hhi > 1500:
            concentration_level = "Moderate"
            
        return {
            "HHI": round(hhi, 2),
            "concentration_level": concentration_level,
            "largest_positions": positions[:3]  # Top 3 positions
        }
        
    def calculate_correlation_risk(
        self,
        portfolio_positions: Dict[str, Any],
        market_data: Dict[str, Any],
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate correlation-based risk metrics for the portfolio.
        
        Args:
            portfolio_positions: Current portfolio positions
            market_data: Historical price data for positions
            lookback_days: Number of days to look back for correlation
            
        Returns:
            Dict of correlation risk metrics
        """
        if not portfolio_positions:
            return {"max_correlation": 0.0, "avg_correlation": 0.0}
            
        # Create returns DataFrame
        returns_data = {}
        for symbol in portfolio_positions:
            if symbol not in market_data:
                continue
            prices = market_data[symbol]['prices'][-lookback_days:]
            returns = np.diff(np.log(prices))
            returns_data[symbol] = returns
            
        if not returns_data:
            return {"max_correlation": 0.0, "avg_correlation": 0.0}
            
        # Calculate correlation matrix
        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr()
        
        # Get upper triangle of correlation matrix
        upper_triangle = np.triu(corr_matrix, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        if len(correlations) == 0:
            return {"max_correlation": 0.0, "avg_correlation": 0.0}
            
        return {
            "max_correlation": float(np.max(np.abs(correlations))),
            "avg_correlation": float(np.mean(np.abs(correlations)))
        }
        
    def calculate_portfolio_risk_metrics(
        self,
        portfolio_positions: Dict[str, Any],
        market_data: Dict[str, Any],
        lookback_days: int = 30
    ) -> PortfolioRiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            portfolio_positions: Current portfolio positions
            market_data: Historical price data
            lookback_days: Analysis lookback period
            
        Returns:
            PortfolioRiskMetrics object with risk calculations
        """
        # Calculate VaR
        var = self.calculate_portfolio_var(portfolio_positions, market_data)
        
        # Calculate Expected Shortfall (CVaR)
        es = var * 1.2  # Simplified calculation, typically ES > VaR
        
        # Get concentration metrics
        concentration = self.assess_concentration_risk(portfolio_positions)
        
        # Get correlation metrics
        correlation = self.calculate_correlation_risk(
            portfolio_positions, 
            market_data, 
            lookback_days
        )
        
        # Calculate Sharpe ratio
        total_value = sum(pos['current_value'] for pos in portfolio_positions.values())
        total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in portfolio_positions.values())
        sharpe = (total_pnl / total_value) if total_value > 0 else 0
        
        return PortfolioRiskMetrics(
            value_at_risk=var,
            expected_shortfall=es,
            concentration_score=concentration['HHI'],
            correlation_exposure=correlation,
            max_drawdown=self._calculate_max_drawdown(portfolio_positions, market_data),
            sharpe_ratio=sharpe,
            total_exposure=total_value
        )
        
    def _calculate_max_drawdown(
        self,
        portfolio_positions: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate maximum drawdown from historical data."""
        if not portfolio_positions:
            return 0.0
            
        # Get portfolio values over time
        dates = sorted(market_data[list(market_data.keys())[0]]['dates'])
        portfolio_values = []
        
        for date in dates:
            value = 0
            for symbol, position in portfolio_positions.items():
                if symbol in market_data:
                    price_idx = market_data[symbol]['dates'].index(date)
                    price = market_data[symbol]['prices'][price_idx]
                    value += position['size'] * price
            portfolio_values.append(value)
            
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdowns
        drawdowns = (running_max - portfolio_values) / running_max
        
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
