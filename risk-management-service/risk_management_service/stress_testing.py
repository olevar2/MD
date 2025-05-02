"""
Stress Testing Engine for Risk Management.

Implements comprehensive stress testing scenarios and stress metrics
calculation for forex trading portfolios.
"""
from typing import Dict, Any, List, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .risk_manager import RiskManager, Position, RiskType, RiskLevel

logger = logging.getLogger(__name__)

@dataclass
class StressScenario:
    """Represents a stress testing scenario."""
    name: str
    description: str
    price_shocks: Dict[str, float]  # Symbol to price change %
    volatility_multiplier: float = 1.0
    correlation_adjustments: Optional[Dict[str, Dict[str, float]]] = None
    liquidity_factor: float = 1.0  # 1.0 = normal, >1 = stressed
    custom_factors: Optional[Dict[str, float]] = None

class StressTestingEngine:
    """
    Engine for running stress tests on forex portfolios.
    
    Implements various stress testing scenarios including:
    - Historical scenarios (e.g., major market events)
    - Hypothetical scenarios (e.g., market shocks)
    - Sensitivity analysis
    - Correlation breakdown scenarios
    """
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.historical_scenarios = self._initialize_historical_scenarios()
        self.hypothetical_scenarios = self._initialize_hypothetical_scenarios()
        
    def _initialize_historical_scenarios(self) -> Dict[str, StressScenario]:
        """Initialize historical stress scenarios."""
        scenarios = {
            "2008_crisis": StressScenario(
                name="2008 Financial Crisis",
                description="Replicates market movements during 2008 crisis",
                price_shocks={
                    "EUR/USD": -0.15,  # 15% drop
                    "GBP/USD": -0.25,  # 25% drop
                    "USD/JPY": 0.10,   # 10% rise (flight to safety)
                },
                volatility_multiplier=3.0,
                correlation_adjustments={
                    "EUR/USD": {"GBP/USD": 0.9},  # Increased correlation
                },
                liquidity_factor=2.5
            ),
            "brexit_shock": StressScenario(
                name="Brexit Vote Shock",
                description="Simulates market reaction to Brexit vote",
                price_shocks={
                    "GBP/USD": -0.12,
                    "EUR/USD": -0.05,
                    "EUR/GBP": 0.08,
                },
                volatility_multiplier=2.0,
                liquidity_factor=1.8
            ),
            "covid_crash": StressScenario(
                name="COVID-19 Market Crash",
                description="Replicates March 2020 market conditions",
                price_shocks={
                    "EUR/USD": -0.08,
                    "GBP/USD": -0.10,
                    "USD/JPY": -0.05,
                    "AUD/USD": -0.15,
                },
                volatility_multiplier=4.0,
                liquidity_factor=3.0
            )
        }
        return scenarios
        
    def _initialize_hypothetical_scenarios(self) -> Dict[str, StressScenario]:
        """Initialize hypothetical stress scenarios."""
        scenarios = {
            "usd_crisis": StressScenario(
                name="USD Crisis",
                description="Severe USD weakness across major pairs",
                price_shocks={
                    "EUR/USD": 0.20,
                    "GBP/USD": 0.20,
                    "AUD/USD": 0.25,
                    "USD/JPY": -0.15,
                },
                volatility_multiplier=2.5,
                liquidity_factor=2.0
            ),
            "global_crisis": StressScenario(
                name="Global Financial Crisis",
                description="Widespread market panic and correlation breakdown",
                price_shocks={
                    "EUR/USD": -0.15,
                    "GBP/USD": -0.18,
                    "AUD/USD": -0.20,
                    "USD/JPY": 0.10,
                },
                volatility_multiplier=3.5,
                correlation_adjustments={},  # All correlations break down
                liquidity_factor=3.0
            ),
            "extreme_volatility": StressScenario(
                name="Extreme Volatility Spike",
                description="Severe volatility spike across all pairs",
                price_shocks={},  # No directional bias
                volatility_multiplier=5.0,
                liquidity_factor=2.5
            )
        }
        return scenarios

    def run_stress_test(
        self,
        scenario: StressScenario,
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Run a stress test using the specified scenario.
        
        Args:
            scenario: Stress scenario to apply
            current_prices: Current prices for all relevant symbols
            
        Returns:
            Dictionary containing stress test results
        """
        # Calculate stressed prices
        stressed_prices = {}
        for symbol, price in current_prices.items():
            shock = scenario.price_shocks.get(symbol, 0.0)
            stressed_prices[symbol] = price * (1 + shock)
            
        # Calculate portfolio value under stress
        original_value = self.risk_manager.current_balance
        stressed_value = original_value
        max_loss = 0.0
        
        for position in self.risk_manager.positions.values():
            if position.symbol in stressed_prices:
                # Calculate position P&L under stress
                stressed_price = stressed_prices[position.symbol]
                position_pnl = position.calculate_pnl(stressed_price)
                
                # Apply volatility multiplier to the loss
                if position_pnl < 0:
                    position_pnl *= scenario.volatility_multiplier
                    
                stressed_value += position_pnl
                max_loss = min(max_loss, position_pnl)
        
        # Calculate key metrics
        value_at_risk = original_value - stressed_value
        max_drawdown = abs(max_loss) / original_value if original_value > 0 else 0
        
        # Calculate liquidity impact
        liquidation_cost = self._estimate_liquidation_cost(
            scenario.liquidity_factor
        )
        
        # Check for limit breaches under stress
        stressed_metrics = {
            'current_balance': stressed_value,
            'drawdown': (original_value - stressed_value) / original_value,
            'total_exposure': sum(
                p.size * stressed_prices.get(p.symbol, p.entry_price) * p.leverage
                for p in self.risk_manager.positions.values()
            )
        }
        
        breached_limits = []
        for risk_type, limits in self.risk_manager.risk_limits.items():
            for limit in limits:
                if risk_type == RiskType.DRAWDOWN and stressed_metrics['drawdown'] > limit.threshold:
                    breached_limits.append({
                        'risk_type': risk_type,
                        'threshold': limit.threshold,
                        'stressed_value': stressed_metrics['drawdown'],
                        'risk_level': limit.risk_level
                    })
                elif risk_type == RiskType.EXPOSURE:
                    exposure_ratio = stressed_metrics['total_exposure'] / stressed_value
                    if exposure_ratio > limit.threshold:
                        breached_limits.append({
                            'risk_type': risk_type,
                            'threshold': limit.threshold,
                            'stressed_value': exposure_ratio,
                            'risk_level': limit.risk_level
                        })
        
        return {
            'scenario_name': scenario.name,
            'original_value': original_value,
            'stressed_value': stressed_value,
            'value_at_risk': value_at_risk,
            'max_drawdown': max_drawdown,
            'liquidation_cost': liquidation_cost,
            'breached_limits': breached_limits,
            'stressed_prices': stressed_prices,
            'risk_metrics': {
                'volatility_impact': value_at_risk * scenario.volatility_multiplier,
                'correlation_impact': self._calculate_correlation_impact(scenario),
                'liquidity_risk': liquidation_cost / original_value
            }
        }
        
    def run_all_stress_tests(
        self,
        current_prices: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """Run all available stress tests."""
        results = {}
        
        # Run historical scenarios
        for name, scenario in self.historical_scenarios.items():
            results[name] = self.run_stress_test(scenario, current_prices)
            
        # Run hypothetical scenarios
        for name, scenario in self.hypothetical_scenarios.items():
            results[name] = self.run_stress_test(scenario, current_prices)
            
        return results
        
    def _estimate_liquidation_cost(self, liquidity_factor: float) -> float:
        """Estimate the cost of liquidating positions under stress."""
        total_cost = 0.0
        
        for position in self.risk_manager.positions.values():
            # Larger positions have higher impact
            size_factor = (position.size / 100000) ** 0.5  # Square root to dampen effect
            
            # Calculate base spread cost
            spread_cost = position.entry_price * 0.0001  # Assume 1 pip spread
            
            # Apply stress multipliers
            total_cost += spread_cost * size_factor * liquidity_factor * position.leverage
            
        return total_cost
        
    def _calculate_correlation_impact(self, scenario: StressScenario) -> float:
        """Calculate the impact of correlation changes."""
        if not scenario.correlation_adjustments:
            return 0.0
            
        impact = 0.0
        for symbol1, adjustments in scenario.correlation_adjustments.items():
            for symbol2, correlation in adjustments.items():
                if symbol1 in self.risk_manager.positions and symbol2 in self.risk_manager.positions:
                    pos1 = self.risk_manager.positions[symbol1]
                    pos2 = self.risk_manager.positions[symbol2]
                    
                    # Calculate correlation impact based on position sizes
                    impact += (
                        abs(pos1.size * pos2.size) *
                        correlation *
                        0.0001  # Base impact factor
                    )
                    
        return impact
        
    def create_custom_scenario(
        self,
        name: str,
        description: str,
        price_shocks: Dict[str, float],
        volatility_multiplier: float = 1.0,
        correlation_adjustments: Optional[Dict[str, Dict[str, float]]] = None,
        liquidity_factor: float = 1.0,
        custom_factors: Optional[Dict[str, float]] = None
    ) -> StressScenario:
        """Create a custom stress testing scenario."""
        scenario = StressScenario(
            name=name,
            description=description,
            price_shocks=price_shocks,
            volatility_multiplier=volatility_multiplier,
            correlation_adjustments=correlation_adjustments,
            liquidity_factor=liquidity_factor,
            custom_factors=custom_factors
        )
        return scenario

    def get_scenario_recommendations(
        self,
        test_results: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze stress test results and provide recommendations.
        
        Args:
            test_results: Results from stress tests
            
        Returns:
            List of recommendations based on stress test results
        """
        recommendations = []
        
        for scenario_name, results in test_results.items():
            # Check for severe value at risk
            if results['value_at_risk'] > self.risk_manager.current_balance * 0.2:  # 20% VaR
                recommendations.append({
                    'severity': 'high',
                    'scenario': scenario_name,
                    'issue': 'High Value at Risk',
                    'action': 'Consider reducing position sizes or implementing tighter stops'
                })
                
            # Check for excessive drawdown
            if results['max_drawdown'] > self.risk_manager.max_drawdown * 0.8:
                recommendations.append({
                    'severity': 'critical',
                    'scenario': scenario_name,
                    'issue': 'Excessive Potential Drawdown',
                    'action': 'Reduce exposure immediately or adjust position sizing'
                })
                
            # Check for liquidity risk
            if results['liquidation_cost'] > results['original_value'] * 0.05:  # 5% liquidation cost
                recommendations.append({
                    'severity': 'medium',
                    'scenario': scenario_name,
                    'issue': 'High Liquidity Risk',
                    'action': 'Consider reducing position sizes or trading more liquid pairs'
                })
                
            # Check for limit breaches
            if results['breached_limits']:
                for breach in results['breached_limits']:
                    if breach['risk_level'] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                        recommendations.append({
                            'severity': 'high',
                            'scenario': scenario_name,
                            'issue': f"Risk Limit Breach: {breach['risk_type']}",
                            'action': 'Adjust positions to comply with risk limits'
                        })
                        
        return recommendations
