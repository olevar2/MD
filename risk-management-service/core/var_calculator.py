"""
Value at Risk (VaR) Calculator specialized for Forex markets.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Union

class VaRCalculator:
    """
    Calculates Value at Risk (VaR) considering Forex market specifics
    like currency pair correlations.
    """

    def __init__(self, confidence_level=0.99, time_horizon_days=1, correlation_data=None):
        """
        Initializes the VaR calculator.

        Args:
            confidence_level (float): The confidence level for VaR (e.g., 0.99 for 99%).
            time_horizon_days (int): The time horizon in days for the VaR calculation.
            correlation_data: Data structure containing currency pair correlations.
        """
        self.confidence_level = confidence_level
        self.time_horizon_days = time_horizon_days
        self.correlation_data = correlation_data or {}
        self.z_score = self._calculate_z_score(confidence_level)
        
    def _calculate_z_score(self, confidence_level: float) -> float:
        """Calculate the Z-score for a given confidence level using the normal distribution."""
        # For common confidence levels, use pre-calculated values for efficiency
        if confidence_level == 0.99:
            return 2.326  # Z-score for 99% confidence
        elif confidence_level == 0.975:
            return 1.96   # Z-score for 97.5% confidence
        elif confidence_level == 0.95:
            return 1.645  # Z-score for 95% confidence
        else:
            # Calculate using the normal distribution percentile point function
            from scipy import stats
            return stats.norm.ppf(confidence_level)

    def calculate_portfolio_var(self, portfolio: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """
        Calculates VaR for a given portfolio.

        Args:
            portfolio: The portfolio object containing positions (e.g., {'positions': {'EURUSD': 10000}}).
            market_data: Current market data including volatility (e.g., {'volatility': {'EURUSD': 0.005}}).

        Returns:
            The calculated VaR value (as a positive number representing potential loss).
        """
        if not portfolio or 'positions' not in portfolio or not portfolio['positions']:
            return 0.0
            
        positions = portfolio['positions']
        
        # For a single currency pair, use parametric VaR
        if len(positions) == 1:
            return self._calculate_parametric_var_single_asset(positions, market_data)
            
        # For multiple currency pairs, use portfolio VaR with correlations
        return self._calculate_portfolio_var_with_correlation(positions, market_data)

    def _calculate_parametric_var_single_asset(self, positions: Dict[str, float], 
                                              market_data: Dict[str, Any]) -> float:
        """
        Calculate parametric VaR for a single currency pair position.
        """
        instrument = list(positions.keys())[0]
        position_size = positions[instrument]
        
        # Get position value (using current price or assuming 1.0 if not available)
        current_price = market_data.get('price', {}).get(instrument, 1.0)
        position_value = abs(position_size * current_price)
        
        # Get volatility (default to 1% daily volatility if not available)
        volatility = market_data.get('volatility', {}).get(instrument, 0.01)
        
        # Scale volatility by time horizon (sqrt(T) rule)
        scaled_volatility = volatility * np.sqrt(self.time_horizon_days)
        
        # Calculate VaR using the parametric formula: Value * Volatility * Z-score
        var_value = position_value * scaled_volatility * self.z_score
        
        return var_value

    def _calculate_portfolio_var_with_correlation(self, positions: Dict[str, float], 
                                                 market_data: Dict[str, Any]) -> float:
        """
        Calculate portfolio VaR considering correlations between currency pairs.
        
        Uses the variance-covariance approach incorporating the correlation matrix.
        """
        position_values = []
        volatilities = []
        instruments = list(positions.keys())
        
        # Extract position values and volatilities
        for instrument in instruments:
            position_size = positions[instrument]
            current_price = market_data.get('price', {}).get(instrument, 1.0)
            position_value = abs(position_size * current_price)
            volatility = market_data.get('volatility', {}).get(instrument, 0.01)
            
            position_values.append(position_value)
            volatilities.append(volatility)
        
        position_values = np.array(position_values)
        volatilities = np.array(volatilities)
        
        # Create correlation matrix (default to identity if no correlation data)
        n = len(instruments)
        correlation_matrix = np.identity(n)
        
        # Fill correlation matrix with actual correlations if available
        for i in range(n):
            for j in range(i+1, n):
                corr_key = f"{instruments[i]}_{instruments[j]}"
                alt_corr_key = f"{instruments[j]}_{instruments[i]}"
                correlation = self.correlation_data.get(corr_key) or self.correlation_data.get(alt_corr_key) or 0.0
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        # Create the covariance matrix
        volatility_matrix = np.diag(volatilities)
        covariance_matrix = volatility_matrix @ correlation_matrix @ volatility_matrix
        
        # Calculate portfolio variance
        portfolio_variance = position_values @ covariance_matrix @ position_values
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Scale by time horizon and calculate VaR
        scaled_volatility = portfolio_volatility * np.sqrt(self.time_horizon_days)
        portfolio_var = scaled_volatility * self.z_score
        
        return portfolio_var

    def update_correlations(self, new_correlation_data: Dict[str, float]) -> None:
        """Updates the correlation data used by the calculator."""
        self.correlation_data.update(new_correlation_data)

    def calculate_conditional_var(self, portfolio: Dict[str, Any], market_data: Dict[str, Any], 
                                  conditional_level: float = 0.975) -> float:
        """
        Calculate Conditional VaR (CVaR)/Expected Shortfall.
        
        CVaR represents the expected loss given that the loss exceeds VaR.
        
        Args:
            portfolio: The portfolio object containing positions
            market_data: Current market data including volatility
            conditional_level: The confidence level for CVaR calculation (typically 97.5%)
            
        Returns:
            The calculated CVaR value
        """
        # First calculate standard VaR
        var_value = self.calculate_portfolio_var(portfolio, market_data)
        
        # For normal distribution, CVaR can be calculated as:
        # CVaR = VaR * (exp(z_score²/2) / (1-confidence_level) * sqrt(2π))
        z_score = self.z_score
        cvar_multiplier = np.exp(z_score**2 / 2) / ((1 - self.confidence_level) * np.sqrt(2 * np.pi))
        
        return var_value * cvar_multiplier

    def run_var_stress_test(self, portfolio: Dict[str, Any], market_data: Dict[str, Any], 
                          volatility_multiplier: float = 2.0) -> float:
        """
        Run a stress test by increasing volatility by a multiplier.
        
        Args:
            portfolio: The portfolio object containing positions
            market_data: Current market data including volatility
            volatility_multiplier: Factor to multiply volatilities by
            
        Returns:
            The stressed VaR value
        """
        # Create a copy of market data to modify
        stressed_market_data = {k: v.copy() if isinstance(v, dict) else v 
                                for k, v in market_data.items()}
        
        # Multiply all volatilities by the stress factor
        if 'volatility' in stressed_market_data:
            for instrument in stressed_market_data['volatility']:
                stressed_market_data['volatility'][instrument] *= volatility_multiplier
                
        # Calculate VaR using stressed market data
        return self.calculate_portfolio_var(portfolio, stressed_market_data)

    def calculate_historical_var(self, portfolio: Dict[str, Any], 
                               historical_returns: Dict[str, List[float]]) -> float:
        """
        Calculate VaR using historical simulation method.
        
        Args:
            portfolio: The portfolio object containing positions
            historical_returns: Dictionary mapping instruments to lists of historical returns
            
        Returns:
            The historical VaR value
        """
        if not portfolio or 'positions' not in portfolio or not portfolio['positions']:
            return 0.0
        
        positions = portfolio['positions']
        instruments = list(positions.keys())
        
        # Check that we have historical returns for all instruments
        for instrument in instruments:
            if instrument not in historical_returns:
                raise ValueError(f"No historical return data for {instrument}")
        
        # Calculate portfolio returns for each historical date
        portfolio_returns = []
        min_length = min(len(historical_returns[instrument]) for instrument in instruments)
        
        for i in range(min_length):
            day_return = sum(positions[instrument] * historical_returns[instrument][i] 
                             for instrument in instruments)
            portfolio_returns.append(day_return)
        
        # Sort returns and find the VaR at the specified confidence level
        sorted_returns = sorted(portfolio_returns)
        var_index = int((1 - self.confidence_level) * len(sorted_returns))
        
        # Return the absolute value as VaR is typically presented as a positive number
        return abs(sorted_returns[var_index])

# Example usage
if __name__ == "__main__":
    # Create calculator instance
    calculator = VaRCalculator(confidence_level=0.99, time_horizon_days=1)
    
    # Sample portfolio with positions in EURUSD and GBPUSD
    portfolio = {
        "positions": {
            "EURUSD": 100000,  # Long 100K units
            "GBPUSD": -50000   # Short 50K units
        }
    }
    
    # Sample market data with prices and volatilities
    market_data = {
        "price": {
            "EURUSD": 1.0950,
            "GBPUSD": 1.2750
        },
        "volatility": {
            "EURUSD": 0.006,  # 0.6% daily volatility
            "GBPUSD": 0.008   # 0.8% daily volatility
        }
    }
    
    # Sample correlation data
    correlations = {
        "EURUSD_GBPUSD": 0.7  # 70% correlation between EUR/USD and GBP/USD
    }
    calculator.update_correlations(correlations)
    
    # Calculate VaR
    var_result = calculator.calculate_portfolio_var(portfolio, market_data)
    print(f"Portfolio VaR (99%, 1-day): ${var_result:.2f}")
    
    # Calculate Conditional VaR
    cvar_result = calculator.calculate_conditional_var(portfolio, market_data)
    print(f"Conditional VaR (99%, 1-day): ${cvar_result:.2f}")
    
    # Run stress test
    stress_var = calculator.run_var_stress_test(portfolio, market_data, 2.5)
    print(f"Stressed VaR (volatility x2.5): ${stress_var:.2f}")
