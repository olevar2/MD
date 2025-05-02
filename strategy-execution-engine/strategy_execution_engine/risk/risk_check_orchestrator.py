"""
Risk Check Orchestrator that coordinates pre-trade and post-trade risk assessments.
"""
from typing import Dict, Any, List, Tuple, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

class RiskCheckOrchestrator:
    """
    Coordinates pre-trade and post-trade risk assessments across various risk dimensions.
    Acts as the central hub for risk validation before order placement.
    """
    
    def __init__(self, risk_service_client, portfolio_service_client=None, market_data_client=None):
        """
        Initialize the Risk Check Orchestrator.
        
        Args:
            risk_service_client: Client to interact with the risk management service
            portfolio_service_client: Optional client for portfolio data
            market_data_client: Optional client for market data
        """
        self.risk_service_client = risk_service_client
        self.portfolio_service_client = portfolio_service_client
        self.market_data_client = market_data_client
        
        # Configure risk check thresholds
        self.max_position_pct = 5.0  # No single position > 5% of account
        self.max_exposure_per_currency = 20.0  # Max 20% exposure to any currency
        self.max_daily_drawdown_pct = 5.0  # Max 5% daily drawdown allowed
        self.max_correlation_exposure = 3.0  # Max 3x exposure to correlated pairs
        
        # Cache for risk calculations
        self.risk_cache = {}
        
    async def perform_pre_trade_checks(self, 
                                     order: Dict[str, Any], 
                                     account: Dict[str, Any], 
                                     portfolio: Dict[str, Any], 
                                     market_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform all pre-trade risk checks before order submission.
        
        Args:
            order: Order details (instrument, direction, quantity, price, etc.)
            account: Account information including balance, margin, etc.
            portfolio: Current portfolio positions and exposure
            market_data: Current market data including prices, volatility, etc.
            
        Returns:
            Tuple of (passed, results) where:
                - passed: Boolean indicating if all checks passed
                - results: Dictionary of individual check results
        """
        # Run all risk checks in parallel
        check_results = {}
        
        checks = [
            self._check_position_limits(order, account, portfolio),
            self._check_exposure_limits(order, account, portfolio),
            self._check_margin_requirements(order, account, portfolio, market_data),
            self._check_max_drawdown(order, account, portfolio, market_data),
            self._check_concentration_risk(order, portfolio),
            self._check_correlation_risk(order, portfolio, market_data)
        ]
        
        results = await asyncio.gather(*checks)
        
        # Combine results
        for result in results:
            check_results.update(result)
        
        # Overall check passed if all individual checks passed
        all_passed = all(result.get('passed', False) for result in check_results.values())
        
        # Log the results
        logger.info(f"Pre-trade risk checks completed. Passed: {all_passed}")
        if not all_passed:
            failed_checks = [name for name, result in check_results.items() if not result.get('passed', False)]
            logger.warning(f"Failed checks: {', '.join(failed_checks)}")
        
        return all_passed, check_results
    
    async def perform_post_trade_checks(self, 
                                      trade: Dict[str, Any], 
                                      account: Dict[str, Any], 
                                      portfolio: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform post-trade risk checks after an order has been executed.
        
        Args:
            trade: Executed trade details
            account: Updated account information
            portfolio: Updated portfolio information
            
        Returns:
            Tuple of (passed, results) similar to pre-trade checks
        """
        check_results = {}
        
        checks = [
            self._check_portfolio_var(portfolio),
            self._check_drawdown_status(account),
            self._check_margin_utilization(account),
            self._check_trade_size_abnormality(trade, account)
        ]
        
        results = await asyncio.gather(*checks)
        
        # Combine results
        for result in results:
            check_results.update(result)
            
        # Overall check passed if all individual checks passed
        all_passed = all(result.get('passed', False) for result in check_results.values())
        
        # Log any post-trade risk issues
        if not all_passed:
            logger.warning(f"Post-trade checks identified risks: {check_results}")
            
        return all_passed, check_results
    
    async def _check_position_limits(self, order, account, portfolio) -> Dict[str, Dict[str, Any]]:
        """Check if the order would exceed position size limits."""
        result = {
            'position_limit_check': {
                'passed': True,
                'message': "Position limit check passed"
            }
        }
        
        # Calculate account value (balance or equity depending on configuration)
        account_value = account.get('equity', account.get('balance', 0))
        if account_value <= 0:
            result['position_limit_check'] = {
                'passed': False,
                'message': "Unable to determine account value"
            }
            return result
            
        # Get existing position for this instrument
        instrument = order.get('instrument', '')
        existing_position_size = portfolio.get('positions', {}).get(instrument, 0)
        
        # Calculate new total position size
        new_position_size = existing_position_size
        if order.get('direction', '').lower() == 'buy':
            new_position_size += order.get('quantity', 0)
        else:
            new_position_size -= order.get('quantity', 0)
        
        # Calculate position size as percentage of account
        price = order.get('price', 0)
        if price <= 0:
            # Try to get price from market data
            price = 1.0  # Default if no price available
            
        position_value = abs(new_position_size) * price
        position_pct = (position_value / account_value) * 100
        
        if position_pct > self.max_position_pct:
            result['position_limit_check'] = {
                'passed': False,
                'message': f"Position size {position_pct:.2f}% exceeds maximum allowed {self.max_position_pct}%",
                'limit': self.max_position_pct,
                'actual': position_pct
            }
            
        return result
        
    async def _check_exposure_limits(self, order, account, portfolio) -> Dict[str, Dict[str, Any]]:
        """Check if the order would exceed currency exposure limits."""
        result = {
            'exposure_limit_check': {
                'passed': True,
                'message': "Exposure limit check passed"
            }
        }
        
        # Extract currencies from the instrument
        instrument = order.get('instrument', '')
        if len(instrument) != 6:
            result['exposure_limit_check'] = {
                'passed': False,
                'message': f"Invalid instrument format: {instrument}"
            }
            return result
            
        base_currency = instrument[0:3]
        quote_currency = instrument[3:6]
        
        # Calculate current exposure per currency
        currency_exposure = self._calculate_currency_exposure(portfolio)
        
        # Add the new order's impact on exposure
        quantity = order.get('quantity', 0)
        price = order.get('price', 1.0)
        
        if order.get('direction', '').lower() == 'buy':
            # Buying base currency, selling quote currency
            currency_exposure[base_currency] = currency_exposure.get(base_currency, 0) + quantity
            currency_exposure[quote_currency] = currency_exposure.get(quote_currency, 0) - quantity * price
        else:
            # Selling base currency, buying quote currency
            currency_exposure[base_currency] = currency_exposure.get(base_currency, 0) - quantity
            currency_exposure[quote_currency] = currency_exposure.get(quote_currency, 0) + quantity * price
        
        # Calculate account value
        account_value = account.get('equity', account.get('balance', 0))
        if account_value <= 0:
            result['exposure_limit_check'] = {
                'passed': False,
                'message': "Unable to determine account value"
            }
            return result
        
        # Check exposure percentages
        for currency, exposure in currency_exposure.items():
            exposure_pct = (abs(exposure) / account_value) * 100
            if exposure_pct > self.max_exposure_per_currency:
                result['exposure_limit_check'] = {
                    'passed': False,
                    'message': f"{currency} exposure {exposure_pct:.2f}% exceeds maximum {self.max_exposure_per_currency}%",
                    'currency': currency,
                    'limit': self.max_exposure_per_currency,
                    'actual': exposure_pct
                }
                break
                
        return result
        
    async def _check_margin_requirements(self, order, account, portfolio, market_data) -> Dict[str, Dict[str, Any]]:
        """Check if the order would exceed margin requirements."""
        result = {
            'margin_requirement_check': {
                'passed': True,
                'message': "Margin requirement check passed"
            }
        }
        
        # Get available margin
        available_margin = account.get('available_margin', 0)
        
        # Calculate required margin for this order
        required_margin = self._calculate_required_margin(order, market_data)
        
        if required_margin > available_margin:
            result['margin_requirement_check'] = {
                'passed': False,
                'message': f"Insufficient margin: requires {required_margin}, available {available_margin}",
                'required': required_margin,
                'available': available_margin
            }
            
        return result
        
    async def _check_max_drawdown(self, order, account, portfolio, market_data) -> Dict[str, Dict[str, Any]]:
        """Check if the order might lead to exceeding maximum allowed drawdown."""
        result = {
            'max_drawdown_check': {
                'passed': True,
                'message': "Maximum drawdown check passed"
            }
        }
        
        # Calculate current daily drawdown
        current_drawdown = self._calculate_daily_drawdown(account)
        
        # Calculate potential additional drawdown from this trade
        potential_loss = self._calculate_potential_loss(order, market_data)
        
        # Calculate account value
        account_value = account.get('equity', account.get('balance', 0))
        
        # Calculate total potential drawdown
        potential_drawdown_pct = current_drawdown + (potential_loss / account_value * 100)
        
        if potential_drawdown_pct > self.max_daily_drawdown_pct:
            result['max_drawdown_check'] = {
                'passed': False,
                'message': f"Potential drawdown {potential_drawdown_pct:.2f}% exceeds maximum {self.max_daily_drawdown_pct}%",
                'limit': self.max_daily_drawdown_pct,
                'actual': potential_drawdown_pct
            }
            
        return result
        
    async def _check_concentration_risk(self, order, portfolio) -> Dict[str, Dict[str, Any]]:
        """Check if the order would create concentration risk in the portfolio."""
        result = {
            'concentration_risk_check': {
                'passed': True,
                'message': "Concentration risk check passed"
            }
        }
        
        # Get instrument group (e.g., currency pair category)
        instrument = order.get('instrument', '')
        instrument_group = self._get_instrument_group(instrument)
        
        # Calculate current exposure to this instrument group
        group_exposure = self._calculate_group_exposure(portfolio, instrument_group)
        
        # Add the new order's impact
        quantity = order.get('quantity', 0)
        price = order.get('price', 1.0)
        
        # Calculate the value of the new order
        order_value = quantity * price
        
        # Calculate total portfolio value
        portfolio_value = self._calculate_portfolio_value(portfolio)
        
        # Calculate new group concentration
        new_group_exposure = group_exposure + order_value
        concentration_pct = (new_group_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        # Check against limit (example: max 30% in one instrument group)
        max_concentration = 30.0
        if concentration_pct > max_concentration:
            result['concentration_risk_check'] = {
                'passed': False,
                'message': f"Concentration in {instrument_group} ({concentration_pct:.2f}%) exceeds maximum {max_concentration}%",
                'group': instrument_group,
                'limit': max_concentration,
                'actual': concentration_pct
            }
            
        return result
        
    async def _check_correlation_risk(self, order, portfolio, market_data) -> Dict[str, Dict[str, Any]]:
        """Check if the order would create correlation risk in the portfolio."""
        result = {
            'correlation_risk_check': {
                'passed': True,
                'message': "Correlation risk check passed"
            }
        }
        
        # Get instrument
        instrument = order.get('instrument', '')
        
        # Get correlation data
        correlation_data = market_data.get('correlation', {})
        if not correlation_data:
            # If no correlation data available, assume the check passes
            return result
            
        # Calculate correlated exposure
        correlated_exposure = 0
        portfolio_positions = portfolio.get('positions', {})
        
        for pair, size in portfolio_positions.items():
            if pair == instrument:
                continue
                
            # Look for correlation between this pair and our order instrument
            corr_key = f"{instrument}_{pair}"
            alt_corr_key = f"{pair}_{instrument}"
            correlation = correlation_data.get(corr_key, correlation_data.get(alt_corr_key, 0))
            
            # Only consider strong correlations
            if abs(correlation) > 0.7:
                pair_price = market_data.get('price', {}).get(pair, 1.0)
                position_value = abs(size) * pair_price
                correlated_exposure += position_value * abs(correlation)
        
        # Add our new order's value
        quantity = order.get('quantity', 0)
        price = order.get('price', 1.0)
        order_value = quantity * price
        
        # Calculate total portfolio value
        portfolio_value = self._calculate_portfolio_value(portfolio)
        
        # Check if correlated exposure is too high
        if portfolio_value > 0 and correlated_exposure > 0:
            correlated_exposure_ratio = correlated_exposure / portfolio_value
            if correlated_exposure_ratio > self.max_correlation_exposure:
                result['correlation_risk_check'] = {
                    'passed': False,
                    'message': f"Correlated exposure ({correlated_exposure_ratio:.2f}x) exceeds maximum ({self.max_correlation_exposure}x)",
                    'limit': self.max_correlation_exposure,
                    'actual': correlated_exposure_ratio
                }
                
        return result

    async def _check_portfolio_var(self, portfolio) -> Dict[str, Dict[str, Any]]:
        """Check portfolio Value at Risk."""
        result = {
            'portfolio_var_check': {
                'passed': True,
                'message': "Portfolio VaR check passed"
            }
        }
        
        # Call risk service to get VaR calculation
        try:
            portfolio_var = await self._get_portfolio_var(portfolio)
            portfolio_value = self._calculate_portfolio_value(portfolio)
            
            # Calculate VaR as percentage of portfolio
            var_pct = (portfolio_var / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            # Check against limit (example: max 20% VaR)
            max_var_pct = 20.0
            if var_pct > max_var_pct:
                result['portfolio_var_check'] = {
                    'passed': False,
                    'message': f"Portfolio VaR {var_pct:.2f}% exceeds maximum {max_var_pct}%",
                    'limit': max_var_pct,
                    'actual': var_pct
                }
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            result['portfolio_var_check'] = {
                'passed': True,  # Pass but with a warning
                'message': f"Unable to calculate portfolio VaR: {str(e)}",
                'warning': True
            }
            
        return result
        
    async def _check_drawdown_status(self, account) -> Dict[str, Dict[str, Any]]:
        """Check current drawdown status."""
        result = {
            'drawdown_status_check': {
                'passed': True,
                'message': "Drawdown status check passed"
            }
        }
        
        # Calculate current drawdown
        current_drawdown = self._calculate_daily_drawdown(account)
        
        # Check if close to daily limit
        if current_drawdown > self.max_daily_drawdown_pct * 0.8:
            result['drawdown_status_check'] = {
                'passed': True,  # Still passes but with warning
                'message': f"Approaching daily drawdown limit: {current_drawdown:.2f}% of {self.max_daily_drawdown_pct}%",
                'warning': True,
                'limit': self.max_daily_drawdown_pct,
                'actual': current_drawdown
            }
            
        return result
        
    async def _check_margin_utilization(self, account) -> Dict[str, Dict[str, Any]]:
        """Check current margin utilization."""
        result = {
            'margin_utilization_check': {
                'passed': True,
                'message': "Margin utilization check passed"
            }
        }
        
        # Calculate margin utilization
        used_margin = account.get('used_margin', 0)
        total_margin = used_margin + account.get('available_margin', 0)
        
        if total_margin > 0:
            margin_utilization = (used_margin / total_margin) * 100
            
            # Check if margin utilization is high
            if margin_utilization > 80:
                result['margin_utilization_check'] = {
                    'passed': True,  # Still passes but with warning
                    'message': f"High margin utilization: {margin_utilization:.2f}%",
                    'warning': True,
                    'limit': 80,
                    'actual': margin_utilization
                }
                
        return result
        
    async def _check_trade_size_abnormality(self, trade, account) -> Dict[str, Dict[str, Any]]:
        """Check if trade size is abnormal compared to account size or typical trades."""
        result = {
            'trade_size_check': {
                'passed': True,
                'message': "Trade size check passed"
            }
        }
        
        # Calculate trade value
        quantity = trade.get('quantity', 0)
        price = trade.get('price', 0)
        trade_value = quantity * price
        
        # Get account value
        account_value = account.get('equity', account.get('balance', 0))
        
        if account_value > 0:
            # Calculate trade size as percentage of account
            trade_size_pct = (trade_value / account_value) * 100
            
            # Check if trade size is unusually large
            if trade_size_pct > 10:  # Example: flagging trades > 10% of account
                result['trade_size_check'] = {
                    'passed': True,  # Still passes but with warning
                    'message': f"Unusually large trade: {trade_size_pct:.2f}% of account",
                    'warning': True,
                    'limit': 10,
                    'actual': trade_size_pct
                }
                
        return result
    
    def _calculate_currency_exposure(self, portfolio) -> Dict[str, float]:
        """Calculate exposure to each currency in the portfolio."""
        currency_exposure = {}
        
        # Process each position
        for instrument, size in portfolio.get('positions', {}).items():
            if len(instrument) != 6:
                continue
                
            base_currency = instrument[0:3]
            quote_currency = instrument[3:6]
            
            # Use approximate price (1.0) if we don't have real prices
            price = 1.0
            
            # Update base currency exposure (positive for long, negative for short)
            currency_exposure[base_currency] = currency_exposure.get(base_currency, 0) + size
            
            # Update quote currency exposure (negative for long, positive for short)
            quote_exposure = -size * price
            currency_exposure[quote_currency] = currency_exposure.get(quote_currency, 0) + quote_exposure
            
        return currency_exposure
    
    def _calculate_required_margin(self, order, market_data) -> float:
        """Calculate required margin for an order."""
        instrument = order.get('instrument', '')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0)
        
        # Get leverage for this instrument (default to conservative 50:1)
        leverage = market_data.get('leverage', {}).get(instrument, 50)
        
        # Calculate required margin
        required_margin = (quantity * price) / leverage
        
        return required_margin
        
    def _calculate_daily_drawdown(self, account) -> float:
        """Calculate current daily drawdown as a percentage."""
        balance = account.get('balance', 0)
        peak_balance = account.get('peak_balance', balance)
        
        if peak_balance <= 0:
            return 0
            
        drawdown = (peak_balance - balance) / peak_balance * 100
        return max(0, drawdown)
        
    def _calculate_potential_loss(self, order, market_data) -> float:
        """Calculate potential loss from a trade based on stop loss or volatility."""
        instrument = order.get('instrument', '')
        quantity = order.get('quantity', 0)
        price = order.get('price', 0)
        stop_loss = order.get('stop_loss', None)
        
        if stop_loss:
            # Calculate loss based on stop loss
            potential_loss = abs(price - stop_loss) * quantity
        else:
            # Estimate loss based on volatility (e.g., 2 standard deviations)
            volatility = market_data.get('volatility', {}).get(instrument, 0.01)
            potential_loss = price * volatility * 2 * quantity
            
        return potential_loss
        
    def _get_instrument_group(self, instrument) -> str:
        """Classify instrument into a group (e.g., major pairs, cross pairs, etc.)."""
        if not instrument or len(instrument) != 6:
            return 'unknown'
            
        # Define common currency groups
        majors = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'NZD']
        
        base = instrument[0:3]
        quote = instrument[3:6]
        
        # Major pairs contain USD
        if 'USD' in instrument:
            return 'major_pairs'
            
        # Check for crosses between major currencies
        if base in majors and quote in majors:
            return 'cross_pairs'
            
        # Otherwise, consider it an exotic pair
        return 'exotic_pairs'
        
    def _calculate_group_exposure(self, portfolio, group) -> float:
        """Calculate exposure to a specific instrument group."""
        exposure = 0
        
        for instrument, size in portfolio.get('positions', {}).items():
            if self._get_instrument_group(instrument) == group:
                # Simplified: use size * 1.0 as the position value
                exposure += abs(size) * 1.0
                
        return exposure
        
    def _calculate_portfolio_value(self, portfolio) -> float:
        """Calculate total portfolio value."""
        value = 0
        
        for instrument, size in portfolio.get('positions', {}).items():
            # Simplified: use size * 1.0 as the position value
            value += abs(size) * 1.0
            
        return value
        
    async def _get_portfolio_var(self, portfolio) -> float:
        """Get portfolio Value at Risk from risk service."""
        if not self.risk_service_client:
            return 0.0
            
        try:
            # This would be an async call to the risk service in a real implementation
            var_result = await self.risk_service_client.calculate_portfolio_var(portfolio)
            return var_result.get('var', 0.0)
        except Exception as e:
            logger.error(f"Error getting portfolio VaR: {e}")
            return 0.0
            
    def get_risk_limit_status(self) -> Dict[str, Any]:
        """
        Get current status of risk limits and utilization.
        Useful for monitoring dashboards and reports.
        
        Returns:
            Dictionary of risk limits and their current utilization
        """
        return {
            'position_limit': {
                'limit': self.max_position_pct,
                'utilization': 0.0  # Would be populated in a real implementation
            },
            'currency_exposure_limit': {
                'limit': self.max_exposure_per_currency,
                'utilization': {}  # Would contain utilization per currency
            },
            'drawdown_limit': {
                'limit': self.max_daily_drawdown_pct,
                'current': 0.0  # Would be populated from account data
            },
            'correlation_limit': {
                'limit': self.max_correlation_exposure,
                'utilization': 0.0  # Would be populated from portfolio analysis
            }
        }
        
    def update_risk_limits(self, new_limits: Dict[str, Any]) -> None:
        """
        Update risk limit settings.
        
        Args:
            new_limits: Dictionary with new limit values to apply
        """
        if 'max_position_pct' in new_limits:
            self.max_position_pct = float(new_limits['max_position_pct'])
            
        if 'max_exposure_per_currency' in new_limits:
            self.max_exposure_per_currency = float(new_limits['max_exposure_per_currency'])
            
        if 'max_daily_drawdown_pct' in new_limits:
            self.max_daily_drawdown_pct = float(new_limits['max_daily_drawdown_pct'])
            
        if 'max_correlation_exposure' in new_limits:
            self.max_correlation_exposure = float(new_limits['max_correlation_exposure'])
            
        logger.info(f"Risk limits updated: {vars(self)}")

# Example usage
if __name__ == "__main__":
    # Mock risk service client for testing
    class MockRiskServiceClient:
        async def calculate_portfolio_var(self, portfolio):
            # Return mock VaR value
            return {'var': 1000.0}
    
    async def run_test():
        # Create the orchestrator
        orchestrator = RiskCheckOrchestrator(MockRiskServiceClient())
        
        # Test order data
        test_order = {
            'instrument': 'EURUSD',
            'direction': 'buy',
            'quantity': 10000,
            'price': 1.0850,
            'order_time': 1618500000000  # Sample timestamp
        }
        
        test_account = {
            'balance': 10000,
            'peak_balance': 12000,
            'available_margin': 8000
        }
        
        test_portfolio = {
            'positions': {
                'GBPUSD': 5000,
                'USDJPY': -3000
            }
        }
        
        test_market_data = {
            'price': {
                'EURUSD': 1.0855,
                'GBPUSD': 1.3050,
                'USDJPY': 109.50
            }
        }
        
        # Run pre-trade checks
        passed, results = await orchestrator.perform_pre_trade_checks(
            test_order, test_account, test_portfolio, test_market_data
        )
        
        print(f"Pre-trade checks passed: {passed}")
        for check, result in results.items():
            print(f"{check}: {result['message']}")
        
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the test
    import asyncio
    asyncio.run(run_test())
