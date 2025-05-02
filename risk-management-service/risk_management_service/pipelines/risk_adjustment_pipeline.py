"""
Pipeline for intelligent risk adjustments, including stop-loss and take-profit placement.
"""
import math
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

class RiskAdjustmentPipeline:
    """
    Orchestrates risk adjustments based on various inputs like market conditions,
    risk profiles, and potentially confluence analysis.
    """

    def __init__(self, risk_profile_manager, market_data_client=None, confluence_analyzer_client=None):
        """
        Initializes the Risk Adjustment Pipeline.

        Args:
            risk_profile_manager: An instance of ForexRiskProfileManager.
            market_data_client: Client to fetch market data like ATR, volatility, etc.
            confluence_analyzer_client: Client to interact with confluence analysis service.
        """
        self.risk_profile_manager = risk_profile_manager
        self.market_data_client = market_data_client
        self.confluence_analyzer_client = confluence_analyzer_client
        self.atr_periods = 14  # Default ATR period

    def determine_optimal_stops(self, position: Dict[str, Any], market_data: Dict[str, Any], market_regime: str) -> Dict[str, float]:
        """
        Calculates optimal stop-loss and take-profit levels for a position.

        Args:
            position: The position object with keys like 'instrument', 'direction', 'entry_price', 'size'.
            market_data: Current market data including price, volatility, ATR.
            market_regime: Current market regime (trending, ranging, volatile, etc).

        Returns:
            A dictionary containing suggested 'stop_loss' and 'take_profit' levels.
        """
        currency_pair = position.get('instrument', 'UNKNOWN')
        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0.0)
        position_size = position.get('size', 0.0)
        
        # Get volatility or use default
        volatility = market_data.get('volatility', {}).get(currency_pair, 0.005)
        
        # 1. Get base risk profile
        profile = self.risk_profile_manager.get_risk_profile(currency_pair, market_regime, volatility)
        
        # 2. (Optional) Enhance with confluence data
        if self.confluence_analyzer_client:
            try:
                # Fetch confluence zones from the confluence analyzer
                confluence_data = self.confluence_analyzer_client.get_confluence_zones(
                    currency_pair=currency_pair,
                    direction=direction,
                    current_price=entry_price
                )
                profile = self.risk_profile_manager.adjust_profile_based_on_confluence(profile, confluence_data)
            except Exception as e:
                print(f"Warning: Failed to get or apply confluence data: {e}")
        
        # 3. Calculate stops using multiple methods
        atr_stops = self._calculate_atr_based_stops(
            entry_price, direction, market_data.get('atr', {}).get(currency_pair),
            profile.stop_loss_pips, profile.take_profit_pips
        )
        
        volatility_stops = self._calculate_volatility_based_stops(
            entry_price, direction, volatility, 
            profile.stop_loss_pips, profile.take_profit_pips
        )
        
        # 4. Calculate support/resistance based stops
        sr_stops = None
        if self.confluence_analyzer_client and 'support_resistance' in market_data:
            sr_data = market_data.get('support_resistance', {}).get(currency_pair, {})
            if sr_data:
                sr_stops = self._calculate_sr_based_stops(
                    entry_price, direction, sr_data
                )

        # 5. Combine stop calculation methods based on market regime
        final_stops = self._combine_stop_methods(
            entry_price, direction, market_regime,
            atr_stops, volatility_stops, sr_stops
        )
        
        # 6. Apply trailing stop logic if enabled
        if profile.trailing_stop_enabled:
            final_stops['trailing_stop_activation'] = self._calculate_trailing_stop_activation(
                entry_price, direction, profile.trailing_stop_activation_pips, 
                pip_value=self._get_pip_value(currency_pair)
            )
        
        # 7. Validate that stops are within reasonable bounds
        final_stops = self._validate_stops(final_stops, entry_price, direction)
        
        # Record risk metrics for logging/monitoring
        pip_value = self._get_pip_value(currency_pair)
        sl_pips = abs(final_stops['stop_loss'] - entry_price) / pip_value
        tp_pips = abs(final_stops['take_profit'] - entry_price) / pip_value
        risk_reward = tp_pips / sl_pips if sl_pips > 0 else 0
        
        return final_stops
    
    def calculate_position_size(self, 
                               account_balance: float,
                               risk_per_trade: Optional[float],
                               currency_pair: str, 
                               entry_price: float, 
                               stop_loss: float, 
                               market_data: Dict[str, Any],
                               market_regime: str) -> float:
        """
        Calculates the optimal position size based on risk parameters.
        
        Args:
            account_balance: The current account balance
            risk_per_trade: Desired risk per trade as percentage (e.g., 1.0 for 1%)
            currency_pair: The currency pair to trade
            entry_price: The intended entry price
            stop_loss: The intended stop loss price
            market_data: Current market data
            market_regime: Current market regime
            
        Returns:
            The recommended position size in units/lots
        """
        # Get volatility from market data
        volatility = market_data.get('volatility', {}).get(currency_pair, 0.005)
        
        # Get risk profile (contains default risk % if not specified)
        profile = self.risk_profile_manager.get_risk_profile(currency_pair, market_regime, volatility)
        
        # Use provided risk % or profile default
        risk_pct = risk_per_trade if risk_per_trade is not None else profile.risk_per_trade_pct
        
        # Calculate risk amount in account currency
        risk_amount = account_balance * (risk_pct / 100)
        
        # Calculate pip value (depends on account currency, position size, and pair)
        pip_value = self._get_pip_value(currency_pair)  # Simplified
        
        # Calculate SL distance in pips
        sl_distance = abs(entry_price - stop_loss) / pip_value
        
        # Avoid division by zero
        if sl_distance <= 0:
            print("Warning: Stop loss distance is too small, using minimum distance")
            sl_distance = 10 * pip_value  # Default to 10 pips minimum
        
        # Calculate position size based on risk
        # Formula: Position size = Risk amount / (SL distance in pips * pip value)
        # This is simplified and needs refinement for different account currencies
        position_size = risk_amount / (sl_distance * pip_value)
        
        # Limit position size based on profile maximum
        if position_size > profile.max_position_size:
            position_size = profile.max_position_size
            
        # Round to standard lot sizes (might need more sophisticated rounding)
        # Standard lot = 100,000 units
        # Mini lot = 10,000 units
        # Micro lot = 1,000 units
        if position_size > 50000:  # Round to nearest standard lot
            position_size = round(position_size / 100000) * 100000
        elif position_size > 5000:  # Round to nearest mini lot
            position_size = round(position_size / 10000) * 10000
        else:  # Round to nearest micro lot
            position_size = round(position_size / 1000) * 1000
            
        return position_size
        
    def _calculate_atr_based_stops(self, entry_price: float, direction: str, 
                                 atr_value: Optional[float], 
                                 default_sl_pips: int, 
                                 default_tp_pips: int) -> Dict[str, float]:
        """
        Calculate stops based on Average True Range (ATR).
        
        Args:
            entry_price: The entry price of the position
            direction: Trade direction ('long' or 'short')
            atr_value: The ATR value (if None, will use defaults)
            default_sl_pips: Default stop loss in pips 
            default_tp_pips: Default take profit in pips
            
        Returns:
            Dictionary with 'stop_loss' and 'take_profit' prices
        """
        # Use ATR if available, otherwise fall back to defaults
        if atr_value is None:
            # Fall back to default pips, but we need to know pip value
            pip_value = 0.0001  # Default for most forex pairs
            sl_distance = default_sl_pips * pip_value
            tp_distance = default_tp_pips * pip_value
        else:
            # Use ATR-based calculation (typical multipliers: 1.5-3 for SL, 2-4 for TP)
            sl_distance = 2.0 * atr_value  # 2x ATR for stop loss
            tp_distance = 3.0 * atr_value  # 3x ATR for take profit
        
        # Apply distances based on direction
        if direction.lower() == 'long':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
            
        return {'stop_loss': stop_loss, 'take_profit': take_profit}
    
    def _calculate_volatility_based_stops(self, 
                                        entry_price: float, 
                                        direction: str,
                                        volatility: float, 
                                        default_sl_pips: int, 
                                        default_tp_pips: int) -> Dict[str, float]:
        """
        Calculate stops based on market volatility.
        
        Args:
            entry_price: The entry price of the position
            direction: Trade direction ('long' or 'short')
            volatility: The volatility measure (typically standard deviation)
            default_sl_pips: Default stop loss in pips 
            default_tp_pips: Default take profit in pips
            
        Returns:
            Dictionary with 'stop_loss' and 'take_profit' prices
        """
        # If volatility is very low, use minimum defaults
        if volatility < 0.001:  # Below 0.1% volatility
            pip_value = 0.0001  # Default for most forex pairs
            sl_distance = default_sl_pips * pip_value
            tp_distance = default_tp_pips * pip_value
        else:
            # For volatility-based calculation (adjust multipliers based on preference)
            # Typically using 2-3 standard deviations for stops
            sl_distance = 2.5 * volatility * entry_price
            tp_distance = 3.5 * volatility * entry_price
        
        # Apply distances based on direction
        if direction.lower() == 'long':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
            
        return {'stop_loss': stop_loss, 'take_profit': take_profit}
    
    def _calculate_sr_based_stops(self, 
                               entry_price: float, 
                               direction: str,
                               sr_data: Dict[str, List[float]]) -> Optional[Dict[str, float]]:
        """
        Calculate stops based on support/resistance levels.
        
        Args:
            entry_price: The entry price of the position
            direction: Trade direction ('long' or 'short')
            sr_data: Dictionary containing support and resistance levels
            
        Returns:
            Dictionary with 'stop_loss' and 'take_profit' prices or None if insufficient data
        """
        supports = sorted(sr_data.get('supports', []))
        resistances = sorted(sr_data.get('resistances', []))
        
        if not supports or not resistances:
            return None
            
        if direction.lower() == 'long':
            # For long positions: 
            # - Place SL below nearest support below entry
            # - Place TP at nearest resistance above entry
            
            # Find nearest support below entry
            nearest_support_below = None
            for s in reversed(supports):
                if s < entry_price:
                    nearest_support_below = s
                    break
                    
            # Find nearest resistance above entry  
            nearest_resistance_above = None
            for r in resistances:
                if r > entry_price:
                    nearest_resistance_above = r
                    break
                    
            if nearest_support_below is not None and nearest_resistance_above is not None:
                # Add small buffer below support (0.5-1%)
                buffer = nearest_support_below * 0.001  # 0.1% buffer
                stop_loss = nearest_support_below - buffer
                take_profit = nearest_resistance_above
                return {'stop_loss': stop_loss, 'take_profit': take_profit}
                
        else:  # short
            # For short positions:
            # - Place SL above nearest resistance above entry
            # - Place TP at nearest support below entry
            
            # Find nearest resistance above entry
            nearest_resistance_above = None
            for r in resistances:
                if r > entry_price:
                    nearest_resistance_above = r
                    break
                    
            # Find nearest support below entry
            nearest_support_below = None
            for s in reversed(supports):
                if s < entry_price:
                    nearest_support_below = s
                    break
                    
            if nearest_resistance_above is not None and nearest_support_below is not None:
                # Add small buffer above resistance (0.5-1%)
                buffer = nearest_resistance_above * 0.001  # 0.1% buffer
                stop_loss = nearest_resistance_above + buffer
                take_profit = nearest_support_below
                return {'stop_loss': stop_loss, 'take_profit': take_profit}
                
        return None
        
    def _combine_stop_methods(self, 
                           entry_price: float, 
                           direction: str,
                           market_regime: str,
                           atr_stops: Dict[str, float], 
                           volatility_stops: Dict[str, float],
                           sr_stops: Optional[Dict[str, float]]) -> Dict[str, float]:
        """
        Combines different stop calculation methods based on market regime.
        
        Args:
            entry_price: The entry price of the position
            direction: Trade direction ('long' or 'short')
            market_regime: Current market regime
            atr_stops: Stops calculated using ATR
            volatility_stops: Stops calculated using volatility
            sr_stops: Stops calculated using support/resistance levels
            
        Returns:
            Dictionary with final 'stop_loss' and 'take_profit' prices
        """
        # Default result (in case combination fails)
        result = {
            'stop_loss': atr_stops['stop_loss'],
            'take_profit': atr_stops['take_profit']
        }
        
        # Combine methods based on market regime
        if market_regime == 'trending':
            # In trending markets, give more weight to ATR and support/resistance
            if sr_stops:
                # Use S/R levels with higher priority in trending markets
                result = sr_stops
            else:
                # Use ATR as primary method in trending markets
                result = atr_stops
                
        elif market_regime == 'volatile':
            # In volatile markets, prioritize volatility-based stops
            result = volatility_stops
            
            # But still respect major support/resistance
            if sr_stops:
                # Choose the more conservative stop loss
                if direction.lower() == 'long':
                    # For long, higher stop loss is more conservative
                    result['stop_loss'] = max(volatility_stops['stop_loss'], sr_stops['stop_loss'])
                    # Take profit might be adjusted to nearest resistance
                    result['take_profit'] = sr_stops['take_profit']
                else:
                    # For short, lower stop loss is more conservative
                    result['stop_loss'] = min(volatility_stops['stop_loss'], sr_stops['stop_loss']) 
                    # Take profit might be adjusted to nearest support
                    result['take_profit'] = sr_stops['take_profit']
                    
        elif market_regime == 'ranging':
            # In ranging markets, prioritize support/resistance
            if sr_stops:
                result = sr_stops
            else:
                # Use ATR as fallback for ranging markets
                result = atr_stops
                
        else:  # Default/unknown regime
            # Use ATR as default with volatility influence
            result['stop_loss'] = (atr_stops['stop_loss'] * 0.7 + 
                                 volatility_stops['stop_loss'] * 0.3)
            result['take_profit'] = (atr_stops['take_profit'] * 0.7 + 
                                   volatility_stops['take_profit'] * 0.3)
                                   
            # Only use SR if available and we're in default regime
            if sr_stops:
                # Blend in some influence from support/resistance
                if direction.lower() == 'long':
                    # Can use support level if it's not too far from ATR calculation
                    atr_sl_distance = entry_price - atr_stops['stop_loss']
                    sr_sl_distance = entry_price - sr_stops['stop_loss']
                    
                    # If SR stop within 2x ATR distance, consider using it
                    if abs(sr_sl_distance) < 2 * abs(atr_sl_distance):
                        result['stop_loss'] = sr_stops['stop_loss']
                else:  # short
                    # Similar logic for short positions
                    atr_sl_distance = atr_stops['stop_loss'] - entry_price
                    sr_sl_distance = sr_stops['stop_loss'] - entry_price
                    
                    if abs(sr_sl_distance) < 2 * abs(atr_sl_distance):
                        result['stop_loss'] = sr_stops['stop_loss']
        
        return result
    
    def _calculate_trailing_stop_activation(self, 
                                         entry_price: float, 
                                         direction: str,
                                         activation_pips: int, 
                                         pip_value: float) -> float:
        """
        Calculate the price at which trailing stop should activate.
        
        Args:
            entry_price: The entry price
            direction: Trade direction ('long' or 'short')
            activation_pips: Number of pips price should move in favor before activation
            pip_value: The pip value for the currency pair
            
        Returns:
            The price at which trailing stop should activate
        """
        if direction.lower() == 'long':
            return entry_price + (activation_pips * pip_value)
        else:  # short
            return entry_price - (activation_pips * pip_value)
    
    def _validate_stops(self, 
                      stops: Dict[str, float], 
                      entry_price: float, 
                      direction: str) -> Dict[str, float]:
        """
        Validate stop levels to ensure they are sensible.
        
        Args:
            stops: Dictionary with stop prices
            entry_price: The entry price
            direction: Trade direction
            
        Returns:
            Validated stop dictionary
        """
        # Make a copy to avoid modifying the original
        result = stops.copy()
        
        # Make sure stop loss isn't too close to entry price (at least 10 pips)
        min_distance = 0.0010  # 10 pips for most pairs
        
        if direction.lower() == 'long':
            if entry_price - result['stop_loss'] < min_distance:
                result['stop_loss'] = entry_price - min_distance
                
            # Ensure stop loss is below entry and take profit is above
            if result['stop_loss'] >= entry_price:
                result['stop_loss'] = entry_price - min_distance
                
            if result['take_profit'] <= entry_price:
                result['take_profit'] = entry_price + min_distance * 2
                
        else:  # short
            if result['stop_loss'] - entry_price < min_distance:
                result['stop_loss'] = entry_price + min_distance
                
            # Ensure stop loss is above entry and take profit is below
            if result['stop_loss'] <= entry_price:
                result['stop_loss'] = entry_price + min_distance
                
            if result['take_profit'] >= entry_price:
                result['take_profit'] = entry_price - min_distance * 2
        
        # Ensure risk:reward is at least 1:1 (minimum)
        sl_distance = abs(result['stop_loss'] - entry_price)
        tp_distance = abs(result['take_profit'] - entry_price)
        
        if tp_distance < sl_distance:
            # Adjust take profit to provide at least 1:1
            if direction.lower() == 'long':
                result['take_profit'] = entry_price + sl_distance
            else:
                result['take_profit'] = entry_price - sl_distance
            
        return result
    
    def _get_pip_value(self, currency_pair: str) -> float:
        """
        Get the pip value for a currency pair.
        
        Args:
            currency_pair: The currency pair
            
        Returns:
            Pip value (0.0001 for most pairs, 0.01 for JPY pairs)
        """
        if currency_pair.upper().endswith('JPY'):
            return 0.01
        else:
            return 0.0001
            
    def adjust_position_for_correlation_risk(self, 
                                           position_size: float,
                                           currency_pair: str,
                                           portfolio: Dict[str, Any],
                                           correlation_data: Dict[str, float],
                                           max_correlation_exposure: float = 3.0) -> float:
        """
        Adjust position size based on correlation with existing portfolio positions.
        
        Args:
            position_size: The initially calculated position size
            currency_pair: The currency pair being traded
            portfolio: Current portfolio with positions
            correlation_data: Dictionary of correlation coefficients between pairs
            max_correlation_exposure: Maximum exposure to correlated positions
            
        Returns:
            Adjusted position size
        """
        if not portfolio or 'positions' not in portfolio or not portfolio['positions']:
            return position_size
            
        # Get all current positions
        current_positions = portfolio['positions']
        
        # Skip if no existing positions or trying to trade the same pair
        if not current_positions or (len(current_positions) == 1 and currency_pair in current_positions):
            return position_size
            
        # Calculate correlation-weighted exposure
        correlated_exposure = 0
        total_current_exposure = sum(abs(size) for size in current_positions.values())
        
        for pair, size in current_positions.items():
            # Skip if it's the same pair
            if pair == currency_pair:
                continue
                
            # Get correlation between this pair and the one we're trading
            corr_key = f"{currency_pair}_{pair}"
            alt_corr_key = f"{pair}_{currency_pair}"
            correlation = correlation_data.get(corr_key) or correlation_data.get(alt_corr_key) or 0.0
            
            # Only consider significant correlations (positive or negative)
            if abs(correlation) > 0.5:
                # Add to correlated exposure (weighted by correlation)
                correlated_exposure += abs(size) * abs(correlation)
        
        # If correlated exposure is significant, adjust position size
        if correlated_exposure > 0:
            # Calculate what portion of total exposure this represents
            exposure_ratio = correlated_exposure / total_current_exposure if total_current_exposure > 0 else 0
            
            # If our correlation exposure is high, reduce position size
            if exposure_ratio > 0.7:  # 70% of portfolio is correlated
                # Reduce by a factor based on exposure
                reduction_factor = 1.0 - ((exposure_ratio - 0.7) / 0.3) * 0.5  # Max 50% reduction
                reduction_factor = max(0.5, reduction_factor)  # Don't reduce by more than 50%
                
                return position_size * reduction_factor
        
        return position_size
        
    def adjust_for_news_events(self, 
                             stops: Dict[str, float], 
                             entry_price: float, 
                             direction: str,
                             news_impact: Dict[str, Any]) -> Dict[str, float]:
        """
        Adjust stop levels based on upcoming news events.
        
        Args:
            stops: Dictionary with stop prices
            entry_price: The entry price
            direction: Trade direction
            news_impact: Information about upcoming news events
            
        Returns:
            Adjusted stops
        """
        result = stops.copy()
        
        # If high impact news is coming, widen stops
        if news_impact.get('has_high_impact', False):
            # Get time until next high impact event
            hours_until_event = news_impact.get('hours_until_event', 48)
            
            # Only adjust if event is soon but not too soon
            if 1 < hours_until_event < 24:
                # Get expected volatility increase factor
                volatility_factor = news_impact.get('expected_volatility_factor', 1.5)
                
                # Widen stops based on volatility factor
                sl_distance = abs(result['stop_loss'] - entry_price)
                tp_distance = abs(result['take_profit'] - entry_price)
                
                # Apply wider stops
                if direction.lower() == 'long':
                    result['stop_loss'] = entry_price - (sl_distance * volatility_factor)
                    result['take_profit'] = entry_price + (tp_distance * volatility_factor)
                else:
                    result['stop_loss'] = entry_price + (sl_distance * volatility_factor)
                    result['take_profit'] = entry_price - (tp_distance * volatility_factor)
        
        return result

# Example usage
if __name__ == "__main__":
    # Import required classes for testing
    from forex_profile_manager import ForexRiskProfileManager
    
    # Create mocks for testing
    class MockMarketDataClient:
        def get_atr(self, pair, timeframe="1H", periods=14):
            # Mock ATR values
            atrs = {"EURUSD": 0.0012, "GBPUSD": 0.0015, "USDJPY": 0.15}
            return atrs.get(pair, 0.001)
            
    class MockConfluenceClient:
        def get_confluence_zones(self, currency_pair, direction, current_price):
            # Mock confluence zones
            return {
                'support': current_price * 0.995,
                'resistance': current_price * 1.005,
                'profit_targets': [30, 60, 90],
                'confluence_strength': 80
            }
    
    # Create the pipeline with mocks
    risk_manager = ForexRiskProfileManager()
    market_client = MockMarketDataClient()
    confluence_client = MockConfluenceClient()
    
    pipeline = RiskAdjustmentPipeline(
        risk_manager, 
        market_data_client=market_client,
        confluence_analyzer_client=confluence_client
    )
    
    # Test with example position and market data
    position = {
        'instrument': 'EURUSD',
        'direction': 'long',
        'entry_price': 1.0850,
        'size': 10000
    }
    
    market_data = {
        'price': {'EURUSD': 1.0850},
        'volatility': {'EURUSD': 0.006},
        'atr': {'EURUSD': 0.0012},
        'support_resistance': {
            'EURUSD': {
                'supports': [1.0820, 1.0780, 1.0750],
                'resistances': [1.0880, 1.0900, 1.0950]
            }
        }
    }
    
    # Calculate optimal stop levels
    stops = pipeline.determine_optimal_stops(position, market_data, 'trending')
    print(f"Optimal stops for {position['instrument']} {position['direction']}:")
    print(f"Entry: {position['entry_price']}")
    print(f"Stop Loss: {stops['stop_loss']}")
    print(f"Take Profit: {stops['take_profit']}")
    
    # Calculate position size
    position_size = pipeline.calculate_position_size(
        account_balance=10000,
        risk_per_trade=1.0,
        currency_pair='EURUSD',
        entry_price=1.0850,
        stop_loss=stops['stop_loss'],
        market_data=market_data,
        market_regime='trending'
    )
    print(f"Calculated position size: {position_size} units")
