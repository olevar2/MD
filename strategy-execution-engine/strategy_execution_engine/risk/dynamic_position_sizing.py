"""
Dynamic Position Sizing component that leverages adaptive parameters.
"""
from typing import Dict, Any, Optional, List
import logging
import math
logger = logging.getLogger(__name__)


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DynamicPositionSizing:
    """
    Calculates position sizes dynamically based on risk parameters,
    market conditions, and adaptive signals from ML models.
    
    This component integrates with the adaptive layer from Phase 4 to
    adjust position sizes based on signal confidence and market regime.
    """

    def __init__(self, risk_service_client, market_data_client=None,
        adaptive_layer_client=None):
        """
        Initialize the Dynamic Position Sizing component.
        
        Args:
            risk_service_client: Client to interact with risk management service
            market_data_client: Optional client for market data
            adaptive_layer_client: Optional client for adaptive layer parameters
        """
        self.risk_service_client = risk_service_client
        self.market_data_client = market_data_client
        self.adaptive_layer_client = adaptive_layer_client
        self.default_risk_per_trade = 1.0
        self.max_risk_per_trade = 2.0
        self.min_risk_per_trade = 0.5
        self.confidence_scaling_factor = 0.5
        self.regime_scaling_factors = {'trending': 1.2, 'ranging': 0.8,
            'volatile': 0.6, 'breakout': 1.0, 'reversal': 0.7}

    @async_with_exception_handling
    async def calculate_position_size(self, account_balance: float,
        instrument: str, entry_price: float, stop_loss: float, market_data:
        Dict[str, Any], signal_metadata: Dict[str, Any]=None) ->Dict[str, Any]:
        """
        Calculate optimal position size based on risk parameters and market conditions.
        
        Args:
            account_balance: Current account balance
            instrument: The trading instrument (e.g., 'EURUSD')
            entry_price: Planned entry price
            stop_loss: Planned stop loss level
            market_data: Market data including regime, volatility
            signal_metadata: Optional metadata from trading signal including confidence
            
        Returns:
            Dictionary with calculated position size and metadata
        """
        risk_percentage = self.default_risk_per_trade
        market_regime = market_data.get('market_regime', 'unknown')
        volatility = market_data.get('volatility', {}).get(instrument, 0.005)
        position_size = await self._calculate_fixed_percentage_risk(
            account_balance, risk_percentage, entry_price, stop_loss,
            instrument)
        adjustment_factors = {}
        regime_factor = self._adjust_for_market_regime(market_regime)
        adjustment_factors['regime'] = regime_factor
        confidence_factor = 1.0
        if signal_metadata and 'confidence' in signal_metadata:
            confidence_factor = self._adjust_for_signal_confidence(
                signal_metadata['confidence'])
            adjustment_factors['confidence'] = confidence_factor
        confluence_factor = 1.0
        if signal_metadata and 'confluence_strength' in signal_metadata:
            confluence_factor = self._adjust_for_confluence_strength(
                signal_metadata['confluence_strength'])
            adjustment_factors['confluence'] = confluence_factor
        adaptive_factor = 1.0
        if self.adaptive_layer_client:
            try:
                adaptive_params = (await self.adaptive_layer_client.
                    get_adaptive_parameters(instrument, market_regime))
                adaptive_factor = self._apply_adaptive_parameters(
                    adaptive_params)
                adjustment_factors['adaptive'] = adaptive_factor
            except Exception as e:
                logger.warning(f'Failed to get adaptive parameters: {e}')
        final_adjustment = self._combine_adjustment_factors(adjustment_factors)
        adjusted_position_size = position_size * final_adjustment
        max_position_size = await self._get_max_allowed_position_size(
            account_balance, instrument, market_data)
        if adjusted_position_size > max_position_size:
            adjusted_position_size = max_position_size
            logger.info(
                f'Position size reduced to maximum allowed: {max_position_size}'
                )
        final_position_size = self._round_to_lot_size(adjusted_position_size)
        return {'position_size': final_position_size, 'base_position_size':
            position_size, 'adjustment_factors': adjustment_factors,
            'final_adjustment': final_adjustment, 'risk_percentage':
            risk_percentage, 'max_allowed_size': max_position_size,
            'instrument': instrument}

    async def _calculate_fixed_percentage_risk(self, account_balance: float,
        risk_percentage: float, entry_price: float, stop_loss: float,
        instrument: str) ->float:
        """
        Calculate position size based on fixed percentage risk approach.
        
        Args:
            account_balance: Current account balance
            risk_percentage: Risk percentage per trade
            entry_price: Planned entry price
            stop_loss: Planned stop loss level
            instrument: The trading instrument
            
        Returns:
            Calculated position size
        """
        risk_amount = account_balance * (risk_percentage / 100)
        stop_loss_distance = abs(entry_price - stop_loss)
        if stop_loss_distance <= 0:
            logger.warning(
                'Stop loss distance is too small, using minimum distance')
            pip_value = self._get_pip_value(instrument)
            stop_loss_distance = 10 * pip_value
        pip_value = self._get_pip_value(instrument)
        position_size = risk_amount / (stop_loss_distance / pip_value *
            pip_value)
        return position_size

    def _adjust_for_market_regime(self, market_regime: str) ->float:
        """
        Adjust position size based on current market regime.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Adjustment factor
        """
        return self.regime_scaling_factors.get(market_regime, 1.0)

    def _adjust_for_signal_confidence(self, confidence: float) ->float:
        """
        Adjust position size based on signal confidence.
        
        Args:
            confidence: Signal confidence (0-100%)
            
        Returns:
            Adjustment factor
        """
        if confidence > 1.0:
            confidence = confidence / 100.0
        base_factor = 0.75
        max_boost = 1.25
        return base_factor + confidence * self.confidence_scaling_factor

    def _adjust_for_confluence_strength(self, confluence_strength: float
        ) ->float:
        """
        Adjust position size based on confluence strength.
        
        Args:
            confluence_strength: Strength of confluent signals (0-100%)
            
        Returns:
            Adjustment factor
        """
        if confluence_strength > 1.0:
            confluence_strength = confluence_strength / 100.0
        base_factor = 0.8
        max_boost = 0.4
        return base_factor + confluence_strength * max_boost

    def _apply_adaptive_parameters(self, adaptive_params: Dict[str, Any]
        ) ->float:
        """
        Apply adaptive layer parameters to position sizing.
        
        Args:
            adaptive_params: Parameters from the adaptive layer
            
        Returns:
            Adjustment factor
        """
        position_scale = adaptive_params.get('position_scale_factor', 1.0)
        tool_effectiveness = adaptive_params.get('tool_effectiveness', 0.7)
        effectiveness_weight = min(1.0, max(0.0, tool_effectiveness))
        adjustment = position_scale * effectiveness_weight + 1.0 * (1.0 -
            effectiveness_weight)
        return min(1.5, max(0.5, adjustment))

    def _combine_adjustment_factors(self, factors: Dict[str, float]) ->float:
        """
        Combine multiple adjustment factors into a single multiplier.
        
        Args:
            factors: Dictionary of named adjustment factors
            
        Returns:
            Combined adjustment factor
        """
        if not factors:
            return 1.0
        weights = {'regime': 0.3, 'confidence': 0.25, 'confluence': 0.25,
            'adaptive': 0.2}
        default_weight = 1.0 / len(factors)
        weighted_sum = 0
        weight_sum = 0
        for name, factor in factors.items():
            weight = weights.get(name, default_weight)
            weighted_sum += factor * weight
            weight_sum += weight
        if weight_sum == 0:
            return 1.0
        return weighted_sum / weight_sum

    @async_with_exception_handling
    async def _get_max_allowed_position_size(self, account_balance: float,
        instrument: str, market_data: Dict[str, Any]) ->float:
        """
        Get maximum allowed position size based on risk limits.
        
        Args:
            account_balance: Current account balance
            instrument: Trading instrument
            market_data: Current market data
            
        Returns:
            Maximum allowed position size
        """
        if not self.risk_service_client:
            return account_balance * 0.1
        try:
            market_regime = market_data.get('market_regime', 'unknown')
            volatility = market_data.get('volatility', {}).get(instrument, 
                0.005)
            risk_profile = await self.risk_service_client.get_risk_profile(
                instrument, market_regime, volatility)
            max_size = risk_profile.get('max_position_size', 
                account_balance * 0.1)
            return max_size
        except Exception as e:
            logger.error(f'Error getting max position size: {e}')
            return account_balance * 0.1

    def _round_to_lot_size(self, position_size: float) ->float:
        """
        Round position size to standard lot sizes.
        
        Args:
            position_size: Raw position size
            
        Returns:
            Position size rounded to standard lot sizes
        """
        if position_size >= 50000:
            return round(position_size / 100000) * 100000
        elif position_size >= 5000:
            return round(position_size / 10000) * 10000
        else:
            return round(position_size / 1000) * 1000

    def _get_pip_value(self, instrument: str) ->float:
        """
        Get pip value for a currency pair.
        
        Args:
            instrument: Currency pair code
            
        Returns:
            Pip value (0.0001 for most pairs, 0.01 for JPY pairs)
        """
        if instrument.upper().endswith('JPY'):
            return 0.01
        else:
            return 0.0001

    def update_parameters(self, new_params: Dict[str, Any]) ->None:
        """
        Update position sizing parameters.
        
        Args:
            new_params: Dictionary of new parameter values
        """
        if 'default_risk_per_trade' in new_params:
            self.default_risk_per_trade = float(new_params[
                'default_risk_per_trade'])
        if 'max_risk_per_trade' in new_params:
            self.max_risk_per_trade = float(new_params['max_risk_per_trade'])
        if 'min_risk_per_trade' in new_params:
            self.min_risk_per_trade = float(new_params['min_risk_per_trade'])
        if 'confidence_scaling_factor' in new_params:
            self.confidence_scaling_factor = float(new_params[
                'confidence_scaling_factor'])
        if 'regime_scaling_factors' in new_params and isinstance(new_params
            ['regime_scaling_factors'], dict):
            self.regime_scaling_factors.update(new_params[
                'regime_scaling_factors'])
        logger.info(f'Position sizing parameters updated: {vars(self)}')


if __name__ == '__main__':


    class MockRiskServiceClient:
    """
    MockRiskServiceClient class.
    
    Attributes:
        Add attributes here
    """


        async def get_risk_profile(self, instrument, market_regime, volatility
            ):
    """
    Get risk profile.
    
    Args:
        instrument: Description of instrument
        market_regime: Description of market_regime
        volatility: Description of volatility
    
    """

            return {'max_position_size': 50000, 'risk_per_trade_pct': 1.0,
                'max_leverage': 30}


    class MockAdaptiveLayerClient:
    """
    MockAdaptiveLayerClient class.
    
    Attributes:
        Add attributes here
    """


        async def get_adaptive_parameters(self, instrument, market_regime):
    """
    Get adaptive parameters.
    
    Args:
        instrument: Description of instrument
        market_regime: Description of market_regime
    
    """

            return {'position_scale_factor': 1.2, 'tool_effectiveness': 0.85}

    async def run_test():
    """
    Run test.
    
    """

        position_sizer = DynamicPositionSizing(MockRiskServiceClient(),
            adaptive_layer_client=MockAdaptiveLayerClient())
        sizing_result = await position_sizer.calculate_position_size(
            account_balance=10000, instrument='EURUSD', entry_price=1.085,
            stop_loss=1.08, market_data={'market_regime': 'trending',
            'volatility': {'EURUSD': 0.006}}, signal_metadata={'confidence':
            0.8, 'confluence_strength': 75})
        print('Position Sizing Result:')
        for key, value in sizing_result.items():
            print(f'{key}: {value}')
    logging.basicConfig(level=logging.INFO)
    import asyncio
    asyncio.run(run_test())
