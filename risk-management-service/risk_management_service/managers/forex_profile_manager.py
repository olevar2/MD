"""
Manages Forex-specific risk profiles based on market regime and volatility.
"""
from enum import Enum
from typing import Dict, Any, Optional, List, Union

class MarketRegime(Enum):
    """Defines different market regimes for risk management."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"

class RiskProfile:
    """Defines a risk profile for trading with configurable parameters."""
    
    def __init__(
        self, 
        name: str,
        max_position_size: float,
        max_leverage: float,
        stop_loss_pips: int,
        take_profit_pips: int,
        trailing_stop_enabled: bool = False,
        trailing_stop_activation_pips: int = 0,
        max_slippage_pips: int = 5,
        max_spread_pips: int = 3,
        risk_per_trade_pct: float = 1.0,
        volatility_factor: float = 1.0,
        **additional_params
    ):
        self.name = name
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_activation_pips = trailing_stop_activation_pips
        self.max_slippage_pips = max_slippage_pips
        self.max_spread_pips = max_spread_pips
        self.risk_per_trade_pct = risk_per_trade_pct
        self.volatility_factor = volatility_factor
        self.additional_params = additional_params
        
    def adjust_for_volatility(self, volatility: float) -> 'RiskProfile':
        """
        Adjusts the risk profile based on current volatility.
        
        Args:
            volatility: Current market volatility measure
            
        Returns:
            A new adjusted RiskProfile instance
        """
        volatility_ratio = volatility / self.volatility_factor
        
        # If volatility is higher than baseline, reduce position size and tighten stops
        factor = 1.0
        if volatility_ratio > 1.0:
            factor = 1.0 / volatility_ratio
            
        # Create a new profile with adjusted parameters
        return RiskProfile(
            name=f"{self.name}_volatility_adjusted",
            max_position_size=self.max_position_size * factor,
            max_leverage=self.max_leverage * factor,
            # Widen stops in high volatility
            stop_loss_pips=int(self.stop_loss_pips * volatility_ratio),
            take_profit_pips=int(self.take_profit_pips * volatility_ratio),
            trailing_stop_enabled=self.trailing_stop_enabled,
            trailing_stop_activation_pips=int(self.trailing_stop_activation_pips * volatility_ratio),
            max_slippage_pips=int(self.max_slippage_pips * volatility_ratio),
            max_spread_pips=self.max_spread_pips,
            risk_per_trade_pct=self.risk_per_trade_pct * factor,
            volatility_factor=volatility,
            **self.additional_params
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert risk profile to dictionary."""
        result = {
            "name": self.name,
            "max_position_size": self.max_position_size,
            "max_leverage": self.max_leverage,
            "stop_loss_pips": self.stop_loss_pips,
            "take_profit_pips": self.take_profit_pips,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "trailing_stop_activation_pips": self.trailing_stop_activation_pips,
            "max_slippage_pips": self.max_slippage_pips,
            "max_spread_pips": self.max_spread_pips,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "volatility_factor": self.volatility_factor,
        }
        result.update(self.additional_params)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskProfile':
        """Create risk profile from dictionary."""
        # Extract known parameters
        known_params = {
            "name", "max_position_size", "max_leverage", "stop_loss_pips",
            "take_profit_pips", "trailing_stop_enabled", "trailing_stop_activation_pips",
            "max_slippage_pips", "max_spread_pips", "risk_per_trade_pct", "volatility_factor"
        }
        
        # Separate standard parameters from additional ones
        standard_params = {k: v for k, v in data.items() if k in known_params}
        additional_params = {k: v for k, v in data.items() if k not in known_params}
        
        return cls(**standard_params, **additional_params)

class ForexRiskProfileManager:
    """
    Handles risk adjustments based on market regime, volatility,
    and potentially confluence analysis data.
    """

    def __init__(self, initial_profiles=None):
        """
        Initializes the Forex Risk Profile Manager.

        Args:
            initial_profiles: Pre-defined risk profiles if any.
        """
        self.profiles: Dict[str, RiskProfile] = {}
        
        # Load initial profiles if provided
        if initial_profiles:
            for key, profile_data in initial_profiles.items():
                if isinstance(profile_data, RiskProfile):
                    self.profiles[key] = profile_data
                elif isinstance(profile_data, dict):
                    self.profiles[key] = RiskProfile.from_dict(profile_data)
        
        # Add default profiles if none provided
        self._initialize_default_profiles()
    
    def _initialize_default_profiles(self):
        """Initialize default risk profiles if none provided."""
        if not self.profiles:
            # Conservative profile for volatile markets
            self.profiles["conservative"] = RiskProfile(
                name="conservative",
                max_position_size=10000,  # 0.1 standard lot
                max_leverage=20,
                stop_loss_pips=50,
                take_profit_pips=75,
                trailing_stop_enabled=True,
                trailing_stop_activation_pips=30,
                max_slippage_pips=2,
                max_spread_pips=3,
                risk_per_trade_pct=0.5,  # 0.5% risk per trade
                volatility_factor=0.008,  # Higher baseline volatility
            )
            
            # Moderate profile for normal markets
            self.profiles["moderate"] = RiskProfile(
                name="moderate",
                max_position_size=25000,  # 0.25 standard lot
                max_leverage=30,
                stop_loss_pips=40,
                take_profit_pips=80,
                trailing_stop_enabled=True,
                trailing_stop_activation_pips=25,
                max_slippage_pips=3,
                max_spread_pips=4,
                risk_per_trade_pct=1.0,  # 1% risk per trade
                volatility_factor=0.006,  # Medium baseline volatility
            )
            
            # Aggressive profile for trending markets
            self.profiles["aggressive"] = RiskProfile(
                name="aggressive",
                max_position_size=50000,  # 0.5 standard lot
                max_leverage=50,
                stop_loss_pips=60,
                take_profit_pips=120,
                trailing_stop_enabled=True,
                trailing_stop_activation_pips=40,
                max_slippage_pips=4,
                max_spread_pips=5,
                risk_per_trade_pct=2.0,  # 2% risk per trade
                volatility_factor=0.004,  # Lower baseline volatility
            )

    def get_risk_profile(self, currency_pair: str, market_regime: str, volatility: float) -> RiskProfile:
        """
        Retrieves or determines the appropriate risk profile.

        Args:
            currency_pair: The currency pair (e.g., 'EURUSD').
            market_regime: The current market regime (e.g., 'trending', 'ranging').
            volatility: The current market volatility measure.

        Returns:
            A risk profile object adjusted for the current market conditions.
        """
        # Try to find a currency pair specific profile
        pair_specific_key = f"{currency_pair}_{market_regime}"
        
        # Get the base profile based on currency pair and market regime if available
        if pair_specific_key in self.profiles:
            base_profile = self.profiles[pair_specific_key]
        else:
            # Otherwise, choose profile based on market regime
            if market_regime == MarketRegime.TRENDING.value:
                base_profile = self.profiles.get("aggressive", list(self.profiles.values())[0])
            elif market_regime == MarketRegime.VOLATILE.value:
                base_profile = self.profiles.get("conservative", list(self.profiles.values())[0])
            else:  # Default to moderate for ranging, unknown, etc.
                base_profile = self.profiles.get("moderate", list(self.profiles.values())[0])
            
        # Adjust the profile based on current volatility
        adjusted_profile = base_profile.adjust_for_volatility(volatility)
        
        return adjusted_profile

    def update_profile(self, profile_key: str, new_profile_data: Dict[str, Any]) -> None:
        """
        Updates an existing risk profile or creates a new one.
        
        Args:
            profile_key: The key identifying the profile to update
            new_profile_data: Either a RiskProfile object or dictionary of profile data
        """
        if isinstance(new_profile_data, RiskProfile):
            self.profiles[profile_key] = new_profile_data
        elif isinstance(new_profile_data, dict):
            self.profiles[profile_key] = RiskProfile.from_dict(new_profile_data)
        else:
            raise TypeError("new_profile_data must be a RiskProfile object or dictionary")

    def adjust_profile_based_on_confluence(self, profile: RiskProfile, confluence_data: Dict[str, Any]) -> RiskProfile:
        """
        Adjusts a risk profile based on confluence zone analysis.
        (Integration point with Phase 4 components)

        Args:
            profile: The risk profile to adjust.
            confluence_data: Data from confluence analysis (e.g., support/resistance levels).

        Returns:
            The adjusted risk profile.
        """
        # Create a new profile for the adjustments
        adjusted_data = profile.to_dict()
        
        # Adjust stop loss based on nearest support/resistance level
        if 'support' in confluence_data and 'resistance' in confluence_data:
            price = confluence_data.get('current_price', None)
            
            if price:
                # Calculate distance to nearest support/resistance
                distance_to_support = abs(price - confluence_data['support'])
                distance_to_resistance = abs(confluence_data['resistance'] - price)
                
                # Adjust stop loss based on nearest level
                if profile.name == 'aggressive':
                    # For aggressive profiles, set stops closer to support/resistance
                    if confluence_data.get('direction', '') == 'long':
                        # For long positions, stop below support
                        adjusted_data['stop_loss_pips'] = max(15, int(distance_to_support * 0.8 / profile.volatility_factor))
                    else:
                        # For short positions, stop above resistance
                        adjusted_data['stop_loss_pips'] = max(15, int(distance_to_resistance * 0.8 / profile.volatility_factor))
                else:
                    # For other profiles, add some buffer beyond support/resistance
                    if confluence_data.get('direction', '') == 'long':
                        adjusted_data['stop_loss_pips'] = max(20, int(distance_to_support * 1.2 / profile.volatility_factor))
                    else:
                        adjusted_data['stop_loss_pips'] = max(20, int(distance_to_resistance * 1.2 / profile.volatility_factor))
        
        # Adjust take profit based on profit potential zones
        if 'profit_targets' in confluence_data and isinstance(confluence_data['profit_targets'], list) and confluence_data['profit_targets']:
            nearest_target_pips = confluence_data['profit_targets'][0]
            if len(confluence_data['profit_targets']) > 1:
                # If multiple targets, set take profit at second target for better R:R
                adjusted_data['take_profit_pips'] = confluence_data['profit_targets'][1]
                
                # Enable trailing stops with activation at first target
                adjusted_data['trailing_stop_enabled'] = True
                adjusted_data['trailing_stop_activation_pips'] = nearest_target_pips
        
        # Adjust position size based on confluence strength
        if 'confluence_strength' in confluence_data:
            # Scale from 0.5x to 1.5x based on strength (0-100%)
            strength_factor = 0.5 + (confluence_data['confluence_strength'] / 100)
            adjusted_data['max_position_size'] *= strength_factor
        
        # Create new profile with adjustments
        return RiskProfile.from_dict(adjusted_data)

    def get_pair_specific_profile(self, currency_pair: str) -> Optional[Dict[str, RiskProfile]]:
        """
        Returns all profiles specific to a currency pair.
        
        Args:
            currency_pair: The currency pair code, e.g., 'EURUSD'
            
        Returns:
            Dictionary of profiles specific to this currency pair, keyed by market regime
        """
        pair_profiles = {}
        for key, profile in self.profiles.items():
            if key.startswith(f"{currency_pair}_"):
                regime = key.split('_')[1]
                pair_profiles[regime] = profile
        
        return pair_profiles if pair_profiles else None
        
    def create_pair_specific_profile(self, 
                                    currency_pair: str, 
                                    market_regime: str,
                                    base_profile_key: str = None,
                                    adjustments: Dict[str, Any] = None) -> RiskProfile:
        """
        Create a currency pair and market regime specific risk profile.
        
        Args:
            currency_pair: The currency pair code, e.g., 'EURUSD'
            market_regime: The market regime to create a profile for
            base_profile_key: Optional base profile to start from
            adjustments: Optional dictionary of parameter adjustments
            
        Returns:
            The newly created profile
        """
        # Start with a base profile
        if base_profile_key and base_profile_key in self.profiles:
            base_profile = self.profiles[base_profile_key]
        elif market_regime == MarketRegime.TRENDING.value:
            base_profile = self.profiles.get("aggressive", list(self.profiles.values())[0])
        elif market_regime == MarketRegime.VOLATILE.value:
            base_profile = self.profiles.get("conservative", list(self.profiles.values())[0])
        else:
            base_profile = self.profiles.get("moderate", list(self.profiles.values())[0])
        
        # Create new profile data
        profile_data = base_profile.to_dict()
        profile_data["name"] = f"{currency_pair}_{market_regime}"
        
        # Apply any adjustments
        if adjustments:
            profile_data.update(adjustments)
            
        # Create and store the new profile
        profile = RiskProfile.from_dict(profile_data)
        self.profiles[f"{currency_pair}_{market_regime}"] = profile
        
        return profile

    def delete_profile(self, profile_key: str) -> bool:
        """
        Deletes a risk profile.
        
        Args:
            profile_key: The key of the profile to delete
            
        Returns:
            True if the profile was deleted, False if it wasn't found
        """
        if profile_key in self.profiles:
            del self.profiles[profile_key]
            return True
        return False

# Example usage
if __name__ == "__main__":
    manager = ForexRiskProfileManager()
    
    # Get a profile for EURUSD in a trending market with medium volatility
    profile = manager.get_risk_profile('EURUSD', 'trending', 0.005)
    print(f"Retrieved Profile: {profile.to_dict()}")
    
    # Example of creating a pair-specific profile
    custom_profile = manager.create_pair_specific_profile(
        'GBPJPY', 
        'volatile',
        adjustments={
            'max_position_size': 15000,
            'stop_loss_pips': 70,
            'risk_per_trade_pct': 0.75
        }
    )
    print(f"Created custom profile for GBPJPY: {custom_profile.to_dict()}")
    
    # Example confluence data
    confluence = {
        'support': 1.0800, 
        'resistance': 1.0900, 
        'current_price': 1.0850,
        'direction': 'long',
        'profit_targets': [40, 80, 120],
        'confluence_strength': 75
    }
    
    # Adjust profile based on confluence data
    adjusted_profile = manager.adjust_profile_based_on_confluence(profile, confluence)
    print(f"Adjusted Profile with confluence: {adjusted_profile.to_dict()}")
