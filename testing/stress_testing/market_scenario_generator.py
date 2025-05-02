"""
Generates realistic market scenarios and data for stress testing.
"""
import random
import time
import logging
import math # Added for volatility calculations

logger = logging.getLogger(__name__)

class MarketScenarioGenerator:
    """
    Creates simulated market data and events based on configured scenarios.
    Examples: Normal conditions, high volatility, flash crash, low liquidity.
    """

    def __init__(self, config):
        self.config = config # Store the main config
        self.scenario_config = config.get('market_scenario', {}) # Get scenario-specific config
        self.symbols = self.scenario_config.get('symbols', ["EURUSD", "GBPUSD", "USDJPY"])
        self.base_prices = {symbol: self.scenario_config.get('base_prices', {}).get(symbol, 1.1000) for symbol in self.symbols}
        self.current_prices = self.base_prices.copy() # Track current price per symbol
        self.current_scenario = 'normal' # Default scenario
        self.scenario_params = {} # To store parameters of the current scenario
        self._in_flash_crash = False
        self._flash_crash_start_time = 0
        self._flash_crash_symbol = None
        logger.info("Initializing MarketScenarioGenerator...")
        self.set_scenario(self.current_scenario) # Initialize with default scenario params

    def set_scenario(self, scenario_name: str):
        """Sets the active market scenario and loads its parameters."""
        logger.info(f"Setting market scenario to: {scenario_name}")
        self.current_scenario = scenario_name
        # Load parameters specific to this scenario from config, with defaults
        default_params = {
            'volatility': 0.0001, 'liquidity_factor': 1.0, 'spread_multiplier': 1.0,
            'order_rate_multiplier': 1.0, 'flash_crash_chance': 0.0,
            'flash_crash_depth': 0.05, 'flash_crash_duration': 10 # seconds
        }
        self.scenario_params = self.scenario_config.get(scenario_name, default_params)
        # Ensure essential keys exist
        for key, default_value in default_params.items():
            self.scenario_params.setdefault(key, default_value)

        # Reset flash crash state when changing scenarios
        self._in_flash_crash = False
        self._flash_crash_start_time = 0
        self._flash_crash_symbol = None
        logger.debug(f"Scenario '{scenario_name}' parameters: {self.scenario_params}")


    def generate_order_request(self) -> dict | None:
        """
        Generates data for a single order request, potentially influenced by the current scenario.
        """
        # Adjust order generation frequency based on scenario
        if random.random() > self.scenario_params.get('order_rate_multiplier', 1.0):
            return None # Skip generation based on rate multiplier

        symbol = random.choice(self.symbols)
        side = random.choice(["BUY", "SELL"])
        base_quantity = random.randint(1000, 100000)

        # Adjust quantity based on liquidity scenario
        liquidity_factor = self.scenario_params.get('liquidity_factor', 1.0)
        quantity = int(base_quantity * liquidity_factor * random.uniform(0.7, 1.3)) # Add some randomness

        # During flash crash, potentially increase sell orders or decrease buy orders
        if self._in_flash_crash and symbol == self._flash_crash_symbol:
            if side == "BUY":
                quantity = int(quantity * 0.5) # Reduce buy size
            else: # SELL
                quantity = int(quantity * 1.5) # Increase sell size

        order_type = "MARKET" # Keep simple for now

        order = {
            "symbol": symbol,
            "side": side,
            "quantity": max(100, quantity), # Ensure minimum quantity
            "type": order_type,
            "timestamp": time.time(),
            "source_scenario": self.current_scenario
        }
        logger.debug(f"Generated order request: {order}")
        return order

    def _apply_flash_crash(self, symbol: str, current_price: float) -> float:
        """Applies flash crash logic if active."""
        if not self._in_flash_crash:
            # Check if we should trigger a flash crash
            if random.random() < self.scenario_params.get('flash_crash_chance', 0.0):
                self._in_flash_crash = True
                self._flash_crash_start_time = time.time()
                self._flash_crash_symbol = symbol # Crash affects only one symbol at a time
                crash_depth = self.scenario_params.get('flash_crash_depth', 0.05)
                logger.warning(f"FLASH CRASH triggered for {symbol}! Depth: {crash_depth:.2%}")
                return current_price * (1 - crash_depth) # Immediate drop
            else:
                return current_price # No crash triggered
        else:
            # Check if flash crash duration has passed
            crash_duration = self.scenario_params.get('flash_crash_duration', 10)
            if time.time() - self._flash_crash_start_time > crash_duration:
                logger.warning(f"FLASH CRASH ended for {self._flash_crash_symbol}.")
                self._in_flash_crash = False
                self._flash_crash_symbol = None
                # Price might partially recover or stay low depending on model
                return current_price * 1.01 # Simulate small bounce back
            elif symbol == self._flash_crash_symbol:
                # Stay in crashed state (or add some noise)
                return current_price * random.uniform(0.99, 1.01)
            else:
                # Other symbols are not affected
                return current_price


    def generate_market_data_update(self) -> dict | None:
        """
        Generates a simulated market data update (e.g., price tick).
        Influenced by the current scenario (volatility, price jumps, spread, flash crash).
        """
        # Simulate update frequency - maybe tie to volatility? Higher vol = more updates?
        # For now, keep it simple
        if random.random() < 0.2: # Generate update 20% of the time this is called
             return None

        symbol = random.choice(self.symbols)
        last_price = self.current_prices[symbol]

        # Apply flash crash logic first
        price_after_crash_check = self._apply_flash_crash(symbol, last_price)

        # Simulate price movement using Gaussian random walk
        volatility = self.scenario_params.get('volatility', 0.0001)
        # Apply volatility spike effect
        if self.current_scenario == 'volatility_spike':
             volatility *= 5 # Example: 5x volatility during spike

        price_change = random.gauss(0, volatility) * math.sqrt(1/252/8/60) # Simplified time scaling
        current_price = price_after_crash_check * (1 + price_change)

        # Simulate spread, influenced by liquidity and volatility
        base_spread = 0.0001
        spread_multiplier = self.scenario_params.get('spread_multiplier', 1.0)
        liquidity_factor = self.scenario_params.get('liquidity_factor', 1.0)
        # Wider spread in low liquidity or high volatility
        effective_spread_multiplier = spread_multiplier / liquidity_factor # Inverse relationship for liquidity
        if self.current_scenario == 'volatility_spike':
            effective_spread_multiplier *= 1.5 # Further widen spread in spikes

        spread = base_spread * effective_spread_multiplier * random.uniform(0.8, 1.5)

        # Apply flash crash effect on spread (can widen dramatically)
        if self._in_flash_crash and symbol == self._flash_crash_symbol:
            spread *= 3.0 # Dramatically widen spread during crash

        bid = round(current_price - spread / 2, 5)
        ask = round(current_price + spread / 2, 5)

        # Ensure bid < ask
        if bid >= ask:
            bid = round(ask - 0.00001, 5) # Minimal spread if calculation failed

        self.current_prices[symbol] = (bid + ask) / 2 # Update current price based on mid

        update = {
            "symbol": symbol,
            "timestamp": time.time(),
            "bid": bid,
            "ask": ask,
            "source_scenario": self.current_scenario,
            "is_in_flash_crash": self._in_flash_crash and symbol == self._flash_crash_symbol # Add flag
        }
        logger.debug(f"Generated market data update: {update}")
        return update

# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # Mock config with new scenarios and parameters
    mock_config = {
        'market_scenario': {
            'symbols': ['EURUSD', 'GBPUSD'],
            'base_prices': {'EURUSD': 1.1000, 'GBPUSD': 1.2500},
            'normal': {
                'volatility': 0.0001,
                'liquidity_factor': 1.0,
                'spread_multiplier': 1.0,
                'order_rate_multiplier': 1.0,
                'flash_crash_chance': 0.0 # No crashes in normal
            },
            'volatility_spike': {
                'volatility': 0.0005, # Higher base volatility
                'liquidity_factor': 0.8, # Slightly lower liquidity
                'spread_multiplier': 1.5, # Wider spreads
                'order_rate_multiplier': 1.2, # More orders
                'flash_crash_chance': 0.01 # Small chance of crash during spike
            },
            'liquidity_gap': {
                'volatility': 0.0002, # Slightly higher vol
                'liquidity_factor': 0.2, # Significantly lower liquidity
                'spread_multiplier': 2.5, # Much wider spreads
                'order_rate_multiplier': 0.3, # Fewer orders
                'flash_crash_chance': 0.005
            },
            'flash_crash_test': { # Scenario specifically to test crashes
                'volatility': 0.0002,
                'liquidity_factor': 0.5,
                'spread_multiplier': 1.2,
                'order_rate_multiplier': 1.0,
                'flash_crash_chance': 0.1, # High chance to trigger crash for testing
                'flash_crash_depth': 0.08, # 8% drop
                'flash_crash_duration': 5 # seconds
            }
        }
    }

    generator = MarketScenarioGenerator(mock_config)

    scenarios_to_test = ['normal', 'volatility_spike', 'liquidity_gap', 'flash_crash_test']
    for scenario in scenarios_to_test:
        print(f"\n--- Testing Scenario: {scenario.upper()} ---")
        generator.set_scenario(scenario)
        for i in range(10): # Generate a few updates/orders per scenario
            print(f"Iteration {i+1}:")
            order = generator.generate_order_request()
            if order:
                print(f"  Order: {order}")
            update = generator.generate_market_data_update()
            if update:
                print(f"  Update: {update}")
            time.sleep(0.1) # Small delay between iterations

        # If testing flash crash, run longer to observe start/end
        if scenario == 'flash_crash_test':
            print("\n--- Running longer for Flash Crash Test ---")
            start_time = time.time()
            while time.time() - start_time < 15: # Run for 15 seconds
                 update = generator.generate_market_data_update()
                 if update:
                     print(f"  Update: {update}")
                 time.sleep(0.2)
