"""
Market scenarios module.

This module provides functionality for...
"""

import random
import datetime
import logging
import math
from typing import List, Dict, Any, Optional, Generator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Placeholder for data models - replace with actual imports if needed
# from analysis_engine.models.market_data import TickData, BarData

# Placeholder for StressTestEnvironment - replace with actual import if available
# from testing.stress_testing.environment import StressTestEnvironment

class MarketScenarioGenerator:
    """Generates specific market scenarios for stress testing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
    """
      init  .
    
    Args:
        config: Description of config
        Any]]: Description of Any]]
    
    """

        self.config = config or {}
        # self.environment: Optional[StressTestEnvironment] = None # Uncomment if StressTestEnvironment is used

    # def set_environment(self, environment: StressTestEnvironment):
    #     """Set the stress test environment."""
    #     self.environment = environment

    def _generate_base_price(self, symbol: str) -> float:
        """Generates a plausible starting base price for a symbol."""
        # Simple example, could be based on symbol characteristics
        if "JPY" in symbol:
            return random.uniform(100, 150)
        elif "GBP" in symbol:
            return random.uniform(1.2, 1.4)
        else:
            return random.uniform(1.0, 1.2)

    def _generate_tick(self, symbol: str, price: float, timestamp: datetime.datetime) -> Dict[str, Any]:
        """Generates a single tick data point."""
        spread = price * random.uniform(0.00005, 0.0002) # Variable spread
        return {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "bid": price - spread / 2,
            "ask": price + spread / 2,
            "type": "tick"
        }

    def generate_market_crash(self,
                              symbols: List[str],
                              duration_seconds: int = 60,
                              ticks_per_second: int = 10,
                              drop_percentage: float = 0.10, # 10% drop
                              volatility_multiplier: float = 5.0
                              ) -> Generator[Dict[str, Any], None, None]:
        """Generates data simulating a market crash."""
        logger.info(f"Generating market crash scenario: duration={duration_seconds}s, drop={drop_percentage*100}%, symbols={symbols}")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        end_time = start_time + datetime.timedelta(seconds=duration_seconds)
        total_ticks = duration_seconds * ticks_per_second

        base_prices = {symbol: self._generate_base_price(symbol) for symbol in symbols}
        current_prices = base_prices.copy()
        target_prices = {symbol: price * (1 - drop_percentage) for symbol, price in base_prices.items()}

        for i in range(total_ticks):
            current_time = start_time + datetime.timedelta(seconds=i / ticks_per_second)
            if current_time > end_time:
                break

            progress = (current_time - start_time).total_seconds() / duration_seconds

            for symbol in symbols:
                # Calculate expected price based on linear drop
                expected_price = base_prices[symbol] + (target_prices[symbol] - base_prices[symbol]) * progress

                # Add volatility - higher volatility during the crash
                noise = random.gauss(0, base_prices[symbol] * 0.0001 * volatility_multiplier)
                current_prices[symbol] = expected_price + noise

                # Ensure price doesn't go negative
                current_prices[symbol] = max(0.0001, current_prices[symbol])

                tick_data = self._generate_tick(symbol, current_prices[symbol], current_time)
                yield tick_data

            # Small delay to simulate real-time generation
            # time.sleep(1.0 / (ticks_per_second * len(symbols)))

        logger.info("Market crash scenario generation complete.")


    def generate_flash_rally(self,
                             symbols: List[str],
                             duration_seconds: int = 30,
                             ticks_per_second: int = 20,
                             rise_percentage: float = 0.08, # 8% rise
                             volatility_multiplier: float = 6.0
                             ) -> Generator[Dict[str, Any], None, None]:
        """Generates data simulating a flash rally."""
        logger.info(f"Generating flash rally scenario: duration={duration_seconds}s, rise={rise_percentage*100}%, symbols={symbols}")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        end_time = start_time + datetime.timedelta(seconds=duration_seconds)
        total_ticks = duration_seconds * ticks_per_second

        base_prices = {symbol: self._generate_base_price(symbol) for symbol in symbols}
        current_prices = base_prices.copy()
        target_prices = {symbol: price * (1 + rise_percentage) for symbol, price in base_prices.items()}

        for i in range(total_ticks):
            current_time = start_time + datetime.timedelta(seconds=i / ticks_per_second)
            if current_time > end_time:
                break

            progress = (current_time - start_time).total_seconds() / duration_seconds

            for symbol in symbols:
                # Calculate expected price based on linear rise
                expected_price = base_prices[symbol] + (target_prices[symbol] - base_prices[symbol]) * progress

                # Add volatility - higher volatility during the rally
                noise = random.gauss(0, base_prices[symbol] * 0.0001 * volatility_multiplier)
                current_prices[symbol] = expected_price + noise

                tick_data = self._generate_tick(symbol, current_prices[symbol], current_time)
                yield tick_data

        logger.info("Flash rally scenario generation complete.")


    def generate_high_volatility(self,
                                 symbols: List[str],
                                 duration_seconds: int = 120,
                                 ticks_per_second: int = 15,
                                 volatility_factor: float = 0.001 # Base volatility factor
                                 ) -> Generator[Dict[str, Any], None, None]:
        """Generates data simulating a period of high volatility without a strong trend."""
        logger.info(f"Generating high volatility scenario: duration={duration_seconds}s, volatility_factor={volatility_factor}, symbols={symbols}")
        start_time = datetime.datetime.now(datetime.timezone.utc)
        end_time = start_time + datetime.timedelta(seconds=duration_seconds)
        total_ticks = duration_seconds * ticks_per_second

        current_prices = {symbol: self._generate_base_price(symbol) for symbol in symbols}

        for i in range(total_ticks):
            current_time = start_time + datetime.timedelta(seconds=i / ticks_per_second)
            if current_time > end_time:
                break

            for symbol in symbols:
                # Generate price movement based on random walk with higher steps
                price_change = random.gauss(0, current_prices[symbol] * volatility_factor)
                current_prices[symbol] += price_change
                # Ensure price doesn't go negative
                current_prices[symbol] = max(0.0001, current_prices[symbol])

                tick_data = self._generate_tick(symbol, current_prices[symbol], current_time)
                yield tick_data

        logger.info("High volatility scenario generation complete.")

    def validate_scenario_data(self, data_generator: Generator[Dict[str, Any], None, None]) -> bool:
        """Basic validation for generated scenario data."""
        count = 0
        max_to_check = 100 # Limit checks for performance
        try:
            for tick in data_generator:
                if not isinstance(tick, dict):
                    logger.error(f"Validation failed: Item is not a dict: {tick}")
                    return False
                required_keys = ["symbol", "timestamp", "bid", "ask", "type"]
                if not all(key in tick for key in required_keys):
                    logger.error(f"Validation failed: Missing keys in tick: {tick}")
                    return False
                if not isinstance(tick["symbol"], str) or not tick["symbol"]:
                    logger.error(f"Validation failed: Invalid symbol: {tick}")
                    return False
                if not isinstance(tick["bid"], (int, float)) or tick["bid"] <= 0:
                    logger.error(f"Validation failed: Invalid bid price: {tick}")
                    return False
                if not isinstance(tick["ask"], (int, float)) or tick["ask"] <= 0:
                     logger.error(f"Validation failed: Invalid ask price: {tick}")
                     return False
                if tick["ask"] < tick["bid"]:
                     logger.error(f"Validation failed: Ask price lower than bid: {tick}")
                     return False
                # TODO: Add timestamp validation (format, sequence)
                count += 1
                if count >= max_to_check:
                    break
            logger.info(f"Validation passed for the first {count} data points.")
            return True
        except Exception as e:
            logger.error(f"Validation failed due to exception: {e}", exc_info=True)
            return False

# --- Example Usage --- 
if __name__ == "__main__":
    scenario_gen = MarketScenarioGenerator()
    test_symbols = ["EURUSD", "GBPJPY"]

    print("\n--- Testing Market Crash ---")
    crash_generator = scenario_gen.generate_market_crash(symbols=test_symbols, duration_seconds=5, ticks_per_second=2)
    if scenario_gen.validate_scenario_data(crash_generator):
        # Need to re-create generator as validation consumes it
        crash_generator = scenario_gen.generate_market_crash(symbols=test_symbols, duration_seconds=5, ticks_per_second=2)
        for i, tick_data in enumerate(crash_generator):
            print(f"Crash Tick {i+1}: {tick_data}")
            if i >= 9: # Print first 10 ticks
                break

    print("\n--- Testing Flash Rally ---")
    rally_generator = scenario_gen.generate_flash_rally(symbols=test_symbols, duration_seconds=5, ticks_per_second=3)
    if scenario_gen.validate_scenario_data(rally_generator):
        rally_generator = scenario_gen.generate_flash_rally(symbols=test_symbols, duration_seconds=5, ticks_per_second=3)
        for i, tick_data in enumerate(rally_generator):
            print(f"Rally Tick {i+1}: {tick_data}")
            if i >= 9: # Print first 10 ticks
                break

    print("\n--- Testing High Volatility ---")
    volatility_generator = scenario_gen.generate_high_volatility(symbols=test_symbols, duration_seconds=5, ticks_per_second=2)
    if scenario_gen.validate_scenario_data(volatility_generator):
        volatility_generator = scenario_gen.generate_high_volatility(symbols=test_symbols, duration_seconds=5, ticks_per_second=2)
        for i, tick_data in enumerate(volatility_generator):
            print(f"Volatility Tick {i+1}: {tick_data}")
            if i >= 9: # Print first 10 ticks
                break

