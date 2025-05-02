\
import asyncio
import httpx
import random
import time
import datetime
import logging
import statistics # Added for latency calculations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple # Added Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Placeholder for StressTestEnvironment - replace with actual import if available
# from testing.stress_testing.environment import StressTestEnvironment

class LoadGenerator(ABC):
    """Abstract base class for different types of load generators."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        # Added latency_stats
        self._metrics = {"generated_count": 0, "errors": 0, "start_time": None, "end_time": None, "latency_ms": []}
        # self.environment: Optional[StressTestEnvironment] = None # Uncomment if StressTestEnvironment is used

    # def set_environment(self, environment: StressTestEnvironment):
    #     """Set the stress test environment."""
    #     self.environment = environment

    @abstractmethod
    async def generate_load(self):
        """The core method to generate the specific type of load."""
        pass

    async def start(self):
        """Start the load generation."""
        if not self.running:
            self.running = True
            self._metrics["start_time"] = time.time()
            logger.info(f"Starting {self.__class__.__name__} with config: {self.config}")
            await self.generate_load()
            self._metrics["end_time"] = time.time()
            logger.info(f"Stopped {self.__class__.__name__}. Metrics: {self.get_metrics()}")

    def stop(self):
        """Signal the load generator to stop."""
        logger.info(f"Stopping {self.__class__.__name__}...")
        self.running = False

    def get_metrics(self) -> Dict[str, Any]:
        """Return the collected metrics."""
        if self._metrics["start_time"] and self._metrics["end_time"]:
            duration = self._metrics["end_time"] - self._metrics["start_time"]
            self._metrics["duration_seconds"] = duration
            if duration > 0:
                self._metrics["rate_per_second"] = self._metrics["generated_count"] / duration
        # Calculate latency statistics
        latencies = self._metrics.get("latency_ms", [])
        if latencies:
            self._metrics["latency_avg_ms"] = statistics.mean(latencies)
            self._metrics["latency_p95_ms"] = statistics.quantiles(latencies, n=100)[94] # 95th percentile
            self._metrics["latency_max_ms"] = max(latencies)
            self._metrics["latency_min_ms"] = min(latencies)
        # Remove raw latency list from final metrics to avoid large output
        self._metrics.pop("latency_ms", None)
        return self._metrics

class MarketDataGenerator(LoadGenerator):
    """Generates synthetic market data (ticks or bars) with support for high volumes and scenarios."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_type = config.get("data_type", "tick") # 'tick' or 'bar'
        self.symbols = config.get("symbols", ["EURUSD", "GBPUSD"])
        self.rate_per_second = config.get("rate_per_second", 10)
        self.batch_size = config.get("batch_size", 1)
        
        # Enhanced configuration for realistic market data
        self.volatility = config.get("volatility", "medium")  # low, medium, high, extreme
        self.spread_model = config.get("spread_model", "normal")  # normal, wide, tight, variable
        self.price_jump_probability = config.get("price_jump_probability", 0.005)  # sudden price movements
        self.correlation_matrix = config.get("correlation_matrix", {})  # Symbol correlation for multi-asset
        
        # Scenario specific patterns
        self.scenario = config.get("scenario", None)  # market_crash, high_volatility, flash_crash, trending
        self.scenario_params = config.get("scenario_params", {})
        
        # Base prices for symbols (will be updated as ticks are generated)
        self._base_prices = {symbol: random.uniform(1.0, 1.3) for symbol in self.symbols}
        if self.scenario == "trending":
            self._trend_directions = {symbol: random.choice([-1, 1]) for symbol in self.symbols}
        
        # Support for multi-core processing for extremely high volumes
        self.use_multiprocessing = config.get("use_multiprocessing", False)
        if self.use_multiprocessing and self.rate_per_second > 50000:
            logger.info("High volume detected, enabling multiprocessing")
            # This would require additional setup with multiprocessing Pool
        
        logger.info(f"MarketDataGenerator initialized with scenario: {self.scenario}, "
                   f"volatility: {self.volatility}, spread: {self.spread_model}")

    def _get_volatility_factor(self) -> float:
        """Returns a multiplier based on configured volatility level."""
        volatility_levels = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.5,
            "extreme": 5.0
        }
        return volatility_levels.get(self.volatility, 1.0)
    
    def _get_spread_factor(self, symbol: str) -> float:
        """Returns appropriate spread based on symbol and configured spread model."""
        base_spread = 0.0001  # 1 pip for majors
        
        # Symbol-specific adjustments
        if "JPY" in symbol:
            base_spread = 0.01  # JPY pairs have different pip values
        elif any(minor in symbol for minor in ["AUD", "NZD", "CAD"]):
            base_spread *= 1.5  # Wider for some pairs
            
        # Apply spread model multiplier
        spread_models = {
            "tight": 0.6,
            "normal": 1.0,
            "wide": 2.0,
            "variable": random.uniform(0.7, 2.5)  # Changes with each call
        }
        
        # In high volatility scenarios, spreads typically widen
        if self.scenario in ["market_crash", "flash_crash"]:
            return base_spread * spread_models.get(self.spread_model, 1.0) * 3.0
        
        return base_spread * spread_models.get(self.spread_model, 1.0)

    async def _generate_tick(self, symbol: str) -> Dict[str, Any]:
        """Generate a realistic market tick with configurable behavior."""
        volatility = self._get_volatility_factor()
        base_price = self._base_prices[symbol]
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        # Determine price movement based on scenario
        price_change = 0
        
        # Handle different scenarios
        if self.scenario == "trending":
            # Create a directional trend
            trend_strength = self.scenario_params.get("trend_strength", 0.00001)
            price_change = self._trend_directions[symbol] * trend_strength
            
            # Add some noise to the trend
            price_change += random.gauss(0, 0.0001 * volatility)
            
        elif self.scenario == "market_crash":
            # Simulated market crash: strong negative bias
            crash_intensity = self.scenario_params.get("crash_intensity", 1.0)
            price_change = random.gauss(-0.0003 * crash_intensity, 0.0005 * volatility)
            
        elif self.scenario == "flash_crash":
            # Occasional extreme drops followed by partial recovery
            if random.random() < 0.01:  # 1% chance of flash crash event
                price_change = -0.01 * random.uniform(0.5, 1.5)  # Sharp drop
                logger.info(f"Flash crash event generated for {symbol}: {price_change:.5f}")
            else:
                price_change = random.gauss(0.0001, 0.0002 * volatility)  # Normal movement
                
        elif self.scenario == "high_volatility":
            # Higher standard deviation in price movements
            price_change = random.gauss(0, 0.0005 * volatility)
            
        else:
            # Default price movement model - random walk with slight mean reversion
            mean_reversion = (1.15 - base_price) * 0.0001  # Tends toward 1.15
            price_change = random.gauss(mean_reversion, 0.0002 * volatility)
        
        # Occasional price jumps
        if random.random() < self.price_jump_probability:
            jump_direction = random.choice([-1, 1])
            jump_size = random.uniform(0.001, 0.003) * volatility
            price_change += jump_direction * jump_size
            logger.debug(f"Price jump in {symbol}: {jump_direction * jump_size:.5f}")
        
        # Update the base price
        new_price = base_price + price_change
        self._base_prices[symbol] = new_price
        
        # Calculate bid/ask with appropriate spread
        spread = self._get_spread_factor(symbol)
        bid = new_price - (spread / 2)
        ask = new_price + (spread / 2)
        
        return {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "bid": bid,
            "ask": ask,
            "type": "tick",
            "volume": random.randint(1, 10),  # Optional volume indicator
            "spread_pips": int(spread * 10000)  # For tracking spread in pips
        }

    async def _generate_bar(self, symbol: str) -> Dict[str, Any]:
        """Generate a realistic OHLC bar with configurable behavior."""
        volatility = self._get_volatility_factor()
        base_price = self._base_prices[symbol]
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        # For bars, we generate multiple movements to simulate intra-bar behavior
        bar_interval = self.config.get("bar_interval", "1m")
        interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(bar_interval, 1)
        
        # Determine number of price movements to simulate inside this bar
        movements = max(5, int(interval_minutes * 60 / 5))  # Simulate a movement every ~5 seconds
        
        prices = [base_price]
        for _ in range(movements):
            last_price = prices[-1]
            
            # Similar logic as tick generation but simplified
            if self.scenario == "trending":
                trend_strength = self.scenario_params.get("trend_strength", 0.00005) 
                change = self._trend_directions[symbol] * trend_strength + random.gauss(0, 0.0002 * volatility)
            elif self.scenario in ["market_crash", "flash_crash"]:
                change = random.gauss(-0.0003, 0.0004 * volatility)
            else:
                mean_reversion = (1.15 - last_price) * 0.0001
                change = random.gauss(mean_reversion, 0.0003 * volatility)
                
            prices.append(last_price + change)
        
        # Extract OHLC from simulated price movements
        open_price = prices[0]
        high_price = max(prices)
        low_price = min(prices)
        close_price = prices[-1]
        
        # Update base price for next bar
        self._base_prices[symbol] = close_price
        
        # Calculate volume with some correlation to price movement
        price_movement = abs(close_price - open_price)
        volume_base = random.randint(100, 1000)
        volume = int(volume_base * (1 + price_movement * 20000))  # Higher volume on larger movements
        
        return {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "type": "bar",
            "interval": bar_interval
        }

    async def generate_load(self):
        """Generates market data and sends it (e.g., to Kafka, websocket)."""
        target_delay = 1.0 / self.rate_per_second if self.rate_per_second > 0 else 0
        
        # For extremely high rates, adjust the batch size dynamically
        if self.rate_per_second > 10000 and self.batch_size < 100:
            logger.info(f"Automatically increasing batch size for high volume generation ({self.rate_per_second}/sec)")
            self.batch_size = min(1000, max(100, self.rate_per_second // 100))
        
        while self.running:
            start_time = time.monotonic()
            batch = []
            try:
                for _ in range(self.batch_size):
                    symbol = random.choice(self.symbols)
                    if self.data_type == "tick":
                        data = await self._generate_tick(symbol)
                    else: # bar
                        data = await self._generate_bar(symbol)
                    batch.append(data)
                    self._metrics["generated_count"] += 1

                # Publish market data batch to configured endpoint
                if batch:
                    logger.debug(f"Generated batch of {len(batch)} market data points.")
                    
                    # Simulate publishing with network latency
                    publish_start = time.monotonic()
                    
                    # TODO: Implement actual publishing logic
                    # Example: await self.environment.kafka_producer.send_batch("market_data", batch)
                    # Example: await self.environment.websocket.broadcast(batch)
                    
                    # Simulate sending time with realistic backpressure
                    if self.rate_per_second > 5000:
                        # Higher volumes can cause destination systems to slow down
                        await asyncio.sleep(0.001 * len(batch) * (1 + random.uniform(0, 0.5)))
                    else:
                        await asyncio.sleep(0.001 * len(batch))
                        
                    publish_duration = (time.monotonic() - publish_start) * 1000
                    self._metrics["latency_ms"].append(publish_duration)
                    
                    # Track publishing latency stats
                    if "max_publish_latency_ms" not in self._metrics:
                        self._metrics["max_publish_latency_ms"] = publish_duration
                    else:
                        self._metrics["max_publish_latency_ms"] = max(
                            self._metrics["max_publish_latency_ms"], publish_duration)

            except Exception as e:
                logger.error(f"Error generating market data: {e}", exc_info=True)
                self._metrics["errors"] += 1
                # Add more detailed error tracking
                error_type = e.__class__.__name__
                if "error_types" not in self._metrics:
                    self._metrics["error_types"] = {}
                self._metrics["error_types"][error_type] = self._metrics["error_types"].get(error_type, 0) + 1

            # Rate limiting with adaptive delay
            elapsed = time.monotonic() - start_time
            sleep_time = target_delay - elapsed
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            elif elapsed > target_delay * 1.5:
                # We're falling behind, log a warning
                if "rate_misses" not in self._metrics:
                    self._metrics["rate_misses"] = 0
                self._metrics["rate_misses"] += 1
                
                if self._metrics["rate_misses"] % 50 == 0:
                    logger.warning(
                        f"MarketDataGenerator falling behind target rate. "
                        f"Target: {self.rate_per_second}/s, Actual: {self.batch_size/elapsed:.1f}/s")

            # Uncomment below for testing high volume with delays
            # await asyncio.sleep(0.1)  # Sleep to simulate time between batches

class ApiRequestGenerator(LoadGenerator):
    """Generates concurrent API requests."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:8000") # Example API base URL
        self.endpoints = config.get("endpoints", [
            {"method": "GET", "path": "/api/v1/portfolio"},
            {"method": "POST", "path": "/api/v1/orders", "data": {"symbol": "EURUSD", "amount": 100, "type": "MARKET", "side": "BUY"}}
        ])
        self.concurrency = config.get("concurrency", 10)
        self.rate_per_second = config.get("rate_per_second", 5) # Target total rate across all concurrent tasks

    async def _make_request(self, client: httpx.AsyncClient, endpoint_config: Dict[str, Any]):
        """Makes a single API request and measures latency."""
        method = endpoint_config.get("method", "GET").upper()
        path = endpoint_config.get("path", "/")
        url = f"{self.base_url}{path}"
        data = endpoint_config.get("data")
        params = endpoint_config.get("params")
        headers = endpoint_config.get("headers") # TODO: Add auth headers if needed

        start_time = time.monotonic()
        try:
            logger.debug(f"Making {method} request to {url}")
            response = await client.request(method, url, json=data, params=params, headers=headers, timeout=10.0)
            latency_ms = (time.monotonic() - start_time) * 1000
            response.raise_for_status() # Raise exception for 4xx/5xx status codes
            self._metrics["generated_count"] += 1
            self._metrics["latency_ms"].append(latency_ms) # Record latency
            logger.debug(f"Request to {url} successful (Status: {response.status_code}, Latency: {latency_ms:.2f}ms)")
        except httpx.RequestError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.warning(f"API Request failed: {e.__class__.__name__} - {e.request.url} (Latency: {latency_ms:.2f}ms)")
            self._metrics["errors"] += 1
            # Optionally record latency even for errors, depending on requirements
            # self._metrics["latency_ms"].append(latency_ms)
        except httpx.HTTPStatusError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.warning(f"API Request failed: Status {e.response.status_code} for url {e.request.url} (Latency: {latency_ms:.2f}ms)")
            self._metrics["errors"] += 1
            # Optionally record latency even for errors
            # self._metrics["latency_ms"].append(latency_ms)
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"Unexpected error during API request: {e} (Latency: {latency_ms:.2f}ms)", exc_info=True)
            self._metrics["errors"] += 1

    async def _worker(self, client: httpx.AsyncClient, worker_id: int):
        """A single worker task making requests."""
        target_delay = (1.0 / self.rate_per_second) * self.concurrency if self.rate_per_second > 0 else 0
        logger.info(f"Worker {worker_id} started. Target delay: {target_delay:.4f}s")
        while self.running:
            start_time = time.monotonic()
            endpoint_config = random.choice(self.endpoints)
            await self._make_request(client, endpoint_config)

            # Rate limiting per worker
            elapsed = time.monotonic() - start_time
            sleep_time = target_delay - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                 # Yield control if running behind schedule
                 await asyncio.sleep(0)
        logger.info(f"Worker {worker_id} stopped.")


    async def generate_load(self):
        """Starts concurrent workers to make API requests."""
        async with httpx.AsyncClient() as client:
            tasks = [asyncio.create_task(self._worker(client, i)) for i in range(self.concurrency)]
            # Keep running until stop() is called
            while self.running:
                await asyncio.sleep(0.5) # Check running flag periodically
            # Wait for tasks to finish after stop() is called
            await asyncio.gather(*tasks, return_exceptions=True)


class UserActionSimulator(LoadGenerator):
    """Simulates realistic user actions (e.g., placing orders, checking portfolio)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Example config: sequence of actions, probabilities, user IDs
        self.user_count = config.get("user_count", 5)
        self.actions_per_user_per_minute = config.get("actions_per_user_per_minute", 2)
        # Define possible actions and their simulation logic (e.g., calling API endpoints)
        self.possible_actions = [
            self._check_portfolio,
            self._place_market_order,
            self._check_market_data
        ]

    async def _check_portfolio(self, user_id: int):
        logger.info(f"User {user_id}: Checking portfolio")
        # TODO: Implement API call to portfolio service
        await asyncio.sleep(random.uniform(0.1, 0.5)) # Simulate action time
        self._metrics["generated_count"] += 1

    async def _place_market_order(self, user_id: int):
        symbol = random.choice(["EURUSD", "USDJPY", "AUDUSD"])
        amount = random.randint(1, 10) * 1000
        side = random.choice(["BUY", "SELL"])
        logger.info(f"User {user_id}: Placing {side} order for {amount} {symbol}")
        # TODO: Implement API call to order execution service
        await asyncio.sleep(random.uniform(0.2, 1.0)) # Simulate action time
        self._metrics["generated_count"] += 1

    async def _check_market_data(self, user_id: int):
        symbol = random.choice(["EURUSD", "USDJPY", "AUDUSD"])
        logger.info(f"User {user_id}: Checking market data for {symbol}")
        # TODO: Implement API call to data service or simulate interaction
        await asyncio.sleep(random.uniform(0.1, 0.3)) # Simulate action time
        self._metrics["generated_count"] += 1

    async def _user_simulator(self, user_id: int):
        """Simulates actions for a single user."""
        actions_per_second = self.actions_per_user_per_minute / 60.0
        target_delay = 1.0 / actions_per_second if actions_per_second > 0 else float('inf')
        logger.info(f"User simulator {user_id} started. Target delay: {target_delay:.2f}s")

        while self.running:
            start_time = time.monotonic()
            try:
                action = random.choice(self.possible_actions)
                await action(user_id)
            except Exception as e:
                logger.error(f"Error simulating action for user {user_id}: {e}", exc_info=True)
                self._metrics["errors"] += 1

            # Rate limiting per user
            elapsed = time.monotonic() - start_time
            sleep_time = target_delay - elapsed
            if sleep_time > 0 and sleep_time != float('inf'):
                await asyncio.sleep(sleep_time)
            elif sleep_time != float('inf'):
                 # Yield control if running behind schedule
                 await asyncio.sleep(0)
            else:
                # If rate is 0 or less, wait indefinitely until stopped
                while self.running:
                    await asyncio.sleep(1)


        logger.info(f"User simulator {user_id} stopped.")


    async def generate_load(self):
        """Starts concurrent user simulators."""
        tasks = [asyncio.create_task(self._user_simulator(i)) for i in range(self.user_count)]
        # Keep running until stop() is called
        while self.running:
            await asyncio.sleep(0.5) # Check running flag periodically
        # Wait for tasks to finish after stop() is called
        await asyncio.gather(*tasks, return_exceptions=True)


# --- Example Usage ---
async def main():
    # Example configurations
    market_data_config = {
        "data_type": "tick",
        "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
        "rate_per_second": 50,
        "batch_size": 5
    }
    api_config = {
        "base_url": "http://localhost:8001", # Adjust if your API runs elsewhere
        "endpoints": [
            {"method": "GET", "path": "/api/v1/health"},
            {"method": "GET", "path": "/api/v1/marketdata/latest/EURUSD"},
            # Add more realistic endpoints for your services
        ],
        "concurrency": 20,
        "rate_per_second": 100 # Total target rate
    }
    user_sim_config = {
        "user_count": 10,
        "actions_per_user_per_minute": 5
    }

    # Create generator instances
    market_gen = MarketDataGenerator(market_data_config)
    api_gen = ApiRequestGenerator(api_config)
    user_sim = UserActionSimulator(user_sim_config)

    # Start generators (run them concurrently)
    logger.info("Starting load generators...")
    tasks = [
        asyncio.create_task(market_gen.start()),
        asyncio.create_task(api_gen.start()),
        asyncio.create_task(user_sim.start())
    ]

    # Run for a specific duration (e.g., 60 seconds)
    duration = 60
    await asyncio.sleep(duration)

    # Stop generators
    logger.info("Stopping load generators...")
    market_gen.stop()
    api_gen.stop()
    user_sim.stop()

    # Wait for generators to finish cleanly
    await asyncio.gather(*tasks)

    # Print metrics
    logger.info("--- Metrics ---")
    logger.info(f"Market Data Generator: {market_gen.get_metrics()}")
    logger.info(f"API Request Generator: {api_gen.get_metrics()}")
    logger.info(f"User Action Simulator: {user_sim.get_metrics()}")
    logger.info("Load generation finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Load generation interrupted by user.")

