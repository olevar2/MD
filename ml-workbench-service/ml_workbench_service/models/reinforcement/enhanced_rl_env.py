"""
Enhanced Reinforcement Learning Environment for Forex Trading

This module extends the basic RL agent with advanced features including:
- Multi-timeframe observation space
- Enhanced reward function with risk adjustment
- Comprehensive state representation
- Support for curriculum learning
- Integration with the advanced forex broker simulator
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from datetime import datetime, timedelta
import random
from dataclasses import dataclass, field

from core_foundations.utils.logger import get_logger
from common_lib.simulation.interfaces import (
    IBrokerSimulator, IMarketRegimeSimulator, MarketRegimeType
)
from common_lib.reinforcement.interfaces import IRLEnvironment
from ml_workbench_service.adapters.simulation_adapters import (
    BrokerSimulatorAdapter, MarketRegimeSimulatorAdapter
)

# Import for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from common_lib.risk.interfaces import IRiskParameters

logger = get_logger(__name__)


@dataclass
class RewardComponent:
    """A component of the reward function with weight and function."""
    name: str
    weight: float
    function: Callable
    description: str = ""
    enabled: bool = True
    history: List[float] = field(default_factory=list)

    def calculate(self, *args, **kwargs) -> float:
        """Calculate this reward component's value."""
        if not self.enabled:
            return 0.0

        value = self.function(*args, **kwargs)
        self.history.append(value)
        return value * self.weight


class EnhancedForexTradingEnv(gym.Env):
    """
    Enhanced environment for reinforcement learning in forex trading.

    Features:
    - Multi-timeframe observation space
    - Comprehensive state representation with technical indicators
    - Risk-adjusted reward function
    - Integration with ForexBrokerSimulator for realistic market conditions
    - Support for curriculum learning
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        broker_simulator: IBrokerSimulator,
        symbol: str = "EUR/USD",
        timeframes: List[str] = None,  # e.g. ["1m", "5m", "15m", "1h"]
        lookback_periods: int = 50,
        features: List[str] = None,
        position_sizing_type: str = "fixed",  # "fixed", "dynamic", "risk_based"
        max_position_size: float = 1.0,  # in lots
        trading_fee_percent: float = 0.002,
        reward_mode: str = "risk_adjusted",  # "pnl", "risk_adjusted", "custom"
        risk_free_rate: float = 0.02,  # Annual risk-free rate for Sharpe ratio
        episode_timesteps: int = 1000,
        time_step_seconds: int = 60,  # 1 minute by default
        random_episode_start: bool = True,
        curriculum_level: int = 0,  # 0 = easiest, higher = harder
        include_broker_state: bool = True,
        include_order_book: bool = True,
        include_technical_indicators: bool = True,
        include_news_sentiment: bool = True,  # Include news and sentiment features
        news_sentiment_simulator: Optional[Any] = None,  # News simulator
        custom_reward_function: Optional[Callable] = None,
        observation_normalization: bool = True,
        custom_indicators: Dict[str, Callable] = None
    ):
        """
        Initialize the enhanced forex trading environment.

        Args:
            broker_simulator: Forex broker simulator instance
            symbol: Trading symbol
            timeframes: List of timeframes to include in observation
            lookback_periods: Number of past periods to include in observation
            features: Features to include from market data
            position_sizing_type: How to determine position sizes
            max_position_size: Maximum position size in lots
            trading_fee_percent: Trading fee as percentage
            reward_mode: Type of reward calculation
            risk_free_rate: Annual risk-free rate for risk-adjusted metrics
            episode_timesteps: Maximum timesteps per episode
            time_step_seconds: Seconds to advance per step
            random_episode_start: Whether to start episodes at random positions
            curriculum_level: Difficulty level for curriculum learning
            include_broker_state: Include broker state in observation
            include_order_book: Include order book data in observation
            include_technical_indicators: Include technical indicators in observation
            include_news_sentiment: Include news and sentiment features
            news_sentiment_simulator: Optional news sentiment simulator instance
            custom_reward_function: Optional custom reward function
            observation_normalization: Whether to normalize observations
            custom_indicators: Optional dictionary of custom indicator functions
        """
        super(EnhancedForexTradingEnv, self).__init__()

        self.broker_simulator = broker_simulator
        self.symbol = symbol
        self.timeframes = timeframes if timeframes else ["1m"]
        self.lookback_periods = lookback_periods
        self.features = features if features else ["open", "high", "low", "close", "volume"]
        self.position_sizing_type = position_sizing_type
        self.max_position_size = max_position_size
        self.trading_fee_percent = trading_fee_percent
        self.reward_mode = reward_mode
        self.risk_free_rate = risk_free_rate
        self.max_episode_steps = episode_timesteps
        self.time_step_seconds = time_step_seconds
        self.random_episode_start = random_episode_start
        self.curriculum_level = curriculum_level
        self.include_broker_state = include_broker_state
        self.include_order_book = include_order_book
        self.include_technical_indicators = include_technical_indicators
        self.include_news_sentiment = include_news_sentiment
        self.news_sentiment_simulator = news_sentiment_simulator
        self.custom_reward_function = custom_reward_function
        self.observation_normalization = observation_normalization
        self.custom_indicators = custom_indicators or {}

        # Internal state
        self.current_step = 0
        self.account_balance = 10000.0  # Starting balance
        self.current_pnl = 0.0
        self.current_position = 0.0
        self.trade_history = []
        self.episode_returns = []
        self.total_trades = 0
        self.profitable_trades = 0
        self.current_timestamp = None
        self.state_history = []
        self.reward_history = []
        self.market_data = {}

        # Configure reward components
        self.reward_components = self._setup_reward_components()

        # Set up observation and action spaces
        self._setup_spaces()

    def _setup_spaces(self):
        """Set up the observation and action spaces."""
        # Observation space dimensions
        observation_dim = 0

        # Market data features
        for tf in self.timeframes:
            # For each timeframe, we have lookback_periods of OHLCV data
            observation_dim += len(self.features) * self.lookback_periods

        # Technical indicators
        if self.include_technical_indicators:
            # Common technical indicators
            observation_dim += 10 * len(self.timeframes)  # Basic indicators per timeframe

            # Custom indicators
            observation_dim += len(self.custom_indicators) * len(self.timeframes)

        # Order book features
        if self.include_order_book:
            observation_dim += 20  # Bid/ask prices and volumes at different levels

        # Broker state features
        if self.include_broker_state:
            observation_dim += 5  # Account balance, position, unrealized P&L, etc.

        # News and sentiment features
        if self.include_news_sentiment and self.news_sentiment_simulator:
            observation_dim += self._get_news_sentiment_dimension()

        # Other features based on curriculum level
        if self.curriculum_level > 0:
            observation_dim += self.curriculum_level * 2  # Additional features per level

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32
        )

        # Define action space
        # Actions: [position_size_pct] where -1.0 = max short, 0 = no position, 1.0 = max long
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def _setup_reward_components(self) -> List[RewardComponent]:
        """Set up the reward components based on the selected mode."""
        components = []

        # PnL-based reward
        components.append(
            RewardComponent(
                name="pnl",
                weight=1.0,
                function=lambda env: env.current_pnl,
                description="Profit and loss for the current step"
            )
        )

        if self.reward_mode == "risk_adjusted":
            # Risk-adjusted rewards
            components.extend([
                RewardComponent(
                    name="drawdown_penalty",
                    weight=-0.5,
                    function=lambda env: max(0, calculate_max_drawdown(env.episode_returns)),
                    description="Penalty for maximum drawdown"
                ),
                RewardComponent(
                    name="sharpe_bonus",
                    weight=0.3,
                    function=lambda env: calculate_sharpe_ratio(env.episode_returns, self.risk_free_rate)
                                       if len(env.episode_returns) > 1 else 0.0,
                    description="Bonus for high Sharpe ratio"
                ),
                RewardComponent(
                    name="trade_frequency_penalty",
                    weight=-0.1,
                    function=lambda env: 1.0 if env.total_trades > 0 and env.current_step > 0
                                       and (env.total_trades / env.current_step) > 0.4 else 0.0,
                    description="Penalty for excessive trading"
                )
            ])

            # Additional risk components for higher curriculum levels
            if self.curriculum_level > 1:
                components.append(
                    RewardComponent(
                        name="volatility_penalty",
                        weight=-0.2,
                        function=lambda env: np.std(env.episode_returns) if len(env.episode_returns) > 1 else 0.0,
                        description="Penalty for high return volatility"
                    )
                )

        # Add news sensitivity component if using news
        if self.include_news_sentiment and self.news_sentiment_simulator:
            components.append(
                RewardComponent(
                    name="news_alignment_bonus",
                    weight=0.3,
                    function=self._calculate_news_alignment_reward,
                    description="Bonus for trading aligned with significant news events"
                )
            )

        return components

    def _get_news_sentiment_dimension(self) -> int:
        """Get the dimension size for news and sentiment features."""
        # Basic features: recent events count, average sentiment, highest impact
        base_dim = 3

        # Impact levels
        impact_levels = len(NewsImpactLevel)

        # Sentiment levels
        sentiment_levels = len(SentimentLevel)

        # Currency specific sentiment
        currencies = set()
        if "/" in self.symbol:
            currencies.update(self.symbol.split('/'))

        # Total features
        total_dim = base_dim + impact_levels + sentiment_levels + len(currencies)
        return total_dim

    def _get_news_sentiment_features(self) -> np.ndarray:
        """Extract news and sentiment features relevant to the current time and symbol."""
        if not self.news_sentiment_simulator:
            return np.zeros(self._get_news_sentiment_dimension())

        # Get recent news events
        recent_events = self.news_sentiment_simulator.get_recent_events(
            self.current_timestamp,
            lookback_hours=24,
            relevant_currencies=self.symbol.replace('/', ',')
        )

        # Initialize feature vector
        features = []

        # Basic features
        features.append(len(recent_events))  # Number of recent events

        # Average sentiment (normalized to [-1, 1])
        if recent_events:
            avg_sentiment = sum(event.sentiment_score for event in recent_events) / len(recent_events)
        else:
            avg_sentiment = 0
        features.append(avg_sentiment)

        # Highest impact (normalized to [0, 1])
        if recent_events:
            max_impact = max(event.impact.value for event in recent_events) / len(NewsImpactLevel)
        else:
            max_impact = 0
        features.append(max_impact)

        # Impact level counts (normalized)
        impact_counts = {level: 0 for level in NewsImpactLevel}
        for event in recent_events:
            impact_counts[event.impact] += 1

        for level in NewsImpactLevel:
            features.append(impact_counts[level] / (len(recent_events) if recent_events else 1))

        # Sentiment distributions
        sentiment_counts = {level: 0 for level in SentimentLevel}
        for event in recent_events:
            # Map the continuous sentiment score to a discrete level
            if event.sentiment_score >= 0.5:
                sentiment_counts[SentimentLevel.POSITIVE] += 1
            elif event.sentiment_score <= -0.5:
                sentiment_counts[SentimentLevel.NEGATIVE] += 1
            else:
                sentiment_counts[SentimentLevel.NEUTRAL] += 1

        for level in SentimentLevel:
            features.append(sentiment_counts[level] / (len(recent_events) if recent_events else 1))

        # Currency-specific sentiment
        if "/" in self.symbol:
            currencies = self.symbol.split('/')
            currency_sentiment = {currency: 0 for currency in currencies}

            for event in recent_events:
                for currency in currencies:
                    if currency in event.currencies:
                        currency_sentiment[currency] += event.sentiment_score

            for currency in currencies:
                if recent_events:
                    features.append(currency_sentiment[currency] / len(recent_events))
                else:
                    features.append(0)

        return np.array(features, dtype=np.float32)

    def _calculate_news_alignment_reward(self, env) -> float:
        """Calculate reward component for alignment with significant news events."""
        if not self.news_sentiment_simulator or not env.current_position:
            return 0.0

        # Get very recent high-impact news
        recent_events = self.news_sentiment_simulator.get_recent_events(
            env.current_timestamp,
            lookback_hours=2,
            min_impact=NewsImpactLevel.HIGH,
            relevant_currencies=env.symbol.replace('/', ',')
        )

        if not recent_events:
            return 0.0

        # Calculate average sentiment for these events
        avg_sentiment = sum(event.sentiment_score for event in recent_events) / len(recent_events)

        # If position aligns with sentiment direction, give reward
        position_sign = np.sign(env.current_position)
        sentiment_sign = np.sign(avg_sentiment)

        alignment_score = position_sign * sentiment_sign

        # Scale by the magnitude of sentiment and position
        scaled_score = alignment_score * abs(avg_sentiment) * min(1.0, abs(env.current_position))

        return scaled_score

    def _get_observation(self) -> np.ndarray:
        """Get the current observation based on the environment state."""
        observation = []

        # Market data features for each timeframe
        for tf in self.timeframes:
            # Get market data for this timeframe
            data = self.broker_simulator.get_historical_data(
                symbol=self.symbol,
                timeframe=tf,
                periods=self.lookback_periods,
                end_time=self.current_timestamp
            )

            # Extract features
            for feature in self.features:
                if feature in data.columns:
                    values = data[feature].values

                    # Apply normalization if enabled
                    if self.observation_normalization:
                        # Z-score normalization within the window
                        mean = np.mean(values)
                        std = np.std(values)
                        if std > 0:
                            values = (values - mean) / std

                    observation.extend(values)

        # Technical indicators
        if self.include_technical_indicators:
            for tf in self.timeframes:
                data = self.broker_simulator.get_historical_data(
                    symbol=self.symbol,
                    timeframe=tf,
                    periods=self.lookback_periods + 50,  # Extra periods for indicator calculation
                    end_time=self.current_timestamp
                )

                if len(data) > 0:
                    indicators = self._calculate_indicators(data, tf)
                    observation.extend(indicators)

        # Order book features
        if self.include_order_book:
            order_book = self.broker_simulator.get_order_book(
                symbol=self.symbol,
                timestamp=self.current_timestamp
            )

            if order_book:
                # Top 5 bid and ask prices and volumes
                for i in range(min(5, len(order_book['bids']))):
                    observation.append(order_book['bids'][i][0])  # Price
                    observation.append(order_book['bids'][i][1])  # Volume

                # Fill with zeros if there are fewer than 5 levels
                for i in range(5 - min(5, len(order_book['bids']))):
                    observation.extend([0, 0])

                for i in range(min(5, len(order_book['asks']))):
                    observation.append(order_book['asks'][i][0])  # Price
                    observation.append(order_book['asks'][i][1])  # Volume

                # Fill with zeros if there are fewer than 5 levels
                for i in range(5 - min(5, len(order_book['asks']))):
                    observation.extend([0, 0])

        # Broker state features
        if self.include_broker_state:
            broker_state = [
                self.account_balance / 10000.0,  # Normalize by initial balance
                self.current_position / self.max_position_size,  # Normalize by max position
                self.current_pnl / 1000.0,  # Normalize P&L
                self.total_trades / 100.0,  # Normalize trade count
                self.profitable_trades / max(1, self.total_trades)  # Win rate
            ]
            observation.extend(broker_state)

        # News and sentiment features
        if self.include_news_sentiment and self.news_sentiment_simulator:
            news_features = self._get_news_sentiment_features()
            observation.extend(news_features)

        # Ensure observation has the correct length
        expected_length = self.observation_space.shape[0]
        actual_length = len(observation)

        if actual_length < expected_length:
            # Pad with zeros if shorter than expected
            observation.extend([0] * (expected_length - actual_length))
        elif actual_length > expected_length:
            # Truncate if longer than expected
            observation = observation[:expected_length]

        return np.array(observation, dtype=np.float32)

    def _calculate_indicators(self, data: pd.DataFrame, timeframe: str) -> List[float]:
        """Calculate technical indicators for a specific timeframe."""
        if len(data) < 20:  # Minimum data required
            return [0] * 10

        # Make a copy to avoid modifying the original dataframe
        df = data.copy()

        indicators = []

        # Simple Moving Averages
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()

        # Exponential Moving Averages
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()

        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

        # Add the latest values of these indicators
        for col in ['sma5', 'sma20', 'ema5', 'ema20', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                   'bb_upper', 'bb_lower']:
            # Normalize indicators to comparable scales
            if col == 'rsi':
                # RSI is already in [0, 100], normalize to [0, 1]
                indicators.append(df[col].iloc[-1] / 100 if not pd.isna(df[col].iloc[-1]) else 0.5)
            elif col in ['macd', 'macd_signal', 'macd_hist']:
                # MACD can be both positive and negative, normalize to [-1, 1]
                max_abs = df[col].abs().max()
                indicators.append(df[col].iloc[-1] / max_abs if max_abs > 0 and not pd.isna(df[col].iloc[-1]) else 0)
            else:
                # Price-based indicators, normalize to last close price
                last_close = df['close'].iloc[-1]
                indicators.append((df[col].iloc[-1] / last_close - 1) if last_close > 0 and not pd.isna(df[col].iloc[-1]) else 0)

        # Add custom indicators
        for name, indicator_func in self.custom_indicators.items():
            try:
                value = indicator_func(df)
                indicators.append(value)
            except Exception as e:
                logger.error(f"Error calculating custom indicator {name}: {str(e)}")
                indicators.append(0.0)

        return indicators

    def _take_action(self, action: np.ndarray) -> Tuple[float, Dict]:
        """
        Execute the action in the trading environment.

        Args:
            action: Normalized position size (-1.0 to 1.0)

        Returns:
            reward: The reward for this action
            info: Additional information
        """
        # Convert normalized action to actual position size
        target_position_size = float(action[0]) * self.max_position_size

        # Determine size change needed
        position_change = target_position_size - self.current_position

        # Execute trade if there's a position change
        trade_result = None
        if abs(position_change) > 0.01:  # Minimum position change threshold
            # Determine order details
            side = "buy" if position_change > 0 else "sell"
            size = abs(position_change)

            # Execute order through broker simulator
            trade_result = self.broker_simulator.execute_order(
                symbol=self.symbol,
                side=side,
                order_type="market",
                size=size,
                timestamp=self.current_timestamp
            )

            # Update trade statistics
            self.total_trades += 1

            # Record the trade
            self.trade_history.append({
                'timestamp': self.current_timestamp,
                'side': side,
                'size': size,
                'price': trade_result['avg_price'],
                'fee': trade_result['fee']
            })

        # Update current position
        self.current_position = target_position_size

        # Update P&L based on position and price changes
        current_price = self.broker_simulator.get_current_price(self.symbol, self.current_timestamp)
        previous_pnl = self.current_pnl

        if trade_result:
            # Calculate P&L from the trade
            trade_pnl = trade_result['realized_pnl']

            # Apply trading fees
            fee = trade_result['fee']

            # Update account balance
            self.account_balance += trade_pnl - fee

            # Update current P&L with this step's change
            self.current_pnl = trade_pnl - fee

            # Track profitability
            if trade_pnl > 0:
                self.profitable_trades += 1
        else:
            # No trade executed, P&L comes from holding position
            if self.current_position != 0:
                # Calculate P&L from price change on the existing position
                price_change = self.broker_simulator.get_price_change(
                    symbol=self.symbol,
                    from_time=self.current_timestamp - timedelta(seconds=self.time_step_seconds),
                    to_time=self.current_timestamp
                )

                holding_pnl = self.current_position * price_change
                self.account_balance += holding_pnl
                self.current_pnl = holding_pnl
            else:
                self.current_pnl = 0.0

        # Keep track of returns for risk metrics
        self.episode_returns.append(self.current_pnl)

        # Calculate reward based on components
        reward = self._calculate_reward()

        # Store reward for history
        self.reward_history.append(reward)

        # Return reward and info
        info = {
            'account_balance': self.account_balance,
            'current_position': self.current_position,
            'current_pnl': self.current_pnl,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': self.profitable_trades / max(1, self.total_trades),
            'current_price': current_price,
        }

        return reward, info

    def _calculate_reward(self) -> float:
        """Calculate the reward based on the selected reward mode and components."""
        if self.custom_reward_function:
            return self.custom_reward_function(self)

        total_reward = 0.0

        for component in self.reward_components:
            reward_value = component.calculate(self)
            total_reward += reward_value

        return total_reward

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment by executing the given action.

        Args:
            action: The action to take

        Returns:
            observation: The new observation
            reward: The reward for the action
            done: Whether the episode is finished
            info: Additional information
        """
        # Advance the simulation time
        self.current_timestamp += timedelta(seconds=self.time_step_seconds)
        self.current_step += 1

        # Execute the action and get reward
        reward, info = self._take_action(action)

        # Check if episode is done
        done = False

        # Episode ends if maximum steps reached
        if self.current_step >= self.max_episode_steps:
            done = True

        # Episode ends if account balance drops too low (risk management)
        if self.account_balance <= self.account_balance * 0.1:  # 90% drawdown
            done = True
            # Apply penalty for bankruptcy
            reward -= 10.0

        # Get the next observation
        observation = self._get_observation()

        # Store state for history
        self.state_history.append({
            'timestamp': self.current_timestamp,
            'observation': observation,
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        })

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        """
        Reset the environment to start a new episode.

        Returns:
            observation: The initial observation
        """
        # Reset environment state
        self.current_step = 0
        self.account_balance = 10000.0
        self.current_pnl = 0.0
        self.current_position = 0.0
        self.trade_history = []
        self.episode_returns = []
        self.total_trades = 0
        self.profitable_trades = 0
        self.state_history = []
        self.reward_history = []

        # Reset market data cache
        self.market_data = {}

        # Set initial timestamp
        if self.random_episode_start:
            # Start at a random time within the available data range
            available_times = self.broker_simulator.get_available_times(self.symbol)
            if available_times:
                # Choose a random time that leaves enough room for an episode
                max_idx = max(0, len(available_times) - self.max_episode_steps - 1)
                if max_idx > 0:
                    start_idx = random.randint(0, max_idx)
                    self.current_timestamp = available_times[start_idx]
                else:
                    self.current_timestamp = available_times[0]
            else:
                # Default to current time if no data available
                self.current_timestamp = datetime.now()
        else:
            # Start at the beginning of available data
            available_times = self.broker_simulator.get_available_times(self.symbol)
            if available_times:
                self.current_timestamp = available_times[0]
            else:
                self.current_timestamp = datetime.now()

        # Set the simulation in the broker to the correct timestamp
        self.broker_simulator.set_current_time(self.current_timestamp)

        # Also set the news simulator to the same time if available
        if self.news_sentiment_simulator:
            self.news_sentiment_simulator.set_current_time(self.current_timestamp)

        # Configure environment difficulty based on curriculum level
        self._configure_difficulty()

        # Get initial observation
        observation = self._get_observation()

        return observation

    def _configure_difficulty(self):
        """Configure the environment difficulty based on the curriculum level."""
        if self.curriculum_level == 0:
            # Easiest: Stable market, low volatility, no extreme events
            self.broker_simulator.set_market_regime(MarketRegimeType.NORMAL)
            self.broker_simulator.set_volatility_factor(0.5)
            self.broker_simulator.enable_extreme_events(False)

        elif self.curriculum_level == 1:
            # Medium: Mixed regimes, normal volatility
            self.broker_simulator.set_market_regime(MarketRegimeType.RANGING)
            self.broker_simulator.set_volatility_factor(1.0)
            self.broker_simulator.enable_extreme_events(False)

        elif self.curriculum_level == 2:
            # Hard: More challenging regimes, higher volatility
            regimes = [MarketRegimeType.TRENDING, MarketRegimeType.VOLATILE, MarketRegimeType.RANGING]
            self.broker_simulator.set_market_regime(random.choice(regimes))
            self.broker_simulator.set_volatility_factor(1.5)
            self.broker_simulator.enable_extreme_events(True)
            self.broker_simulator.set_extreme_event_probability(0.01)  # 1% chance per step

        else:  # Level 3+
            # Very hard: All regimes including crisis, high volatility, frequent extreme events
            regimes = list(MarketRegimeType)
            self.broker_simulator.set_market_regime(random.choice(regimes))
            self.broker_simulator.set_volatility_factor(2.0)
            self.broker_simulator.enable_extreme_events(True)
            self.broker_simulator.set_extreme_event_probability(0.02)  # 2% chance per step

    def render(self, mode='human'):
        """Render the environment."""
        if mode != 'human':
            return

        print(f"Step: {self.current_step}, Time: {self.current_timestamp}")
        print(f"Account Balance: ${self.account_balance:.2f}")
        print(f"Current Position: {self.current_position:.4f} lots")
        print(f"Current P&L: ${self.current_pnl:.2f}")
        print(f"Total Trades: {self.total_trades}, Profitable: {self.profitable_trades}")
        if self.total_trades > 0:
            print(f"Win Rate: {self.profitable_trades / self.total_trades:.2%}")
        print(f"Current Price: {self.broker_simulator.get_current_price(self.symbol, self.current_timestamp)}")
        print(f"Current Market Regime: {self.broker_simulator.get_current_market_regime().name}")
        print("-" * 50)

    def close(self):
        """Clean up resources."""
        pass

    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]

    def get_episode_summary(self) -> Dict:
        """Get summary statistics for the completed episode."""
        if not self.trade_history:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_return_per_trade": 0.0,
                "reward_components": {}
            }

        # Calculate performance metrics
        total_return = sum(self.episode_returns)
        sharpe = calculate_sharpe_ratio(self.episode_returns, self.risk_free_rate) if len(self.episode_returns) > 1 else 0.0
        max_dd = calculate_max_drawdown(self.episode_returns)
        win_rate = self.profitable_trades / max(1, self.total_trades)
        avg_return = total_return / max(1, self.total_trades)

        # Get reward component contributions
        reward_components = {}
        for component in self.reward_components:
            if component.history:
                reward_components[component.name] = {
                    "total": sum(component.history),
                    "mean": np.mean(component.history),
                    "min": np.min(component.history),
                    "max": np.max(component.history)
                }

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": self.total_trades,
            "win_rate": win_rate,
            "avg_return_per_trade": avg_return,
            "reward_components": reward_components,
            "final_balance": self.account_balance
        }
