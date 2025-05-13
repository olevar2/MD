"""
Market Regime Simulator for Forex Trading Platform.

This module simulates different market regimes with configurable parameters,
providing a realistic basis for testing trading strategies and RL models.
"""
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, AsyncGenerator
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)


from trading_gateway_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MarketRegimeType(Enum):
    """Enum representing different market regime types."""
    TRENDING_BULLISH = 'trending_bullish'
    TRENDING_BEARISH = 'trending_bearish'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    BREAKOUT = 'breakout'
    LIQUIDITY_CRISIS = 'liquidity_crisis'
    CHOPPY = 'choppy'
    LOW_VOLATILITY = 'low_volatility'


class TransitionSpeed(Enum):
    """Enum representing the speed of transition between market regimes."""
    INSTANT = 'instant'
    FAST = 'fast'
    MEDIUM = 'medium'
    SLOW = 'slow'


class MarketRegimeParameters:
    """Class to hold parameters defining a market regime."""

    def __init__(self, regime_type: MarketRegimeType, volatility: float,
        trend_strength: float, mean_reversion_strength: float,
        liquidity_factor: float, jump_probability: float, jump_size_mean:
        float, jump_size_std: float, autocorrelation: float,
        bid_ask_spread_bps: float):
        """
        Initialize market regime parameters.
        
        Args:
            regime_type: Type of market regime
            volatility: Base volatility level (annualized)
            trend_strength: Strength of trending behavior (0 = no trend, 1 = strong trend)
            mean_reversion_strength: Strength of mean reversion (0 = none, 1 = strong mean reversion)
            liquidity_factor: Measure of market liquidity (0 = illiquid, 1 = highly liquid)
            jump_probability: Probability of price jumps per candle
            jump_size_mean: Mean jump size as percentage of price
            jump_size_std: Standard deviation of jump size
            autocorrelation: Price return autocorrelation (-1 to 1)
            bid_ask_spread_bps: Typical bid-ask spread in basis points
        """
        self.regime_type = regime_type
        self.volatility = volatility
        self.trend_strength = trend_strength
        self.mean_reversion_strength = mean_reversion_strength
        self.liquidity_factor = liquidity_factor
        self.jump_probability = jump_probability
        self.jump_size_mean = jump_size_mean
        self.jump_size_std = jump_size_std
        self.autocorrelation = autocorrelation
        self.bid_ask_spread_bps = bid_ask_spread_bps

    @classmethod
    def create_default_parameters(cls, regime_type: MarketRegimeType
        ) ->'MarketRegimeParameters':
        """
        Create default parameters for a given market regime type.
        
        Args:
            regime_type: The market regime type
            
        Returns:
            MarketRegimeParameters with typical default values for the regime type
        """
        defaults = {MarketRegimeType.TRENDING_BULLISH: {'volatility': 0.12,
            'trend_strength': 0.7, 'mean_reversion_strength': 0.1,
            'liquidity_factor': 0.9, 'jump_probability': 0.01,
            'jump_size_mean': 0.001, 'jump_size_std': 0.002,
            'autocorrelation': 0.2, 'bid_ask_spread_bps': 1.0},
            MarketRegimeType.TRENDING_BEARISH: {'volatility': 0.15,
            'trend_strength': -0.7, 'mean_reversion_strength': 0.1,
            'liquidity_factor': 0.85, 'jump_probability': 0.015,
            'jump_size_mean': -0.001, 'jump_size_std': 0.0025,
            'autocorrelation': 0.2, 'bid_ask_spread_bps': 1.2},
            MarketRegimeType.RANGING: {'volatility': 0.08, 'trend_strength':
            0.0, 'mean_reversion_strength': 0.6, 'liquidity_factor': 0.95,
            'jump_probability': 0.005, 'jump_size_mean': 0.0,
            'jump_size_std': 0.001, 'autocorrelation': -0.1,
            'bid_ask_spread_bps': 0.8}, MarketRegimeType.VOLATILE: {
            'volatility': 0.25, 'trend_strength': 0.2,
            'mean_reversion_strength': 0.2, 'liquidity_factor': 0.7,
            'jump_probability': 0.03, 'jump_size_mean': 0.0,
            'jump_size_std': 0.005, 'autocorrelation': 0.05,
            'bid_ask_spread_bps': 2.0}, MarketRegimeType.BREAKOUT: {
            'volatility': 0.18, 'trend_strength': 0.8,
            'mean_reversion_strength': 0.05, 'liquidity_factor': 0.8,
            'jump_probability': 0.08, 'jump_size_mean': 0.002,
            'jump_size_std': 0.004, 'autocorrelation': 0.4,
            'bid_ask_spread_bps': 1.5}, MarketRegimeType.LIQUIDITY_CRISIS:
            {'volatility': 0.4, 'trend_strength': -0.5,
            'mean_reversion_strength': 0.0, 'liquidity_factor': 0.2,
            'jump_probability': 0.15, 'jump_size_mean': -0.005,
            'jump_size_std': 0.01, 'autocorrelation': 0.6,
            'bid_ask_spread_bps': 8.0}, MarketRegimeType.CHOPPY: {
            'volatility': 0.14, 'trend_strength': 0.0,
            'mean_reversion_strength': 0.3, 'liquidity_factor': 0.75,
            'jump_probability': 0.02, 'jump_size_mean': 0.0,
            'jump_size_std': 0.002, 'autocorrelation': -0.2,
            'bid_ask_spread_bps': 1.3}, MarketRegimeType.LOW_VOLATILITY: {
            'volatility': 0.05, 'trend_strength': 0.1,
            'mean_reversion_strength': 0.2, 'liquidity_factor': 0.9,
            'jump_probability': 0.002, 'jump_size_mean': 0.0001,
            'jump_size_std': 0.0005, 'autocorrelation': 0.05,
            'bid_ask_spread_bps': 0.7}}
        params = defaults[regime_type]
        return cls(regime_type=regime_type, **params)


class MarketRegimeGenerator:
    """Class for generating synthetic price data based on market regime parameters."""

    def __init__(self, initial_price: float=1.0, seed: Optional[int]=None):
        """
        Initialize the market regime generator.
        
        Args:
            initial_price: The starting price for the simulation
            seed: Random seed for reproducibility
        """
        self.initial_price = initial_price
        self.rng = np.random.RandomState(seed)
        self.price_history = []
        self.current_price = initial_price

    def _generate_single_candle(self, regime_params: MarketRegimeParameters,
        prev_return: float=0.0) ->Dict[str, float]:
        """
        Generate a single OHLCV candle based on regime parameters.
        
        Args:
            regime_params: Parameters defining the market regime
            prev_return: Previous candle's return for autocorrelation
            
        Returns:
            Dict containing OHLCV values for a single candle
        """
        vol = regime_params.volatility / np.sqrt(252 * 24)
        trend = regime_params.trend_strength * vol * 0.2
        mean_rev = regime_params.mean_reversion_strength
        autocorr = regime_params.autocorrelation
        base_return = trend
        if mean_rev > 0 and len(self.price_history) > 0:
            last_price = self.price_history[-1]['close']
            mean_price = np.mean([c['close'] for c in self.price_history[-20:]]
                )
            if mean_price > 0:
                mean_rev_component = mean_rev * (mean_price - last_price
                    ) / mean_price
                base_return += mean_rev_component
        base_return += autocorr * prev_return
        random_component = self.rng.normal(0, vol)
        jump = 0.0
        if self.rng.random() < regime_params.jump_probability:
            jump = self.rng.normal(regime_params.jump_size_mean,
                regime_params.jump_size_std)
        total_return = base_return + random_component + jump
        open_price = self.current_price
        close_price = open_price * (1 + total_return)
        intracandle_vol = vol * 1.5
        if close_price > open_price:
            high_price = max(open_price, close_price) * (1 + abs(self.rng.
                normal(0, intracandle_vol)))
            low_price = min(open_price, close_price) * (1 - abs(self.rng.
                normal(0, intracandle_vol)))
        else:
            high_price = max(open_price, close_price) * (1 + abs(self.rng.
                normal(0, intracandle_vol)))
            low_price = min(open_price, close_price) * (1 - abs(self.rng.
                normal(0, intracandle_vol)))
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        base_volume = 1000 * regime_params.liquidity_factor
        volume_modifier = 1.0 + 2.0 * abs(total_return) / vol
        volume = int(base_volume * volume_modifier * self.rng.lognormal(0, 0.2)
            )
        candle = {'open': open_price, 'high': high_price, 'low': low_price,
            'close': close_price, 'volume': volume, 'return': total_return,
            'spread_bps': regime_params.bid_ask_spread_bps * (1 + 0.2 * abs
            (self.rng.normal(0, 1)))}
        self.current_price = close_price
        return candle

    def generate_candles(self, regime_params: MarketRegimeParameters,
        num_candles: int, with_timestamp: bool=True, timeframe_minutes: int
        =60, start_time: Optional[datetime]=None) ->pd.DataFrame:
        """
        Generate a sequence of OHLCV candles based on regime parameters.
        
        Args:
            regime_params: Parameters defining the market regime
            num_candles: Number of candles to generate
            with_timestamp: Whether to include timestamps
            timeframe_minutes: Timeframe in minutes (default: 60 for hourly)
            start_time: Starting timestamp (default: current time)
            
        Returns:
            DataFrame containing OHLCV data
        """
        candles = []
        prev_return = 0.0
        self.price_history = []
        for _ in range(num_candles):
            candle = self._generate_single_candle(regime_params, prev_return)
            candles.append(candle)
            self.price_history.append(candle)
            prev_return = candle['return']
        df = pd.DataFrame(candles)
        if with_timestamp:
            if start_time is None:
                start_time = datetime.now().replace(microsecond=0, second=0,
                    minute=0)
            timestamps = [(start_time + timedelta(minutes=i *
                timeframe_minutes)) for i in range(num_candles)]
            df['timestamp'] = timestamps
            df.set_index('timestamp', inplace=True)
        return df


class MarketRegimeSimulator:
    """
    Class for simulating market regime transitions and generating synthetic price data.
    """

    def __init__(self, initial_regime: MarketRegimeType=MarketRegimeType.
        RANGING, initial_price: float=1.0, transition_matrix: Optional[Dict
        [MarketRegimeType, Dict[MarketRegimeType, float]]]=None, seed:
        Optional[int]=None):
        """
        Initialize the market regime simulator.
        
        Args:
            initial_regime: Starting market regime
            initial_price: Starting price
            transition_matrix: Dict mapping regimes to transition probabilities
            seed: Random seed for reproducibility
        """
        self.initial_regime = initial_regime
        self.current_regime = initial_regime
        self.initial_price = initial_price
        self.current_price = initial_price
        self.transition_matrix = (transition_matrix or self.
            _create_default_transition_matrix())
        self.rng = np.random.RandomState(seed)
        self.generator = MarketRegimeGenerator(initial_price=initial_price,
            seed=seed)
        self.regime_history = []
        self.regime_params = MarketRegimeParameters.create_default_parameters(
            initial_regime)

    def _create_default_transition_matrix(self) ->Dict[MarketRegimeType,
        Dict[MarketRegimeType, float]]:
        """
        Create a default transition probability matrix for market regimes.
        
        Returns:
            Dict of dicts representing transition probabilities between regimes
        """
        transition_matrix = {regime: {other: (0.0) for other in
            MarketRegimeType} for regime in MarketRegimeType}
        transition_matrix[MarketRegimeType.TRENDING_BULLISH][MarketRegimeType
            .TRENDING_BULLISH] = 0.9
        transition_matrix[MarketRegimeType.TRENDING_BULLISH][MarketRegimeType
            .RANGING] = 0.05
        transition_matrix[MarketRegimeType.TRENDING_BULLISH][MarketRegimeType
            .VOLATILE] = 0.03
        transition_matrix[MarketRegimeType.TRENDING_BULLISH][MarketRegimeType
            .BREAKOUT] = 0.01
        transition_matrix[MarketRegimeType.TRENDING_BULLISH][MarketRegimeType
            .TRENDING_BEARISH] = 0.01
        transition_matrix[MarketRegimeType.TRENDING_BEARISH][MarketRegimeType
            .TRENDING_BEARISH] = 0.88
        transition_matrix[MarketRegimeType.TRENDING_BEARISH][MarketRegimeType
            .RANGING] = 0.05
        transition_matrix[MarketRegimeType.TRENDING_BEARISH][MarketRegimeType
            .VOLATILE] = 0.04
        transition_matrix[MarketRegimeType.TRENDING_BEARISH][MarketRegimeType
            .LIQUIDITY_CRISIS] = 0.02
        transition_matrix[MarketRegimeType.TRENDING_BEARISH][MarketRegimeType
            .TRENDING_BULLISH] = 0.01
        transition_matrix[MarketRegimeType.RANGING][MarketRegimeType.RANGING
            ] = 0.85
        transition_matrix[MarketRegimeType.RANGING][MarketRegimeType.
            TRENDING_BULLISH] = 0.05
        transition_matrix[MarketRegimeType.RANGING][MarketRegimeType.
            TRENDING_BEARISH] = 0.05
        transition_matrix[MarketRegimeType.RANGING][MarketRegimeType.BREAKOUT
            ] = 0.03
        transition_matrix[MarketRegimeType.RANGING][MarketRegimeType.CHOPPY
            ] = 0.02
        transition_matrix[MarketRegimeType.VOLATILE][MarketRegimeType.VOLATILE
            ] = 0.75
        transition_matrix[MarketRegimeType.VOLATILE][MarketRegimeType.
            TRENDING_BEARISH] = 0.1
        transition_matrix[MarketRegimeType.VOLATILE][MarketRegimeType.
            LIQUIDITY_CRISIS] = 0.05
        transition_matrix[MarketRegimeType.VOLATILE][MarketRegimeType.RANGING
            ] = 0.05
        transition_matrix[MarketRegimeType.VOLATILE][MarketRegimeType.BREAKOUT
            ] = 0.05
        transition_matrix[MarketRegimeType.BREAKOUT][MarketRegimeType.BREAKOUT
            ] = 0.3
        transition_matrix[MarketRegimeType.BREAKOUT][MarketRegimeType.
            TRENDING_BULLISH] = 0.3
        transition_matrix[MarketRegimeType.BREAKOUT][MarketRegimeType.
            TRENDING_BEARISH] = 0.2
        transition_matrix[MarketRegimeType.BREAKOUT][MarketRegimeType.VOLATILE
            ] = 0.15
        transition_matrix[MarketRegimeType.BREAKOUT][MarketRegimeType.RANGING
            ] = 0.05
        transition_matrix[MarketRegimeType.LIQUIDITY_CRISIS][MarketRegimeType
            .LIQUIDITY_CRISIS] = 0.7
        transition_matrix[MarketRegimeType.LIQUIDITY_CRISIS][MarketRegimeType
            .VOLATILE] = 0.2
        transition_matrix[MarketRegimeType.LIQUIDITY_CRISIS][MarketRegimeType
            .TRENDING_BEARISH] = 0.1
        transition_matrix[MarketRegimeType.CHOPPY][MarketRegimeType.CHOPPY
            ] = 0.7
        transition_matrix[MarketRegimeType.CHOPPY][MarketRegimeType.RANGING
            ] = 0.15
        transition_matrix[MarketRegimeType.CHOPPY][MarketRegimeType.
            LOW_VOLATILITY] = 0.1
        transition_matrix[MarketRegimeType.CHOPPY][MarketRegimeType.VOLATILE
            ] = 0.05
        transition_matrix[MarketRegimeType.LOW_VOLATILITY][MarketRegimeType
            .LOW_VOLATILITY] = 0.8
        transition_matrix[MarketRegimeType.LOW_VOLATILITY][MarketRegimeType
            .RANGING] = 0.1
        transition_matrix[MarketRegimeType.LOW_VOLATILITY][MarketRegimeType
            .CHOPPY] = 0.05
        transition_matrix[MarketRegimeType.LOW_VOLATILITY][MarketRegimeType
            .TRENDING_BULLISH] = 0.05
        return transition_matrix

    def _determine_next_regime(self) ->MarketRegimeType:
        """
        Determine the next market regime based on transition probabilities.
        
        Returns:
            Next market regime type
        """
        if self.current_regime not in self.transition_matrix:
            logger.warning(
                f'Current regime {self.current_regime} not found in transition matrix'
                )
            return self.current_regime
        transitions = self.transition_matrix[self.current_regime]
        regimes = list(transitions.keys())
        probabilities = list(transitions.values())
        probabilities = np.array(probabilities)
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
        else:
            return self.current_regime
        next_regime = self.rng.choice(regimes, p=probabilities)
        return next_regime

    def _blend_regime_parameters(self, from_params: MarketRegimeParameters,
        to_params: MarketRegimeParameters, blend_factor: float
        ) ->MarketRegimeParameters:
        """
        Blend parameters between two regimes for smooth transitions.
        
        Args:
            from_params: Starting regime parameters
            to_params: Target regime parameters
            blend_factor: Blending factor (0 = from, 1 = to)
            
        Returns:
            Blended MarketRegimeParameters
        """
        blend_factor = max(0.0, min(1.0, blend_factor))
        blended_params = MarketRegimeParameters(regime_type=to_params.
            regime_type, volatility=from_params.volatility * (1 -
            blend_factor) + to_params.volatility * blend_factor,
            trend_strength=from_params.trend_strength * (1 - blend_factor) +
            to_params.trend_strength * blend_factor,
            mean_reversion_strength=from_params.mean_reversion_strength * (
            1 - blend_factor) + to_params.mean_reversion_strength *
            blend_factor, liquidity_factor=from_params.liquidity_factor * (
            1 - blend_factor) + to_params.liquidity_factor * blend_factor,
            jump_probability=from_params.jump_probability * (1 -
            blend_factor) + to_params.jump_probability * blend_factor,
            jump_size_mean=from_params.jump_size_mean * (1 - blend_factor) +
            to_params.jump_size_mean * blend_factor, jump_size_std=
            from_params.jump_size_std * (1 - blend_factor) + to_params.
            jump_size_std * blend_factor, autocorrelation=from_params.
            autocorrelation * (1 - blend_factor) + to_params.
            autocorrelation * blend_factor, bid_ask_spread_bps=from_params.
            bid_ask_spread_bps * (1 - blend_factor) + to_params.
            bid_ask_spread_bps * blend_factor)
        return blended_params

    def simulate(self, duration: int, timeframe_minutes: int=60,
        regime_change_prob: float=0.01, transition_speed: TransitionSpeed=
        TransitionSpeed.MEDIUM, start_time: Optional[datetime]=None,
        force_regime_changes: Optional[List[Tuple[int, MarketRegimeType]]]=
        None, symbols: Optional[List[str]]=None) ->Tuple[pd.DataFrame, List
        [Dict]]:
        """
        Simulate market data with regime changes.
        
        Args:
            duration: Number of candles to simulate
            timeframe_minutes: Timeframe in minutes
            regime_change_prob: Probability of regime change per candle
            transition_speed: Speed of transition between regimes
            start_time: Starting timestamp
            force_regime_changes: List of tuples (candle_idx, new_regime) for forced regime changes
            
        Returns:
            Tuple of (DataFrame with OHLCV data, List of regime change events)
        """
        if start_time is None:
            start_time = datetime.now().replace(microsecond=0, second=0,
                minute=0)
        self.current_price = self.initial_price
        self.current_regime = self.initial_regime
        self.regime_params = MarketRegimeParameters.create_default_parameters(
            self.current_regime)
        self.generator = MarketRegimeGenerator(initial_price=self.
            initial_price, seed=self.rng.randint(0, 10000))
        self.regime_history = []
        all_candles = []
        regime_changes = []
        transition_durations = {TransitionSpeed.INSTANT: 1, TransitionSpeed
            .FAST: 15, TransitionSpeed.MEDIUM: 75, TransitionSpeed.SLOW: 250}
        transition_duration = transition_durations[transition_speed]
        forced_changes = {}
        if force_regime_changes:
            for candle_idx, new_regime in force_regime_changes:
                forced_changes[candle_idx] = new_regime
        in_transition = False
        transition_progress = 0
        transition_target = None
        from_params = self.regime_params
        to_params = None
        for i in range(duration):
            if i in forced_changes:
                new_regime = forced_changes[i]
                if new_regime != self.current_regime:
                    in_transition = True
                    transition_progress = 0
                    transition_target = new_regime
                    from_params = self.regime_params
                    to_params = (MarketRegimeParameters.
                        create_default_parameters(new_regime))
                    regime_changes.append({'index': i, 'timestamp': 
                        start_time + timedelta(minutes=i *
                        timeframe_minutes), 'from_regime': self.
                        current_regime, 'to_regime': new_regime, 'type':
                        'forced'})
                    self.current_regime = new_regime
            elif not in_transition and self.rng.random() < regime_change_prob:
                new_regime = self._determine_next_regime()
                if new_regime != self.current_regime:
                    in_transition = True
                    transition_progress = 0
                    transition_target = new_regime
                    from_params = self.regime_params
                    to_params = (MarketRegimeParameters.
                        create_default_parameters(new_regime))
                    regime_changes.append({'index': i, 'timestamp': 
                        start_time + timedelta(minutes=i *
                        timeframe_minutes), 'from_regime': self.
                        current_regime, 'to_regime': new_regime, 'type':
                        'probabilistic'})
                    self.current_regime = new_regime
            if in_transition:
                transition_progress += 1
                blend_factor = min(1.0, transition_progress /
                    transition_duration)
                self.regime_params = self._blend_regime_parameters(from_params,
                    to_params, blend_factor)
                if blend_factor >= 1.0:
                    in_transition = False
            candle = self.generator._generate_single_candle(self.regime_params)
            candle['timestamp'] = start_time + timedelta(minutes=i *
                timeframe_minutes)
            candle['regime'] = self.current_regime.value
            all_candles.append(candle)
            self.regime_history.append(self.current_regime)
        df = pd.DataFrame(all_candles)
        if len(df) > 0:
            df.set_index('timestamp', inplace=True)
        return df, regime_changes

    @async_with_exception_handling
    async def generate_data_stream(self, start_date: datetime, end_date:
        datetime, symbols: List[str], timeframe: str='1H') ->AsyncGenerator[
        Tuple[datetime, Dict[str, Dict]], None]:
        """
        Asynchronously streams generated market data for the given symbols and date range.

        This acts as a wrapper around the simulate method, adapting it to the
        streaming interface required by data providers.

        Args:
            start_date: The starting timestamp for data generation.
            end_date: The ending timestamp for data generation.
            symbols: List of symbols to generate data for (Note: current generator is single-symbol).
            timeframe: Pandas frequency string for the timeframe (e.g., '1H', '15T').

        Yields:
            A tuple containing the timestamp and a dictionary mapping symbol
            to its OHLCV data dict for that timestamp.
        """
        logger.info(
            f'Starting data stream generation from {start_date} to {end_date} for {symbols} ({timeframe})'
            )
        try:
            timeframe_delta = pd.to_timedelta(timeframe)
            timeframe_minutes = int(timeframe_delta.total_seconds() / 60)
        except ValueError:
            logger.error(
                f'Invalid timeframe string: {timeframe}. Using default 60 minutes.'
                )
            timeframe_minutes = 60
            timeframe_delta = timedelta(minutes=60)
        duration = int((end_date - start_date) / timeframe_delta)
        if duration <= 0:
            logger.warning(
                'End date is not after start date. No data will be generated.')
            return
        target_symbol = symbols[0] if symbols else 'SYNTHETIC_EURUSD'
        logger.warning(
            f'MarketRegimeSimulator currently generates data for one symbol ({target_symbol}) and replicates it for all requested symbols: {symbols}'
            )
        df_generated, regime_changes = self.simulate(duration=duration,
            timeframe_minutes=timeframe_minutes, start_time=start_date)
        if df_generated.empty:
            logger.warning('Simulation generated an empty DataFrame.')
            return
        for timestamp, row in df_generated.iterrows():
            candle_data = row.to_dict()
            market_data_batch = {}
            for symbol in symbols:
                market_data_batch[symbol] = candle_data.copy()
            yield timestamp, market_data_batch
        logger.info(f'Finished data stream generation for {symbols}')


class MarketReplaySimulator:
    """
    Class for replaying historical market data with regime changes.
    """

    def __init__(self, historical_data: pd.DataFrame, regime_detector=None,
        seed: Optional[int]=None):
        """
        Initialize the market replay simulator.
        
        Args:
            historical_data: DataFrame with OHLCV data
            regime_detector: Optional detector to classify regimes in the historical data
            seed: Random seed for reproducibility
        """
        self.historical_data = historical_data.copy()
        self.rng = np.random.RandomState(seed)
        self.regime_detector = regime_detector
        self.current_index = 0
        self.regime_history = []
        if regime_detector:
            self._detect_regimes()
        else:
            self.historical_data['regime'] = MarketRegimeType.RANGING.value

    def _detect_regimes(self):
        """Detect market regimes in the historical data using the provided detector."""
        if not self.regime_detector:
            return
        regimes = []
        for i in range(len(self.historical_data)):
            window = self.historical_data.iloc[max(0, i - 50):i + 1]
            regime = self.regime_detector.detect_regime(window)
            regimes.append(regime.value)
        self.historical_data['regime'] = regimes

    def inject_regime_change(self, index: int, new_regime: MarketRegimeType,
        duration: int=100, intensity: float=1.0):
        """
        Inject a regime change at a specific index in the historical data.
        
        Args:
            index: Index at which to inject the regime change
            new_regime: The new regime to inject
            duration: How many candles the regime should last
            intensity: Strength of the regime characteristics (0.0-1.0)
        """
        if index < 0 or index >= len(self.historical_data):
            logger.warning(f'Index {index} out of bounds for historical data')
            return
        generator = MarketRegimeGenerator(initial_price=self.
            historical_data.iloc[index]['close'], seed=self.rng.randint(0, 
            10000))
        regime_params = MarketRegimeParameters.create_default_parameters(
            new_regime)
        regime_params.volatility *= intensity
        regime_params.trend_strength *= intensity
        regime_params.mean_reversion_strength *= intensity
        regime_params.jump_probability *= intensity
        synthetic_data = generator.generate_candles(regime_params=
            regime_params, num_candles=duration, with_timestamp=False)
        for i in range(duration):
            if index + i >= len(self.historical_data):
                break
            if duration > 1:
                blend_factor = 1.0 - abs(i - duration / 2) / (duration / 2)
            else:
                blend_factor = 1.0
            blend_factor *= intensity
            orig_candle = self.historical_data.iloc[index + i]
            synth_candle = synthetic_data.iloc[i]
            for col in ['open', 'high', 'low', 'close']:
                orig_val = orig_candle[col]
                synth_val = synth_candle[col]
                scale_factor = orig_val / synth_val if synth_val != 0 else 1.0
                adjusted_synth_val = synth_val * scale_factor
                blended_val = orig_val * (1 - blend_factor
                    ) + adjusted_synth_val * blend_factor
                self.historical_data.at[self.historical_data.index[index +
                    i], col] = blended_val
            self.historical_data.at[self.historical_data.index[index + i],
                'regime'] = new_regime.value

    def inject_market_event(self, index: int, event_type: str, magnitude:
        float=1.0, recovery_duration: int=20):
        """
        Inject a market event like a flash crash at a specific index.
        
        Args:
            index: Index at which to inject the event
            event_type: Type of event ('flash_crash', 'gap_up', 'liquidity_crisis', etc.)
            magnitude: Size of the event impact (0.0-1.0)
            recovery_duration: How many candles until markets normalize
        """
        if index < 0 or index >= len(self.historical_data):
            logger.warning(f'Index {index} out of bounds for historical data')
            return
        current_price = self.historical_data.iloc[index]['close']
        event_params = {'flash_crash': {'price_change': -0.05 * magnitude,
            'volatility_multiplier': 3.0 * magnitude, 'recovery_type':
            'v_shaped'}, 'gap_up': {'price_change': 0.03 * magnitude,
            'volatility_multiplier': 2.0 * magnitude, 'recovery_type':
            'new_level'}, 'liquidity_crisis': {'price_change': -0.08 *
            magnitude, 'volatility_multiplier': 4.0 * magnitude,
            'recovery_type': 'slow'}, 'news_spike': {'price_change': 0.04 *
            magnitude if self.rng.random() > 0.5 else -0.04 * magnitude,
            'volatility_multiplier': 2.5 * magnitude, 'recovery_type':
            'partial'}}
        params = event_params.get(event_type, {'price_change': 0.02 *
            magnitude * (1 if self.rng.random() > 0.5 else -1),
            'volatility_multiplier': 2.0 * magnitude, 'recovery_type':
            'partial'})
        price_change = params['price_change']
        new_price = current_price * (1 + price_change)
        self.historical_data.at[self.historical_data.index[index], 'close'
            ] = new_price
        if price_change < 0:
            self.historical_data.at[self.historical_data.index[index], 'low'
                ] = min(self.historical_data.iloc[index]['low'], new_price *
                (1 - 0.01 * magnitude * self.rng.random()))
        else:
            self.historical_data.at[self.historical_data.index[index], 'high'
                ] = max(self.historical_data.iloc[index]['high'], new_price *
                (1 + 0.01 * magnitude * self.rng.random()))
        recovery_type = params['recovery_type']
        for i in range(1, recovery_duration + 1):
            if index + i >= len(self.historical_data):
                break
            if recovery_type == 'v_shaped':
                recovery_factor = min(1.0, (i / (recovery_duration * 0.3)) ** 2
                    )
            elif recovery_type == 'new_level':
                recovery_factor = 0.1 * (i / recovery_duration)
            elif recovery_type == 'slow':
                recovery_factor = 0.7 * (i / recovery_duration)
            elif recovery_type == 'partial':
                recovery_factor = 0.5 * (1 - np.exp(-3 * i / recovery_duration)
                    )
            else:
                recovery_factor = i / recovery_duration
            original_price = current_price
            target_price = original_price * (1 - price_change * (1 -
                recovery_factor))
            volatility_factor = params['volatility_multiplier'] * (1 - (i /
                recovery_duration) ** 2)
            random_factor = 1.0 + (self.rng.random() - 0.5
                ) * 0.02 * volatility_factor
            adjusted_price = target_price * random_factor
            self.historical_data.at[self.historical_data.index[index + i],
                'close'] = adjusted_price
            if i == 1:
                open_gap = new_price * (1 + (self.rng.random() - 0.5) * 
                    0.01 * volatility_factor)
                self.historical_data.at[self.historical_data.index[index +
                    i], 'open'] = open_gap
            orig_high = self.historical_data.iloc[index + i]['high']
            orig_low = self.historical_data.iloc[index + i]['low']
            orig_open = self.historical_data.iloc[index + i]['open']
            high_increase = (orig_high - orig_open) * volatility_factor
            low_decrease = (orig_open - orig_low) * volatility_factor
            new_high = max(adjusted_price, orig_open) + high_increase
            new_low = min(adjusted_price, orig_open) - low_decrease
            self.historical_data.at[self.historical_data.index[index + i],
                'high'] = new_high
            self.historical_data.at[self.historical_data.index[index + i],
                'low'] = new_low

    def replay(self, start_index: int=0, end_index: Optional[int]=None,
        speed_multiplier: float=1.0, include_regime_changes: bool=True
        ) ->pd.DataFrame:
        """
        Replay the historical data from start to end index.
        
        Args:
            start_index: Starting index for replay
            end_index: Ending index for replay (default: end of data)
            speed_multiplier: Speed of replay (1.0 = realtime)
            include_regime_changes: Whether to include regime information
            
        Returns:
            DataFrame with replayed market data
        """
        if end_index is None:
            end_index = len(self.historical_data) - 1
        start_index = max(0, min(start_index, len(self.historical_data) - 1))
        end_index = max(start_index, min(end_index, len(self.
            historical_data) - 1))
        self.current_index = start_index
        self.regime_history = []
        replay_data = self.historical_data.iloc[start_index:end_index + 1
            ].copy()
        if include_regime_changes:
            for idx, row in replay_data.iterrows():
                regime_str = row.get('regime', MarketRegimeType.RANGING.value)
                if isinstance(regime_str, str):
                    for regime_type in MarketRegimeType:
                        if regime_type.value == regime_str:
                            self.regime_history.append(regime_type)
                            break
                    else:
                        self.regime_history.append(MarketRegimeType.RANGING)
                else:
                    self.regime_history.append(MarketRegimeType.RANGING)
        return replay_data
