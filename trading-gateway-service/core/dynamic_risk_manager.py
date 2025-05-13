"""
Dynamic Risk Management module for Forex Trading Platform.

This module provides adaptive risk parameter management based on market conditions,
enabling trading strategies to adjust risk parameters in real-time as market regimes change.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import datetime
from dataclasses import dataclass
import logging
import json
import os
from abc import ABC, abstractmethod
from core.forex_broker_simulator import ForexBrokerSimulator
from core.market_regime_simulator import MarketRegimeType
from core.advanced_market_regime_simulator import AdvancedMarketRegimeSimulator, MarketCondition
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class RiskTolerance(str, Enum):
    """Risk tolerance levels for trading systems."""
    VERY_CONSERVATIVE = 'very_conservative'
    CONSERVATIVE = 'conservative'
    MODERATE = 'moderate'
    AGGRESSIVE = 'aggressive'
    VERY_AGGRESSIVE = 'very_aggressive'


class RiskParameter(str, Enum):
    """Risk parameter types that can be dynamically adjusted."""
    POSITION_SIZE = 'position_size'
    MAX_DRAWDOWN = 'max_drawdown'
    STOP_LOSS_DISTANCE = 'stop_loss_distance'
    TAKE_PROFIT_DISTANCE = 'take_profit_distance'
    MAX_POSITIONS = 'max_positions'
    LEVERAGE = 'leverage'
    MAX_CORRELATION = 'max_correlation'
    MIN_REWARD_RISK = 'min_reward_risk'
    MAX_DAILY_LOSS = 'max_daily_loss'
    MAX_TRADE_SIZE = 'max_trade_size'


@dataclass
class RiskProfile:
    """Risk profile configuration with adjustable parameters."""
    name: str
    tolerance: RiskTolerance
    position_size_pct: float
    max_drawdown_pct: float
    stop_loss_pips: Dict[str, float]
    take_profit_pips: Dict[str, float]
    max_positions: int
    leverage: float
    max_correlation: float
    min_reward_risk: float
    max_daily_loss_pct: float
    max_trade_size_pct: float

    @classmethod
    def create_default(cls, tolerance: RiskTolerance) ->'RiskProfile':
        """Create a default risk profile for a given tolerance level."""
        if tolerance == RiskTolerance.VERY_CONSERVATIVE:
            return cls(name='Very Conservative', tolerance=tolerance,
                position_size_pct=0.5, max_drawdown_pct=2.0, stop_loss_pips
                ={'DEFAULT': 20.0}, take_profit_pips={'DEFAULT': 40.0},
                max_positions=2, leverage=1.0, max_correlation=0.3,
                min_reward_risk=3.0, max_daily_loss_pct=1.0,
                max_trade_size_pct=1.0)
        elif tolerance == RiskTolerance.CONSERVATIVE:
            return cls(name='Conservative', tolerance=tolerance,
                position_size_pct=1.0, max_drawdown_pct=5.0, stop_loss_pips
                ={'DEFAULT': 25.0}, take_profit_pips={'DEFAULT': 37.5},
                max_positions=4, leverage=2.0, max_correlation=0.4,
                min_reward_risk=2.5, max_daily_loss_pct=2.0,
                max_trade_size_pct=2.0)
        elif tolerance == RiskTolerance.MODERATE:
            return cls(name='Moderate', tolerance=tolerance,
                position_size_pct=2.0, max_drawdown_pct=10.0,
                stop_loss_pips={'DEFAULT': 30.0}, take_profit_pips={
                'DEFAULT': 45.0}, max_positions=6, leverage=5.0,
                max_correlation=0.6, min_reward_risk=2.0,
                max_daily_loss_pct=3.0, max_trade_size_pct=3.0)
        elif tolerance == RiskTolerance.AGGRESSIVE:
            return cls(name='Aggressive', tolerance=tolerance,
                position_size_pct=3.0, max_drawdown_pct=15.0,
                stop_loss_pips={'DEFAULT': 40.0}, take_profit_pips={
                'DEFAULT': 60.0}, max_positions=10, leverage=10.0,
                max_correlation=0.7, min_reward_risk=1.5,
                max_daily_loss_pct=5.0, max_trade_size_pct=5.0)
        else:
            return cls(name='Very Aggressive', tolerance=tolerance,
                position_size_pct=5.0, max_drawdown_pct=25.0,
                stop_loss_pips={'DEFAULT': 50.0}, take_profit_pips={
                'DEFAULT': 75.0}, max_positions=15, leverage=20.0,
                max_correlation=0.8, min_reward_risk=1.0,
                max_daily_loss_pct=8.0, max_trade_size_pct=8.0)

    def adjust_for_symbol(self, symbol: str, volatility_factor: float=1.0
        ) ->None:
        """Adjust risk parameters for a specific symbol based on its volatility."""
        if symbol not in self.stop_loss_pips:
            self.stop_loss_pips[symbol] = self.stop_loss_pips['DEFAULT'
                ] * volatility_factor
        if symbol not in self.take_profit_pips:
            self.take_profit_pips[symbol] = self.take_profit_pips['DEFAULT'
                ] * volatility_factor

    def to_dict(self) ->Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {'name': self.name, 'tolerance': self.tolerance.value,
            'position_size_pct': self.position_size_pct, 'max_drawdown_pct':
            self.max_drawdown_pct, 'stop_loss_pips': self.stop_loss_pips,
            'take_profit_pips': self.take_profit_pips, 'max_positions':
            self.max_positions, 'leverage': self.leverage,
            'max_correlation': self.max_correlation, 'min_reward_risk':
            self.min_reward_risk, 'max_daily_loss_pct': self.
            max_daily_loss_pct, 'max_trade_size_pct': self.max_trade_size_pct}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->'RiskProfile':
        """Create from dictionary."""
        return cls(name=data['name'], tolerance=RiskTolerance(data[
            'tolerance']), position_size_pct=data['position_size_pct'],
            max_drawdown_pct=data['max_drawdown_pct'], stop_loss_pips=data[
            'stop_loss_pips'], take_profit_pips=data['take_profit_pips'],
            max_positions=data['max_positions'], leverage=data['leverage'],
            max_correlation=data['max_correlation'], min_reward_risk=data[
            'min_reward_risk'], max_daily_loss_pct=data[
            'max_daily_loss_pct'], max_trade_size_pct=data[
            'max_trade_size_pct'])


class RiskAdjustmentStrategy(ABC):
    """Base class for risk adjustment strategies."""

    @abstractmethod
    def adjust_risk_profile(self, profile: RiskProfile, market_condition:
        MarketCondition, volatility: float, account_stats: Dict[str, Any]
        ) ->RiskProfile:
        """
        Adjust risk profile based on market conditions and account statistics.
        
        Args:
            profile: Base risk profile
            market_condition: Current market condition
            volatility: Current market volatility
            account_stats: Account statistics
            
        Returns:
            Adjusted risk profile
        """
        pass


class ConservativeTrendFollowingRiskStrategy(RiskAdjustmentStrategy):
    """
    Risk strategy that increases position size and reward-risk ratio in trending markets,
    but reduces risk in volatile or choppy conditions.
    """

    def adjust_risk_profile(self, profile: RiskProfile, market_condition:
        MarketCondition, volatility: float, account_stats: Dict[str, Any]
        ) ->RiskProfile:
    """
    Adjust risk profile.
    
    Args:
        profile: Description of profile
        market_condition: Description of market_condition
        volatility: Description of volatility
        account_stats: Description of account_stats
        Any]: Description of Any]
    
    Returns:
        RiskProfile: Description of return value
    
    """

        adjusted = RiskProfile.from_dict(profile.to_dict())
        if market_condition in [MarketCondition.TRENDING_BULLISH,
            MarketCondition.TRENDING_BEARISH]:
            adjusted.position_size_pct *= 1.2
            for symbol in adjusted.take_profit_pips:
                adjusted.take_profit_pips[symbol] *= 1.3
            adjusted.min_reward_risk *= 1.2
        elif market_condition in [MarketCondition.HIGH_VOLATILITY,
            MarketCondition.FLASH_CRASH, MarketCondition.FLASH_SPIKE]:
            adjusted.position_size_pct *= 0.5
            for symbol in adjusted.stop_loss_pips:
                adjusted.stop_loss_pips[symbol] *= 1.5
            adjusted.max_positions = max(1, adjusted.max_positions // 2)
            adjusted.leverage = max(1.0, adjusted.leverage * 0.5)
        elif market_condition in [MarketCondition.CHOPPY, MarketCondition.
            RANGING_NARROW]:
            adjusted.position_size_pct *= 0.8
            for symbol in adjusted.stop_loss_pips:
                adjusted.stop_loss_pips[symbol] *= 0.8
            for symbol in adjusted.take_profit_pips:
                adjusted.take_profit_pips[symbol] *= 0.8
        volatility_adjustment = min(2.0, max(0.5, volatility))
        for symbol in adjusted.stop_loss_pips:
            adjusted.stop_loss_pips[symbol] *= volatility_adjustment
        if account_stats.get('consecutive_losses', 0) > 3:
            adjusted.position_size_pct *= 0.7
            adjusted.max_positions = max(1, adjusted.max_positions - 1)
        if account_stats.get('current_drawdown_pct', 0
            ) > adjusted.max_drawdown_pct * 0.7:
            adjusted.position_size_pct *= 0.5
            adjusted.max_positions = max(1, adjusted.max_positions // 2)
        adjusted.position_size_pct = min(profile.position_size_pct * 1.5,
            max(profile.position_size_pct * 0.25, adjusted.position_size_pct))
        adjusted.max_positions = min(profile.max_positions * 2, max(1,
            adjusted.max_positions))
        adjusted.leverage = min(profile.leverage * 1.5, max(1.0, adjusted.
            leverage))
        return adjusted


class AggressiveBreakoutRiskStrategy(RiskAdjustmentStrategy):
    """
    Risk strategy that increases position size and leverage during breakouts,
    but maintains tight stops to control risk.
    """

    def adjust_risk_profile(self, profile: RiskProfile, market_condition:
        MarketCondition, volatility: float, account_stats: Dict[str, Any]
        ) ->RiskProfile:
    """
    Adjust risk profile.
    
    Args:
        profile: Description of profile
        market_condition: Description of market_condition
        volatility: Description of volatility
        account_stats: Description of account_stats
        Any]: Description of Any]
    
    Returns:
        RiskProfile: Description of return value
    
    """

        adjusted = RiskProfile.from_dict(profile.to_dict())
        if market_condition in [MarketCondition.BREAKOUT_BULLISH,
            MarketCondition.BREAKOUT_BEARISH]:
            adjusted.position_size_pct *= 1.5
            adjusted.leverage *= 1.3
        elif market_condition in [MarketCondition.REVERSAL_BULLISH,
            MarketCondition.REVERSAL_BEARISH]:
            adjusted.position_size_pct *= 1.2
            for symbol in adjusted.stop_loss_pips:
                adjusted.stop_loss_pips[symbol] *= 1.2
        elif market_condition in [MarketCondition.RANGING_WIDE,
            MarketCondition.RANGING_NARROW]:
            adjusted.position_size_pct *= 0.7
            adjusted.leverage *= 0.8
        if account_stats.get('profit_factor', 1.0) > 1.5:
            adjusted.position_size_pct *= 1.2
            adjusted.max_positions += 1
        volatility_adjustment = min(2.0, max(0.5, volatility))
        if volatility_adjustment > 1.5:
            adjusted.position_size_pct *= 0.8
            adjusted.leverage = max(1.0, adjusted.leverage * 0.7)
        adjusted.position_size_pct = min(profile.position_size_pct * 2.0,
            max(profile.position_size_pct * 0.5, adjusted.position_size_pct))
        adjusted.max_positions = min(profile.max_positions * 2, max(1,
            adjusted.max_positions))
        adjusted.leverage = min(profile.leverage * 1.5, max(1.0, adjusted.
            leverage))
        return adjusted


class MeanReversionRiskStrategy(RiskAdjustmentStrategy):
    """
    Risk strategy optimized for mean-reversion trading, with tighter stops
    during trending markets and wider stops in ranging conditions.
    """

    def adjust_risk_profile(self, profile: RiskProfile, market_condition:
        MarketCondition, volatility: float, account_stats: Dict[str, Any]
        ) ->RiskProfile:
    """
    Adjust risk profile.
    
    Args:
        profile: Description of profile
        market_condition: Description of market_condition
        volatility: Description of volatility
        account_stats: Description of account_stats
        Any]: Description of Any]
    
    Returns:
        RiskProfile: Description of return value
    
    """

        adjusted = RiskProfile.from_dict(profile.to_dict())
        if market_condition in [MarketCondition.RANGING_WIDE,
            MarketCondition.RANGING_NARROW]:
            adjusted.position_size_pct *= 1.3
            for symbol in adjusted.take_profit_pips:
                adjusted.take_profit_pips[symbol] *= 0.8
        elif market_condition in [MarketCondition.TRENDING_BULLISH,
            MarketCondition.TRENDING_BEARISH]:
            adjusted.position_size_pct *= 0.6
            for symbol in adjusted.stop_loss_pips:
                adjusted.stop_loss_pips[symbol] *= 0.7
        elif market_condition in [MarketCondition.CHOPPY]:
            adjusted.position_size_pct *= 1.4
            adjusted.max_positions = min(adjusted.max_positions + 2, 
                adjusted.max_positions * 1.5)
        if 'market_extremity' in account_stats:
            if abs(account_stats['market_extremity']) > 0.8:
                adjusted.position_size_pct *= 1.5
        if volatility > 1.3:
            adjusted.position_size_pct *= 0.7
            adjusted.leverage *= 0.7
        elif volatility < 0.7:
            adjusted.position_size_pct *= 0.8
        adjusted.position_size_pct = min(profile.position_size_pct * 1.5,
            max(profile.position_size_pct * 0.3, adjusted.position_size_pct))
        adjusted.max_positions = min(profile.max_positions * 1.5, max(1,
            adjusted.max_positions))
        adjusted.leverage = min(profile.leverage * 1.2, max(1.0, adjusted.
            leverage))
        return adjusted


class DynamicRiskManager:
    """
    Dynamic risk management system that adjusts risk parameters based on 
    market conditions, volatility, and account performance.
    """

    def __init__(self, broker_simulator: ForexBrokerSimulator,
        market_simulator: AdvancedMarketRegimeSimulator, base_risk_profile:
        Optional[RiskProfile]=None, risk_strategy: Optional[
        RiskAdjustmentStrategy]=None, adjustment_frequency_seconds: int=300,
        config_path: Optional[str]=None):
        """
        Initialize the dynamic risk manager.
        
        Args:
            broker_simulator: Forex broker simulator
            market_simulator: Market regime simulator
            base_risk_profile: Base risk profile to use
            risk_strategy: Strategy for adjusting risk parameters
            adjustment_frequency_seconds: How often to adjust risk
            config_path: Path to load/save configurations
        """
        self.broker_simulator = broker_simulator
        self.market_simulator = market_simulator
        if base_risk_profile is None:
            base_risk_profile = RiskProfile.create_default(RiskTolerance.
                MODERATE)
        self.base_risk_profile = base_risk_profile
        self.current_risk_profile = RiskProfile.from_dict(base_risk_profile
            .to_dict())
        if risk_strategy is None:
            risk_strategy = ConservativeTrendFollowingRiskStrategy()
        self.risk_strategy = risk_strategy
        self.adjustment_frequency_seconds = adjustment_frequency_seconds
        self.config_path = config_path
        self.last_adjustment_time = datetime.datetime.now()
        self.historical_risk_profiles = []
        self.account_statistics = {'consecutive_wins': 0,
            'consecutive_losses': 0, 'profit_factor': 1.0,
            'current_drawdown_pct': 0.0, 'win_rate_30_trades': 0.5}
        if config_path and os.path.exists(config_path):
            self.load_configuration(config_path)

    def update(self, current_time: datetime.datetime) ->None:
        """
        Update the risk manager, adjusting parameters if needed.
        
        Args:
            current_time: Current simulation time
        """
        time_since_last = (current_time - self.last_adjustment_time
            ).total_seconds()
        if time_since_last >= self.adjustment_frequency_seconds:
            self._update_account_statistics()
            market_conditions = self._get_market_conditions()
            self._adjust_risk_parameters(market_conditions)
            self.last_adjustment_time = current_time

    def _update_account_statistics(self) ->None:
        """Update account statistics from broker."""
        balance = self.broker_simulator.balance
        initial_balance = self.broker_simulator.initial_balance
        max_balance = max(balance, initial_balance)
        current_drawdown_pct = (max_balance - balance) / max_balance * 100
        recent_trades = []
        wins = sum(1 for trade in recent_trades if trade.get('profit', 0) > 0)
        if recent_trades:
            win_rate = wins / len(recent_trades)
        else:
            win_rate = self.account_statistics.get('win_rate_30_trades', 0.5)
        consecutive_wins = self.account_statistics.get('consecutive_wins', 0)
        consecutive_losses = self.account_statistics.get('consecutive_losses',
            0)
        gross_profit = sum(trade.get('profit', 0) for trade in
            recent_trades if trade.get('profit', 0) > 0)
        gross_loss = abs(sum(trade.get('profit', 0) for trade in
            recent_trades if trade.get('profit', 0) < 0))
        profit_factor = gross_profit / max(0.01, gross_loss)
        self.account_statistics.update({'consecutive_wins':
            consecutive_wins, 'consecutive_losses': consecutive_losses,
            'profit_factor': profit_factor, 'current_drawdown_pct':
            current_drawdown_pct, 'win_rate_30_trades': win_rate})

    @with_exception_handling
    def _get_market_conditions(self) ->Dict[str, Any]:
        """
        Get current market conditions from market simulator.
        
        Returns:
            Dictionary of market conditions
        """
        symbols = list(self.broker_simulator.prices.keys())
        conditions = {}
        volatilities = {}
        for symbol in symbols:
            condition = MarketCondition.RANGING_NARROW
            if hasattr(self.market_simulator, 'active_conditions'):
                if symbol in self.market_simulator.active_conditions:
                    condition = self.market_simulator.active_conditions[symbol]
            conditions[symbol] = condition
            volatility_factor = 1.0
            if hasattr(self.market_simulator, 'get_symbol_volatility'):
                try:
                    volatility_factor = (self.market_simulator.
                        get_symbol_volatility(symbol))
                except:
                    pass
            volatilities[symbol] = volatility_factor
        return {'conditions': conditions, 'volatilities': volatilities,
            'overall_condition': self._get_dominant_condition(conditions)}

    def _get_dominant_condition(self, conditions: Dict[str, MarketCondition]
        ) ->MarketCondition:
        """Determine the dominant market condition across all symbols."""
        if not conditions:
            return MarketCondition.NORMAL
        condition_counts = {}
        for condition in conditions.values():
            if condition not in condition_counts:
                condition_counts[condition] = 0
            condition_counts[condition] += 1
        if not condition_counts:
            return MarketCondition.NORMAL
        return max(condition_counts.items(), key=lambda x: x[1])[0]

    def _adjust_risk_parameters(self, market_conditions: Dict[str, Any]
        ) ->None:
        """
        Adjust risk parameters based on market conditions.
        
        Args:
            market_conditions: Current market conditions
        """
        self.historical_risk_profiles.append({'timestamp': datetime.
            datetime.now().isoformat(), 'profile': self.
            current_risk_profile.to_dict()})
        max_history = 100
        if len(self.historical_risk_profiles) > max_history:
            self.historical_risk_profiles = self.historical_risk_profiles[-
                max_history:]
        overall_condition = market_conditions['overall_condition']
        volatilities = market_conditions['volatilities']
        avg_volatility = sum(volatilities.values()) / max(1, len(volatilities))
        self.current_risk_profile = self.risk_strategy.adjust_risk_profile(
            profile=self.base_risk_profile, market_condition=
            overall_condition, volatility=avg_volatility, account_stats=
            self.account_statistics)
        for symbol, volatility in volatilities.items():
            self.current_risk_profile.adjust_for_symbol(symbol, volatility)
        logger.info(
            f'Risk parameters adjusted for {overall_condition.value} condition with volatility factor {avg_volatility:.2f}'
            )

    @with_risk_management_resilience('get_risk_parameter')
    def get_risk_parameter(self, parameter: RiskParameter, symbol: Optional
        [str]=None) ->Union[float, int]:
        """
        Get a specific risk parameter value.
        
        Args:
            parameter: The risk parameter to retrieve
            symbol: Optional symbol for symbol-specific parameters
            
        Returns:
            Risk parameter value
        """
        if parameter == RiskParameter.POSITION_SIZE:
            return self.current_risk_profile.position_size_pct
        elif parameter == RiskParameter.MAX_DRAWDOWN:
            return self.current_risk_profile.max_drawdown_pct
        elif parameter == RiskParameter.STOP_LOSS_DISTANCE:
            if symbol and symbol in self.current_risk_profile.stop_loss_pips:
                return self.current_risk_profile.stop_loss_pips[symbol]
            return self.current_risk_profile.stop_loss_pips['DEFAULT']
        elif parameter == RiskParameter.TAKE_PROFIT_DISTANCE:
            if symbol and symbol in self.current_risk_profile.take_profit_pips:
                return self.current_risk_profile.take_profit_pips[symbol]
            return self.current_risk_profile.take_profit_pips['DEFAULT']
        elif parameter == RiskParameter.MAX_POSITIONS:
            return self.current_risk_profile.max_positions
        elif parameter == RiskParameter.LEVERAGE:
            return self.current_risk_profile.leverage
        elif parameter == RiskParameter.MAX_CORRELATION:
            return self.current_risk_profile.max_correlation
        elif parameter == RiskParameter.MIN_REWARD_RISK:
            return self.current_risk_profile.min_reward_risk
        elif parameter == RiskParameter.MAX_DAILY_LOSS:
            return self.current_risk_profile.max_daily_loss_pct
        elif parameter == RiskParameter.MAX_TRADE_SIZE:
            return self.current_risk_profile.max_trade_size_pct
        return 0.0

    @with_risk_management_resilience('get_current_risk_profile')
    def get_current_risk_profile(self) ->Dict[str, Any]:
        """
        Get the current risk profile as a dictionary.
        
        Returns:
            Risk profile as dictionary
        """
        return self.current_risk_profile.to_dict()

    def reset_risk_profile(self) ->None:
        """Reset the risk profile to the base profile."""
        self.current_risk_profile = RiskProfile.from_dict(self.
            base_risk_profile.to_dict())

    @with_exception_handling
    def save_configuration(self, filepath: Optional[str]=None) ->None:
        """
        Save the current configuration to a file.
        
        Args:
            filepath: Path to save file to (defaults to self.config_path)
        """
        if filepath is None:
            filepath = self.config_path
        if filepath is None:
            logger.warning('No filepath specified for saving configuration')
            return
        config = {'base_risk_profile': self.base_risk_profile.to_dict(),
            'current_risk_profile': self.current_risk_profile.to_dict(),
            'adjustment_frequency_seconds': self.
            adjustment_frequency_seconds, 'account_statistics': self.
            account_statistics, 'risk_strategy': self.risk_strategy.
            __class__.__name__, 'historical_risk_profiles': self.
            historical_risk_profiles[-10:] if self.historical_risk_profiles
             else []}
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f'Risk configuration saved to {filepath}')
        except Exception as e:
            logger.error(f'Failed to save risk configuration: {str(e)}')

    @with_database_resilience('load_configuration')
    @with_exception_handling
    def load_configuration(self, filepath: str) ->None:
        """
        Load configuration from a file.
        
        Args:
            filepath: Path to load file from
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            if 'base_risk_profile' in config:
                self.base_risk_profile = RiskProfile.from_dict(config[
                    'base_risk_profile'])
            if 'current_risk_profile' in config:
                self.current_risk_profile = RiskProfile.from_dict(config[
                    'current_risk_profile'])
            if 'adjustment_frequency_seconds' in config:
                self.adjustment_frequency_seconds = config[
                    'adjustment_frequency_seconds']
            if 'account_statistics' in config:
                self.account_statistics.update(config['account_statistics'])
            if 'risk_strategy' in config:
                strategy_name = config['risk_strategy']
                if strategy_name == 'ConservativeTrendFollowingRiskStrategy':
                    self.risk_strategy = (
                        ConservativeTrendFollowingRiskStrategy())
                elif strategy_name == 'AggressiveBreakoutRiskStrategy':
                    self.risk_strategy = AggressiveBreakoutRiskStrategy()
                elif strategy_name == 'MeanReversionRiskStrategy':
                    self.risk_strategy = MeanReversionRiskStrategy()
            logger.info(f'Risk configuration loaded from {filepath}')
        except Exception as e:
            logger.error(f'Failed to load risk configuration: {str(e)}')

    @with_risk_management_resilience('get_risk_adjustment_history')
    def get_risk_adjustment_history(self) ->pd.DataFrame:
        """
        Get the history of risk parameter adjustments as a DataFrame.
        
        Returns:
            DataFrame of risk adjustment history
        """
        if not self.historical_risk_profiles:
            return pd.DataFrame()
        data = []
        for entry in self.historical_risk_profiles:
            record = {'timestamp': entry['timestamp']}
            record.update(entry['profile'])
            data.append(record)
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        return df


class RLRiskOptimizer:
    """
    Reinforcement learning-based risk parameter optimizer.
    
    This class uses reinforcement learning to optimize risk parameters
    based on market conditions and trading performance.
    """

    def __init__(self, risk_manager: DynamicRiskManager, learning_rate:
        float=0.01, exploration_rate: float=0.2, model_save_path: Optional[
        str]=None):
        """
        Initialize the RL risk optimizer.
        
        Args:
            risk_manager: Dynamic risk manager
            learning_rate: Learning rate for optimizer
            exploration_rate: Exploration rate for optimizer
            model_save_path: Path to save the model
        """
        self.risk_manager = risk_manager
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.model_save_path = model_save_path
        self.parameter_ranges = {'position_size_pct': (0.1, 10.0),
            'max_drawdown_pct': (1.0, 30.0), 'stop_loss_factor': (0.5, 2.0),
            'take_profit_factor': (0.5, 2.0), 'max_positions': (1, 20),
            'leverage': (1.0, 30.0), 'max_correlation': (0.1, 0.9),
            'min_reward_risk': (0.5, 5.0), 'max_daily_loss_pct': (0.5, 10.0
            ), 'max_trade_size_pct': (0.1, 10.0)}
        self.model = self._initialize_model()
        self.training_history = []

    @with_exception_handling
    def _initialize_model(self) ->Any:
        """
        Initialize the reinforcement learning model.
        
        Returns:
            Initialized model
        """
        try:
            import torch
            model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.
                ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.
                Linear(64, 10))
            return model
        except ImportError:
            logger.warning('PyTorch not available, using placeholder model')
            return {'type': 'placeholder'}

    def optimize(self, current_state: Dict[str, Any], reward: float) ->None:
        """
        Optimize risk parameters based on current state and reward.
        
        Args:
            current_state: Current market and account state
            reward: Reward from previous action
        """
        if random.random() > self.exploration_rate:
            market_condition = current_state.get('market_condition',
                MarketCondition.NORMAL)
            volatility = current_state.get('volatility', 1.0)
            profit_factor = current_state.get('profit_factor', 1.0)
            drawdown_pct = current_state.get('drawdown_pct', 0.0)
            if profit_factor > 1.5 and drawdown_pct < 5.0:
                if market_condition in [MarketCondition.TRENDING_BULLISH,
                    MarketCondition.TRENDING_BEARISH, MarketCondition.
                    BREAKOUT_BULLISH, MarketCondition.BREAKOUT_BEARISH]:
                    self.risk_manager.risk_strategy = (
                        AggressiveBreakoutRiskStrategy())
                elif market_condition in [MarketCondition.RANGING_NARROW,
                    MarketCondition.RANGING_WIDE, MarketCondition.CHOPPY]:
                    self.risk_manager.risk_strategy = (
                        MeanReversionRiskStrategy())
            else:
                self.risk_manager.risk_strategy = (
                    ConservativeTrendFollowingRiskStrategy())
            self.training_history.append({'timestamp': datetime.datetime.
                now().isoformat(), 'state': {'market_condition':
                market_condition.value, 'volatility': volatility,
                'profit_factor': profit_factor, 'drawdown_pct':
                drawdown_pct}, 'action': self.risk_manager.risk_strategy.
                __class__.__name__, 'reward': reward})

    @with_exception_handling
    def save_model(self, filepath: Optional[str]=None) ->None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save model to
        """
        if filepath is None:
            filepath = self.model_save_path
        if filepath is None:
            logger.warning('No filepath specified for saving model')
            return
        try:
            if isinstance(self.model, dict) and self.model.get('type'
                ) == 'placeholder':
                with open(filepath, 'w') as f:
                    json.dump({'type': 'placeholder', 'training_history':
                        self.training_history[-100:]}, f, indent=2)
            else:
                import torch
                torch.save(self.model.state_dict(), filepath)
                history_path = filepath + '.history.json'
                with open(history_path, 'w') as f:
                    json.dump({'training_history': self.training_history[-
                        100:]}, f, indent=2)
            logger.info(f'RL risk optimizer model saved to {filepath}')
        except Exception as e:
            logger.error(f'Failed to save model: {str(e)}')

    @with_database_resilience('load_model')
    @with_exception_handling
    def load_model(self, filepath: str) ->None:
        """
        Load the model from a file.
        
        Args:
            filepath: Path to load model from
        """
        try:
            if os.path.exists(filepath):
                try:
                    import torch
                    self.model.load_state_dict(torch.load(filepath))
                except:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if 'training_history' in data:
                            self.training_history = data['training_history']
                logger.info(f'RL risk optimizer model loaded from {filepath}')
        except Exception as e:
            logger.error(f'Failed to load model: {str(e)}')


if __name__ == '__main__':
    from core.forex_broker_simulator import ForexBrokerSimulator
    from core.advanced_market_regime_simulator import AdvancedMarketRegimeSimulator
    broker_sim = ForexBrokerSimulator()
    market_sim = AdvancedMarketRegimeSimulator(broker_simulator=broker_sim)
    risk_profile = RiskProfile.create_default(RiskTolerance.MODERATE)
    risk_strategy = ConservativeTrendFollowingRiskStrategy()
    risk_manager = DynamicRiskManager(broker_simulator=broker_sim,
        market_simulator=market_sim, base_risk_profile=risk_profile,
        risk_strategy=risk_strategy)
    rl_optimizer = RLRiskOptimizer(risk_manager=risk_manager,
        model_save_path='./models/rl_risk_optimizer.pth')
    current_time = datetime.datetime.now()
    for i in range(10):
        current_time += datetime.timedelta(minutes=15)
        risk_manager.update(current_time)
        state = {'market_condition': MarketCondition.RANGING_NARROW,
            'volatility': 1.0 + i * 0.1, 'profit_factor': 1.0 + i * 0.05,
            'drawdown_pct': max(0.0, 5.0 - i * 0.5)}
        reward = 0.1 * i
        rl_optimizer.optimize(state, reward)
    final_profile = risk_manager.get_current_risk_profile()
    print(f'Final risk profile: {final_profile}')
