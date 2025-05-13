"""
Curriculum Learning Framework for Reinforcement Learning in Forex Trading

This module provides a framework for implementing curriculum learning in
forex trading RL training. Curriculum learning involves training an agent
on increasingly difficult scenarios to improve learning efficiency and
final performance.
"""
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from enum import Enum
import logging
from dataclasses import dataclass, field
from core.advanced_market_regime_simulator import MarketCondition, MarketSession, LiquidityProfile, SimulationScenario
from core.enhanced_market_condition_generator import EnhancedMarketConditionGenerator
from core.forex_broker_simulator import ForexBrokerSimulator
from core.news_sentiment_simulator import NewsAndSentimentSimulator, NewsImpactLevel
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

@dataclass
class DifficultyLevel:
    """Defines a difficulty level for curriculum learning."""
    level_id: int
    name: str
    description: str
    volatility_scale: float = 1.0
    allowed_conditions: List[MarketCondition] = field(default_factory=list)
    allowed_patterns: List[str] = field(default_factory=list)
    liquidity_profiles: List[LiquidityProfile] = field(default_factory=list)
    anomaly_probability: float = 0.1
    news_event_probability: float = 0.1
    max_drawdown_percent: float = 5.0
    max_adverse_excursion: float = 3.0
    validation_metrics: Dict[str, float] = field(default_factory=dict)


class CurriculumLearningFramework:
    """
    Curriculum Learning Framework that progressively increases difficulty
    of trading scenarios for more effective RL agent training.
    
    Key features:
    - Predefined curriculum progression with difficulty levels
    - Performance-based progression
    - Customizable progression criteria
    - Automatic scenario generation for each level
    - Detailed performance tracking and analysis
    """

    def __init__(self, broker_simulator: ForexBrokerSimulator,
        market_generator: EnhancedMarketConditionGenerator, num_levels: int
        =5, symbols: List[str]=None, performance_threshold: float=0.7,
        consecutive_successes_required: int=3, session_duration: timedelta=
        timedelta(hours=8), random_seed: Optional[int]=None):
        """
        Initialize the curriculum learning framework.
        
        Args:
            broker_simulator: Forex broker simulator instance
            market_generator: Enhanced market condition generator
            num_levels: Number of difficulty levels
            symbols: List of trading symbols to use
            performance_threshold: Performance threshold to advance levels (0-1)
            consecutive_successes_required: Number of consecutive successful
                training sessions required to advance to the next level
            session_duration: Duration of each training session
            random_seed: Optional seed for reproducibility
        """
        self.broker_simulator = broker_simulator
        self.market_generator = market_generator
        self.num_levels = num_levels
        self.symbols = symbols or ['EUR/USD']
        self.performance_threshold = performance_threshold
        self.consecutive_successes_required = consecutive_successes_required
        self.session_duration = session_duration
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.difficulty_levels = self._initialize_difficulty_levels()
        self.current_level = 1
        self.consecutive_successes = 0
        self.training_history = []
        self.curriculum = self._generate_curriculum()

    def _initialize_difficulty_levels(self) ->Dict[int, DifficultyLevel]:
        """Initialize the difficulty levels for the curriculum."""
        levels = {}
        levels[1] = DifficultyLevel(level_id=1, name='Beginner',
            description='Stable trending markets with high liquidity',
            volatility_scale=0.8, allowed_conditions=[MarketCondition.
            NORMAL, MarketCondition.TRENDING_BULLISH, MarketCondition.
            TRENDING_BEARISH, MarketCondition.RANGING_NARROW],
            liquidity_profiles=[LiquidityProfile.HIGH, LiquidityProfile.
            MEDIUM], anomaly_probability=0.0, news_event_probability=0.1,
            max_drawdown_percent=2.0, max_adverse_excursion=1.0,
            validation_metrics={'min_sharpe_ratio': 1.0,
            'min_profit_factor': 1.2, 'max_drawdown_pct': 5.0})
        levels[2] = DifficultyLevel(level_id=2, name='Intermediate',
            description=
            'Varied market conditions with increased volatility',
            volatility_scale=1.0, allowed_conditions=[MarketCondition.
            TRENDING_BULLISH, MarketCondition.TRENDING_BEARISH,
            MarketCondition.RANGING_WIDE, MarketCondition.RANGING_NARROW,
            MarketCondition.REVERSAL_BULLISH, MarketCondition.
            REVERSAL_BEARISH], liquidity_profiles=[LiquidityProfile.HIGH,
            LiquidityProfile.MEDIUM], anomaly_probability=0.1,
            news_event_probability=0.2, max_drawdown_percent=4.0,
            max_adverse_excursion=2.0, validation_metrics={
            'min_sharpe_ratio': 0.8, 'min_profit_factor': 1.1,
            'max_drawdown_pct': 8.0})
        levels[3] = DifficultyLevel(level_id=3, name='Advanced',
            description=
            'Breakouts and higher volatility with moderate news impact',
            volatility_scale=1.3, allowed_conditions=[MarketCondition.
            HIGH_VOLATILITY, MarketCondition.BREAKOUT_BULLISH,
            MarketCondition.BREAKOUT_BEARISH, MarketCondition.RANGING_WIDE,
            MarketCondition.NEWS_REACTION], liquidity_profiles=[
            LiquidityProfile.MEDIUM, LiquidityProfile.LOW],
            anomaly_probability=0.2, news_event_probability=0.4,
            max_drawdown_percent=6.0, max_adverse_excursion=3.0,
            validation_metrics={'min_sharpe_ratio': 0.6,
            'min_profit_factor': 1.05, 'max_drawdown_pct': 10.0})
        levels[4] = DifficultyLevel(level_id=4, name='Expert', description=
            'Complex scenarios with low liquidity and significant news impact',
            volatility_scale=1.6, allowed_conditions=[MarketCondition.
            HIGH_VOLATILITY, MarketCondition.LIQUIDITY_GAP, MarketCondition
            .NEWS_REACTION, MarketCondition.REVERSAL_BEARISH,
            MarketCondition.REVERSAL_BULLISH], liquidity_profiles=[
            LiquidityProfile.MEDIUM, LiquidityProfile.LOW],
            anomaly_probability=0.4, news_event_probability=0.6,
            max_drawdown_percent=8.0, max_adverse_excursion=4.0,
            validation_metrics={'min_sharpe_ratio': 0.4,
            'min_profit_factor': 1.02, 'max_drawdown_pct': 12.0})
        levels[5] = DifficultyLevel(level_id=5, name='Master', description=
            'Extreme market conditions and stress testing',
            volatility_scale=2.0, allowed_conditions=[MarketCondition.
            FLASH_CRASH, MarketCondition.FLASH_SPIKE, MarketCondition.
            LIQUIDITY_GAP, MarketCondition.HIGH_VOLATILITY, MarketCondition
            .NEWS_REACTION], liquidity_profiles=[LiquidityProfile.LOW,
            LiquidityProfile.VERY_LOW], anomaly_probability=0.6,
            news_event_probability=0.8, max_drawdown_percent=15.0,
            max_adverse_excursion=7.0, validation_metrics={
            'min_sharpe_ratio': 0.2, 'min_profit_factor': 1.0,
            'max_drawdown_pct': 15.0})
        return levels

    def _generate_curriculum(self) ->Dict[int, List[SimulationScenario]]:
        """Generate training scenarios for all difficulty levels."""
        curriculum = {}
        for level_id, level in self.difficulty_levels.items():
            curriculum[level_id] = []
            for scenario_index in range(5):
                symbol = random.choice(self.symbols)
                condition = random.choice(level.allowed_conditions)
                liquidity = random.choice(level.liquidity_profiles)
                include_news = random.random() < level.news_event_probability
                news_events = None
                if include_news and self.market_generator.news_simulator:
                    if level_id <= 2:
                        impact_level = NewsImpactLevel.LOW
                    elif level_id <= 4:
                        impact_level = NewsImpactLevel.MEDIUM
                    else:
                        impact_level = NewsImpactLevel.HIGH
                    offset_minutes = int(self.session_duration.
                        total_seconds() * random.uniform(0.3, 0.7) / 60)
                    news_events = [{'event_type': 'ECONOMIC_DATA',
                        'impact_level': impact_level, 'time_offset_minutes':
                        offset_minutes, 'volatility_impact': 1.0 + 
                        impact_level.value * 0.5, 'price_impact': 0.001 *
                        impact_level.value * (1 if random.random() > 0.5 else
                        -1)}]
                scenario = self.market_generator.generate_market_scenario(
                    symbol=symbol, condition=condition, duration=self.
                    session_duration, news_events=news_events)
                scenario.volatility_factor *= level.volatility_scale
                curriculum[level_id].append(scenario)
        return curriculum

    @with_broker_api_resilience('get_current_level_scenarios')
    def get_current_level_scenarios(self) ->List[SimulationScenario]:
        """Get the scenarios for the current difficulty level."""
        return self.curriculum[self.current_level]

    def report_training_results(self, scenario_index: int,
        performance_metrics: Dict[str, float]) ->None:
        """
        Report training results for a specific scenario.
        
        Args:
            scenario_index: Index of the scenario that was trained on
            performance_metrics: Dictionary of performance metrics
        """
        result = {'level': self.current_level, 'scenario_index':
            scenario_index, 'metrics': performance_metrics, 'timestamp':
            datetime.now(), 'success': self._evaluate_success(
            performance_metrics)}
        self.training_history.append(result)
        if result['success']:
            self.consecutive_successes += 1
        else:
            self.consecutive_successes = 0
        self._check_level_advancement()

    def _evaluate_success(self, metrics: Dict[str, float]) ->bool:
        """
        Evaluate if the training was successful based on metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            True if training met the success criteria, False otherwise
        """
        level = self.difficulty_levels[self.current_level]
        validation_metrics = level.validation_metrics
        success = True
        if ('sharpe_ratio' in metrics and 'min_sharpe_ratio' in
            validation_metrics):
            if metrics['sharpe_ratio'] < validation_metrics['min_sharpe_ratio'
                ]:
                success = False
        if ('profit_factor' in metrics and 'min_profit_factor' in
            validation_metrics):
            if metrics['profit_factor'] < validation_metrics[
                'min_profit_factor']:
                success = False
        if ('max_drawdown_pct' in metrics and 'max_drawdown_pct' in
            validation_metrics):
            if metrics['max_drawdown_pct'] > validation_metrics[
                'max_drawdown_pct']:
                success = False
        return success

    def _check_level_advancement(self) ->None:
        """Check if the agent should advance to the next level."""
        if (self.consecutive_successes >= self.
            consecutive_successes_required and self.current_level < self.
            num_levels):
            self.current_level += 1
            self.consecutive_successes = 0
            logger.info(
                f'Agent advanced to curriculum level {self.current_level}: {self.difficulty_levels[self.current_level].name}'
                )

    def reset_progress(self) ->None:
        """Reset the agent's progress to the first level."""
        self.current_level = 1
        self.consecutive_successes = 0
        logger.info('Agent progress has been reset to level 1')

    @with_broker_api_resilience('get_progress_summary')
    def get_progress_summary(self) ->Dict[str, Any]:
        """Get a summary of the agent's progress through the curriculum."""
        level_stats = {}
        for level_id in range(1, self.num_levels + 1):
            level_history = [r for r in self.training_history if r['level'] ==
                level_id]
            successes = sum(1 for r in level_history if r['success'])
            attempts = len(level_history)
            if attempts > 0:
                success_rate = successes / attempts
            else:
                success_rate = 0.0
            level_stats[level_id] = {'name': self.difficulty_levels[
                level_id].name, 'attempts': attempts, 'successes':
                successes, 'success_rate': success_rate, 'completed': 
                level_id < self.current_level}
        return {'current_level': self.current_level, 'current_level_name':
            self.difficulty_levels[self.current_level].name,
            'consecutive_successes': self.consecutive_successes,
            'required_successes': self.consecutive_successes_required,
            'total_training_sessions': len(self.training_history),
            'level_statistics': level_stats, 'curriculum_completion': (self
            .current_level - 1) / self.num_levels}

    def generate_validation_scenarios(self, num_scenarios: int=3) ->List[
        SimulationScenario]:
        """
        Generate validation scenarios at the current difficulty level.
        
        Args:
            num_scenarios: Number of validation scenarios to generate
            
        Returns:
            List of simulation scenarios for validation
        """
        validation_scenarios = []
        level = self.difficulty_levels[self.current_level]
        for _ in range(num_scenarios):
            symbol = random.choice(self.symbols)
            condition = random.choice(level.allowed_conditions)
            scenario = self.market_generator.generate_market_scenario(symbol
                =symbol, condition=condition, duration=self.session_duration)
            scenario.volatility_factor *= level.volatility_scale
            validation_scenarios.append(scenario)
        return validation_scenarios

    def export_curriculum_configuration(self) ->Dict[str, Any]:
        """Export the complete curriculum configuration."""
        config = {'num_levels': self.num_levels, 'symbols': self.symbols,
            'performance_threshold': self.performance_threshold,
            'consecutive_successes_required': self.
            consecutive_successes_required, 'session_duration_hours': self.
            session_duration.total_seconds() / 3600, 'levels': []}
        for level_id, level in self.difficulty_levels.items():
            level_config = {'id': level.level_id, 'name': level.name,
                'description': level.description, 'volatility_scale': level
                .volatility_scale, 'allowed_conditions': [c.value for c in
                level.allowed_conditions], 'liquidity_profiles': [lp.value for
                lp in level.liquidity_profiles], 'anomaly_probability':
                level.anomaly_probability, 'news_event_probability': level.
                news_event_probability, 'validation_metrics': level.
                validation_metrics}
            config['levels'].append(level_config)
        return config

    def import_curriculum_configuration(self, config: Dict[str, Any]) ->None:
        """
        Import a curriculum configuration.
        
        Args:
            config: Dictionary with curriculum configuration
        """
        self.num_levels = config_manager.get('num_levels', 5)
        self.symbols = config_manager.get('symbols', ['EUR/USD'])
        self.performance_threshold = config_manager.get('performance_threshold', 0.7)
        self.consecutive_successes_required = config.get(
            'consecutive_successes_required', 3)
        self.session_duration = timedelta(hours=config.get(
            'session_duration_hours', 8))
        self.reset_progress()
        if 'levels' in config:
            new_levels = {}
            for level_config in config['levels']:
                level_id = level_config['id']
                allowed_conditions = []
                for condition_str in level_config.get('allowed_conditions', []
                    ):
                    for condition in MarketCondition:
                        if condition.value == condition_str:
                            allowed_conditions.append(condition)
                            break
                liquidity_profiles = []
                for profile_str in level_config_manager.get('liquidity_profiles', []):
                    for profile in LiquidityProfile:
                        if profile.value == profile_str:
                            liquidity_profiles.append(profile)
                            break
                new_levels[level_id] = DifficultyLevel(level_id=level_id,
                    name=level_config_manager.get('name', f'Level {level_id}'),
                    description=level_config_manager.get('description', ''),
                    volatility_scale=level_config.get('volatility_scale', 
                    1.0), allowed_conditions=allowed_conditions,
                    liquidity_profiles=liquidity_profiles,
                    anomaly_probability=level_config.get(
                    'anomaly_probability', 0.1), news_event_probability=
                    level_config_manager.get('news_event_probability', 0.1),
                    validation_metrics=level_config.get(
                    'validation_metrics', {}))
            if new_levels:
                self.difficulty_levels = new_levels
        self.curriculum = self._generate_curriculum()
