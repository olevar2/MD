"""
Enhanced Market Condition Generator for Phase 6 Implementation.

This module extends the existing market simulation capabilities with more
fine-grained control over market conditions, advanced pattern generation,
and integration with news and sentiment data.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from enum import Enum
import logging
from scipy import stats
from functools import lru_cache

from trading_gateway_service.simulation.forex_broker_simulator import (
    ForexBrokerSimulator, MarketRegimeType, MarketConditionConfig, LiquidityLevel
)
from trading_gateway_service.simulation.advanced_market_regime_simulator import (
    MarketCondition, MarketSession, LiquidityProfile, SimulationScenario
)
from trading_gateway_service.simulation.news_sentiment_simulator import (
    NewsAndSentimentSimulator, NewsEvent, NewsImpactLevel, SentimentLevel
)

from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class PatternType(str, Enum):
    """Fine-grained market patterns for simulation."""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    CHANNEL_ASCENDING = "channel_ascending"
    CHANNEL_DESCENDING = "channel_descending"
    CHANNEL_HORIZONTAL = "channel_horizontal"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    CUP_AND_HANDLE = "cup_and_handle"
    INVERSE_CUP_AND_HANDLE = "inverse_cup_and_handle"


class MarketAnomalyType(str, Enum):
    """Market anomalies and special conditions."""
    MOMENTUM_EXHAUSTION = "momentum_exhaustion"
    LIQUIDITY_CASCADE = "liquidity_cascade"
    SHORT_SQUEEZE = "short_squeeze"
    STOP_HUNT = "stop_hunt"
    MOMENTUM_IGNITION = "momentum_ignition"
    DEAD_CAT_BOUNCE = "dead_cat_bounce"
    OVERNIGHT_GAP = "overnight_gap"
    ORDER_IMBALANCE = "order_imbalance"
    THIN_MARKET = "thin_market"


class EnhancedMarketConditionGenerator:
    """
    Advanced market condition generator with fine-grained control over price patterns,
    market anomalies, and realistic market behavior for reinforcement learning training.
    
    This generator can produce:
    - Price series with specific technical patterns
    - Market anomalies and special conditions
    - Complex multi-regime scenarios
    - News-affected price action
    - Realistic order book dynamics
    """
    
    def __init__(
        self,
        broker_simulator: ForexBrokerSimulator,
        news_simulator: Optional[NewsAndSentimentSimulator] = None,
        base_volatility_map: Optional[Dict[str, float]] = None,
        base_spread_map: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the advanced market condition generator.
        
        Args:
            broker_simulator: Forex broker simulator instance
            news_simulator: News and sentiment simulator for event integration
            base_volatility_map: Base volatility levels by symbol
            base_spread_map: Base spread levels by symbol
            random_seed: Optional seed for reproducibility
        """
        self.broker_simulator = broker_simulator
        self.news_simulator = news_simulator
        self.base_volatility_map = base_volatility_map or {}
        self.base_spread_map = base_spread_map or {}
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Internal state
        self.pattern_library = self._initialize_pattern_library()
        self.anomaly_library = self._initialize_anomaly_library()
        self.current_patterns = {}  # Symbol -> active pattern
        self.historical_patterns = {}  # Symbol -> List of (timestamp, pattern)

    def _initialize_pattern_library(self) -> Dict[PatternType, Dict[str, Any]]:
        """Initialize the technical pattern generation library."""
        library = {}
        
        # Head and Shoulders
        library[PatternType.HEAD_AND_SHOULDERS] = {
            'duration_bars': 50,
            'success_rate': 0.75,  # Probability of pattern completing as expected
            'volatility_profile': {
                'pre': 1.0,  # Normal volatility before pattern
                'during': 0.8,  # Slightly lower during pattern formation
                'post': 1.5  # Higher after pattern completes
            },
            'volume_profile': {
                'left_shoulder': 1.2,
                'head': 1.5,
                'right_shoulder': 0.9,
                'breakdown': 2.0
            },
            'expected_move': -0.03  # -3% move after completion
        }
        
        # Double Top
        library[PatternType.DOUBLE_TOP] = {
            'duration_bars': 35,
            'success_rate': 0.7,
            'volatility_profile': {
                'pre': 0.9,
                'during': 1.1,
                'post': 1.4
            },
            'volume_profile': {
                'first_top': 1.3,
                'second_top': 1.1,
                'breakdown': 1.7
            },
            'expected_move': -0.025
        }
        
        # Double Bottom
        library[PatternType.DOUBLE_BOTTOM] = {
            'duration_bars': 35,
            'success_rate': 0.7,
            'volatility_profile': {
                'pre': 1.1,
                'during': 0.9,
                'post': 1.3
            },
            'volume_profile': {
                'first_bottom': 1.4,
                'second_bottom': 1.2,
                'breakout': 1.7
            },
            'expected_move': 0.025
        }
        
        # Ascending Triangle
        library[PatternType.TRIANGLE_ASCENDING] = {
            'duration_bars': 30,
            'success_rate': 0.65,
            'volatility_profile': {
                'pre': 1.2,
                'during': 0.7,  # Decreasing volatility during consolidation
                'post': 1.6
            },
            'volume_profile': {
                'during': 0.8,  # Decreasing volume during pattern
                'breakout': 2.0
            },
            'expected_move': 0.02
        }
        
        # Descending Triangle
        library[PatternType.TRIANGLE_DESCENDING] = {
            'duration_bars': 30,
            'success_rate': 0.65,
            'volatility_profile': {
                'pre': 1.1,
                'during': 0.7,
                'post': 1.5
            },
            'volume_profile': {
                'during': 0.8,
                'breakdown': 1.8
            },
            'expected_move': -0.02
        }
        
        # Add other patterns...
        library[PatternType.TRIANGLE_SYMMETRICAL] = {
            'duration_bars': 25,
            'success_rate': 0.6,
            'volatility_profile': {
                'pre': 1.2,
                'during': 0.6,
                'post': 1.8
            },
            'volume_profile': {
                'during': 0.7,
                'breakout': 2.2
            },
            'expected_move': 0.0,  # Direction determined at runtime (50/50)
            'direction_bias': 0.0  # No bias, equal chance up/down
        }
        
        # Bullish Flag
        library[PatternType.FLAG_BULLISH] = {
            'duration_bars': 15,
            'success_rate': 0.7,
            'volatility_profile': {
                'pre': 1.5,  # High volatility in the flagpole
                'during': 0.6,  # Low volatility in the flag
                'post': 1.3
            },
            'volume_profile': {
                'flagpole': 1.8,
                'flag': 0.6,  # Decreasing volume in flag
                'breakout': 1.7
            },
            'expected_move': 0.02
        }
        
        # Bearish Flag
        library[PatternType.FLAG_BEARISH] = {
            'duration_bars': 15,
            'success_rate': 0.7,
            'volatility_profile': {
                'pre': 1.5,
                'during': 0.6,
                'post': 1.3
            },
            'volume_profile': {
                'flagpole': 1.7,
                'flag': 0.6,
                'breakdown': 1.6
            },
            'expected_move': -0.02
        }
        
        return library
    
    def _initialize_anomaly_library(self) -> Dict[MarketAnomalyType, Dict[str, Any]]:
        """Initialize the market anomaly generation library."""
        library = {}
        
        # Stop Hunt
        library[MarketAnomalyType.STOP_HUNT] = {
            'duration_bars': 8,
            'recovery_rate': 0.8,  # How much of the move is typically recovered
            'typical_magnitude': 0.01,  # 1% move
            'liquidity_profile': LiquidityProfile.LOW,
            'volume_profile': {
                'pre': 0.7,
                'during': 2.0,
                'post': 1.5
            }
        }
        
        # Liquidity Cascade
        library[MarketAnomalyType.LIQUIDITY_CASCADE] = {
            'duration_bars': 10,
            'recovery_rate': 0.4,  # Only 40% recovery typically
            'typical_magnitude': 0.02,  # 2% move
            'liquidity_profile': LiquidityProfile.VERY_LOW,
            'volume_profile': {
                'pre': 0.8,
                'during': 3.0,
                'post': 2.2
            }
        }
        
        # Short Squeeze
        library[MarketAnomalyType.SHORT_SQUEEZE] = {
            'duration_bars': 15,
            'recovery_rate': 0.1,  # Very little reversal
            'typical_magnitude': 0.03,  # 3% move
            'liquidity_profile': LiquidityProfile.LOW,
            'volume_profile': {
                'pre': 0.9,
                'during': 4.0,
                'post': 1.8
            }
        }
        
        # Momentum Ignition
        library[MarketAnomalyType.MOMENTUM_IGNITION] = {
            'duration_bars': 20,
            'recovery_rate': 0.3,
            'typical_magnitude': 0.015,
            'liquidity_profile': LiquidityProfile.MEDIUM,
            'volume_profile': {
                'pre': 0.8,
                'ignition': 3.0,
                'continuation': 2.0,
                'exhaustion': 1.5
            }
        }
        
        # Overnight Gap
        library[MarketAnomalyType.OVERNIGHT_GAP] = {
            'fill_probability': 0.65,  # Probability gap will fill
            'typical_magnitude': 0.01,
            'liquidity_profile': LiquidityProfile.MEDIUM,
            'volume_profile': {
                'open': 2.5,
                'post': 1.3
            }
        }
        
        # Add other anomalies...
        library[MarketAnomalyType.THIN_MARKET] = {
            'duration_bars': 30,
            'volatility_increase': 1.8,
            'liquidity_profile': LiquidityProfile.VERY_LOW,
            'spread_factor': 3.0,
            'typical_magnitude': 0.005,
            'whipsaw_probability': 0.7
        }
        
        return library

    def generate_market_scenario(
        self, 
        symbol: str,
        condition: MarketCondition,
        duration: timedelta,
        pattern: Optional[PatternType] = None,
        anomalies: Optional[List[MarketAnomalyType]] = None,
        news_events: Optional[List[Dict[str, Any]]] = None,
        start_time: Optional[datetime] = None
    ) -> SimulationScenario:
        """
        Generate a complete market scenario with specified conditions.
        
        Args:
            symbol: Trading symbol
            condition: Main market condition
            duration: Scenario duration
            pattern: Optional technical pattern to include
            anomalies: Optional market anomalies to include
            news_events: Optional news events to include
            start_time: Starting time (defaults to now)
            
        Returns:
            A fully configured simulation scenario
        """
        if start_time is None:
            start_time = datetime.now()
        
        # Base scenario parameters from the condition
        scenario_params = self._get_base_scenario_parameters(condition)
        
        # Adjust parameters based on the pattern if specified
        if pattern is not None:
            pattern_params = self.pattern_library.get(pattern, {})
            scenario_params = self._adjust_for_pattern(scenario_params, pattern_params)
        
        # Apply anomaly effects if specified
        if anomalies:
            for anomaly in anomalies:
                anomaly_params = self.anomaly_library.get(anomaly, {})
                scenario_params = self._adjust_for_anomaly(scenario_params, anomaly_params)
        
        # Create special events configuration
        special_events = []
        
        # Add pattern-driven events
        if pattern is not None:
            pattern_event = self._create_pattern_event(pattern, duration)
            if pattern_event:
                special_events.append(pattern_event)
        
        # Add anomaly-driven events
        if anomalies:
            for anomaly in anomalies:
                anomaly_event = self._create_anomaly_event(anomaly, duration)
                if anomaly_event:
                    special_events.append(anomaly_event)
        
        # Add news events if specified
        if news_events:
            special_events.extend(news_events)
        
        # Create the final scenario
        scenario_name = f"{condition.value}"
        if pattern:
            scenario_name += f"_{pattern.value}"
        
        scenario = SimulationScenario(
            name=scenario_name,
            symbol=symbol,
            duration=duration,
            market_condition=condition,
            liquidity_profile=scenario_params['liquidity_profile'],
            volatility_factor=scenario_params['volatility_factor'],
            spread_factor=scenario_params['spread_factor'],
            trend_strength=scenario_params['trend_strength'],
            mean_reversion_strength=scenario_params['mean_reversion_strength'],
            price_jump_probability=scenario_params['price_jump_probability'],
            price_jump_magnitude=scenario_params['price_jump_magnitude'],
            special_events=special_events,
            description=self._generate_scenario_description(condition, pattern, anomalies)
        )
        
        return scenario
    
    def _get_base_scenario_parameters(self, condition: MarketCondition) -> Dict[str, Any]:
        """Get base scenario parameters for the specified market condition."""
        # Default parameters
        params = {
            'liquidity_profile': LiquidityProfile.MEDIUM,
            'volatility_factor': 1.0,
            'spread_factor': 1.0,
            'trend_strength': 0.0,
            'mean_reversion_strength': 0.0,
            'price_jump_probability': 0.0,
            'price_jump_magnitude': 0.0
        }
        
        # Adjust based on market condition
        if condition == MarketCondition.TRENDING_BULLISH:
            params.update({
                'liquidity_profile': LiquidityProfile.HIGH,
                'volatility_factor': 0.9,
                'trend_strength': 0.6,
                'mean_reversion_strength': -0.1
            })
        elif condition == MarketCondition.TRENDING_BEARISH:
            params.update({
                'liquidity_profile': LiquidityProfile.HIGH,
                'volatility_factor': 1.0,
                'trend_strength': -0.6,
                'mean_reversion_strength': -0.1
            })
        elif condition == MarketCondition.RANGING_NARROW:
            params.update({
                'liquidity_profile': LiquidityProfile.MEDIUM,
                'volatility_factor': 0.7,
                'trend_strength': 0.0,
                'mean_reversion_strength': 0.6
            })
        elif condition == MarketCondition.RANGING_WIDE:
            params.update({
                'liquidity_profile': LiquidityProfile.MEDIUM,
                'volatility_factor': 1.2,
                'trend_strength': 0.0,
                'mean_reversion_strength': 0.4
            })
        elif condition == MarketCondition.BREAKOUT_BULLISH:
            params.update({
                'liquidity_profile': LiquidityProfile.MEDIUM,
                'volatility_factor': 1.5,
                'trend_strength': 0.8,
                'price_jump_probability': 0.05,
                'price_jump_magnitude': 0.002
            })
        elif condition == MarketCondition.BREAKOUT_BEARISH:
            params.update({
                'liquidity_profile': LiquidityProfile.MEDIUM,
                'volatility_factor': 1.5,
                'trend_strength': -0.8,
                'price_jump_probability': 0.05,
                'price_jump_magnitude': 0.002
            })
        elif condition == MarketCondition.HIGH_VOLATILITY:
            params.update({
                'liquidity_profile': LiquidityProfile.MEDIUM,
                'volatility_factor': 2.2,
                'price_jump_probability': 0.08,
                'price_jump_magnitude': 0.003
            })
        elif condition == MarketCondition.LIQUIDITY_GAP:
            params.update({
                'liquidity_profile': LiquidityProfile.VERY_LOW,
                'volatility_factor': 1.8,
                'spread_factor': 3.0,
                'price_jump_probability': 0.1,
                'price_jump_magnitude': 0.004
            })
        elif condition == MarketCondition.FLASH_CRASH:
            params.update({
                'liquidity_profile': LiquidityProfile.VERY_LOW,
                'volatility_factor': 4.0,
                'spread_factor': 5.0,
                'trend_strength': -2.0,
                'price_jump_probability': 0.3,
                'price_jump_magnitude': 0.01
            })
        elif condition == MarketCondition.NEWS_REACTION:
            params.update({
                'liquidity_profile': LiquidityProfile.LOW,
                'volatility_factor': 2.0,
                'price_jump_probability': 0.15,
                'price_jump_magnitude': 0.005
            })
            
        return params
    
    def _adjust_for_pattern(
        self, 
        base_params: Dict[str, Any], 
        pattern_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust scenario parameters for a specific technical pattern."""
        result = base_params.copy()
        
        # Apply volatility adjustment
        volatility_profile = pattern_params.get('volatility_profile', {})
        # For now we'll use the 'during' value, later we can make this time-dependent
        if 'during' in volatility_profile:
            result['volatility_factor'] *= volatility_profile['during']
        
        # Adjust trend based on expected move
        expected_move = pattern_params.get('expected_move', 0.0)
        if expected_move != 0:
            # Convert expected price change to trend strength
            # This is a simplified mapping
            result['trend_strength'] = min(max(-1.0, expected_move * 20), 1.0)
        
        # Direction bias for symmetric patterns
        if 'direction_bias' in pattern_params:
            bias = pattern_params['direction_bias']
            if bias != 0 and result['trend_strength'] == 0:
                result['trend_strength'] = bias * 0.3  # Subtle bias
        
        return result
    
    def _adjust_for_anomaly(
        self, 
        base_params: Dict[str, Any], 
        anomaly_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adjust scenario parameters for a market anomaly."""
        result = base_params.copy()
        
        # Apply liquidity profile if specified
        if 'liquidity_profile' in anomaly_params:
            # Use the lower liquidity between base and anomaly
            base_liquidity_idx = list(LiquidityProfile).index(base_params['liquidity_profile'])
            anomaly_liquidity_idx = list(LiquidityProfile).index(anomaly_params['liquidity_profile'])
            result['liquidity_profile'] = list(LiquidityProfile)[min(base_liquidity_idx, anomaly_liquidity_idx)]
        
        # Apply volatility increase if specified
        if 'volatility_increase' in anomaly_params:
            result['volatility_factor'] *= anomaly_params['volatility_increase']
        
        # Apply spread factor if specified
        if 'spread_factor' in anomaly_params:
            result['spread_factor'] *= anomaly_params['spread_factor']
        
        # Apply price jump parameters if specified
        if 'typical_magnitude' in anomaly_params:
            # Increase jump probability and magnitude
            magnitude = anomaly_params['typical_magnitude']
            result['price_jump_probability'] = max(
                result['price_jump_probability'],
                0.1  # Base probability for anomalies
            )
            result['price_jump_magnitude'] = max(
                result['price_jump_magnitude'],
                magnitude
            )
        
        return result
    
    def _create_pattern_event(
        self, 
        pattern: PatternType, 
        duration: timedelta
    ) -> Optional[Dict[str, Any]]:
        """Create a special event configuration for a pattern formation."""
        pattern_params = self.pattern_library.get(pattern, {})
        if not pattern_params:
            return None
        
        expected_move = pattern_params.get('expected_move', 0.0)
        
        # For symmetric patterns, randomly determine direction
        if expected_move == 0.0 and pattern == PatternType.TRIANGLE_SYMMETRICAL:
            expected_move = 0.02 if random.random() > 0.5 else -0.02
            
        # Calculate when the pattern should complete
        # Place it around 2/3 through the scenario duration
        completion_offset = int(duration.total_seconds() * 0.67 / 60)
        
        event = {
            'event_type': 'PATTERN_COMPLETION',
            'pattern_type': pattern.value,
            'time_offset_minutes': completion_offset,
            'price_impact': expected_move,
            'volatility_impact': pattern_params.get('volatility_profile', {}).get('post', 1.0),
            'volume_impact': pattern_params.get('volume_profile', {}).get('breakout', 1.5),
            'success_rate': pattern_params.get('success_rate', 0.7)
        }
        
        return event
    
    def _create_anomaly_event(
        self, 
        anomaly: MarketAnomalyType, 
        duration: timedelta
    ) -> Optional[Dict[str, Any]]:
        """Create a special event configuration for a market anomaly."""
        anomaly_params = self.anomaly_library.get(anomaly, {})
        if not anomaly_params:
            return None
        
        # Calculate when the anomaly should occur
        # For most anomalies, place them in the second half of the scenario
        min_offset = int(duration.total_seconds() * 0.5 / 60)
        max_offset = int(duration.total_seconds() * 0.9 / 60)
        anomaly_offset = random.randint(min_offset, max_offset)
        
        typical_magnitude = anomaly_params.get('typical_magnitude', 0.01)
        # Determine direction (-1 for down, +1 for up)
        direction = 1 if random.random() > 0.5 else -1
        
        event = {
            'event_type': 'MARKET_ANOMALY',
            'anomaly_type': anomaly.value,
            'time_offset_minutes': anomaly_offset,
            'duration_minutes': anomaly_params.get('duration_bars', 10) * 5,  # Assume 5 minutes per bar
            'price_impact': typical_magnitude * direction,
            'volatility_impact': anomaly_params.get('volatility_increase', 2.0),
            'recovery_rate': anomaly_params.get('recovery_rate', 0.5),
        }
        
        return event
    
    def _generate_scenario_description(
        self, 
        condition: MarketCondition,
        pattern: Optional[PatternType] = None,
        anomalies: Optional[List[MarketAnomalyType]] = None
    ) -> str:
        """Generate a descriptive text for the scenario."""
        desc_parts = []
        
        # Base condition description
        condition_desc = {
            MarketCondition.TRENDING_BULLISH: "Strong bullish trend",
            MarketCondition.TRENDING_BEARISH: "Strong bearish trend",
            MarketCondition.RANGING_NARROW: "Narrow ranging market",
            MarketCondition.RANGING_WIDE: "Wide ranging market with increased volatility",
            MarketCondition.BREAKOUT_BULLISH: "Bullish breakout with momentum",
            MarketCondition.BREAKOUT_BEARISH: "Bearish breakdown with momentum",
            MarketCondition.REVERSAL_BULLISH: "Bullish reversal after downtrend",
            MarketCondition.REVERSAL_BEARISH: "Bearish reversal after uptrend",
            MarketCondition.HIGH_VOLATILITY: "High volatility market with erratic price action",
            MarketCondition.LIQUIDITY_GAP: "Low liquidity conditions with potential price gaps",
            MarketCondition.FLASH_CRASH: "Flash crash with severe price decline",
            MarketCondition.FLASH_SPIKE: "Flash spike with rapid price increase",
            MarketCondition.NEWS_REACTION: "Market reacting to significant news",
            MarketCondition.NORMAL: "Normal market conditions"
        }
        
        desc_parts.append(condition_desc.get(condition, str(condition)))
        
        # Add pattern description
        if pattern:
            pattern_desc = {
                PatternType.HEAD_AND_SHOULDERS: "with head and shoulders pattern formation",
                PatternType.DOUBLE_TOP: "with double top pattern formation",
                PatternType.DOUBLE_BOTTOM: "with double bottom pattern formation",
                PatternType.TRIANGLE_ASCENDING: "with ascending triangle pattern",
                PatternType.TRIANGLE_DESCENDING: "with descending triangle pattern",
                PatternType.TRIANGLE_SYMMETRICAL: "with symmetrical triangle pattern",
                PatternType.CHANNEL_ASCENDING: "in ascending channel",
                PatternType.CHANNEL_DESCENDING: "in descending channel",
                PatternType.FLAG_BULLISH: "with bullish flag pattern",
                PatternType.FLAG_BEARISH: "with bearish flag pattern"
            }
            desc_parts.append(pattern_desc.get(pattern, f"with {pattern.value} pattern"))
        
        # Add anomaly descriptions
        if anomalies:
            anomaly_descriptors = []
            for anomaly in anomalies:
                anomaly_desc = {
                    MarketAnomalyType.STOP_HUNT: "stop hunting activity",
                    MarketAnomalyType.LIQUIDITY_CASCADE: "liquidity cascade",
                    MarketAnomalyType.SHORT_SQUEEZE: "short squeeze",
                    MarketAnomalyType.MOMENTUM_IGNITION: "momentum ignition",
                    MarketAnomalyType.OVERNIGHT_GAP: "overnight gap",
                    MarketAnomalyType.THIN_MARKET: "thin market conditions"
                }
                anomaly_descriptors.append(anomaly_desc.get(anomaly, str(anomaly)))
            
            if anomaly_descriptors:
                desc_parts.append(f"featuring {', '.join(anomaly_descriptors)}")
        
        return " ".join(desc_parts)
    
    def generate_multi_session_scenario(
        self, 
        symbol: str,
        duration: timedelta = timedelta(days=1),
        session_specific_conditions: Optional[Dict[MarketSession, MarketCondition]] = None
    ) -> List[SimulationScenario]:
        """
        Generate a complete day scenario with different market conditions for each session.
        
        Args:
            symbol: Trading symbol
            duration: Total scenario duration
            session_specific_conditions: Optional mapping of sessions to conditions
            
        Returns:
            List of session-specific scenarios
        """
        # Default conditions if not specified
        if not session_specific_conditions:
            session_specific_conditions = {
                MarketSession.SYDNEY: MarketCondition.RANGING_NARROW,
                MarketSession.TOKYO: MarketCondition.RANGING_WIDE,
                MarketSession.LONDON: MarketCondition.TRENDING_BULLISH,
                MarketSession.NEWYORK: MarketCondition.HIGH_VOLATILITY,
                MarketSession.OVERLAP_LONDON_NEWYORK: MarketCondition.BREAKOUT_BULLISH
            }
        
        # Get session times
        session_times = {
            MarketSession.SYDNEY: (0, 7),  # 00:00-07:00 UTC
            MarketSession.TOKYO: (0, 9),   # 00:00-09:00 UTC
            MarketSession.LONDON: (8, 17),  # 08:00-17:00 UTC
            MarketSession.NEWYORK: (13, 21),  # 13:00-21:00 UTC
            MarketSession.OVERLAP_TOKYO_LONDON: (8, 9),  # 08:00-09:00 UTC
            MarketSession.OVERLAP_LONDON_NEWYORK: (13, 17),  # 13:00-17:00 UTC
        }
        
        # Create scenarios for each session
        scenarios = []
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for session, condition in session_specific_conditions.items():
            if session in session_times:
                start_hour, end_hour = session_times[session]
                session_start = base_time.replace(hour=start_hour)
                session_end = base_time.replace(hour=end_hour)
                session_duration = timedelta(hours=end_hour - start_hour)
                
                # Generate random pattern with 30% probability
                pattern = None
                if random.random() < 0.3:
                    pattern = random.choice(list(PatternType))
                
                # Generate random anomaly with 20% probability
                anomalies = []
                if random.random() < 0.2:
                    anomalies = [random.choice(list(MarketAnomalyType))]
                
                scenario = self.generate_market_scenario(
                    symbol=symbol,
                    condition=condition,
                    duration=session_duration,
                    pattern=pattern,
                    anomalies=anomalies,
                    start_time=session_start
                )
                
                scenarios.append(scenario)
        
        return scenarios
    
    def generate_multi_day_curriculum(
        self,
        symbol: str,
        num_days: int = 5,
        difficulty_progression: bool = True
    ) -> Dict[int, List[SimulationScenario]]:
        """
        Generate a multi-day curriculum with progressive difficulty.
        
        Args:
            symbol: Trading symbol
            num_days: Number of days to generate
            difficulty_progression: Whether to increase difficulty over days
            
        Returns:
            Dictionary of day -> list of scenarios
        """
        curriculum = {}
        
        # Define difficulty progression
        if difficulty_progression:
            difficulty_levels = {
                1: {  # Day 1 - Easy
                    'volatility_scale': 0.8,
                    'conditions': [
                        MarketCondition.NORMAL,
                        MarketCondition.TRENDING_BULLISH,
                        MarketCondition.RANGING_NARROW
                    ],
                    'anomaly_prob': 0.0,
                    'pattern_prob': 0.2,
                    'news_prob': 0.1
                },
                2: {  # Day 2 - Medium-Easy
                    'volatility_scale': 0.9,
                    'conditions': [
                        MarketCondition.TRENDING_BULLISH,
                        MarketCondition.TRENDING_BEARISH,
                        MarketCondition.RANGING_WIDE
                    ],
                    'anomaly_prob': 0.1,
                    'pattern_prob': 0.3,
                    'news_prob': 0.2
                },
                3: {  # Day 3 - Medium
                    'volatility_scale': 1.0,
                    'conditions': [
                        MarketCondition.RANGING_WIDE,
                        MarketCondition.BREAKOUT_BULLISH,
                        MarketCondition.BREAKOUT_BEARISH,
                        MarketCondition.REVERSAL_BULLISH
                    ],
                    'anomaly_prob': 0.2,
                    'pattern_prob': 0.4,
                    'news_prob': 0.3
                },
                4: {  # Day 4 - Medium-Hard
                    'volatility_scale': 1.2,
                    'conditions': [
                        MarketCondition.BREAKOUT_BEARISH,
                        MarketCondition.HIGH_VOLATILITY,
                        MarketCondition.REVERSAL_BEARISH
                    ],
                    'anomaly_prob': 0.3,
                    'pattern_prob': 0.5,
                    'news_prob': 0.4
                },
                5: {  # Day 5 - Hard
                    'volatility_scale': 1.5,
                    'conditions': [
                        MarketCondition.LIQUIDITY_GAP,
                        MarketCondition.HIGH_VOLATILITY,
                        MarketCondition.NEWS_REACTION,
                        MarketCondition.FLASH_CRASH
                    ],
                    'anomaly_prob': 0.5,
                    'pattern_prob': 0.6,
                    'news_prob': 0.6
                }
            }
        else:
            # Default to medium difficulty for all days
            difficulty_levels = {day: {
                'volatility_scale': 1.0,
                'conditions': list(MarketCondition),
                'anomaly_prob': 0.2,
                'pattern_prob': 0.3,
                'news_prob': 0.3
            } for day in range(1, num_days + 1)}
        
        # Generate scenarios for each day
        for day in range(1, num_days + 1):
            curriculum[day] = []
            
            # Get difficulty settings for this day
            settings = difficulty_levels.get(day, difficulty_levels.get(1))
            
            # Create session-specific conditions
            sessions = [
                MarketSession.SYDNEY,
                MarketSession.TOKYO,
                MarketSession.LONDON,
                MarketSession.NEWYORK
            ]
            
            session_conditions = {}
            for session in sessions:
                # Select random condition from allowed list for this difficulty
                condition = random.choice(settings['conditions'])
                session_conditions[session] = condition
            
            # Generate multi-session scenario
            day_scenarios = self.generate_multi_session_scenario(
                symbol=symbol,
                session_specific_conditions=session_conditions
            )
            
            # Apply difficulty settings to each scenario
            for scenario in day_scenarios:
                # Scale volatility
                scenario.volatility_factor *= settings['volatility_scale']
                
                # Add anomaly with probability
                if random.random() < settings['anomaly_prob']:
                    anomaly = random.choice(list(MarketAnomalyType))
                    anomaly_event = self._create_anomaly_event(
                        anomaly, 
                        scenario.duration
                    )
                    if anomaly_event and scenario.special_events is not None:
                        scenario.special_events.append(anomaly_event)
                
                # Add news event with probability
                if random.random() < settings['news_prob'] and self.news_simulator:
                    impact_levels = [NewsImpactLevel.LOW, NewsImpactLevel.MEDIUM, NewsImpactLevel.HIGH]
                    impact_weights = [0.5, 0.3, 0.2]
                    
                    # Higher difficulty = higher chance of high impact news
                    if day > 3:
                        impact_weights = [0.2, 0.4, 0.4]
                    
                    impact = random.choices(impact_levels, weights=impact_weights)[0]
                    
                    offset_minutes = int(scenario.duration.total_seconds() * random.uniform(0.3, 0.7) / 60)
                    
                    news_event = {
                        'event_type': 'ECONOMIC_DATA',
                        'impact_level': impact,
                        'time_offset_minutes': offset_minutes,
                        'volatility_impact': 1.0 + (impact.value * 0.5),
                        'price_impact': 0.001 * impact.value * (1 if random.random() > 0.5 else -1)
                    }
                    
                    if scenario.special_events is not None:
                        scenario.special_events.append(news_event)
                
                curriculum[day].append(scenario)
        
        return curriculum
