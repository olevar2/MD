"""
ElliottWaveTradingStrategy

This module implements a trading strategy based on Elliott Wave analysis
that identifies wave patterns and generates trading signals at optimal entry points.

Part of Phase 4 implementation to enhance the adaptive trading capabilities.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from strategy_execution_engine.strategies.advanced_ta_strategy import AdvancedTAStrategy
from analysis_engine.services.tool_effectiveness import MarketRegime
from analysis_engine.analysis.elliott_wave_analyzer import ElliottWaveAnalyzer
from analysis_engine.analysis.fibonacci_tools import FibonacciTools


class ElliottWaveTradingStrategy(AdvancedTAStrategy):
    """
    Trading strategy that leverages Elliott Wave analysis to identify optimal
    entry and exit points at key wave formations and completions.
    """
    
    def _init_strategy_config(self) -> None:
        """Initialize strategy-specific configuration parameters."""
        # Initialize required services
        self.elliott_wave_analyzer = ElliottWaveAnalyzer()
        self.fibonacci_tools = FibonacciTools()
        
        # Strategy-specific parameters
        self.adaptive_params = {
            "min_wave_confidence": 0.75,     # Minimum confidence for wave pattern detection
            "preferred_entry_waves": [2, 4], # Preferred waves for entry
            "correction_entry_filter": True, # Filter for entries during corrections
            "trend_continuation_only": False,# Only take trades in direction of larger trend
            "wave_completion_threshold": 0.8,# Threshold for wave completion
            "fib_extension_targets": [1.618, 2.618], # Fibonacci targets for extensions
            "fib_retracement_targets": [0.382, 0.618], # Fibonacci targets for retracements
            "sl_prior_wave_pct": 110,        # Stop loss as percentage of prior wave
            "atr_multiple_sl": 2.0,          # ATR multiple for stop loss
            "profit_target_wave_ratio": 1.0, # Target profit as ratio of expected wave move
        }
        
        # Configuration specific to Elliott Wave strategies
        self.config.update({
            "use_alternative_counts": True,   # Consider alternative wave counts
            "use_fib_confluence": True,      # Use Fibonacci confluence for confirmation
            "filter_by_higher_tf": True,     # Filter signals by higher timeframe wave count
            "use_momentum_filter": True,     # Use momentum indicators as filter
            "min_risk_reward": 1.5,         # Minimum risk-reward ratio
            "max_signals_per_day": 3,       # Maximum number of signals per day
            "wave_validation_checks": True,  # Perform extra validation on wave patterns
            "use_time_cycles": False,        # Use time cycles for confirmation (future enhancement)
        })
        
        self.logger.info(f"Initialized {self.name} with Elliott Wave parameters")

    def _adapt_parameters_to_regime(self, regime: MarketRegime) -> None:
        """Adjust strategy parameters based on the current market regime."""
        self.logger.info(f"Adapting parameters to {regime} regime")
        
        if regime == MarketRegime.TRENDING:
            # In trending markets, focus on impulse waves
            self.adaptive_params["min_wave_confidence"] = 0.7
            self.adaptive_params["preferred_entry_waves"] = [2]  # Enter at wave 2 pullbacks
            self.adaptive_params["correction_entry_filter"] = False
            self.adaptive_params["trend_continuation_only"] = True
            self.adaptive_params["fib_extension_targets"] = [1.618, 2.618]
            self.config["use_alternative_counts"] = False  # More conviction in primary count
            
        elif regime == MarketRegime.RANGING:
            # In ranging markets, focus on corrective patterns
            self.adaptive_params["min_wave_confidence"] = 0.8  # Require higher confidence
            self.adaptive_params["preferred_entry_waves"] = [4, 2]  # Prefer wave 4 entries
            self.adaptive_params["correction_entry_filter"] = True
            self.adaptive_params["trend_continuation_only"] = False
            self.adaptive_params["fib_extension_targets"] = [1.0, 1.382]  # More conservative targets
            self.adaptive_params["fib_retracement_targets"] = [0.5, 0.618]
            self.config["use_alternative_counts"] = True  # Consider alternatives
            
        elif regime == MarketRegime.VOLATILE:
            # In volatile markets, be more selective with entries
            self.adaptive_params["min_wave_confidence"] = 0.85  # Very high confidence required
            self.adaptive_params["sl_prior_wave_pct"] = 120  # Wider stops
            self.adaptive_params["atr_multiple_sl"] = 2.5
            self.adaptive_params["wave_completion_threshold"] = 0.9  # Wait for more completion
            self.config["min_risk_reward"] = 2.0  # Require higher R:R
            self.config["max_signals_per_day"] = 1  # Very selective
            
        elif regime == MarketRegime.BREAKOUT:
            # In breakout regimes, focus on wave 3 momentum
            self.adaptive_params["preferred_entry_waves"] = [1]  # Enter at wave 1 completion
            self.adaptive_params["fib_extension_targets"] = [2.618, 4.236]  # More aggressive targets
            self.adaptive_params["profit_target_wave_ratio"] = 1.5
            self.config["use_momentum_filter"] = True
            self.config["max_signals_per_day"] = 2
        
        self.logger.info(
            f"Adapted Elliott Wave parameters - Min confidence: {self.adaptive_params['min_wave_confidence']}, "
            f"Preferred entry waves: {self.adaptive_params['preferred_entry_waves']}"
        )

    def _perform_strategy_analysis(
        self,
        symbol: str,
        price_data: Dict[str, pd.DataFrame],
        confluence_results: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform Elliott Wave analysis on the provided price data.
        
        Args:
            symbol: Trading symbol
            price_data: Dict of price data frames indexed by timeframe
            confluence_results: Results from confluence analysis
            additional_data: Any additional data required
            
        Returns:
            Dictionary with strategy-specific analysis results
        """
        self.logger.info(f"Performing Elliott Wave analysis for {symbol}")
        
        analysis_result = {
            "wave_patterns": {},
            "wave_signals": {},
            "fibonacci_levels": {},
            "higher_tf_context": {},
            "filter_results": {},
            "combined_score": 0.0
        }
        
        try:
            # Get primary and higher timeframes for analysis
            primary_df = price_data.get(self.primary_timeframe)
            if primary_df is None or primary_df.empty:
                return analysis_result
                
            # Define higher timeframe for context
            higher_tfs = [tf for tf in self.timeframes if tf != self.primary_timeframe]
            if higher_tfs:
                higher_tf = higher_tfs[0]  # Use first higher timeframe
                higher_df = price_data.get(higher_tf)
            else:
                higher_tf = None
                higher_df = None
            
            # Analyze Elliott Wave patterns for each timeframe
            for timeframe, df in price_data.items():
                if df is None or df.empty:
                    continue
                
                # Calculate ATR for sizing
                atr = self.technical_indicators.calculate_atr(df, period=14).iloc[-1]
                
                # Get Elliott Wave count
                wave_count = self.elliott_wave_analyzer.analyze_waves(
                    price_data=df,
                    min_confidence=self.adaptive_params["min_wave_confidence"]
                )
                
                # Skip timeframe if no valid wave count
                if not wave_count or wave_count.get("confidence", 0) < self.adaptive_params["min_wave_confidence"]:
                    continue
                
                # Store wave patterns
                analysis_result["wave_patterns"][timeframe] = wave_count
                
                # Get current wave and expected next wave
                current_wave = wave_count.get("current_wave", {})
                next_wave = wave_count.get("next_wave", {})
                
                # Get wave signals based on current and next waves
                wave_signals = self._generate_wave_signals(
                    df=df,
                    timeframe=timeframe,
                    current_wave=current_wave,
                    next_wave=next_wave,
                    atr=atr
                )
                
                # Store wave signals
                analysis_result["wave_signals"][timeframe] = wave_signals
                
                # Calculate Fibonacci levels for current wave structure
                fib_levels = self._calculate_fibonacci_levels(
                    df=df,
                    wave_count=wave_count
                )
                
                # Store Fibonacci levels
                analysis_result["fibonacci_levels"][timeframe] = fib_levels
            
            # Analyze higher timeframe context if available
            if higher_df is not None and higher_tf is not None:
                higher_tf_context = self._analyze_higher_timeframe_context(
                    primary_df=primary_df,
                    higher_df=higher_df,
                    primary_wave_count=analysis_result["wave_patterns"].get(self.primary_timeframe, {}),
                    higher_wave_count=analysis_result["wave_patterns"].get(higher_tf, {})
                )
                
                analysis_result["higher_tf_context"] = higher_tf_context
            
            # Apply filters
            filter_results = self._apply_filters(
                price_data=price_data,
                analysis_result=analysis_result,
                confluence_results=confluence_results
            )
            
            analysis_result["filter_results"] = filter_results
            
            # Calculate combined score
            analysis_result["combined_score"] = self._calculate_combined_score(
                analysis_result=analysis_result,
                confluence_results=confluence_results
            )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in Elliott Wave analysis: {str(e)}", exc_info=True)
            return analysis_result

    def _generate_wave_signals(
        self,
        df: pd.DataFrame,
        timeframe: str,
        current_wave: Dict[str, Any],
        next_wave: Dict[str, Any],
        atr: float
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on Elliott Wave analysis."""
        signals = []
        
        # Get current data
        current_price = df["close"].iloc[-1]
        
        # Skip if no current wave data
        if not current_wave:
            return signals
            
        # Check if we have a valid entry opportunity based on the current wave
        wave_num = current_wave.get("wave_num")
        wave_degree = current_wave.get("wave_degree", "intermediate")
        wave_type = current_wave.get("wave_type", "")
        completion = current_wave.get("completion", 0.0)
        direction = current_wave.get("direction", "")
        
        # Valid entry waves
        preferred_waves = self.adaptive_params["preferred_entry_waves"]
        
        # Check for entry conditions
        if wave_num in preferred_waves:
            # For wave 2 or 4 entries (countertrend corrections that are ending)
            if wave_num in [2, 4] and completion >= self.adaptive_params["wave_completion_threshold"]:
                # Wave 2/4 should be ending, which means we prepare for wave 3/5 in the main trend direction
                entry_direction = "buy" if direction == "bullish" else "sell"
                
                # Calculate entry, stop and target prices
                entry_price = current_price
                
                # Stop loss based on wave criteria
                wave_extreme = current_wave.get("extreme_price", 0)
                if wave_extreme > 0:
                    # For bullish signals, stop below the wave extreme
                    if entry_direction == "buy":
                        buffer = (current_price - wave_extreme) * (self.adaptive_params["sl_prior_wave_pct"] / 100 - 1)
                        stop_loss = wave_extreme - buffer
                    # For bearish signals, stop above the wave extreme
                    else:
                        buffer = (wave_extreme - current_price) * (self.adaptive_params["sl_prior_wave_pct"] / 100 - 1)
                        stop_loss = wave_extreme + buffer
                else:
                    # Fallback to ATR-based stop
                    if entry_direction == "buy":
                        stop_loss = current_price - (atr * self.adaptive_params["atr_multiple_sl"])
                    else:
                        stop_loss = current_price + (atr * self.adaptive_params["atr_multiple_sl"])
                
                # Target based on expected wave movement
                expected_move = next_wave.get("expected_magnitude", 0)
                if expected_move > 0:
                    if entry_direction == "buy":
                        take_profit = current_price + (expected_move * self.adaptive_params["profit_target_wave_ratio"])
                    else:
                        take_profit = current_price - (expected_move * self.adaptive_params["profit_target_wave_ratio"])
                else:
                    # Fallback to risk-reward based target
                    risk = abs(entry_price - stop_loss)
                    if entry_direction == "buy":
                        take_profit = entry_price + (risk * self.config["min_risk_reward"])
                    else:
                        take_profit = entry_price - (risk * self.config["min_risk_reward"])
                
                # Create signal
                signals.append({
                    "type": f"wave_{wave_num}_completion",
                    "direction": entry_direction,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "wave_info": {
                        "current_wave": wave_num,
                        "next_wave": next_wave.get("wave_num"),
                        "completion": completion,
                        "confidence": current_wave.get("confidence", 0.0)
                    },
                    "description": f"{entry_direction.capitalize()} signal on {timeframe} at Wave {wave_num} completion",
                    "strength": current_wave.get("confidence", 0.0) * 0.8
                })
            
            # For wave 1 completion entries (catching the new trend early)
            elif wave_num == 1 and completion >= self.adaptive_params["wave_completion_threshold"]:
                # Wave 1 is ending, which means we prepare for wave 2 pullback then wave 3
                entry_direction = "buy" if direction == "bullish" else "sell"
                
                # For wave 1, we might want to wait for the wave 2 pullback to complete
                # This is often added as a pending signal
                signals.append({
                    "type": "wave_1_completion_alert",
                    "direction": "neutral",  # This is more of an alert than an entry
                    "description": f"Wave 1 completing on {timeframe}, prepare for Wave 2 pullback",
                    "wave_info": {
                        "current_wave": wave_num,
                        "next_wave": next_wave.get("wave_num"),
                        "completion": completion,
                        "confidence": current_wave.get("confidence", 0.0)
                    },
                    "strength": current_wave.get("confidence", 0.0) * 0.6
                })
                
            # For wave 3 confirmation entries (trend confirmation, often with breakout)
            elif wave_num == 3 and completion <= 0.3:  # Early in wave 3
                # Early wave 3 detection, strong trend signal
                entry_direction = "buy" if direction == "bullish" else "sell"
                
                # Calculate entry, stop and target prices
                entry_price = current_price
                
                # Stop is typically below the start of wave 3
                wave_start = current_wave.get("start_price", 0)
                if wave_start > 0:
                    if entry_direction == "buy":
                        stop_loss = min(wave_start, current_price - (atr * self.adaptive_params["atr_multiple_sl"]))
                    else:
                        stop_loss = max(wave_start, current_price + (atr * self.adaptive_params["atr_multiple_sl"]))
                else:
                    # Fallback to ATR-based stop
                    if entry_direction == "buy":
                        stop_loss = current_price - (atr * self.adaptive_params["atr_multiple_sl"])
                    else:
                        stop_loss = current_price + (atr * self.adaptive_params["atr_multiple_sl"])
                
                # Wave 3 targets are often extended
                expected_move = next_wave.get("expected_magnitude", 0)
                if expected_move > 0:
                    # Wave 3 often extends to 1.618 or 2.618 of wave 1
                    extension_factor = self.adaptive_params["fib_extension_targets"][0]  # Use first extension target
                    if entry_direction == "buy":
                        take_profit = current_price + (expected_move * extension_factor)
                    else:
                        take_profit = current_price - (expected_move * extension_factor)
                else:
                    # Fallback to risk-reward based target
                    risk = abs(entry_price - stop_loss)
                    reward_factor = max(2.0, self.config["min_risk_reward"])  # Higher reward for wave 3
                    if entry_direction == "buy":
                        take_profit = entry_price + (risk * reward_factor)
                    else:
                        take_profit = entry_price - (risk * reward_factor)
                
                # Create signal
                signals.append({
                    "type": "wave_3_confirmation",
                    "direction": entry_direction,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "wave_info": {
                        "current_wave": wave_num,
                        "next_wave": next_wave.get("wave_num"),
                        "completion": completion,
                        "confidence": current_wave.get("confidence", 0.0)
                    },
                    "description": f"Strong {entry_direction.capitalize()} trend on {timeframe} with Wave 3 confirmation",
                    "strength": current_wave.get("confidence", 0.0) * 0.9  # Wave 3 signals are strong
                })
        
        return signals

    def _calculate_fibonacci_levels(
        self,
        df: pd.DataFrame,
        wave_count: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate Fibonacci levels based on the current wave structure."""
        fib_levels = {
            "retracements": {},
            "extensions": {},
            "projections": {}
        }
        
        # Get current wave information
        current_wave = wave_count.get("current_wave", {})
        wave_num = current_wave.get("wave_num")
        
        if not wave_num:
            return fib_levels
            
        # Get wave points for Fibonacci calculations
        wave_points = wave_count.get("wave_points", {})
        
        # Calculate retracements for the current wave
        if wave_num in [2, 4, "b"]:  # Correction waves
            start_point = wave_points.get(f"wave_{wave_num-1}_start")
            end_point = wave_points.get(f"wave_{wave_num-1}_end")
            
            if start_point and end_point and start_point.get("price") and end_point.get("price"):
                retracements = self.fibonacci_tools.calculate_retracements(
                    start_price=start_point.get("price"),
                    end_price=end_point.get("price"),
                    direction="up" if end_point.get("price") > start_point.get("price") else "down"
                )
                
                fib_levels["retracements"] = retracements
        
        # Calculate extensions for potential future waves
        if wave_num in [1, 3]:  # After impulse waves
            # For wave 3 extensions based on wave 1
            if wave_num == 3:
                wave1_start = wave_points.get("wave_1_start")
                wave1_end = wave_points.get("wave_1_end")
                
                if wave1_start and wave1_end and wave1_start.get("price") and wave1_end.get("price"):
                    wave1_height = abs(wave1_end.get("price") - wave1_start.get("price"))
                    direction = "up" if wave1_end.get("price") > wave1_start.get("price") else "down"
                    
                    # Calculate wave 3 extensions
                    extensions = self.fibonacci_tools.calculate_extensions(
                        base_move=wave1_height,
                        start_price=wave_points.get("wave_2_end", {}).get("price", df["close"].iloc[-1]),
                        direction=direction
                    )
                    
                    fib_levels["extensions"] = extensions
        
        # Calculate projections for ABC patterns in corrections
        if wave_num in ["a", "c"]:  # Within corrective patterns
            wave_a_start = wave_points.get("wave_a_start")
            wave_a_end = wave_points.get("wave_a_end")
            wave_b_end = wave_points.get("wave_b_end")
            
            if wave_a_start and wave_a_end and wave_b_end:
                projections = self.fibonacci_tools.calculate_abc_projections(
                    a_start_price=wave_a_start.get("price"),
                    a_end_price=wave_a_end.get("price"),
                    b_end_price=wave_b_end.get("price")
                )
                
                fib_levels["projections"] = projections
            
        return fib_levels

    def _analyze_higher_timeframe_context(
        self,
        primary_df: pd.DataFrame,
        higher_df: pd.DataFrame,
        primary_wave_count: Dict[str, Any],
        higher_wave_count: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the wave count context from higher timeframes."""
        context = {
            "alignment": False,
            "higher_tf_wave": None,
            "direction_match": False,
            "strength": 0.0
        }
        
        # Skip if missing data
        if not primary_wave_count or not higher_wave_count:
            return context
            
        # Get wave information
        primary_current_wave = primary_wave_count.get("current_wave", {})
        higher_current_wave = higher_wave_count.get("current_wave", {})
        
        # Get the directions
        primary_direction = primary_current_wave.get("direction")
        higher_direction = higher_current_wave.get("direction")
        
        # Check if directions match
        direction_match = primary_direction == higher_direction
        context["direction_match"] = direction_match
        
        # Get higher timeframe wave
        context["higher_tf_wave"] = higher_current_wave.get("wave_num")
        
        # Check for alignment based on wave theory
        # e.g., if higher timeframe is in wave 3 (strong trend), lower timeframe impulse waves are more reliable
        higher_wave_num = higher_current_wave.get("wave_num")
        primary_wave_num = primary_current_wave.get("wave_num")
        
        # Strong alignment examples:
        # Higher TF in wave 3 (strong trend) and lower TF showing impulsive wave in same direction
        # Higher TF in wave 2/4 (correction) and lower TF showing corrective pattern
        
        if higher_wave_num in [1, 3, 5] and primary_wave_num in [1, 3, 5] and direction_match:
            alignment = True
            strength = 0.9
        elif higher_wave_num in [2, 4] and primary_wave_num in [2, 4, "a", "b", "c"] and direction_match:
            alignment = True
            strength = 0.8
        elif higher_wave_num in [1, 3, 5] and not direction_match:
            # Counter-trend on lower timeframe - be cautious
            alignment = False
            strength = 0.3
        else:
            # Default alignment check
            alignment = direction_match
            strength = 0.6 if direction_match else 0.4
        
        context["alignment"] = alignment
        context["strength"] = strength
        
        return context

    def _apply_filters(
        self,
        price_data: Dict[str, pd.DataFrame],
        analysis_result: Dict[str, Any],
        confluence_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply additional filters to enhance signal quality."""
        filter_results = {
            "momentum_aligned": False,
            "higher_tf_aligned": False,
            "fib_confluence": False,
            "pattern_validated": False
        }
        
        try:
            primary_df = price_data.get(self.primary_timeframe)
            if primary_df is None or primary_df.empty:
                return filter_results
                
            # Get wave signals for primary timeframe
            primary_signals = analysis_result.get("wave_signals", {}).get(self.primary_timeframe, [])
            if not primary_signals:
                return filter_results
                
            # Get most recent signal direction
            signal_direction = "neutral"
            for signal in primary_signals:
                if signal.get("direction") in ["buy", "sell"]:
                    signal_direction = signal["direction"]
                    break
                    
            # Check momentum alignment if configured
            if self.config["use_momentum_filter"]:
                # Get momentum indicators
                rsi = self.technical_indicators.calculate_rsi(primary_df)
                macd = self.technical_indicators.calculate_macd(primary_df)
                
                # Check momentum alignment
                if signal_direction == "buy":
                    momentum_aligned = (rsi.iloc[-1] > 40 and  # RSI above 40
                                      rsi.iloc[-1] > rsi.iloc[-2] and  # RSI rising
                                      macd["histogram"].iloc[-1] > 0)  # MACD histogram positive
                    filter_results["momentum_aligned"] = momentum_aligned
                
                elif signal_direction == "sell":
                    momentum_aligned = (rsi.iloc[-1] < 60 and  # RSI below 60
                                      rsi.iloc[-1] < rsi.iloc[-2] and  # RSI falling
                                      macd["histogram"].iloc[-1] < 0)  # MACD histogram negative
                    filter_results["momentum_aligned"] = momentum_aligned
            
            # Check higher timeframe alignment if configured
            if self.config["filter_by_higher_tf"]:
                higher_tf_context = analysis_result.get("higher_tf_context", {})
                filter_results["higher_tf_aligned"] = higher_tf_context.get("alignment", False)
            
            # Check Fibonacci confluence if configured
            if self.config["use_fib_confluence"]:
                # Get Fibonacci levels
                fib_levels = analysis_result.get("fibonacci_levels", {}).get(self.primary_timeframe, {})
                
                # Get support/resistance levels from confluence analysis
                sr_zones = confluence_results.get("support_resistance_confluence", [])
                
                # Check if any Fibonacci level aligns with S/R zones
                fib_points = []
                
                # Collect all Fibonacci points
                for category, levels in fib_levels.items():
                    for level, price in levels.items():
                        fib_points.append(price)
                
                # Get current price
                current_price = primary_df["close"].iloc[-1]
                
                # Check for confluence between Fibonacci levels and S/R zones
                fib_sr_confluence = False
                for fib_price in fib_points:
                    for zone in sr_zones:
                        zone_price = zone.get("price", 0)
                        # If Fibonacci level and S/R zone are within 0.5% of each other
                        if abs(fib_price - zone_price) / current_price < 0.005:
                            fib_sr_confluence = True
                            break
                    if fib_sr_confluence:
                        break
                
                filter_results["fib_confluence"] = fib_sr_confluence
            
            # Validate wave pattern if configured
            if self.config["wave_validation_checks"]:
                # Get wave count for validation
                wave_count = analysis_result.get("wave_patterns", {}).get(self.primary_timeframe, {})
                current_wave = wave_count.get("current_wave", {})
                
                # Run validation checks on the wave pattern
                validation_passed = self._validate_wave_pattern(primary_df, current_wave)
                filter_results["pattern_validated"] = validation_passed
            
            return filter_results
            
        except Exception as e:
            self.logger.error(f"Error applying Elliott Wave filters: {str(e)}")
            return filter_results

    def _validate_wave_pattern(
        self,
        df: pd.DataFrame,
        current_wave: Dict[str, Any]
    ) -> bool:
        """Validate the Elliott Wave pattern using additional checks."""
        # Skip if no wave data
        if not current_wave:
            return False
            
        # Get wave information
        wave_num = current_wave.get("wave_num")
        direction = current_wave.get("direction")
        
        if not wave_num or not direction:
            return False
            
        # Basic validation passed
        return True

    def _calculate_combined_score(
        self,
        analysis_result: Dict[str, Any],
        confluence_results: Dict[str, Any]
    ) -> float:
        """Calculate combined strategy score."""
        score = 0.0
        
        try:
            # Get wave signals from primary timeframe
            primary_signals = analysis_result.get("wave_signals", {}).get(self.primary_timeframe, [])
            
            if not primary_signals:
                return score
                
            # Base score on the strongest signal's strength
            signal_strengths = [s.get("strength", 0) for s in primary_signals if s.get("direction") in ["buy", "sell"]]
            if signal_strengths:
                max_signal_strength = max(signal_strengths)
                score += max_signal_strength * 0.5  # 50% weight for signal strength
                
            # Add filter confirmations
            filter_results = analysis_result.get("filter_results", {})
            filter_count = sum(1 for v in filter_results.values() if v)
            max_filters = len(filter_results)
            
            if max_filters > 0:
                filter_score = filter_count / max_filters
                score += filter_score * 0.3  # 30% weight for filters
                
            # Add confluence from S/R and pattern recognition
            confluence_score = confluence_results.get("confluence_score", 0)
            score += confluence_score * 0.2  # 20% weight for confluence
            
            # Ensure score is between 0 and 1
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating combined score: {str(e)}")
            return score

    def _generate_signals(
        self,
        symbol: str,
        strategy_analysis: Dict[str, Any],
        confluence_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on strategy analysis."""
        signals = []
        
        try:
            # Get wave signals from primary timeframe
            primary_signals = strategy_analysis.get("wave_signals", {}).get(self.primary_timeframe, [])
            
            # Skip if no signals
            if not primary_signals:
                return []
                
            # Get combined score
            combined_score = strategy_analysis.get("combined_score", 0.0)
            
            # Skip if score is too low
            min_score_threshold = 0.65  # Higher threshold for Elliott Wave signals
            if combined_score < min_score_threshold:
                return []
                
            # Get filter results
            filter_results = strategy_analysis.get("filter_results", {})
            
            # Check if we have enough filter confirmations
            min_filter_count = 2 if self.market_regime == MarketRegime.VOLATILE else 1
            filter_count = sum(1 for v in filter_results.values() if v)
            
            if filter_count < min_filter_count:
                return []
                
            # Process signals that match our entry criteria
            for wave_signal in primary_signals:
                if wave_signal.get("type") in ["wave_2_completion", "wave_4_completion", "wave_3_confirmation"]:
                    direction = wave_signal.get("direction")
                    
                    if direction not in ["buy", "sell"]:
                        continue
                        
                    # Get signal details
                    entry_price = wave_signal.get("entry_price")
                    stop_loss = wave_signal.get("stop_loss")
                    take_profit = wave_signal.get("take_profit")
                    
                    if not entry_price or not stop_loss or not take_profit:
                        continue
                        
                    # Calculate risk-reward ratio
                    risk = abs(entry_price - stop_loss)
                    reward = abs(entry_price - take_profit)
                    
                    if risk > 0:
                        reward_risk_ratio = reward / risk
                    else:
                        reward_risk_ratio = 0
                        
                    # Check if reward-risk meets minimum requirement
                    if reward_risk_ratio < self.config["min_risk_reward"]:
                        continue
                        
                    # Create final signal with all metadata
                    signals.append({
                        "symbol": symbol,
                        "strategy": self.name,
                        "direction": direction,
                        "type": wave_signal.get("type"),
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "timeframe": self.primary_timeframe,
                        "confidence": combined_score,
                        "timestamp": datetime.now().isoformat(),
                        "reward_risk_ratio": reward_risk_ratio,
                        "metadata": {
                            "wave_info": wave_signal.get("wave_info", {}),
                            "filter_confirmations": filter_results,
                            "signal_description": wave_signal.get("description", ""),
                            "signal_strength": wave_signal.get("strength", 0)
                        }
                    })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating Elliott Wave signals: {str(e)}", exc_info=True)
            
        return signals
