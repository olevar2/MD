"""
Pivot Point Confluence Strategy

This module implements a trading strategy based on pivot point confluence across multiple timeframes,
identifying strong support and resistance levels for high-probability trades.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

from strategy_execution_engine.strategies.advanced_ta_strategy import AdvancedTAStrategy
from analysis_engine.analysis.advanced_ta.pivot_points import PivotPointAnalyzer
from analysis_engine.services.tool_effectiveness import MarketRegime, TimeFrame
from optimization.caching.calculation_cache import memoize_with_ttl


class PivotConfluenceStrategy(AdvancedTAStrategy):
    """
    Trading strategy that identifies high-probability trading opportunities at pivot point confluence zones
    across multiple timeframes, with adaptive parameters based on market regimes.
    """

    def __init__(
        self,
        name: str,
        timeframes: List[str],
        primary_timeframe: str,
        symbols: List[str],
        risk_per_trade_pct: float = 1.0,
        **kwargs
    ):
        super().__init__(
            name=name,
            timeframes=timeframes,
            primary_timeframe=primary_timeframe,
            symbols=symbols,
            risk_per_trade_pct=risk_per_trade_pct,
            **kwargs
        )
        self._init_strategy_config()
        self.logger.info(f"PivotConfluenceStrategy '{name}' initialized")

    def _init_strategy_config(self) -> None:
        # Initialize pivot point analyzer with multiple formula types
        self.pivot_analyzer = PivotPointAnalyzer(
            methods=["standard", "fibonacci", "woodie", "camarilla"],
            lookback_period=10,
            include_midpoints=True
        )
        
        # Adaptive parameters defaults
        self.adaptive_params = {
            "confluence_radius_pips": 10,
            "min_confluence_count": 3,
            "atr_multiple_sl": 1.0,
            "use_adr_adjustment": True,
            "batch_processing": True,
            "max_pivot_age": 5,  # Max age in periods to consider pivot valid
            "score_threshold": 7  # Minimum confluence score to generate signal
        }
        
        # Regime-specific adaptive settings
        self.regime_parameters = {
            MarketRegime.TRENDING.value: {
                "confluence_radius_pips": 15,
                "min_confluence_count": 4,
                "max_pivot_age": 3
            },
            MarketRegime.RANGING.value: {
                "confluence_radius_pips": 8,
                "min_confluence_count": 3,
                "max_pivot_age": 7
            },
            MarketRegime.VOLATILE.value: {
                "confluence_radius_pips": 20,
                "min_confluence_count": 5,
                "max_pivot_age": 2
            },
            MarketRegime.BREAKOUT.value: {
                "confluence_radius_pips": 12,
                "min_confluence_count": 3,
                "max_pivot_age": 4
            }
        }
        
        # Strategy config
        self.config.update({
            "preferred_direction": "both",
            "min_confidence": 0.6,
            "pivot_types": ["PP", "S1", "S2", "R1", "R2", "S3", "R3"]
        })
        
    def _adapt_parameters_to_regime(self, regime: MarketRegime) -> None:
        """
        Adjust adaptive parameters based on market regime for pivot confluence strategy.
        
        Args:
            regime: Current market regime
        """
        params = self.regime_parameters.get(regime.value)
        if not params:
            return
        
        # Apply only parameters that exist in adaptive_params
        adaptive_updates = {k: v for k, v in params.items() if k in self.adaptive_params}
        for key, value in adaptive_updates.items():
            self.adaptive_params[key] = value
            
        self.logger.info(f"Adapted pivot confluence parameters to regime {regime}: {adaptive_updates}")
        
        # Adjust analyzer settings based on regime
        if regime == MarketRegime.TRENDING:
            self.pivot_analyzer.include_midpoints = False
        elif regime == MarketRegime.RANGING:
            self.pivot_analyzer.include_midpoints = True
            
    @memoize_with_ttl(ttl=300)  # Cache results for 5 minutes
    def _calculate_pivot_points(self, symbol: str, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Calculate pivot points with caching for performance optimization.
        
        Args:
            symbol: The trading symbol
            df: Price data DataFrame
            method: Pivot calculation method
            
        Returns:
            DataFrame with pivot calculations
        """
        return self.pivot_analyzer.calculate(df, method=method)
        
    def _find_confluence_zones(
        self,
        price_data: Dict[str, pd.DataFrame],
        symbol: str,
        radius_pips: float
    ) -> List[Dict[str, Any]]:
        """
        Find zones where multiple pivot points from different timeframes are in close proximity.
        
        Args:
            price_data: Dictionary of price data by timeframe
            symbol: Trading symbol
            radius_pips: Radius in pips to consider pivots in confluence
            
        Returns:
            List of confluence zones with scores and pivot details
        """
        all_pivots = []
        current_price = None
        
        # Extract all pivot points from all timeframes
        for tf, df in price_data.items():
            if df.empty:
                continue
                
            # Get current price from the most recent data
            if current_price is None and len(df) > 0:
                current_price = df['close'].iloc[-1]
                
            # Calculate pivots for each method
            for method in ["standard", "fibonacci", "woodie", "camarilla"]:
                try:
                    # Use cached calculation when possible
                    pivot_df = self._calculate_pivot_points(symbol, df, method)
                    
                    # Extract pivot levels
                    for pivot_type in self.config["pivot_types"]:
                        if f"{method}_{pivot_type}" in pivot_df.columns:
                            pivot_value = pivot_df[f"{method}_{pivot_type}"].iloc[-1]
                            if pd.notna(pivot_value):
                                all_pivots.append({
                                    "timeframe": tf,
                                    "method": method,
                                    "type": pivot_type,
                                    "value": pivot_value,
                                    "age": 0  # Age of the most recent pivot
                                })
                                
                    # Add historical pivots with age
                    max_age = self.adaptive_params["max_pivot_age"]
                    for i in range(1, min(max_age + 1, len(pivot_df))):
                        for pivot_type in self.config["pivot_types"]:
                            col_name = f"{method}_{pivot_type}"
                            if col_name in pivot_df.columns:
                                pivot_value = pivot_df[col_name].iloc[-(i+1)]
                                if pd.notna(pivot_value):
                                    all_pivots.append({
                                        "timeframe": tf,
                                        "method": method,
                                        "type": pivot_type,
                                        "value": pivot_value,
                                        "age": i
                                    })
                
                except Exception as e:
                    self.logger.error(f"Error calculating {method} pivots for {tf}: {e}")
        
        # No pivots found
        if not all_pivots:
            return []
            
        # Find confluence zones
        pips_factor = 0.0001  # Adjust based on symbol, this is for major forex pairs
        radius = radius_pips * pips_factor
        
        # Sort pivots by value for efficient processing
        all_pivots.sort(key=lambda x: x["value"])
        
        # Find clusters of pivot points within the radius
        zones = []
        i = 0
        while i < len(all_pivots):
            zone_pivots = [all_pivots[i]]
            center = all_pivots[i]["value"]
            
            # Find all pivots within radius
            j = i + 1
            while j < len(all_pivots) and abs(all_pivots[j]["value"] - center) <= radius:
                zone_pivots.append(all_pivots[j])
                j += 1
                
            # Only include zones with multiple pivots
            if len(zone_pivots) >= self.adaptive_params["min_confluence_count"]:
                # Calculate zone score based on pivot types, methods, timeframes, and age
                score = self._calculate_zone_score(zone_pivots)
                
                # Create zone data
                zone = {
                    "center": sum(p["value"] for p in zone_pivots) / len(zone_pivots),
                    "pivots": zone_pivots,
                    "count": len(zone_pivots),
                    "score": score,
                    "unique_timeframes": len(set(p["timeframe"] for p in zone_pivots)),
                    "unique_methods": len(set(p["method"] for p in zone_pivots))
                }
                
                # Determine zone type (support or resistance)
                if current_price is not None:
                    zone["type"] = "support" if zone["center"] < current_price else "resistance"
                    zone["distance"] = abs(zone["center"] - current_price)
                
                zones.append(zone)
                
            # Move to next potential zone
            i = j
                
        return zones
        
    def _calculate_zone_score(self, pivots: List[Dict[str, Any]]) -> float:
        """
        Calculate a score for a confluence zone based on pivot properties.
        
        Args:
            pivots: List of pivots in the zone
            
        Returns:
            Zone score
        """
        base_score = len(pivots)
        
        # Bonus for diversity
        unique_timeframes = set(p["timeframe"] for p in pivots)
        unique_methods = set(p["method"] for p in pivots)
        unique_types = set(p["type"] for p in pivots)
        
        # Penalties for older pivots
        age_penalty = sum(p["age"] for p in pivots) / len(pivots)
        
        # Bonuses for higher priority pivot types
        type_weights = {"PP": 1.5, "S1": 1.2, "R1": 1.2, "S2": 1.1, "R2": 1.1, "S3": 1.0, "R3": 1.0}
        type_bonus = sum(type_weights.get(p["type"], 1.0) for p in pivots)
        
        # Calculate final score
        score = (base_score * 1.5) + (len(unique_timeframes) * 2) + (len(unique_methods) * 1.5) + (len(unique_types) * 0.5)
        score += type_bonus - age_penalty
        
        return score
        
    def _perform_strategy_analysis(
        self,
        symbol: str,
        price_data: Dict[str, pd.DataFrame],
        confluence_results: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform pivot confluence analysis across timeframes.
        
        Args:
            symbol: Trading symbol
            price_data: Dictionary of price data by timeframe
            confluence_results: Results from other indicators
            additional_data: Any additional data for analysis
            
        Returns:
            Analysis results
        """
        results = {}
        
        # Get current price
        current_price = None
        for tf in sorted(price_data.keys()):
            df = price_data[tf]
            if not df.empty:
                current_price = df['close'].iloc[-1]
                break
                
        if current_price is None:
            self.logger.warning(f"No price data available for {symbol}")
            return results
            
        # Find confluence zones
        radius_pips = self.adaptive_params["confluence_radius_pips"]
        zones = self._find_confluence_zones(price_data, symbol, radius_pips)
        
        if not zones:
            self.logger.info(f"No confluence zones found for {symbol}")
            return results
            
        # Sort zones by score descending
        zones.sort(key=lambda z: z["score"], reverse=True)
        
        # Find nearest support and resistance zones
        supports = [z for z in zones if z["type"] == "support"]
        resistances = [z for z in zones if z["type"] == "resistance"]
        
        nearest_support = min(supports, key=lambda z: z["distance"]) if supports else None
        nearest_resistance = min(resistances, key=lambda z: z["distance"]) if resistances else None
        
        # Store results
        results["zones"] = zones
        results["best_zone"] = zones[0] if zones else None
        results["nearest_support"] = nearest_support
        results["nearest_resistance"] = nearest_resistance
        results["price"] = current_price
        
        # Calculate signal strength based on best zone score
        if zones:
            best_score = zones[0]["score"]
            score_threshold = self.adaptive_params["score_threshold"]
            min_count = self.adaptive_params["min_confluence_count"]
            
            if best_score >= score_threshold and zones[0]["count"] >= min_count:
                results["signal_strength"] = min(10, int(best_score / score_threshold * 5))
            else:
                results["signal_strength"] = 0
        else:
            results["signal_strength"] = 0
            
        return results
        
    def _generate_signals(
        self,
        symbol: str,
        strategy_analysis: Dict[str, Any],
        confluence_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on pivot confluence analysis.
        
        Args:
            symbol: Trading symbol
            strategy_analysis: Results from pivot confluence analysis
            confluence_results: Results from other indicators
            
        Returns:
            List of trading signals
        """
        signals = []
        strength = strategy_analysis.get("signal_strength", 0)
        
        # Check if signal is strong enough
        if strength < 1:
            return signals
            
        # Get current price and best zone
        current_price = strategy_analysis.get("price")
        best_zone = strategy_analysis.get("best_zone")
        nearest_support = strategy_analysis.get("nearest_support")
        nearest_resistance = strategy_analysis.get("nearest_resistance")
        
        if not current_price or not best_zone:
            return signals
            
        # Determine trade direction and entry
        if best_zone["type"] == "support":
            direction = "bullish"
            entry_price = nearest_support["center"] + (0.0001 * 5)  # 5 pips above support
            stop_loss = nearest_support["center"] - (0.0001 * 10)  # 10 pips below support
            
            # Target nearest resistance or use default risk:reward
            if nearest_resistance:
                take_profit = nearest_resistance["center"] - (0.0001 * 5)  # 5 pips below resistance
            else:
                # Use 1:2 risk:reward ratio
                take_profit = entry_price + (entry_price - stop_loss) * 2
                
        else:  # Resistance
            direction = "bearish"
            entry_price = nearest_resistance["center"] - (0.0001 * 5)  # 5 pips below resistance
            stop_loss = nearest_resistance["center"] + (0.0001 * 10)  # 10 pips above resistance
            
            # Target nearest support or use default risk:reward
            if nearest_support:
                take_profit = nearest_support["center"] + (0.0001 * 5)  # 5 pips above support
            else:
                # Use 1:2 risk:reward ratio
                take_profit = entry_price - (stop_loss - entry_price) * 2
        
        # Create detailed zone description
        zone_details = f"{best_zone['count']} pivots"
        if best_zone.get("unique_timeframes"):
            zone_details += f" across {best_zone['unique_timeframes']} timeframes"
        if best_zone.get("unique_methods"):
            zone_details += f" using {best_zone['unique_methods']} pivot methods"
            
        # Create trade signal
        signal = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "signal_type": "pivot_confluence",
            "direction": direction,
            "strength": strength,
            "confidence": min(0.5 + (strength * 0.05), 0.95),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reason": f"Pivot confluence zone ({zone_details}) detected as {best_zone['type']} with score {best_zone['score']:.1f}"
        }
        
        signals.append(signal)
        return signals
""""""
