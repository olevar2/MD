"""
Confluence Analyzer

This module provides comprehensive functionality to identify confluence zones where multiple
technical analysis factors align to provide stronger trading signals.

Confluence is detected when different technical indicators, price patterns, or
key levels align to suggest the same market direction or pivot point.

Features:
- Multi-timeframe analysis
- Support/Resistance confluence
- Indicator alignment detection
- Pattern completion analysis
- Fibonacci level convergence
- Elliott wave pivot points
- Gann level analysis
- Harmonic pattern detection
- Adaptive effectiveness tracking
- External signal aggregation
- Volume profile analysis
- Market structure analysis
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
from enum import Enum
import logging
from dataclasses import dataclass

from analysis_engine.core.base.components import BaseAnalyzer, AnalysisResult
from analysis_engine.analysis.indicators import IndicatorClient
from analysis_engine.services.tool_effectiveness import ToolEffectivenessTracker, MarketRegime
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository

logger = logging.getLogger(__name__)

class ConfluenceStrength(Enum):
    """Strength of confluence signal"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXCEPTIONAL = 5

@dataclass
class ConfluenceZone:
    """Container for confluence zone data"""
    center: float
    lower_bound: float
    upper_bound: float
    strength: ConfluenceStrength
    contributing_tools: List[str]
    timeframes: List[str]
    confluence_types: List[str]
    levels: List[Dict[str, Any]]
    score: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "center": self.center,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "strength": self.strength.name,
            "contributing_tools": self.contributing_tools,
            "timeframes": self.timeframes,
            "confluence_types": self.confluence_types,
            "levels": self.levels,
            "score": self.score,
            "metadata": self.metadata or {}
        }

class ConfluenceAnalyzer(BaseAnalyzer):
    """
    Comprehensive analyzer for identifying confluence points across multiple
    indicators, timeframes, and analysis methods.
    
    Features:
    - Multi-timeframe analysis
    - Support/Resistance confluence
    - Indicator alignment detection
    - Pattern completion analysis
    - Fibonacci level convergence
    - Elliott wave pivot points
    - Gann level analysis
    - Harmonic pattern detection
    - Adaptive effectiveness tracking
    - External signal aggregation
    - Volume profile analysis
    - Market structure analysis
    """
    
    DEFAULT_PARAMS = {
        "min_tools_for_confluence": 2,
        "effectiveness_threshold": 0.5,
        "sr_proximity_threshold": 0.0015,  # 0.15% price proximity for S/R confluence
        "indicator_agreement_threshold": 0.7,  # 70% agreement required
        "timeframe_alignment_required": 3,  # Number of timeframes needed for confluence
        "zone_width_pips": 20,  # Width of confluence zones in pips
        "source_weights": {  # Weights for different signal sources
            "price_action": 1.0,
            "indicator": 0.8,
            "pattern": 0.9,
            "support_resistance": 1.0,
            "trend": 0.9,
            "momentum": 0.8,
            "volatility": 0.7,
            "volume": 0.8,
            "fibonacci": 0.9,
            "elliott_wave": 0.9,
            "sentiment": 0.6,
            "other": 0.5
        },
        "conflicting_signals_penalty": 0.3,  # Penalty for conflicting signals
        "use_volume_profile": True,  # Use volume profile/POC
        "use_market_structure": True,  # Use market structure points
        "use_fibonacci_levels": True,  # Use Fibonacci retracements/extensions
        "use_moving_averages": True,  # Use moving averages as support/resistance
        "moving_averages": [20, 50, 100, 200],  # MA periods to use
        "use_pivots": True,
        "pivot_type": "traditional",  # traditional, fibonacci, camarilla, etc.
        "session_barriers": True,  # Use previous session high/low/close
        "max_levels": 10,  # Maximum number of confluent levels to return
    }
    
    def __init__(
        self,
        tool_effectiveness_repository: Optional[ToolEffectivenessRepository] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the confluence analyzer
        
        Args:
            tool_effectiveness_repository: Repository for tracking tool effectiveness
            parameters: Optional configuration parameters
        """
        resolved_params = self.DEFAULT_PARAMS.copy()
        if parameters:
            resolved_params.update(parameters)
            
        super().__init__(name="ConfluenceAnalyzer", parameters=resolved_params)
        
        self.effectiveness_repository = tool_effectiveness_repository
        self.tool_tracker = ToolEffectivenessTracker() if tool_effectiveness_repository else None
        self.indicator_client = IndicatorClient()
        
        # Initialize signal cache for external signals
        self.signal_cache = []
        self.last_analysis_time = None
        
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Analyze market data to identify confluence zones.
        
        Args:
            data: Dictionary containing market data and parameters
                {
                    "symbol": str,
                    "timeframe": str,
                    "market_data": {
                        "open": List[float],
                        "high": List[float],
                        "low": List[float],
                        "close": List[float],
                        "volume": List[float],
                        "timestamp": List[str]
                    }
                }
                
        Returns:
            AnalysisResult containing identified confluence zones
        """
        try:
            # Extract data
            symbol = data["symbol"]
            timeframe = data["timeframe"]
            market_data = data["market_data"]
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(market_data)
            
            # Get current price
            current_price = df["close"].iloc[-1]
            
            # Collect all levels from different sources
            all_levels = self._collect_all_levels(df, current_price)
            
            # Get effective tools if repository is available
            effective_tools = {}
            if self.effectiveness_repository and "market_regime" in data:
                effective_tools = await self._get_effective_tools(data["market_regime"])
            
            # Group levels into zones
            confluence_zones = self._group_levels_into_zones(
                all_levels,
                current_price,
                effective_tools
            )
            
            # Calculate zone strength
            scored_zones = self._calculate_zone_strength(
                confluence_zones,
                data.get("market_regime"),
                effective_tools
            )
            
            # Sort zones by strength
            sorted_zones = sorted(
                scored_zones,
                key=lambda z: z.strength.value,
                reverse=True
            )[:self.parameters["max_levels"]]
            
            # Prepare result
            result = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "current_price": current_price,
                "confluence_zones": [zone.to_dict() for zone in sorted_zones],
                "market_regime": data.get("market_regime"),
                "effective_tools": effective_tools
            }
            
            return AnalysisResult(
                analyzer_name=self.name,
                result=result,
                metadata={
                    "timeframe": timeframe,
                    "zone_count": len(sorted_zones)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in confluence analysis: {str(e)}", exc_info=True)
            return AnalysisResult(
                analyzer_name=self.name,
                result={"error": f"Analysis failed: {str(e)}"},
                is_valid=False
            )
            
    def _collect_all_levels(self, df: pd.DataFrame, current_price: float) -> List[Dict[str, Any]]:
        """Collect all potential support/resistance levels"""
        levels = []
        
        # Add support/resistance levels
        if self.parameters["use_market_structure"]:
            sr_levels = self._get_support_resistance_levels(df)
            levels.extend(sr_levels)
        
        # Add Fibonacci levels
        if self.parameters["use_fibonacci_levels"]:
            fib_levels = self._get_fibonacci_levels(df)
            levels.extend(fib_levels)
        
        # Add moving averages
        if self.parameters["use_moving_averages"]:
            ma_levels = self._get_moving_average_levels(df)
            levels.extend(ma_levels)
        
        # Add pivot points
        if self.parameters["use_pivots"]:
            pivot_levels = self._get_pivot_levels(df)
            levels.extend(pivot_levels)
        
        # Add volume profile levels
        if self.parameters["use_volume_profile"]:
            volume_levels = self._get_volume_profile_levels(df)
            levels.extend(volume_levels)
        
        return levels
        
    def _get_support_resistance_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get support and resistance levels from market structure"""
        levels = []
        
        # Get swing highs and lows
        highs = df["high"].values
        lows = df["low"].values
        
        # Find swing points
        for i in range(2, len(df) - 2):
            # Swing high
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                levels.append({
                    "price": highs[i],
                    "type": "resistance",
                    "tool_id": "market_structure",
                    "timeframe": "current",
                    "strength": 1.0
                })
            
            # Swing low
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                levels.append({
                    "price": lows[i],
                    "type": "support",
                    "tool_id": "market_structure",
                    "timeframe": "current",
                    "strength": 1.0
                })
        
        return levels
        
    def _get_fibonacci_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get Fibonacci retracement and extension levels"""
        levels = []
        
        # Get recent swing high and low
        recent_high = df["high"].max()
        recent_low = df["low"].min()
        
        # Fibonacci retracement levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for level in fib_levels:
            # Retracement from high to low
            price = recent_high - (recent_high - recent_low) * level
            levels.append({
                "price": price,
                "type": "fibonacci",
                "tool_id": "fibonacci",
                "timeframe": "current",
                "strength": 0.9,
                "level": level
            })
            
            # Retracement from low to high
            price = recent_low + (recent_high - recent_low) * level
            levels.append({
                "price": price,
                "type": "fibonacci",
                "tool_id": "fibonacci",
                "timeframe": "current",
                "strength": 0.9,
                "level": level
            })
        
        return levels
        
    def _get_moving_average_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get moving average levels"""
        levels = []
        
        for period in self.parameters["moving_averages"]:
            ma = self.indicator_client.calculate_sma(df["close"], period)
            current_ma = ma.iloc[-1]
            
            levels.append({
                "price": current_ma,
                "type": "moving_average",
                "tool_id": f"ma_{period}",
                "timeframe": "current",
                "strength": 0.8,
                "period": period
            })
        
        return levels
        
    def _get_pivot_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get pivot point levels"""
        levels = []
        
        # Get previous day's data
        prev_high = df["high"].iloc[-2]
        prev_low = df["low"].iloc[-2]
        prev_close = df["close"].iloc[-2]
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Calculate support and resistance levels
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        
        # Add levels
        levels.extend([
            {
                "price": pivot,
                "type": "pivot",
                "tool_id": "pivot",
                "timeframe": "daily",
                "strength": 0.9,
                "level": "pivot"
            },
            {
                "price": r1,
                "type": "resistance",
                "tool_id": "pivot",
                "timeframe": "daily",
                "strength": 0.8,
                "level": "r1"
            },
            {
                "price": s1,
                "type": "support",
                "tool_id": "pivot",
                "timeframe": "daily",
                "strength": 0.8,
                "level": "s1"
            },
            {
                "price": r2,
                "type": "resistance",
                "tool_id": "pivot",
                "timeframe": "daily",
                "strength": 0.7,
                "level": "r2"
            },
            {
                "price": s2,
                "type": "support",
                "tool_id": "pivot",
                "timeframe": "daily",
                "strength": 0.7,
                "level": "s2"
            }
        ])
        
        return levels
        
    def _get_volume_profile_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get volume profile levels"""
        levels = []
        
        # Calculate volume profile
        price_range = df["high"].max() - df["low"].min()
        num_bins = 50
        bin_size = price_range / num_bins
        
        # Create price bins
        bins = np.arange(df["low"].min(), df["high"].max() + bin_size, bin_size)
        
        # Calculate volume profile
        volume_profile = np.zeros(len(bins) - 1)
        for i in range(len(df)):
            price = df["close"].iloc[i]
            volume = df["volume"].iloc[i]
            bin_idx = np.digitize(price, bins) - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile[bin_idx] += volume
        
        # Find POC (Point of Control)
        poc_idx = np.argmax(volume_profile)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Find value area (70% of volume)
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * 0.7
        
        # Sort bins by volume
        sorted_indices = np.argsort(volume_profile)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumulative_volume += volume_profile[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= target_volume:
                break
        
        # Calculate value area bounds
        value_area_prices = [bins[i] for i in value_area_indices]
        va_high = max(value_area_prices)
        va_low = min(value_area_prices)
        
        # Add levels
        levels.extend([
            {
                "price": poc_price,
                "type": "volume_profile",
                "tool_id": "volume_profile",
                "timeframe": "current",
                "strength": 1.0,
                "level": "poc"
            },
            {
                "price": va_high,
                "type": "volume_profile",
                "tool_id": "volume_profile",
                "timeframe": "current",
                "strength": 0.8,
                "level": "va_high"
            },
            {
                "price": va_low,
                "type": "volume_profile",
                "tool_id": "volume_profile",
                "timeframe": "current",
                "strength": 0.8,
                "level": "va_low"
            }
        ])
        
        return levels
        
    def _group_levels_into_zones(
        self,
        levels: List[Dict[str, Any]],
        current_price: float,
        effective_tools: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Group similar price levels into confluence zones"""
        if not levels:
            return []
            
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x["price"])
        
        # Calculate base zone width
        base_zone_width = current_price * self.parameters["zone_width_pips"] / 10000
        
        # Initialize zones
        zones = []
        
        # Process each level
        for level in sorted_levels:
            # Check if this level fits into an existing zone
            added_to_existing = False
            for zone in zones:
                # If price is within the zone's bounds, add to zone
                if zone["lower_bound"] <= level["price"] <= zone["upper_bound"]:
                    zone["levels"].append(level)
                    # Update contributing tools
                    if level["tool_id"] not in zone["contributing_tools"]:
                        zone["contributing_tools"].append(level["tool_id"])
                    # Update timeframes
                    if level["timeframe"] not in zone["timeframes"]:
                        zone["timeframes"].append(level["timeframe"])
                    # Update confluence types
                    level_type = level["type"]
                    if level_type not in zone["confluence_types"]:
                        zone["confluence_types"].append(level_type)
                    
                    added_to_existing = True
                    break
            
            # If didn't fit existing zone, create new zone
            if not added_to_existing:
                zone_width = base_zone_width
                
                # Create new zone
                zones.append({
                    "center": level["price"],
                    "lower_bound": level["price"] - (zone_width / 2),
                    "upper_bound": level["price"] + (zone_width / 2),
                    "levels": [level],
                    "contributing_tools": [level["tool_id"]],
                    "timeframes": [level["timeframe"]],
                    "confluence_types": [level["type"]]
                })
        
        # Filter zones that have confluence
        confluent_zones = []
        for zone in zones:
            # Only keep zone if it has multiple tools or meets minimum requirements
            effective_tools_in_zone = [
                tool for tool in zone["contributing_tools"] 
                if tool in effective_tools
            ]
            
            # Check for sufficient confluence
            if (len(zone["contributing_tools"]) >= self.parameters["min_tools_for_confluence"] or 
                len(effective_tools_in_zone) > 0 or
                len(zone["timeframes"]) > 1):
                confluent_zones.append(zone)
        
        return confluent_zones
        
    def _calculate_zone_strength(
        self, 
        zones: List[Dict[str, Any]],
        market_regime: Optional[MarketRegime],
        effective_tools: Dict[str, float]
    ) -> List[ConfluenceZone]:
        """Calculate strength of confluence zones"""
        confluence_zones = []
        
        for zone in zones:
            # Calculate base score from number of tools and timeframes
            base_score = (
                len(zone["contributing_tools"]) * 0.4 +
                len(zone["timeframes"]) * 0.3 +
                len(zone["confluence_types"]) * 0.3
            )
            
            # Apply effectiveness weights
            effective_tools_in_zone = [
                tool for tool in zone["contributing_tools"] 
                if tool in effective_tools
            ]
            
            if effective_tools_in_zone:
                effectiveness_score = sum(
                    effective_tools[tool]
                    for tool in effective_tools_in_zone
                ) / len(effective_tools_in_zone)
                base_score *= (1 + effectiveness_score)
            
            # Determine strength based on score
            if base_score >= 4.0:
                strength = ConfluenceStrength.EXCEPTIONAL
            elif base_score >= 3.0:
                strength = ConfluenceStrength.VERY_STRONG
            elif base_score >= 2.0:
                strength = ConfluenceStrength.STRONG
            elif base_score >= 1.5:
                strength = ConfluenceStrength.MODERATE
            else:
                strength = ConfluenceStrength.WEAK
            
            # Create ConfluenceZone object
            confluence_zone = ConfluenceZone(
                center=zone["center"],
                lower_bound=zone["lower_bound"],
                upper_bound=zone["upper_bound"],
                strength=strength,
                contributing_tools=zone["contributing_tools"],
                timeframes=zone["timeframes"],
                confluence_types=zone["confluence_types"],
                levels=zone["levels"],
                score=base_score
            )
            
            confluence_zones.append(confluence_zone)
        
        return confluence_zones
        
    async def _get_effective_tools(self, market_regime: MarketRegime) -> Dict[str, float]:
        """Get effective tools for current market regime"""
        if not self.effectiveness_repository:
            return {}
            
        return await self.effectiveness_repository.get_effective_tools(
            market_regime,
            min_score=self.parameters["effectiveness_threshold"]
        ) 