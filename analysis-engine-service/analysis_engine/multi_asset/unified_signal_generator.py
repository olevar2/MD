"""
Unified Signal Generator Module

This module provides a consistent approach to signal generation across
different asset classes while respecting asset-specific characteristics.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from analysis_engine.multi_asset.asset_registry import AssetClass, AssetRegistry
from analysis_engine.multi_asset.indicator_adapter import IndicatorAdapter
from analysis_engine.services.multi_asset_service import MultiAssetService
from analysis_engine.models.signal import SignalStrength, SignalDirection, Signal

logger = logging.getLogger(__name__)


class UnifiedSignalGenerator:
    """
    Generates consistent trading signals across different asset classes
    
    This class ensures that signal generation logic is consistent while still
    respecting the unique characteristics of each asset class.
    """
    
    def __init__(
        self, 
        multi_asset_service: Optional[MultiAssetService] = None,
        indicator_adapter: Optional[IndicatorAdapter] = None
    ):
        """Initialize the unified signal generator"""
        self.multi_asset_service = multi_asset_service or MultiAssetService()
        self.indicator_adapter = indicator_adapter or IndicatorAdapter(multi_asset_service)
        self.logger = logging.getLogger(__name__)
        
    async def generate_signals(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        timeframe: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate trading signals for any asset class
        
        Args:
            data: Price data DataFrame
            symbol: Symbol for the asset
            timeframe: Timeframe of the data
            config: Configuration parameters
            
        Returns:
            Dictionary with generated signals
        """
        # Get asset information
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            self.logger.warning(f"Asset info not found for {symbol}, using default parameters")
            return {"error": f"Asset not found: {symbol}"}
        
        asset_class = asset_info.get("asset_class")
        
        # Get default or provided configuration
        config = config or self._get_default_config(asset_class)
        
        # Apply indicators with asset-specific parameters
        enhanced_data = self._apply_indicators(data, symbol, config)
        
        # Generate signals using asset-specific logic
        signals = self._generate_signals_for_asset(enhanced_data, symbol, asset_class, config)
        
        # Calculate signal consensus
        consensus = self._calculate_signal_consensus(signals, asset_class)
        
        # Format result
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "asset_class": asset_class,
            "signals": signals,
            "consensus": consensus,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _apply_indicators(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply technical indicators to the price data"""
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Apply moving averages
        ma_config = config.get("moving_averages", {})
        if ma_config.get("enabled", True):
            for period in ma_config.get("periods", [20, 50, 200]):
                for ma_type in ma_config.get("types", ["sma"]):
                    df = self.indicator_adapter.moving_average(
                        df, symbol, period=period, ma_type=ma_type
                    )
        
        # Apply RSI
        rsi_config = config.get("rsi", {})
        if rsi_config.get("enabled", True):
            df = self.indicator_adapter.relative_strength_index(
                df, symbol, period=rsi_config.get("period", 14)
            )
        
        # Apply Bollinger Bands
        bb_config = config.get("bollinger_bands", {})
        if bb_config.get("enabled", True):
            df = self.indicator_adapter.bollinger_bands(
                df, symbol, 
                period=bb_config.get("period", 20),
                std_dev=bb_config.get("std_dev", 2.0)
            )
        
        # Apply ATR
        atr_config = config.get("atr", {})
        if atr_config.get("enabled", True):
            df = self.indicator_adapter.average_true_range(
                df, symbol, period=atr_config.get("period", 14)
            )
        
        # Apply MACD
        macd_config = config.get("macd", {})
        if macd_config.get("enabled", True):
            df = self.indicator_adapter.macd(
                df, symbol,
                fast_period=macd_config.get("fast_period", 12),
                slow_period=macd_config.get("slow_period", 26),
                signal_period=macd_config.get("signal_period", 9)
            )
            
        return df
    
    def _generate_signals_for_asset(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        asset_class: AssetClass,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate signals based on asset class"""
        # Common signal generation
        signals = {}
        
        # Trend signals based on moving averages
        signals["trend"] = self._generate_trend_signals(data, asset_class)
        
        # Momentum signals based on RSI
        signals["momentum"] = self._generate_momentum_signals(data, asset_class)
        
        # Volatility signals based on Bollinger Bands and ATR
        signals["volatility"] = self._generate_volatility_signals(data, asset_class)
        
        # MACD signals
        signals["macd"] = self._generate_macd_signals(data, asset_class)
        
        # Add asset-specific signals
        if asset_class == AssetClass.FOREX:
            signals["forex_specific"] = self._generate_forex_specific_signals(data, symbol, config)
        elif asset_class == AssetClass.CRYPTO:
            signals["crypto_specific"] = self._generate_crypto_specific_signals(data, symbol, config)
        elif asset_class == AssetClass.STOCKS:
            signals["stock_specific"] = self._generate_stock_specific_signals(data, symbol, config)
            
        return signals
    
    def _calculate_signal_consensus(
        self, 
        signals: Dict[str, Any], 
        asset_class: AssetClass
    ) -> Dict[str, Any]:
        """Calculate overall signal consensus with asset-specific weighting"""
        # Extract all directional signals
        directions = []
        weights = []
        
        # Process trend signals
        if "trend" in signals:
            trend_signal = signals["trend"].get("direction")
            if trend_signal:
                directions.append(trend_signal)
                weights.append(self._get_signal_weight("trend", asset_class))
        
        # Process momentum signals
        if "momentum" in signals:
            momentum_signal = signals["momentum"].get("direction")
            if momentum_signal:
                directions.append(momentum_signal)
                weights.append(self._get_signal_weight("momentum", asset_class))
        
        # Process volatility signals
        if "volatility" in signals:
            volatility_signal = signals["volatility"].get("direction")
            if volatility_signal:
                directions.append(volatility_signal)
                weights.append(self._get_signal_weight("volatility", asset_class))
                
        # Process MACD signals
        if "macd" in signals:
            macd_signal = signals["macd"].get("direction")
            if macd_signal:
                directions.append(macd_signal)
                weights.append(self._get_signal_weight("macd", asset_class))
        
        # Process asset-specific signals
        if asset_class == AssetClass.FOREX and "forex_specific" in signals:
            forex_signal = signals["forex_specific"].get("direction")
            if forex_signal:
                directions.append(forex_signal)
                weights.append(self._get_signal_weight("forex_specific", asset_class))
                
        elif asset_class == AssetClass.CRYPTO and "crypto_specific" in signals:
            crypto_signal = signals["crypto_specific"].get("direction")
            if crypto_signal:
                directions.append(crypto_signal)
                weights.append(self._get_signal_weight("crypto_specific", asset_class))
                
        elif asset_class == AssetClass.STOCKS and "stock_specific" in signals:
            stock_signal = signals["stock_specific"].get("direction")
            if stock_signal:
                directions.append(stock_signal)
                weights.append(self._get_signal_weight("stock_specific", asset_class))
        
        # Calculate weighted consensus
        if not directions or not weights:
            return {
                "direction": None,
                "strength": SignalStrength.NEUTRAL,
                "confidence": 0.0
            }
        
        # Convert directions to numeric values
        numeric_directions = []
        for direction in directions:
            if direction == SignalDirection.LONG:
                numeric_directions.append(1.0)
            elif direction == SignalDirection.SHORT:
                numeric_directions.append(-1.0)
            else:
                numeric_directions.append(0.0)
        
        # Calculate weighted average
        weighted_sum = np.sum(np.array(numeric_directions) * np.array(weights))
        weight_sum = np.sum(weights)
        
        if weight_sum > 0:
            consensus_value = weighted_sum / weight_sum
        else:
            consensus_value = 0.0
        
        # Convert to signal direction and strength
        direction, strength = self._convert_consensus_to_signal(consensus_value)
        
        # Calculate confidence based on agreement among signals
        agreement = 1.0 - (np.std(numeric_directions) if len(numeric_directions) > 1 else 0.0)
        confidence = min(1.0, max(0.0, (0.5 + agreement / 2)))
        
        return {
            "direction": direction,
            "strength": strength,
            "consensus_value": consensus_value,
            "confidence": confidence
        }
    
    def _convert_consensus_to_signal(self, consensus_value: float) -> Tuple[str, str]:
        """Convert consensus value to signal direction and strength"""
        if consensus_value > 0.7:
            return SignalDirection.LONG, SignalStrength.STRONG
        elif consensus_value > 0.3:
            return SignalDirection.LONG, SignalStrength.MODERATE
        elif consensus_value > 0.1:
            return SignalDirection.LONG, SignalStrength.WEAK
        elif consensus_value < -0.7:
            return SignalDirection.SHORT, SignalStrength.STRONG
        elif consensus_value < -0.3:
            return SignalDirection.SHORT, SignalStrength.MODERATE
        elif consensus_value < -0.1:
            return SignalDirection.SHORT, SignalStrength.WEAK
        else:
            return SignalDirection.NEUTRAL, SignalStrength.NEUTRAL
    
    def _get_signal_weight(self, signal_type: str, asset_class: AssetClass) -> float:
        """Get weight for a signal type based on asset class"""
        # Default weights
        default_weights = {
            "trend": 1.0,
            "momentum": 0.8,
            "volatility": 0.6,
            "macd": 0.8,
            "forex_specific": 1.0,
            "crypto_specific": 1.0,
            "stock_specific": 1.0
        }
        
        # Asset-specific weight adjustments
        asset_adjustments = {
            AssetClass.FOREX: {
                "trend": 1.2,
                "momentum": 0.9,
                "volatility": 0.7
            },
            AssetClass.CRYPTO: {
                "trend": 0.8,
                "momentum": 1.2,
                "volatility": 1.0
            },
            AssetClass.STOCKS: {
                "trend": 1.0,
                "momentum": 1.0,
                "volatility": 0.8
            }
        }
        
        # Get base weight
        weight = default_weights.get(signal_type, 0.5)
        
        # Apply asset-specific adjustment if available
        if asset_class in asset_adjustments and signal_type in asset_adjustments[asset_class]:
            weight *= asset_adjustments[asset_class][signal_type]
            
        return weight
    
    def _get_default_config(self, asset_class: AssetClass) -> Dict[str, Any]:
        """Get default configuration based on asset class"""
        # Common config
        config = {
            "moving_averages": {
                "enabled": True,
                "periods": [20, 50, 200],
                "types": ["sma", "ema"]
            },
            "rsi": {
                "enabled": True,
                "period": 14
            },
            "bollinger_bands": {
                "enabled": True,
                "period": 20,
                "std_dev": 2.0
            },
            "atr": {
                "enabled": True,
                "period": 14
            },
            "macd": {
                "enabled": True,
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            }
        }
        
        # Asset-specific adjustments
        if asset_class == AssetClass.FOREX:
            # Forex standard parameters work well
            pass
        elif asset_class == AssetClass.CRYPTO:
            # Adjust for crypto's higher volatility
            config["rsi"]["period"] = 12  # More responsive RSI
            config["bollinger_bands"]["std_dev"] = 2.5  # Wider bands
            config["atr"]["period"] = 12  # More responsive ATR
        elif asset_class == AssetClass.STOCKS:
            # Adjustments for stocks
            config["moving_averages"]["periods"] = [10, 30, 200]  # Different MA periods
            
        return config
    
    def _generate_trend_signals(self, data: pd.DataFrame, asset_class: AssetClass) -> Dict[str, Any]:
        """Generate trend signals from moving averages"""
        # Look for available MAs in the data
        ma_columns = [col for col in data.columns if col.startswith(('sma_', 'ema_'))]
        if not ma_columns:
            return {"direction": None}
            
        # Get the latest row
        latest = data.iloc[-1]
        
        # Check for short-term vs long-term MA crossovers
        short_term_mas = []
        long_term_mas = []
        
        for col in ma_columns:
            parts = col.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                period = int(parts[1])
                if period <= 50:
                    short_term_mas.append((col, period))
                else:
                    long_term_mas.append((col, period))
        
        # Sort by period
        short_term_mas.sort(key=lambda x: x[1])
        long_term_mas.sort(key=lambda x: x[1])
        
        # Check if current price is above/below moving averages
        price = latest['close']
        above_short = sum(1 for col, _ in short_term_mas if price > latest[col])
        below_short = len(short_term_mas) - above_short
        
        above_long = sum(1 for col, _ in long_term_mas if price > latest[col])
        below_long = len(long_term_mas) - above_long
        
        # Check MA alignment (short-term MAs above/below long-term MAs)
        alignment_bullish = 0
        alignment_bearish = 0
        alignment_count = 0
        
        if short_term_mas and long_term_mas:
            for short_col, _ in short_term_mas:
                for long_col, _ in long_term_mas:
                    alignment_count += 1
                    if latest[short_col] > latest[long_col]:
                        alignment_bullish += 1
                    else:
                        alignment_bearish += 1
        
        # Determine trend direction and strength
        direction = SignalDirection.NEUTRAL
        strength = SignalStrength.NEUTRAL
        
        # Price above/below MAs
        if short_term_mas:
            price_vs_ma_score = (above_short - below_short) / len(short_term_mas)
        else:
            price_vs_ma_score = 0
            
        # MA alignment
        if alignment_count > 0:
            alignment_score = (alignment_bullish - alignment_bearish) / alignment_count
        else:
            alignment_score = 0
            
        # Combined score (-1 to +1)
        trend_score = 0.6 * price_vs_ma_score + 0.4 * alignment_score
        
        # Determine direction and strength
        if trend_score > 0.7:
            direction = SignalDirection.LONG
            strength = SignalStrength.STRONG
        elif trend_score > 0.3:
            direction = SignalDirection.LONG
            strength = SignalStrength.MODERATE
        elif trend_score > 0.1:
            direction = SignalDirection.LONG
            strength = SignalStrength.WEAK
        elif trend_score < -0.7:
            direction = SignalDirection.SHORT
            strength = SignalStrength.STRONG
        elif trend_score < -0.3:
            direction = SignalDirection.SHORT
            strength = SignalStrength.MODERATE
        elif trend_score < -0.1:
            direction = SignalDirection.SHORT
            strength = SignalStrength.WEAK
            
        return {
            "direction": direction,
            "strength": strength,
            "score": trend_score
        }
    
    def _generate_momentum_signals(self, data: pd.DataFrame, asset_class: AssetClass) -> Dict[str, Any]:
        """Generate momentum signals from RSI"""
        # Check for RSI
        rsi_columns = [col for col in data.columns if col.startswith('rsi_')]
        if not rsi_columns:
            return {"direction": None}
            
        # Get the latest row
        latest = data.iloc[-1]
        
        # Use the shortest period RSI available
        rsi_column = min(rsi_columns, key=lambda col: int(col.split('_')[1]) 
                         if col.split('_')[1].isdigit() else 999)
        
        rsi_value = latest[rsi_column]
        
        # Adjust thresholds based on asset class
        if asset_class == AssetClass.FOREX:
            oversold = 30
            overbought = 70
        elif asset_class == AssetClass.CRYPTO:
            # Crypto often has wider RSI swings
            oversold = 25
            overbought = 75
        else:
            oversold = 30
            overbought = 70
            
        # Determine signal
        direction = SignalDirection.NEUTRAL
        strength = SignalStrength.NEUTRAL
        
        if rsi_value <= oversold:
            direction = SignalDirection.LONG
            if rsi_value <= oversold - 10:
                strength = SignalStrength.STRONG
            elif rsi_value <= oversold - 5:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
                
        elif rsi_value >= overbought:
            direction = SignalDirection.SHORT
            if rsi_value >= overbought + 10:
                strength = SignalStrength.STRONG
            elif rsi_value >= overbought + 5:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
                
        return {
            "direction": direction,
            "strength": strength,
            "rsi_value": rsi_value
        }
    
    def _generate_volatility_signals(self, data: pd.DataFrame, asset_class: AssetClass) -> Dict[str, Any]:
        """Generate signals based on volatility indicators"""
        # Check for Bollinger Bands
        bb_columns = [col for col in data.columns if col.startswith('bb_')]
        
        # We need all three bands
        bb_upper = next((col for col in bb_columns if 'upper' in col), None)
        bb_middle = next((col for col in bb_columns if 'middle' in col), None)
        bb_lower = next((col for col in bb_columns if 'lower' in col), None)
        
        if not (bb_upper and bb_middle and bb_lower):
            return {"direction": None}
            
        # Get the latest row
        latest = data.iloc[-1]
        price = latest['close']
        
        # Check position relative to bands
        upper = latest[bb_upper]
        middle = latest[bb_middle]
        lower = latest[bb_lower]
        
        # Band width as volatility measure
        band_width = (upper - lower) / middle
        
        # Check for price breakout of bands
        direction = SignalDirection.NEUTRAL
        strength = SignalStrength.NEUTRAL
        
        if price >= upper:
            # Price above upper band
            if asset_class == AssetClass.CRYPTO:
                # In crypto, breakouts often continue
                direction = SignalDirection.LONG
                strength = SignalStrength.MODERATE
            else:
                # In other assets, this can indicate overbought
                direction = SignalDirection.SHORT
                strength = SignalStrength.WEAK
                
        elif price <= lower:
            # Price below lower band
            if asset_class == AssetClass.CRYPTO:
                # In crypto, breakdowns often continue
                direction = SignalDirection.SHORT
                strength = SignalStrength.MODERATE
            else:
                # In other assets, this can indicate oversold
                direction = SignalDirection.LONG
                strength = SignalStrength.WEAK
                
        # Check for band squeeze (low volatility)
        avg_band_width = data[bb_upper].subtract(data[bb_lower]).divide(data[bb_middle]).rolling(20).mean().iloc[-1]
        
        is_squeeze = band_width < avg_band_width * 0.85
        
        return {
            "direction": direction,
            "strength": strength,
            "band_width": band_width,
            "is_squeeze": is_squeeze
        }
    
    def _generate_macd_signals(self, data: pd.DataFrame, asset_class: AssetClass) -> Dict[str, Any]:
        """Generate signals from MACD indicator"""
        # Check for MACD columns
        if not all(col in data.columns for col in ['macd_line', 'macd_signal', 'macd_histogram']):
            return {"direction": None}
            
        # Get the current and previous rows
        if len(data) < 2:
            return {"direction": None}
            
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # MACD line crossing signal line
        macd_cross_up = previous['macd_line'] < previous['macd_signal'] and latest['macd_line'] > latest['macd_signal']
        macd_cross_down = previous['macd_line'] > previous['macd_signal'] and latest['macd_line'] < latest['macd_signal']
        
        # MACD histogram direction change
        hist_direction_change = (
            previous['macd_histogram'] < 0 and latest['macd_histogram'] > 0 or
            previous['macd_histogram'] > 0 and latest['macd_histogram'] < 0
        )
        
        # Determine signal
        direction = SignalDirection.NEUTRAL
        strength = SignalStrength.NEUTRAL
        
        if macd_cross_up:
            direction = SignalDirection.LONG
            strength = SignalStrength.MODERATE if latest['macd_line'] > 0 else SignalStrength.WEAK
            
        elif macd_cross_down:
            direction = SignalDirection.SHORT
            strength = SignalStrength.MODERATE if latest['macd_line'] < 0 else SignalStrength.WEAK
            
        elif latest['macd_histogram'] > previous['macd_histogram'] and latest['macd_histogram'] > 0:
            # Increasing positive histogram = strengthening bullish momentum
            direction = SignalDirection.LONG
            strength = SignalStrength.WEAK
            
        elif latest['macd_histogram'] < previous['macd_histogram'] and latest['macd_histogram'] < 0:
            # Decreasing negative histogram = strengthening bearish momentum
            direction = SignalDirection.SHORT
            strength = SignalStrength.WEAK
            
        return {
            "direction": direction,
            "strength": strength,
            "macd_line": latest['macd_line'],
            "macd_signal": latest['macd_signal'],
            "macd_histogram": latest['macd_histogram'],
            "crossed_up": macd_cross_up,
            "crossed_down": macd_cross_down
        }
    
    def _generate_forex_specific_signals(self, data: pd.DataFrame, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forex-specific signals"""
        # For forex, we might check specific things like:
        # - Pip movement relative to average
        # - Support/resistance levels in pips
        # - Session-based volatility
        
        # Just a basic implementation for now
        return {
            "direction": None,
            "strength": SignalStrength.NEUTRAL
        }
    
    def _generate_crypto_specific_signals(self, data: pd.DataFrame, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate crypto-specific signals"""
        # For crypto, check:
        # - Volume trends
        # - Price action around whole numbers
        # - Market dominance (for BTC)
        
        # Just a basic implementation for now
        return {
            "direction": None,
            "strength": SignalStrength.NEUTRAL
        }
    
    def _generate_stock_specific_signals(self, data: pd.DataFrame, symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate stock-specific signals"""
        # For stocks, check:
        # - Volume trends
        # - Gap analysis
        # - Previous session close/open
        
        # Just a basic implementation for now
        return {
            "direction": None,
            "strength": SignalStrength.NEUTRAL
        }
