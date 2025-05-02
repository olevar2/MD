"""
Execution Analytics Module.

This module provides comprehensive analysis of order executions, measuring quality
and providing feedback for strategy improvement.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import statistics
from uuid import uuid4

from core_foundations.utils.logger import get_logger
from trading_gateway_service.interfaces.broker_adapter import ExecutionReport, OrderStatus

logger = get_logger("execution-analytics")

class TimeFrame(Enum):
    """Time frames for analytics."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class MarketCondition(Enum):
    """Market conditions for analytics."""
    NORMAL = "normal"
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"


class ExecutionAnalytics:
    """
    Provides comprehensive analysis of order executions to measure quality
    and provide feedback for strategy improvement.

    Key capabilities:
    - Measuring slippage under different market conditions
    - Analyzing execution latency and fill rates
    - Calculating execution quality metrics
    - Correlating execution quality with market conditions
    - Providing data for ML model feedback loops
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the execution analytics.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}

        # Execution records storage
        self.executions = {}  # order_id -> execution data
        self.signals = {}  # signal_id -> signal data
        self.market_snapshots = {}  # timestamp -> market data

        # Analytics data storage
        self.slippage_stats = {
            "total": [],
            "by_instrument": {},
            "by_condition": {},
            "by_size": {
                "small": [],  # < 10k units
                "medium": [],  # 10k-100k units
                "large": []   # > 100k units
            }
        }

        self.latency_stats = {
            "signal_to_order_ms": [],
            "order_to_execution_ms": [],
            "signal_to_execution_ms": []
        }

        self.fill_rate_stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "partial_fills": 0,
            "rejected_orders": 0,
            "cancelled_orders": 0,
            "by_instrument": {}
        }

        self.execution_quality_scores = []

        # Add storage for VWAP and IS metrics
        self.vwap_deviations = []
        self.implementation_shortfalls = []

    def record_execution(self, execution_id: str, execution_data: Dict[str, Any]) -> None:
        """
        Record an execution for later analysis.

        Args:
            execution_id: Unique identifier for the execution
            execution_data: Execution data including order_id, instrument, direction, size,
                filled_size, execution_price, etc.
        """
        try:
            self.executions[execution_id] = execution_data
            
            # Schedule analysis if order is final (filled, cancelled, rejected, etc.)
            order_status = execution_data.get("status")
            if order_status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED,
                               OrderStatus.CANCELED, OrderStatus.REJECTED]:
                asyncio.create_task(self._analyze_execution(execution_id))
                
            logger.info(f"Recorded execution {execution_id} for analysis")
            
        except Exception as e:
            logger.error(f"Error recording execution: {e}")

    def record_signal(self, signal_id: str, signal_data: Dict[str, Any]) -> None:
        """
        Record a trading signal for later correlation with executions.

        Args:
            signal_id: Unique identifier for the signal
            signal_data: Signal data including timestamp, instrument, direction,
                size, expected_price, etc.
        """
        try:
            self.signals[signal_id] = signal_data
            logger.info(f"Recorded signal {signal_id} for analysis")
            
        except Exception as e:
            logger.error(f"Error recording signal: {e}")

    def record_market_snapshot(self, timestamp: str, instrument: str, data: Dict[str, Any]) -> None:
        """
        Record a market data snapshot for later analysis.

        Args:
            timestamp: ISO timestamp for the snapshot
            instrument: Trading instrument
            data: Market data including bid, ask, etc.
        """
        try:
            # Initialize timestamp entry if needed
            if timestamp not in self.market_snapshots:
                self.market_snapshots[timestamp] = {}
                
            self.market_snapshots[timestamp][instrument] = data

            # Cleanup old snapshots to prevent memory issues
            self._cleanup_old_snapshots()

            logger.debug(f"Recorded market snapshot for {instrument}")

        except Exception as e:
            logger.error(f"Error recording market snapshot: {e}")

    async def _analyze_execution(self, execution_id: str) -> None:
        """
        Analyze execution quality for a specific order.

        Args:
            execution_id: ID of the execution to analyze
        """
        try:
            if execution_id not in self.executions:
                logger.warning(f"Execution record not found: {execution_id}")
                return

            execution = self.executions[execution_id]
            instrument = execution.get("instrument")

            if not instrument:
                logger.warning(f"Missing instrument in execution: {execution_id}")
                return

            # Calculate slippage if we have the necessary data
            if execution.get("execution_price") is not None:
                expected_price = self._get_expected_price(execution)
                if expected_price:
                    slippage = self._calculate_slippage(
                        instrument,
                        execution.get("direction"),
                        expected_price,
                        execution.get("execution_price")
                    )
                    execution["slippage_pips"] = slippage

                    # Update slippage statistics
                    self._update_slippage_stats(execution)

            # Calculate latency if we have the necessary timestamps
            order_time = execution.get("order_time")
            signal_time = execution.get("signal_time")
            execution_time = execution.get("execution_time")

            if order_time and execution_time:
                order_time_dt = datetime.fromisoformat(order_time)
                execution_time_dt = datetime.fromisoformat(execution_time)
                latency_ms = (execution_time_dt - order_time_dt).total_seconds() * 1000
                execution["latency_ms"] = latency_ms

                self.latency_stats["order_to_execution_ms"].append(latency_ms)

            if signal_time and execution_time:
                signal_time_dt = datetime.fromisoformat(signal_time)
                execution_time_dt = datetime.fromisoformat(execution_time)
                total_latency_ms = (execution_time_dt - signal_time_dt).total_seconds() * 1000
                execution["total_latency_ms"] = total_latency_ms

                self.latency_stats["signal_to_execution_ms"].append(total_latency_ms)

            # Estimate market condition
            market_condition = self._estimate_market_condition(instrument)
            execution["market_condition"] = market_condition.value if market_condition else None

            # Calculate execution quality score
            quality_score = self._calculate_execution_quality(execution)
            execution["execution_quality"] = quality_score

            if quality_score is not None:
                self.execution_quality_scores.append(quality_score)

            # Calculate VWAP deviation
            vwap_deviation = await self._calculate_vwap_deviation(execution)
            execution["vwap_deviation_bps"] = vwap_deviation
            if vwap_deviation is not None:
                self.vwap_deviations.append(vwap_deviation)

            # Calculate Implementation Shortfall
            imp_shortfall = await self._calculate_implementation_shortfall(execution)
            execution["implementation_shortfall_bps"] = imp_shortfall
            if imp_shortfall is not None:
                self.implementation_shortfalls.append(imp_shortfall)

            logger.info(f"Analyzed execution {execution_id}: slippage={execution.get('slippage_pips')} pips, quality={quality_score}, vwap_dev={vwap_deviation} bps, is={imp_shortfall} bps")

        except Exception as e:
            logger.error(f"Error analyzing execution: {e}")

    def _get_expected_price(self, execution: Dict[str, Any]) -> Optional[float]:
        """
        Get the expected price for an execution.

        Args:
            execution: Execution record

        Returns:
            Expected price or None if not available
        """
        # Try to get expected price from signal
        signal_id = execution.get("signal_id")
        if signal_id and signal_id in self.signals:
            signal = self.signals[signal_id].get("signal", {})

            # For market orders, use the expected price from signal if available
            if signal.get("type", "").upper() == "MARKET":
                return signal.get("expected_price")

            # For limit orders, use the limit price
            if signal.get("type", "").upper() == "LIMIT":
                return signal.get("limit_price")

            # For stop orders, use the stop price
            if signal.get("type", "").upper() == "STOP":
                return signal.get("stop_price")

        # Fallback to market snapshot at time of order
        if execution.get("order_time") and execution.get("direction"):
            instrument = execution.get("instrument")
            snapshot = self._get_nearest_market_snapshot(execution.get("order_time"), instrument)
            
            if snapshot:
                direction = execution.get("direction")
                # For buy orders, use ask price; for sell orders, use bid price
                if direction == "BUY":
                    return snapshot.get("ask")
                elif direction == "SELL":
                    return snapshot.get("bid")

        return None

    def _get_nearest_market_snapshot(self, timestamp_str: str, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Get the nearest market snapshot to a timestamp.

        Args:
            timestamp_str: ISO timestamp string
            instrument: Trading instrument

        Returns:
            Market data snapshot or None if not available
        """
        try:
            target_time = datetime.fromisoformat(timestamp_str)

            closest_timestamp = None
            min_time_diff = timedelta(minutes=5)  # Only consider snapshots within 5 minutes

            for ts_str, data in self.market_snapshots.items():
                if instrument in data:
                    ts = datetime.fromisoformat(ts_str)
                    diff = abs(ts - target_time)

                    if diff < min_time_diff:
                        min_time_diff = diff
                        closest_timestamp = ts_str

            if closest_timestamp:
                return self.market_snapshots[closest_timestamp][instrument]

            return None

        except Exception as e:
            logger.error(f"Error getting nearest market snapshot: {e}")
            return None

    def _calculate_slippage(self, instrument: str, direction: str,
                          expected_price: float, execution_price: float) -> float:
        """
        Calculate slippage in pips.

        Args:
            instrument: Trading instrument
            direction: Order direction ("BUY" or "SELL")
            expected_price: Expected price
            execution_price: Actual execution price

        Returns:
            Slippage in pips (positive means unfavorable)
        """
        # Determine pip size
        pip_size = 0.01 if instrument.endswith("JPY") else 0.0001

        # Calculate raw price difference
        price_diff = execution_price - expected_price

        # For sell orders, reverse the sign since higher execution price is better
        if direction == "SELL":
            price_diff = -price_diff

        # Convert to pips
        slippage_pips = price_diff / pip_size

        return slippage_pips
        
    async def _calculate_vwap_deviation(self, execution: Dict[str, Any]) -> Optional[float]:
        """
        Calculate deviation from Volume Weighted Average Price (VWAP) in basis points.
        Requires market trade data (price, volume) during execution.

        Args:
            execution: Execution record

        Returns:
            VWAP deviation in basis points (positive means worse than VWAP) or None
        """
        # Placeholder: Needs actual market trade data access
        # 1. Get execution start and end times
        # 2. Fetch market trades (price, volume) for the instrument during this period
        # 3. Calculate market VWAP = sum(price * volume) / sum(volume)
        # 4. Get average execution price for this order
        # 5. Calculate deviation: (avg_exec_price - market_vwap) / market_vwap * 10000
        # 6. Adjust sign based on direction (BUY: positive dev is bad, SELL: negative dev is bad)

        avg_exec_price = execution.get("execution_price")
        direction = execution.get("direction")

        if avg_exec_price is None or direction is None:
            return None

        # Get market VWAP from trade data during execution period
        try:
            market_vwap = await self._get_market_vwap(
                instrument=execution.get("instrument"),
                start_time_str=execution.get("order_time"), # Or a more precise start
                end_time_str=execution.get("execution_time")
            )
        except Exception as e:
            logger.error(f"Error getting market VWAP: {e}")
            market_vwap = None

        if market_vwap is None or market_vwap == 0:
            return None

        deviation = (avg_exec_price - market_vwap) / market_vwap * 10000 # In basis points

        # Adjust sign: positive deviation should always be "bad"
        if direction == "SELL":
            deviation = -deviation

        return deviation
        
    async def _calculate_implementation_shortfall(self, execution: Dict[str, Any]) -> Optional[float]:
        """
        Calculate Implementation Shortfall (IS) in basis points.
        Compares execution cost to a benchmark price (e.g., arrival price).

        Args:
            execution: Execution record

        Returns:
            Implementation Shortfall in basis points (positive means cost/shortfall) or None
        """
        avg_exec_price = execution.get("execution_price")
        direction = execution.get("direction")
        order_size = execution.get("size", 0)
        filled_size = execution.get("filled_size", 0)

        if avg_exec_price is None or direction is None or order_size == 0:
            return None

        # Get benchmark price from the time the decision/signal occurred
        try:
            benchmark_price = await self._get_benchmark_price(execution)
        except Exception as e:
            logger.error(f"Error getting benchmark price: {e}")
            benchmark_price = None

        if benchmark_price is None or benchmark_price == 0:
            return None

        # Simplified IS calculation (Execution Cost component only, ignoring opportunity cost/fees for now)
        # Difference per share/unit
        price_diff = avg_exec_price - benchmark_price

        # Adjust sign based on direction (Buy: higher exec price is cost, Sell: lower exec price is cost)
        if direction == "SELL":
            price_diff = -price_diff

        # Normalize by benchmark price and convert to basis points
        implementation_shortfall_bps = (price_diff / benchmark_price) * 10000

        # Consider partial fills? For now, assume full fill or ignore opportunity cost.
        # A more complete calculation would factor in (order_size - filled_size) * (current_price - benchmark_price)

        return implementation_shortfall_bps
        
    async def _get_market_vwap(self, instrument: str, start_time_str: Optional[str], end_time_str: Optional[str]) -> Optional[float]:
        """
        Get Volume Weighted Average Price (VWAP) for an instrument during a specific period.
        
        Args:
            instrument: The trading instrument (e.g., 'EUR/USD')
            start_time_str: Start time as ISO string
            end_time_str: End time as ISO string
            
        Returns:
            VWAP as a float or None if data is unavailable
        """
        if not instrument or not start_time_str or not end_time_str:
            logger.warning(f"Missing required parameters for VWAP calculation: instrument={instrument}, start_time={start_time_str}, end_time={end_time_str}")
            return None
        
        try:
            # Get access to MarketDataService
            market_data_service = self.config.get('market_data_service')
            
            if market_data_service and hasattr(market_data_service, 'get_historical_data'):
                # Fetch historical data with the finest timeframe available ('1m')
                historical_data = await market_data_service.get_historical_data(
                    instrument=instrument,
                    start_time=start_time_str,
                    end_time=end_time_str,
                    timeframe='1m'
                )
                
                if historical_data is not None and not historical_data.empty:
                    # Calculate VWAP using close prices and volume
                    # VWAP = sum(price * volume) / sum(volume)
                    if 'close' in historical_data.columns and 'volume' in historical_data.columns:
                        total_volume = historical_data['volume'].sum()
                        
                        if total_volume > 0:
                            weighted_price_sum = (historical_data['close'] * historical_data['volume']).sum()
                            vwap = weighted_price_sum / total_volume
                            logger.info(f"Calculated VWAP for {instrument} between {start_time_str} and {end_time_str}: {vwap}")
                            return vwap
                        else:
                            logger.warning(f"Zero total volume for {instrument} between {start_time_str} and {end_time_str}")
                    else:
                        logger.warning(f"Historical data missing required columns (close, volume) for VWAP calculation")
                else:
                    logger.warning(f"No historical data available for {instrument} between {start_time_str} and {end_time_str}")
            else:
                logger.warning("Market data service unavailable for fetching historical data")
                
            return None
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return None
            
    async def _get_benchmark_price(self, execution: Dict[str, Any]) -> Optional[float]:
        """
        Get benchmark price for calculating implementation shortfall.
        Uses either arrival price from signal/order time or decision price if available.
        
        Args:
            execution: Execution record with details about the order

        Returns:
            Benchmark price or None if unavailable
        """
        # First check if a decision price is explicitly provided in the signal
        signal_id = execution.get("signal_id")
        if signal_id and signal_id in self.signals:
            signal_data = self.signals[signal_id]
            signal = signal_data.get("signal", {})
            
            # Check if the signal includes a reference/decision price
            if "reference_price" in signal:
                logger.info(f"Using explicit reference price from signal for execution {execution.get('order_id')}")
                return signal.get("reference_price")
            
            # For limit orders, the limit price might be an appropriate benchmark
            if signal.get("type", "").upper() == "LIMIT" and "limit_price" in signal:
                logger.info(f"Using limit price as benchmark for execution {execution.get('order_id')}")
                return signal.get("limit_price")
        
        # If no explicit price in signal, try to get market price at signal/order time
        arrival_time = execution.get("signal_time") or execution.get("order_time")
        instrument = execution.get("instrument")
        direction = execution.get("direction")
        
        if arrival_time and instrument:
            # First try to get a snapshot from our local cache
            snapshot = self._get_nearest_market_snapshot(arrival_time, instrument)
            if snapshot and "bid" in snapshot and "ask" in snapshot:
                # For buy orders, the ask price is the benchmark (cost to buy)
                # For sell orders, the bid price is the benchmark (revenue from selling)
                if direction == "BUY":
                    benchmark = snapshot["ask"]
                elif direction == "SELL":
                    benchmark = snapshot["bid"]
                else:
                    # If direction is unknown, use mid price
                    benchmark = (snapshot["bid"] + snapshot["ask"]) / 2
                    
                logger.info(f"Using market snapshot price as benchmark for execution {execution.get('order_id')}")
                return benchmark
            
            # If no snapshot in cache, try market data service if available
            market_data_service = self.config.get('market_data_service')
            if market_data_service:
                try:
                    # Try to get historical data from the service
                    if hasattr(market_data_service, 'get_historical_data'):
                        # Convert arrival time to datetime if it's a string
                        if isinstance(arrival_time, str):
                            arrival_time_dt = datetime.fromisoformat(arrival_time.replace('Z', '+00:00'))
                        else:
                            arrival_time_dt = arrival_time
                            
                        # Get a small window of data around the arrival time
                        window_start = arrival_time_dt - timedelta(minutes=1)
                        window_end = arrival_time_dt + timedelta(minutes=1)
                        
                        historical_data = await market_data_service.get_historical_data(
                            instrument=instrument,
                            start_time=window_start,
                            end_time=window_end,
                            timeframe='1m'
                        )
                        
                        if historical_data is not None and not historical_data.empty:
                            # Get the closest data point to arrival time
                            if 'open' in historical_data.columns:
                                benchmark = historical_data['open'].iloc[0]  # Use the first candle's open price
                                logger.info(f"Using historical data price as benchmark for execution {execution.get('order_id')}")
                                return benchmark
                except Exception as e:
                    logger.error(f"Error getting historical benchmark price: {e}")

        # If we get here, we couldn't determine a benchmark price
        logger.warning(f"Could not determine benchmark price for execution {execution.get('order_id')}. Implementation shortfall calculation will be unavailable.")
        return None

    def _update_slippage_stats(self, execution: Dict[str, Any]) -> None:
        """
        Update slippage statistics with a new execution.

        Args:
            execution: Execution record
        """
        slippage = execution.get("slippage_pips")
        if slippage is None:
            return

        instrument = execution.get("instrument")
        market_condition = execution.get("market_condition")
        size = execution.get("size", 0)

        # Update total slippage stats
        self.slippage_stats["total"].append(slippage)

        # Update instrument-specific stats
        if instrument:
            if instrument not in self.slippage_stats["by_instrument"]:
                self.slippage_stats["by_instrument"][instrument] = []
            self.slippage_stats["by_instrument"][instrument].append(slippage)

        # Update market condition stats
        if market_condition:
            if market_condition not in self.slippage_stats["by_condition"]:
                self.slippage_stats["by_condition"][market_condition] = []
            self.slippage_stats["by_condition"][market_condition].append(slippage)

        # Update size-based stats
        if size < 10000:
            self.slippage_stats["by_size"]["small"].append(slippage)
        elif size < 100000:
            self.slippage_stats["by_size"]["medium"].append(slippage)
        else:
            self.slippage_stats["by_size"]["large"].append(slippage)

    def _cleanup_old_snapshots(self) -> None:
        """Clean up old market snapshots to prevent memory issues."""
        # Keep only the last 1000 snapshots
        max_snapshots = 1000
        if len(self.market_snapshots) > max_snapshots:
            # Sort by timestamp
            sorted_timestamps = sorted(self.market_snapshots.keys())
            # Remove oldest snapshots
            for ts in sorted_timestamps[:-max_snapshots]:
                self.market_snapshots.pop(ts)

    def _estimate_market_condition(self, instrument: str) -> MarketCondition:
        """
        Estimate the current market condition based on recent volatility, 
        price action, and other factors.

        Args:
            instrument: Trading instrument

        Returns:
            Market condition enum
        """
        # Placeholder: This would need to analyze recent price action,
        # volatility, spread, and order book depth to determine
        # For now, assume normal market
        return MarketCondition.NORMAL

    def _calculate_execution_quality(self, execution: Dict[str, Any]) -> Optional[float]:
        """
        Calculate overall execution quality score.
        Combines multiple factors: slippage, latency, market impact, etc.

        Args:
            execution: Execution record

        Returns:
            Quality score (0-100, higher is better) or None
        """
        # Placeholder for a comprehensive quality score calculation
        # This should combine:
        # - Slippage (weight 0.5)
        # - Latency (weight 0.2)
        # - Market impact (weight 0.3)
        
        factors = {}
        
        # Convert slippage to a score (lower slippage = higher score)
        slippage = execution.get("slippage_pips")
        if slippage is not None:
            # Convert slippage to a 0-100 score
            # 0 slippage = 100, 10 pips slippage = 0
            slippage_score = max(0, 100 - abs(slippage) * 10)
            factors["slippage"] = slippage_score
            
        # Convert latency to a score
        latency = execution.get("latency_ms")
        if latency is not None:
            # Convert latency to a 0-100 score
            # 0ms = 100, 1000ms+ = 0
            latency_score = max(0, 100 - (latency / 10))
            factors["latency"] = latency_score
            
        # If we have any factors, calculate a weighted score
        if factors:
            weights = {
                "slippage": 0.5,
                "latency": 0.2,
                # Market impact would be the other 0.3
            }
            
            # Calculate weighted sum of available factors
            score = 0
            total_weight = 0
            
            for factor, value in factors.items():
                weight = weights.get(factor, 0)
                score += value * weight
                total_weight += weight
                
            # Normalize by total weight used
            if total_weight > 0:
                return score / total_weight
                
        return None

    def get_average_slippage(self, instrument: Optional[str] = None, 
                           market_condition: Optional[str] = None,
                           size_category: Optional[str] = None) -> Optional[float]:
        """
        Get average slippage for various conditions.

        Args:
            instrument: Filter by instrument
            market_condition: Filter by market condition
            size_category: Filter by size category

        Returns:
            Average slippage in pips
        """
        if instrument:
            if instrument not in self.slippage_stats["by_instrument"]:
                return None
            values = self.slippage_stats["by_instrument"][instrument]
        elif market_condition:
            if market_condition not in self.slippage_stats["by_condition"]:
                return None
            values = self.slippage_stats["by_condition"][market_condition]
        elif size_category:
            if size_category not in self.slippage_stats["by_size"]:
                return None
            values = self.slippage_stats["by_size"][size_category]
        else:
            values = self.slippage_stats["total"]

        if not values:
            return None

        return sum(values) / len(values)

    def get_fill_rate(self, instrument: Optional[str] = None) -> float:
        """
        Get the fill rate for all orders or a specific instrument.

        Args:
            instrument: Filter by instrument

        Returns:
            Fill rate as a percentage
        """
        if instrument:
            if instrument not in self.fill_rate_stats["by_instrument"]:
                return 0.0
                
            stats = self.fill_rate_stats["by_instrument"][instrument]
            if stats.get("total_orders", 0) == 0:
                return 0.0
                
            return stats.get("filled_orders", 0) / stats.get("total_orders", 1) * 100.0
        else:
            if self.fill_rate_stats["total_orders"] == 0:
                return 0.0
                
            return self.fill_rate_stats["filled_orders"] / max(1, self.fill_rate_stats["total_orders"]) * 100.0

    def get_average_latency(self) -> Dict[str, float]:
        """
        Get average latencies for different stages of order execution.

        Returns:
            Dictionary of average latencies in milliseconds
        """
        result = {}
        
        for key, values in self.latency_stats.items():
            if values:
                result[key] = sum(values) / len(values)
            else:
                result[key] = 0.0
                
        return result

    def get_execution_quality_stats(self) -> Dict[str, Any]:
        """
        Get statistics about execution quality.

        Returns:
            Dictionary with execution quality statistics
        """
        result = {
            "average_quality": None,
            "median_quality": None,
            "min_quality": None,
            "max_quality": None,
            "stddev_quality": None,
            "count": 0
        }
        
        if self.execution_quality_scores:
            result["average_quality"] = sum(self.execution_quality_scores) / len(self.execution_quality_scores)
            result["median_quality"] = statistics.median(self.execution_quality_scores)
            result["min_quality"] = min(self.execution_quality_scores)
            result["max_quality"] = max(self.execution_quality_scores)
            
            if len(self.execution_quality_scores) > 1:
                result["stddev_quality"] = statistics.stdev(self.execution_quality_scores)
                
            result["count"] = len(self.execution_quality_scores)
            
        return result

    def get_vwap_deviation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about VWAP deviation.

        Returns:
            Dictionary with VWAP deviation statistics
        """
        result = {
            "average_deviation_bps": None,
            "median_deviation_bps": None,
            "min_deviation_bps": None,
            "max_deviation_bps": None,
            "stddev_deviation_bps": None,
            "count": 0
        }
        
        if self.vwap_deviations:
            result["average_deviation_bps"] = sum(self.vwap_deviations) / len(self.vwap_deviations)
            result["median_deviation_bps"] = statistics.median(self.vwap_deviations)
            result["min_deviation_bps"] = min(self.vwap_deviations)
            result["max_deviation_bps"] = max(self.vwap_deviations)
            
            if len(self.vwap_deviations) > 1:
                result["stddev_deviation_bps"] = statistics.stdev(self.vwap_deviations)
                
            result["count"] = len(self.vwap_deviations)
            
        return result

    def get_implementation_shortfall_stats(self) -> Dict[str, Any]:
        """
        Get statistics about implementation shortfall.

        Returns:
            Dictionary with implementation shortfall statistics
        """
        result = {
            "average_shortfall_bps": None,
            "median_shortfall_bps": None,
            "min_shortfall_bps": None,
            "max_shortfall_bps": None,
            "stddev_shortfall_bps": None,
            "count": 0
        }
        
        if self.implementation_shortfalls:
            result["average_shortfall_bps"] = sum(self.implementation_shortfalls) / len(self.implementation_shortfalls)
            result["median_shortfall_bps"] = statistics.median(self.implementation_shortfalls)
            result["min_shortfall_bps"] = min(self.implementation_shortfalls)
            result["max_shortfall_bps"] = max(self.implementation_shortfalls)
            
            if len(self.implementation_shortfalls) > 1:
                result["stddev_shortfall_bps"] = statistics.stdev(self.implementation_shortfalls)
                
            result["count"] = len(self.implementation_shortfalls)
            
        return result
