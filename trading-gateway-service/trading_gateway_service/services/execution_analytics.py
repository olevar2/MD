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
logger = get_logger('execution-analytics')
from trading_gateway_service.error.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class TimeFrame(Enum):
    """Time frames for analytics."""
    MINUTE_1 = '1m'
    MINUTE_5 = '5m'
    MINUTE_15 = '15m'
    MINUTE_30 = '30m'
    HOUR_1 = '1h'
    HOUR_4 = '4h'
    DAY_1 = '1d'
    WEEK_1 = '1w'


class MarketCondition(Enum):
    """Market conditions for analytics."""
    NORMAL = 'normal'
    TRENDING = 'trending'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    ILLIQUID = 'illiquid'


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

    def __init__(self, config: Optional[Dict[str, Any]]=None):
        """
        Initialize the execution analytics.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.executions = {}
        self.signals = {}
        self.market_snapshots = {}
        self.slippage_stats = {'total': [], 'by_instrument': {},
            'by_condition': {}, 'by_size': {'small': [], 'medium': [],
            'large': []}}
        self.latency_stats = {'signal_to_order_ms': [],
            'order_to_execution_ms': [], 'signal_to_execution_ms': []}
        self.fill_rate_stats = {'total_orders': 0, 'filled_orders': 0,
            'partial_fills': 0, 'rejected_orders': 0, 'cancelled_orders': 0,
            'by_instrument': {}}
        self.execution_quality_scores = []
        self.vwap_deviations = []
        self.implementation_shortfalls = []
        self.data_retention_days = self.config_manager.get('data_retention_days', 30)
        self.cleanup_interval_hours = self.config.get('cleanup_interval_hours',
            24)
        self.running = False
        self.tasks = {}
        logger.info('ExecutionAnalytics initialized')

    async def start(self) ->None:
        """Start the execution analytics service."""
        if self.running:
            logger.warning('Execution analytics service already running')
            return
        logger.info('Starting execution analytics service')
        self.running = True
        self.tasks['cleanup'] = asyncio.create_task(self._periodic_cleanup())

    @async_with_exception_handling
    async def stop(self) ->None:
        """Stop the execution analytics service."""
        if not self.running:
            logger.warning('Execution analytics service not running')
            return
        logger.info('Stopping execution analytics service')
        self.running = False
        for name, task in self.tasks.items():
            logger.info(f'Cancelling {name} task')
            task.cancel()
        for name, task in self.tasks.items():
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f'{name} task cancelled')
        self.tasks = {}
        logger.info('Execution analytics service stopped')

    @async_with_exception_handling
    async def record_signal_processing(self, signal: Dict[str, Any],
        order_request: Any) ->None:
        """
        Record signal processing for analysis.

        Args:
            signal: The trading signal
            order_request: The resulting order request
        """
        try:
            signal_id = signal.get('signal_id')
            if not signal_id:
                return
            self.signals[signal_id] = {'signal': signal, 'received_time': 
                signal.get('received_time') or datetime.now().isoformat(),
                'processing_time': datetime.now().isoformat(),
                'order_request': order_request.__dict__ if order_request else
                None, 'order_id': order_request.client_order_id if
                order_request else None}
            logger.debug(f'Recorded signal processing: {signal_id}')
            if order_request and order_request.client_order_id:
                order_id = order_request.client_order_id
                if order_id not in self.executions:
                    self.executions[order_id] = {'order_id': order_id,
                        'signal_id': signal_id, 'instrument': signal.get(
                        'instrument'), 'direction': signal.get('direction'),
                        'size': getattr(order_request, 'size', 0),
                        'signal_time': signal.get('received_time') or
                        datetime.now().isoformat(), 'order_time': datetime.
                        now().isoformat(), 'execution_time': None, 'status':
                        'PENDING', 'slippage_pips': None, 'latency_ms':
                        None, 'market_condition': None, 'execution_quality':
                        None, 'vwap_deviation_bps': None,
                        'implementation_shortfall_bps': None}
        except Exception as e:
            logger.error(f'Error recording signal processing: {e}')

    @with_broker_api_resilience('process_execution_report')
    @async_with_exception_handling
    async def process_execution_report(self, report: ExecutionReport) ->None:
        """
        Process an execution report for analytics.

        Args:
            report: Execution report from the broker
        """
        try:
            order_id = report.order_id
            client_order_id = report.client_order_id
            execution_id = client_order_id or order_id
            if execution_id not in self.executions:
                self.executions[execution_id] = {'order_id': order_id,
                    'client_order_id': client_order_id, 'instrument':
                    report.instrument, 'direction': report.direction,
                    'size': report.size, 'filled_size': report.filled_size,
                    'execution_price': report.execution_price, 'status':
                    report.status.value, 'execution_time': datetime.now().
                    isoformat(), 'order_time': None, 'signal_time': None,
                    'slippage_pips': None, 'latency_ms': None,
                    'market_condition': None, 'execution_quality': None,
                    'vwap_deviation_bps': None,
                    'implementation_shortfall_bps': None}
            else:
                self.executions[execution_id].update({'status': report.
                    status.value, 'execution_time': datetime.now().
                    isoformat(), 'filled_size': report.filled_size,
                    'execution_price': report.execution_price})
            if report.status in [OrderStatus.FILLED, OrderStatus.PARTIAL_FILL]:
                await self._analyze_execution(execution_id)
            self._update_fill_rate_stats(report)
            logger.debug(f'Processed execution report: {execution_id}')
        except Exception as e:
            logger.error(f'Error processing execution report: {e}')

    @async_with_exception_handling
    async def record_market_snapshot(self, instrument: str, data: Dict[str,
        Any]) ->None:
        """
        Record market data snapshot for analysis.

        Args:
            instrument: Trading instrument
            data: Market data snapshot
        """
        try:
            timestamp = datetime.now().isoformat()
            if timestamp not in self.market_snapshots:
                self.market_snapshots[timestamp] = {}
            self.market_snapshots[timestamp][instrument] = data
            self._cleanup_old_snapshots()
            logger.debug(f'Recorded market snapshot for {instrument}')
        except Exception as e:
            logger.error(f'Error recording market snapshot: {e}')

    @async_with_exception_handling
    async def _analyze_execution(self, execution_id: str) ->None:
        """
        Analyze execution quality for a specific order.

        Args:
            execution_id: ID of the execution to analyze
        """
        try:
            if execution_id not in self.executions:
                logger.warning(f'Execution record not found: {execution_id}')
                return
            execution = self.executions[execution_id]
            instrument = execution.get('instrument')
            if not instrument:
                logger.warning(
                    f'Missing instrument in execution: {execution_id}')
                return
            if execution.get('execution_price') is not None:
                expected_price = self._get_expected_price(execution)
                if expected_price:
                    slippage = self._calculate_slippage(instrument,
                        execution.get('direction'), expected_price,
                        execution.get('execution_price'))
                    execution['slippage_pips'] = slippage
                    self._update_slippage_stats(execution)
            order_time = execution.get('order_time')
            signal_time = execution.get('signal_time')
            execution_time = execution.get('execution_time')
            if order_time and execution_time:
                order_time_dt = datetime.fromisoformat(order_time)
                execution_time_dt = datetime.fromisoformat(execution_time)
                latency_ms = (execution_time_dt - order_time_dt).total_seconds(
                    ) * 1000
                execution['latency_ms'] = latency_ms
                self.latency_stats['order_to_execution_ms'].append(latency_ms)
            if signal_time and execution_time:
                signal_time_dt = datetime.fromisoformat(signal_time)
                execution_time_dt = datetime.fromisoformat(execution_time)
                total_latency_ms = (execution_time_dt - signal_time_dt
                    ).total_seconds() * 1000
                execution['total_latency_ms'] = total_latency_ms
                self.latency_stats['signal_to_execution_ms'].append(
                    total_latency_ms)
            market_condition = self._estimate_market_condition(instrument)
            execution['market_condition'
                ] = market_condition.value if market_condition else None
            quality_score = self._calculate_execution_quality(execution)
            execution['execution_quality'] = quality_score
            if quality_score is not None:
                self.execution_quality_scores.append(quality_score)
            vwap_deviation = await self._calculate_vwap_deviation(execution)
            execution['vwap_deviation_bps'] = vwap_deviation
            if vwap_deviation is not None:
                self.vwap_deviations.append(vwap_deviation)
            imp_shortfall = await self._calculate_implementation_shortfall(
                execution)
            execution['implementation_shortfall_bps'] = imp_shortfall
            if imp_shortfall is not None:
                self.implementation_shortfalls.append(imp_shortfall)
            logger.info(
                f"Analyzed execution {execution_id}: slippage={execution.get('slippage_pips')} pips, quality={quality_score}, vwap_dev={vwap_deviation} bps, is={imp_shortfall} bps"
                )
        except Exception as e:
            logger.error(f'Error analyzing execution: {e}')

    def _get_expected_price(self, execution: Dict[str, Any]) ->Optional[float]:
        """
        Get the expected price for an execution.

        Args:
            execution: Execution record

        Returns:
            Expected price or None if not available
        """
        signal_id = execution.get('signal_id')
        if signal_id and signal_id in self.signals:
            signal = self.signals[signal_id].get('signal', {})
            if signal.get('type', '').upper() == 'MARKET':
                return signal.get('expected_price')
            if signal.get('type', '').upper() == 'LIMIT':
                return signal.get('limit_price')
            if signal.get('type', '').upper() == 'STOP':
                return signal.get('stop_price')
        execution_time = execution.get('execution_time')
        if execution_time:
            instrument = execution.get('instrument')
            snapshot = self._get_nearest_market_snapshot(execution_time,
                instrument)
            if snapshot:
                direction = execution.get('direction')
                if direction == 'BUY':
                    return snapshot.get('ask')
                elif direction == 'SELL':
                    return snapshot.get('bid')
        return None

    @with_exception_handling
    def _get_nearest_market_snapshot(self, timestamp_str: str, instrument: str
        ) ->Optional[Dict[str, Any]]:
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
            min_time_diff = timedelta(minutes=5)
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
            logger.error(f'Error getting nearest market snapshot: {e}')
            return None

    def _calculate_slippage(self, instrument: str, direction: str,
        expected_price: float, execution_price: float) ->float:
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
        pip_size = 0.01 if instrument.endswith('JPY') else 0.0001
        price_diff = execution_price - expected_price
        if direction == 'SELL':
            price_diff = -price_diff
        slippage_pips = price_diff / pip_size
        return slippage_pips

    @async_with_exception_handling
    async def _calculate_vwap_deviation(self, execution: Dict[str, Any]
        ) ->Optional[float]:
        """
        Calculate deviation from Volume Weighted Average Price (VWAP) in basis points.
        Uses executed order price against market VWAP over the execution period.

        Args:
            execution: Execution record

        Returns:
            VWAP deviation in basis points (positive means worse than VWAP) or None
        """
        avg_exec_price = execution.get('execution_price')
        direction = execution.get('direction')
        if avg_exec_price is None or direction is None:
            return None
        try:
            market_vwap = await self._get_market_vwap(instrument=execution.
                get('instrument'), start_time_str=execution.get(
                'order_time'), end_time_str=execution.get('execution_time'))
        except Exception as e:
            logger.error(f'Error getting market VWAP: {e}')
            return None
        if not market_vwap:
            return None
        deviation = (avg_exec_price - market_vwap) / market_vwap * 10000
        if direction == 'SELL':
            deviation = -deviation
        return deviation

    @async_with_exception_handling
    async def _calculate_implementation_shortfall(self, execution: Dict[str,
        Any]) ->Optional[float]:
        """
        Calculate Implementation Shortfall (IS) in basis points.
        Compares execution cost to a benchmark price (e.g., arrival price).

        Args:
            execution: Execution record

        Returns:
            Implementation Shortfall in basis points (positive means cost/shortfall) or None
        """
        avg_exec_price = execution.get('execution_price')
        direction = execution.get('direction')
        order_size = execution.get('size', 0)
        filled_size = execution.get('filled_size', 0)
        if avg_exec_price is None or direction is None or order_size == 0:
            return None
        try:
            benchmark_price = await self._get_benchmark_price(execution)
        except Exception as e:
            logger.error(f'Error getting benchmark price: {e}')
            benchmark_price = None
        if benchmark_price is None or benchmark_price == 0:
            return None
        price_diff = avg_exec_price - benchmark_price
        if direction == 'SELL':
            price_diff = -price_diff
        implementation_shortfall_bps = price_diff / benchmark_price * 10000
        return implementation_shortfall_bps

    @async_with_exception_handling
    async def _get_market_vwap(self, instrument: str, start_time_str:
        Optional[str], end_time_str: Optional[str]) ->Optional[float]:
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
            logger.warning(
                f'Missing required parameters for VWAP calculation: instrument={instrument}, start_time={start_time_str}, end_time={end_time_str}'
                )
            return None
        try:
            market_data_service = self.config_manager.get('market_data_service')
            if market_data_service and hasattr(market_data_service,
                'get_historical_data'):
                historical_data = (await market_data_service.
                    get_historical_data(instrument=instrument, start_time=
                    start_time_str, end_time=end_time_str, timeframe='1m'))
                if historical_data is not None and not historical_data.empty:
                    if ('close' in historical_data.columns and 'volume' in
                        historical_data.columns):
                        total_volume = historical_data['volume'].sum()
                        if total_volume > 0:
                            weighted_price_sum = (historical_data['close'] *
                                historical_data['volume']).sum()
                            vwap = weighted_price_sum / total_volume
                            logger.info(
                                f'Calculated VWAP for {instrument} between {start_time_str} and {end_time_str}: {vwap}'
                                )
                            return vwap
                        else:
                            logger.warning(
                                f'Zero total volume for {instrument} between {start_time_str} and {end_time_str}'
                                )
                    else:
                        logger.warning(
                            f'Historical data missing required columns (close, volume) for VWAP calculation'
                            )
                else:
                    logger.warning(
                        f'No historical data available for {instrument} between {start_time_str} and {end_time_str}'
                        )
            else:
                logger.warning(
                    'Market data service unavailable for fetching historical data'
                    )
            return None
        except Exception as e:
            logger.error(f'Error calculating VWAP: {e}')
            return None

    @async_with_exception_handling
    async def _get_benchmark_price(self, execution: Dict[str, Any]) ->Optional[
        float]:
        """
        Get benchmark price for calculating implementation shortfall.
        Uses either arrival price from signal/order time or decision price if available.

        Args:
            execution: Execution record with details about the order

        Returns:
            Benchmark price or None if unavailable
        """
        signal_id = execution.get('signal_id')
        if signal_id and signal_id in self.signals:
            signal_data = self.signals[signal_id]
            signal = signal_data.get('signal', {})
            if 'reference_price' in signal:
                logger.info(
                    f"Using explicit reference price from signal for execution {execution.get('order_id')}"
                    )
                return signal.get('reference_price')
            if signal.get('type', '').upper(
                ) == 'LIMIT' and 'limit_price' in signal:
                logger.info(
                    f"Using limit price as benchmark for execution {execution.get('order_id')}"
                    )
                return signal.get('limit_price')
        arrival_time = execution.get('signal_time') or execution.get(
            'order_time')
        instrument = execution.get('instrument')
        direction = execution.get('direction')
        if arrival_time and instrument:
            snapshot = self._get_nearest_market_snapshot(arrival_time,
                instrument)
            if snapshot and 'bid' in snapshot and 'ask' in snapshot:
                if direction == 'BUY':
                    benchmark = snapshot['ask']
                elif direction == 'SELL':
                    benchmark = snapshot['bid']
                else:
                    benchmark = (snapshot['bid'] + snapshot['ask']) / 2
                logger.info(
                    f"Using market snapshot price as benchmark for execution {execution.get('order_id')}"
                    )
                return benchmark
            market_data_service = self.config_manager.get('market_data_service')
            if market_data_service:
                try:
                    if hasattr(market_data_service, 'get_historical_data'):
                        if isinstance(arrival_time, str):
                            arrival_time_dt = datetime.fromisoformat(
                                arrival_time.replace('Z', '+00:00'))
                        else:
                            arrival_time_dt = arrival_time
                        window_start = arrival_time_dt - timedelta(minutes=1)
                        window_end = arrival_time_dt + timedelta(minutes=1)
                        historical_data = (await market_data_service.
                            get_historical_data(instrument=instrument,
                            start_time=window_start, end_time=window_end,
                            timeframe='1m'))
                        if (historical_data is not None and not
                            historical_data.empty):
                            if 'open' in historical_data.columns:
                                benchmark = historical_data['open'].iloc[0]
                                logger.info(
                                    f"Using historical data price as benchmark for execution {execution.get('order_id')}"
                                    )
                                return benchmark
                except Exception as e:
                    logger.error(
                        f'Error getting historical benchmark price: {e}')
        logger.warning(
            f"Could not determine benchmark price for execution {execution.get('order_id')}. Implementation shortfall calculation will be unavailable."
            )
        return None

    def _update_slippage_stats(self, execution: Dict[str, Any]) ->None:
        """
        Update slippage statistics with a new execution.

        Args:
            execution: Execution record
        """
        slippage = execution.get('slippage_pips')
        if slippage is None:
            return
        instrument = execution.get('instrument')
        market_condition = execution.get('market_condition')
        size = execution.get('size', 0)
        self.slippage_stats['total'].append(slippage)
        if instrument:
            if instrument not in self.slippage_stats['by_instrument']:
                self.slippage_stats['by_instrument'][instrument] = []
            self.slippage_stats['by_instrument'][instrument].append(slippage)
        if market_condition:
            if market_condition not in self.slippage_stats['by_condition']:
                self.slippage_stats['by_condition'][market_condition] = []
            self.slippage_stats['by_condition'][market_condition].append(
                slippage)
        if size:
            if size < 10000:
                self.slippage_stats['by_size']['small'].append(slippage)
            elif size < 100000:
                self.slippage_stats['by_size']['medium'].append(slippage)
            else:
                self.slippage_stats['by_size']['large'].append(slippage)

    def _update_fill_rate_stats(self, report: ExecutionReport) ->None:
        """
        Update fill rate statistics based on execution report.

        Args:
            report: Execution report
        """
        instrument = report.instrument
        status = report.status
        self.fill_rate_stats['total_orders'] += 1
        if status == OrderStatus.FILLED:
            self.fill_rate_stats['filled_orders'] += 1
        elif status == OrderStatus.PARTIAL_FILL:
            self.fill_rate_stats['partial_fills'] += 1
        elif status == OrderStatus.REJECTED:
            self.fill_rate_stats['rejected_orders'] += 1
        elif status == OrderStatus.CANCELLED:
            self.fill_rate_stats['cancelled_orders'] += 1
        if instrument:
            if instrument not in self.fill_rate_stats['by_instrument']:
                self.fill_rate_stats['by_instrument'][instrument] = {
                    'total_orders': 0, 'filled_orders': 0, 'partial_fills':
                    0, 'rejected_orders': 0, 'cancelled_orders': 0}
            self.fill_rate_stats['by_instrument'][instrument]['total_orders'
                ] += 1
            if status == OrderStatus.FILLED:
                self.fill_rate_stats['by_instrument'][instrument][
                    'filled_orders'] += 1
            elif status == OrderStatus.PARTIAL_FILL:
                self.fill_rate_stats['by_instrument'][instrument][
                    'partial_fills'] += 1
            elif status == OrderStatus.REJECTED:
                self.fill_rate_stats['by_instrument'][instrument][
                    'rejected_orders'] += 1
            elif status == OrderStatus.CANCELLED:
                self.fill_rate_stats['by_instrument'][instrument][
                    'cancelled_orders'] += 1

    def _estimate_market_condition(self, instrument: str) ->Optional[
        MarketCondition]:
        """
        Estimate current market condition based on recent snapshots.

        Args:
            instrument: Trading instrument

        Returns:
            Estimated market condition or None if not enough data
        """
        recent_snapshots = []
        for ts in sorted(self.market_snapshots.keys(), reverse=True):
            if len(recent_snapshots) >= 10:
                break
            if instrument in self.market_snapshots[ts]:
                recent_snapshots.append(self.market_snapshots[ts][instrument])
        if len(recent_snapshots) < 3:
            return MarketCondition.NORMAL
        prices = []
        spreads = []
        for snapshot in recent_snapshots:
            if 'bid' in snapshot and 'ask' in snapshot:
                mid = (snapshot['bid'] + snapshot['ask']) / 2
                prices.append(mid)
                spreads.append(snapshot['ask'] - snapshot['bid'])
        if len(prices) < 3:
            return MarketCondition.NORMAL
        price_changes = [(abs(prices[i] - prices[i - 1]) / prices[i - 1] * 
            100) for i in range(1, len(prices))]
        volatility = statistics.stdev(prices) / statistics.mean(prices) * 100
        avg_spread = statistics.mean(spreads) if spreads else 0
        if volatility > 0.5:
            return MarketCondition.VOLATILE
        elif avg_spread > 0.0002 and instrument != 'USDJPY':
            return MarketCondition.ILLIQUID
        elif all(pc < 0.05 for pc in price_changes):
            return MarketCondition.RANGING
        elif sum(price_changes) / len(price_changes) > 0.1:
            return MarketCondition.TRENDING
        else:
            return MarketCondition.NORMAL

    def _calculate_execution_quality(self, execution: Dict[str, Any]
        ) ->Optional[float]:
        """
        Calculate execution quality score (0-100).

        Args:
            execution: Execution record

        Returns:
            Quality score or None if not enough data
        """
        factors = {}
        slippage = execution.get('slippage_pips')
        if slippage is not None:
            if slippage <= 0:
                factors['slippage'] = 40
            else:
                factors['slippage'] = 40 * max(0, min(1, pow(0.6, slippage)))
        latency = execution.get('latency_ms')
        if latency is not None:
            if latency <= 50:
                factors['latency'] = 30
            else:
                factors['latency'] = 30 * max(0, min(1, (1000 - latency) / 950)
                    )
        size = execution.get('size', 0)
        filled_size = execution.get('filled_size', 0)
        if size > 0:
            fill_rate = filled_size / size
            factors['fill_rate'] = 20 * fill_rate
        market_condition = execution.get('market_condition')
        if market_condition:
            condition_scores = {MarketCondition.NORMAL.value: 10,
                MarketCondition.TRENDING.value: 8, MarketCondition.RANGING.
                value: 7, MarketCondition.VOLATILE.value: 5,
                MarketCondition.ILLIQUID.value: 3}
            factors['market_condition'] = condition_scores.get(market_condition
                , 5)
        if not factors:
            return None
        total_score = sum(factors.values())
        max_possible = sum([40, 30, 20, 10])
        max_for_factors = 0
        for key in factors:
            if key == 'slippage':
                max_for_factors += 40
            elif key == 'latency':
                max_for_factors += 30
            elif key == 'fill_rate':
                max_for_factors += 20
            elif key == 'market_condition':
                max_for_factors += 10
        normalized_score = (total_score / max_for_factors * 100 if 
            max_for_factors > 0 else 0)
        return normalized_score

    @async_with_exception_handling
    async def _periodic_cleanup(self) ->None:
        """Periodically clean up old data."""
        logger.info('Starting periodic cleanup task')
        try:
            while self.running:
                try:
                    await asyncio.sleep(self.cleanup_interval_hours * 60 * 60)
                    self._cleanup_old_data()
                except Exception as e:
                    logger.error(f'Error in periodic cleanup: {e}')
                    await asyncio.sleep(60 * 60)
        except asyncio.CancelledError:
            logger.info('Periodic cleanup task cancelled')
            raise

    @with_exception_handling
    def _cleanup_old_data(self) ->None:
        """Clean up data older than retention period."""
        try:
            retention_limit = datetime.now() - timedelta(days=self.
                data_retention_days)
            retention_limit_str = retention_limit.isoformat()
            keys_to_remove = []
            for key, execution in self.executions.items():
                execution_time = execution.get('execution_time')
                if execution_time and execution_time < retention_limit_str:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.executions[key]
            logger.info(f'Cleaned up {len(keys_to_remove)} old executions')
            keys_to_remove = []
            for key, signal_data in self.signals.items():
                received_time = signal_data.get('received_time')
                if received_time and received_time < retention_limit_str:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.signals[key]
            logger.info(f'Cleaned up {len(keys_to_remove)} old signals')
            self._cleanup_old_snapshots(retention_limit_str)
        except Exception as e:
            logger.error(f'Error cleaning up old data: {e}')

    @with_exception_handling
    def _cleanup_old_snapshots(self, older_than: Optional[str]=None) ->None:
        """
        Clean up old market snapshots.

        Args:
            older_than: Optional timestamp string, all snapshots older will be removed
        """
        try:
            if older_than is None:
                if len(self.market_snapshots) > 100:
                    timestamps = sorted(self.market_snapshots.keys())[:-100]
                    for ts in timestamps:
                        del self.market_snapshots[ts]
                    logger.debug(
                        f'Cleaned up {len(timestamps)} old market snapshots')
            else:
                keys_to_remove = [ts for ts in self.market_snapshots.keys() if
                    ts < older_than]
                for ts in keys_to_remove:
                    del self.market_snapshots[ts]
                logger.debug(
                    f'Cleaned up {len(keys_to_remove)} old market snapshots')
        except Exception as e:
            logger.error(f'Error cleaning up old snapshots: {e}')

    @with_broker_api_resilience('get_metrics')
    def get_metrics(self) ->Dict[str, Any]:
        """
        Get execution quality metrics.

        Returns:
            Dictionary with execution metrics
        """
        metrics = {'execution_count': len(self.executions), 'signal_count':
            len(self.signals), 'avg_execution_quality': statistics.mean(
            self.execution_quality_scores) if self.execution_quality_scores
             else None, 'fill_rate': self.fill_rate_stats['filled_orders'] /
            self.fill_rate_stats['total_orders'] if self.fill_rate_stats[
            'total_orders'] > 0 else 0}
        if self.slippage_stats['total']:
            metrics['avg_slippage_pips'] = statistics.mean(self.
                slippage_stats['total'])
            metrics['max_slippage_pips'] = max(self.slippage_stats['total'])
        if self.latency_stats['order_to_execution_ms']:
            metrics['avg_execution_latency_ms'] = statistics.mean(self.
                latency_stats['order_to_execution_ms'])
        if self.latency_stats['signal_to_execution_ms']:
            metrics['avg_total_latency_ms'] = statistics.mean(self.
                latency_stats['signal_to_execution_ms'])
        if self.vwap_deviations:
            metrics['avg_vwap_deviation_bps'] = statistics.mean(self.
                vwap_deviations)
            metrics['max_vwap_deviation_bps'] = max(self.vwap_deviations)
            metrics['min_vwap_deviation_bps'] = min(self.vwap_deviations)
        if self.implementation_shortfalls:
            metrics['avg_implementation_shortfall_bps'] = statistics.mean(self
                .implementation_shortfalls)
            metrics['max_implementation_shortfall_bps'] = max(self.
                implementation_shortfalls)
            metrics['min_implementation_shortfall_bps'] = min(self.
                implementation_shortfalls)
        return metrics

    @with_broker_api_resilience('get_execution_details')
    async def get_execution_details(self, execution_id: str) ->Optional[Dict
        [str, Any]]:
        """
        Get detailed information about a specific execution.

        Args:
            execution_id: ID of the execution

        Returns:
            Dictionary with execution details or None if not found
        """
        if execution_id not in self.executions:
            return None
        return self.executions[execution_id]

    @with_broker_api_resilience('get_executions_by_instrument')
    async def get_executions_by_instrument(self, instrument: str, limit:
        int=100) ->List[Dict[str, Any]]:
        """
        Get executions for a specific instrument.

        Args:
            instrument: Trading instrument
            limit: Maximum number of executions to return

        Returns:
            List of execution records
        """
        matching_executions = []
        for execution in self.executions.values():
            if execution.get('instrument') == instrument:
                matching_executions.append(execution)
                if len(matching_executions) >= limit:
                    break
        return sorted(matching_executions, key=lambda e: e.get(
            'execution_time', ''), reverse=True)

    @with_broker_api_resilience('get_execution_quality_by_condition')
    async def get_execution_quality_by_condition(self) ->Dict[str, Dict[str,
        float]]:
        """
        Get execution quality metrics grouped by market condition.

        Returns:
            Dictionary with metrics by market condition
        """
        by_condition = {}
        for execution in self.executions.values():
            condition = execution.get('market_condition')
            quality = execution.get('execution_quality')
            slippage = execution.get('slippage_pips')
            if condition and quality is not None:
                if condition not in by_condition:
                    by_condition[condition] = {'executions': [],
                        'quality_scores': [], 'slippage_values': []}
                by_condition[condition]['executions'].append(execution)
                by_condition[condition]['quality_scores'].append(quality)
                if slippage is not None:
                    by_condition[condition]['slippage_values'].append(slippage)
        result = {}
        for condition, data in by_condition.items():
            quality_scores = data['quality_scores']
            slippage_values = data['slippage_values']
            if quality_scores:
                result[condition] = {'execution_count': len(quality_scores),
                    'avg_quality': statistics.mean(quality_scores),
                    'min_quality': min(quality_scores), 'max_quality': max(
                    quality_scores)}
                if slippage_values:
                    result[condition]['avg_slippage'] = statistics.mean(
                        slippage_values)
        return result

    @with_database_resilience('get_feedback_for_ml_model')
    async def get_feedback_for_ml_model(self) ->Dict[str, Any]:
        """
        Get aggregated feedback data for ML model integration.

        Returns:
            Dictionary with feedback data for ML models
        """
        ml_data = {'timestamp': datetime.now().isoformat(),
            'execution_metrics': {'count': len(self.executions),
            'avg_quality': statistics.mean(self.execution_quality_scores) if
            self.execution_quality_scores else None, 'avg_slippage': 
            statistics.mean(self.slippage_stats['total']) if self.
            slippage_stats['total'] else None, 'fill_rate': self.
            fill_rate_stats['filled_orders'] / self.fill_rate_stats[
            'total_orders'] if self.fill_rate_stats['total_orders'] > 0 else
            0, 'avg_vwap_deviation_bps': statistics.mean(self.
            vwap_deviations) if self.vwap_deviations else None,
            'avg_implementation_shortfall_bps': statistics.mean(self.
            implementation_shortfalls) if self.implementation_shortfalls else
            None}, 'by_instrument': {}, 'by_condition': {}, 'by_size': {}}
        for instrument, slippage_values in self.slippage_stats['by_instrument'
            ].items():
            if slippage_values:
                ml_data['by_instrument'][instrument] = {'avg_slippage':
                    statistics.mean(slippage_values), 'min_slippage': min(
                    slippage_values), 'max_slippage': max(slippage_values)}
                if instrument in self.fill_rate_stats['by_instrument']:
                    instrument_stats = self.fill_rate_stats['by_instrument'][
                        instrument]
                    if instrument_stats['total_orders'] > 0:
                        ml_data['by_instrument'][instrument]['fill_rate'
                            ] = instrument_stats['filled_orders'
                            ] / instrument_stats['total_orders']
        for condition, slippage_values in self.slippage_stats['by_condition'
            ].items():
            if slippage_values:
                ml_data['by_condition'][condition] = {'avg_slippage':
                    statistics.mean(slippage_values), 'min_slippage': min(
                    slippage_values), 'max_slippage': max(slippage_values)}
        for size_category, slippage_values in self.slippage_stats['by_size'
            ].items():
            if slippage_values:
                ml_data['by_size'][size_category] = {'avg_slippage':
                    statistics.mean(slippage_values), 'min_slippage': min(
                    slippage_values), 'max_slippage': max(slippage_values)}
        return ml_data


async def main():
    """
    Main.
    
    """

    analytics = ExecutionAnalytics()
    await analytics.start()
    signal = {'signal_id': 'test-signal-1', 'instrument': 'EURUSD',
        'direction': 'BUY', 'type': 'MARKET', 'strategy_id':
        'test-strategy', 'size': 10000, 'received_time': datetime.now().
        isoformat()}


    class MockOrderRequest:
    """
    MockOrderRequest class.
    
    Attributes:
        Add attributes here
    """


        def __init__(self):
    """
      init  .
    
    """

            self.client_order_id = 'test-order-1'
            self.size = 10000
    order_request = MockOrderRequest()
    await analytics.record_signal_processing(signal, order_request)
    await analytics.record_market_snapshot('EURUSD', {'bid': 1.085, 'ask': 
        1.0852, 'volume': 100, 'timestamp': datetime.now().isoformat()})


    class MockExecutionReport:
    """
    MockExecutionReport class.
    
    Attributes:
        Add attributes here
    """


        def __init__(self):
    """
      init  .
    
    """

            self.order_id = 'broker-order-1'
            self.client_order_id = 'test-order-1'
            self.instrument = 'EURUSD'
            self.direction = 'BUY'
            self.size = 10000
            self.filled_size = 10000
            self.execution_price = 1.0853
            self.status = OrderStatus.FILLED
            self.transaction_time = datetime.now().isoformat()
    report = MockExecutionReport()
    await analytics.process_execution_report(report)
    await asyncio.sleep(1)
    metrics = analytics.get_metrics()
    print(f'Metrics: {metrics}')
    ml_feedback = await analytics.get_feedback_for_ml_model()
    print(f'ML Feedback: {ml_feedback}')
    await analytics.stop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
