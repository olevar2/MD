"""
PaperTradingCoordinator for establishing the full paper trading loop from signal to execution.

from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

This component coordinates the complete trading process including session management,
signal processing, order execution, and feedback collection for paper trading.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Set
import uuid
from core_foundations.models.market_data import MarketData
from core_foundations.models.signal import Signal, SignalConfidence
from core_foundations.models.order import Order, OrderStatus, OrderType, OrderDirection
from core_foundations.models.position import Position
from core_foundations.utils.logger import get_logger
from strategy_execution_engine.risk.dynamic_position_sizing import DynamicPositionSizing
from strategy_execution_engine.risk.risk_check_orchestrator import RiskCheckOrchestrator
from strategy_execution_engine.signal_aggregation.signal_aggregator import SignalAggregator
from strategy_execution_engine.trading.trading_session_manager import TradingSessionManager
from strategy_execution_engine.trading.feedback_collector import FeedbackCollector


class PaperTradingMode(Enum):
    """Different modes for paper trading."""
    NORMAL = 'normal'
    INSTANT = 'instant'
    BACKTEST = 'backtest'
    SIMULATION = 'simulation'


class PaperTradingCoordinator:
    """
    Coordinates the full paper trading process from signal generation to execution and feedback.
    
    This component acts as the central coordinator for paper trading, managing the flow of
    data between different components like signal aggregation, risk checks, order execution,
    and feedback collection.
    """

    def __init__(self, trading_session_manager: Optional[
        TradingSessionManager]=None, signal_aggregator: Optional[
        SignalAggregator]=None, risk_check_orchestrator: Optional[
        RiskCheckOrchestrator]=None, position_sizer: Optional[
        DynamicPositionSizing]=None, trading_gateway_client=None,
        feedback_collector: Optional[FeedbackCollector]=None, mode:
        PaperTradingMode=PaperTradingMode.NORMAL, config: Optional[Dict[str,
        Any]]=None):
        """
        Initialize the PaperTradingCoordinator.
        
        Args:
            trading_session_manager: Manager for trading sessions and hours
            signal_aggregator: Component for aggregating trading signals
            risk_check_orchestrator: Component for risk validation
            position_sizer: Component for position size calculation
            trading_gateway_client: Client for interacting with the trading gateway
            feedback_collector: Component for gathering execution feedback
            mode: Paper trading mode
            config: Additional configuration parameters
        """
        self.logger = get_logger(self.__class__.__name__)
        self.trading_session_manager = (trading_session_manager or
            TradingSessionManager())
        self.signal_aggregator = signal_aggregator or SignalAggregator()
        self.risk_check_orchestrator = (risk_check_orchestrator or
            RiskCheckOrchestrator())
        self.position_sizer = position_sizer or DynamicPositionSizing()
        self.trading_gateway_client = trading_gateway_client
        self.feedback_collector = feedback_collector or FeedbackCollector()
        self.mode = mode
        self.config = config or {}
        self.active_orders: Dict[str, Order] = {}
        self.active_positions: Dict[str, Position] = {}
        self.pending_signals: List[Signal] = []
        self._running = False
        self._last_execution_time = datetime.now()
        self._performance_metrics: Dict[str, Any] = {'signals_processed': 0,
            'orders_placed': 0, 'orders_filled': 0, 'orders_rejected': 0,
            'average_execution_time_ms': 0}
        self.logger.info(
            f'PaperTradingCoordinator initialized with mode: {mode.value}')

    @async_with_exception_handling
    async def start(self) ->None:
        """Start the paper trading loop."""
        if self._running:
            self.logger.warning('Paper trading loop is already running')
            return
        self._running = True
        self.logger.info('Starting paper trading loop')
        await self.trading_session_manager.start()
        try:
            while self._running:
                if not self.trading_session_manager.is_trading_allowed():
                    await self._handle_outside_trading_hours()
                    await asyncio.sleep(60)
                    continue
                await self._process_pending_signals()
                await self._update_order_statuses()
                await self._collect_feedback()
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f'Error in paper trading loop: {str(e)}')
            raise
        finally:
            self._running = False
            await self.trading_session_manager.stop()
            self.logger.info('Paper trading loop stopped')

    async def stop(self) ->None:
        """Stop the paper trading loop."""
        if not self._running:
            self.logger.warning('Paper trading loop is not running')
            return
        self._running = False
        self.logger.info('Stopping paper trading loop')

    async def add_signal(self, signal: Signal) ->bool:
        """
        Add a new signal for processing.
        
        Args:
            signal: The trading signal to process
            
        Returns:
            bool: True if signal was accepted, False otherwise
        """
        if not signal:
            return False
        if not signal.symbol or not signal.direction:
            self.logger.warning(f'Rejected invalid signal: {signal}')
            return False
        self.pending_signals.append(signal)
        self.logger.debug(
            f'Added signal to queue: {signal.symbol} {signal.direction}')
        return True

    async def get_active_positions(self) ->Dict[str, Position]:
        """
        Get all currently active positions.
        
        Returns:
            Dictionary of position_id to Position objects
        """
        return self.active_positions.copy()

    async def get_active_orders(self) ->Dict[str, Order]:
        """
        Get all currently active orders.
        
        Returns:
            Dictionary of order_id to Order objects
        """
        return self.active_orders.copy()

    async def get_performance_metrics(self) ->Dict[str, Any]:
        """
        Get performance metrics for the paper trading session.
        
        Returns:
            Dictionary of performance metrics
        """
        return self._performance_metrics.copy()

    @async_with_exception_handling
    async def _process_pending_signals(self) ->None:
        """Process any pending trading signals."""
        if not self.pending_signals:
            return
        batch_size = min(10, len(self.pending_signals))
        batch = self.pending_signals[:batch_size]
        self.pending_signals = self.pending_signals[batch_size:]
        for signal in batch:
            try:
                start_time = time.time()
                if signal.confidence == SignalConfidence.LOW:
                    self.logger.debug(
                        f'Skipping low confidence signal: {signal.symbol} {signal.direction}'
                        )
                    continue
                aggregated_signal = await self._aggregate_signal(signal)
                risk_result = await self._perform_pre_trade_risk_check(
                    aggregated_signal)
                if not risk_result.get('approved', False):
                    self.logger.info(
                        f"Signal rejected by risk check: {risk_result.get('reason', 'Unknown')}"
                        )
                    continue
                position_size = await self._calculate_position_size(
                    aggregated_signal, risk_result)
                order_parameters = await self._calculate_order_parameters(
                    aggregated_signal)
                order = await self._create_and_submit_order(aggregated_signal,
                    position_size, order_parameters)
                if order:
                    self.active_orders[order.order_id] = order
                    self._performance_metrics['orders_placed'] += 1
                execution_time_ms = (time.time() - start_time) * 1000
                avg_time = self._performance_metrics[
                    'average_execution_time_ms']
                orders_processed = max(1, self._performance_metrics[
                    'signals_processed'])
                self._performance_metrics['average_execution_time_ms'] = (
                    avg_time * orders_processed + execution_time_ms) / (
                    orders_processed + 1)
                self._performance_metrics['signals_processed'] += 1
            except Exception as e:
                self.logger.error(
                    f'Error processing signal {signal.symbol}: {str(e)}')

    @async_with_exception_handling
    async def _aggregate_signal(self, signal: Signal) ->Signal:
        """
        Aggregate the signal with other related signals.
        
        Args:
            signal: The trading signal to aggregate
            
        Returns:
            The aggregated signal
        """
        if not self.signal_aggregator:
            return signal
        try:
            return await self.signal_aggregator.aggregate_signal(signal)
        except Exception as e:
            self.logger.warning(f'Error aggregating signal: {str(e)}')
            return signal

    @async_with_exception_handling
    async def _perform_pre_trade_risk_check(self, signal: Signal) ->Dict[
        str, Any]:
        """
        Perform pre-trade risk checks.
        
        Args:
            signal: The trading signal to check
            
        Returns:
            Risk check result with approval status
        """
        if not self.risk_check_orchestrator:
            return {'approved': True}
        try:
            return await self.risk_check_orchestrator.perform_pre_trade_check(
                instrument=signal.symbol, direction=signal.direction,
                signal_metadata=signal.metadata)
        except Exception as e:
            self.logger.error(f'Error in risk check: {str(e)}')
            return {'approved': False, 'reason': f'Risk check error: {str(e)}'}

    @async_with_exception_handling
    async def _calculate_position_size(self, signal: Signal, risk_result:
        Dict[str, Any]) ->float:
        """
        Calculate appropriate position size.
        
        Args:
            signal: The trading signal
            risk_result: Result from risk check
            
        Returns:
            Position size
        """
        if not self.position_sizer:
            return self.config_manager.get('default_position_size', 10000)
        try:
            account_balance = await self._get_account_balance()
            market_data = await self._get_market_data(signal.symbol)
            return await self.position_sizer.calculate_position_size(instrument
                =signal.symbol, direction=signal.direction, signal_metadata
                =signal.metadata, account_balance=account_balance,
                risk_percentage=self.config_manager.get('risk_per_trade', 1.0),
                market_data=market_data)
        except Exception as e:
            self.logger.warning(f'Error calculating position size: {str(e)}')
            return self.config_manager.get('default_position_size', 10000)

    @async_with_exception_handling
    async def _calculate_order_parameters(self, signal: Signal) ->Dict[str, Any
        ]:
        """
        Calculate entry, stop loss, and take profit levels.
        
        Args:
            signal: The trading signal
            
        Returns:
            Dictionary with order parameters
        """
        parameters = {'entry_type': OrderType.MARKET, 'entry_price': None,
            'stop_loss': None, 'take_profit': None}
        try:
            market_data = await self._get_market_data(signal.symbol)
            if signal.metadata and 'suggested_entry' in signal.metadata:
                parameters['entry_type'] = OrderType.LIMIT
                parameters['entry_price'] = signal.metadata['suggested_entry']
            if signal.metadata and 'suggested_stop' in signal.metadata:
                parameters['stop_loss'] = signal.metadata['suggested_stop']
            if signal.metadata and 'suggested_target' in signal.metadata:
                parameters['take_profit'] = signal.metadata['suggested_target']
            if not parameters['stop_loss'] or not parameters['take_profit']:
                if self.trading_gateway_client and hasattr(self.
                    trading_gateway_client, 'calculate_stops'):
                    stops = await self.trading_gateway_client.calculate_stops(
                        instrument=signal.symbol, direction=signal.
                        direction, entry_price=parameters['entry_price'] or
                        market_data['price'], atr=market_data.get('atr', None))
                    if not parameters['stop_loss']:
                        parameters['stop_loss'] = stops.get('stop_loss')
                    if not parameters['take_profit']:
                        parameters['take_profit'] = stops.get('take_profit')
        except Exception as e:
            self.logger.warning(f'Error calculating order parameters: {str(e)}'
                )
        return parameters

    @async_with_exception_handling
    async def _create_and_submit_order(self, signal: Signal, position_size:
        float, order_parameters: Dict[str, Any]) ->Optional[Order]:
        """
        Create and submit an order to the trading gateway.
        
        Args:
            signal: The trading signal
            position_size: The calculated position size
            order_parameters: Parameters for the order
            
        Returns:
            The created order, or None if failed
        """
        if not self.trading_gateway_client:
            self.logger.warning(
                'Trading gateway client not available, skipping order submission'
                )
            return None
        try:
            order = Order(order_id=str(uuid.uuid4()), symbol=signal.symbol,
                direction=signal.direction, type=order_parameters[
                'entry_type'], size=position_size, price=order_parameters[
                'entry_price'], stop_loss=order_parameters['stop_loss'],
                take_profit=order_parameters['take_profit'], status=
                OrderStatus.PENDING, signal_id=signal.signal_id, metadata={
                'source': 'paper_trading', 'signal_confidence': signal.
                confidence.value, 'creation_time': datetime.now().isoformat()})
            result = await self.trading_gateway_client.submit_order(order)
            if not result.get('success', False):
                self.logger.warning(
                    f"Order submission failed: {result.get('message', 'Unknown error')}"
                    )
                return None
            if 'order' in result:
                order = result['order']
            self.logger.info(
                f'Order submitted: {order.order_id} {order.symbol} {order.direction} {order.size}'
                )
            return order
        except Exception as e:
            self.logger.error(f'Error creating/submitting order: {str(e)}')
            return None

    @async_with_exception_handling
    async def _update_order_statuses(self) ->None:
        """Update the status of all active orders."""
        if not self.trading_gateway_client or not self.active_orders:
            return
        try:
            order_ids = list(self.active_orders.keys())
            updates = await self.trading_gateway_client.get_orders_status(
                order_ids)
            for order_id, update in updates.items():
                if order_id not in self.active_orders:
                    continue
                self.active_orders[order_id] = update
                if update.status == OrderStatus.FILLED:
                    self._performance_metrics['orders_filled'] += 1
                    position = Position(position_id=str(uuid.uuid4()),
                        symbol=update.symbol, direction=update.direction,
                        size=update.size, entry_price=update.filled_price or
                        update.price, open_time=datetime.now(), stop_loss=
                        update.stop_loss, take_profit=update.take_profit,
                        order_id=update.order_id, metadata=update.metadata)
                    self.active_positions[position.position_id] = position
                    self.logger.info(
                        f'Position opened: {position.position_id} {position.symbol} {position.direction} {position.size}'
                        )
                elif update.status in (OrderStatus.REJECTED, OrderStatus.
                    CANCELLED):
                    if update.status == OrderStatus.REJECTED:
                        self._performance_metrics['orders_rejected'] += 1
                    del self.active_orders[order_id]
                    self.logger.info(
                        f'Order {update.status.value}: {order_id} {update.symbol}'
                        )
        except Exception as e:
            self.logger.error(f'Error updating order statuses: {str(e)}')

    @async_with_exception_handling
    async def _collect_feedback(self) ->None:
        """Collect feedback on executed orders for strategy improvement."""
        if not self.feedback_collector or not self.active_positions:
            return
        try:
            position_ids = list(self.active_positions.keys())
            position_updates = await self._get_position_updates(position_ids)
            for position_id, update in position_updates.items():
                if position_id not in self.active_positions:
                    continue
                original_position = self.active_positions[position_id]
                self.active_positions[position_id] = update
                if update.is_closed:
                    pnl = self._calculate_position_pnl(update)
                    await self.feedback_collector.collect_execution_feedback(
                        position=update, original_position=
                        original_position, pnl=pnl, market_conditions=await
                        self._get_market_conditions(update.symbol))
                    del self.active_positions[position_id]
                    self.logger.info(
                        f'Position closed: {position_id} {update.symbol} P&L: {pnl}'
                        )
        except Exception as e:
            self.logger.error(f'Error collecting feedback: {str(e)}')

    @async_with_exception_handling
    async def _handle_outside_trading_hours(self) ->None:
        """Handle trading outside of allowed hours."""
        session_info = self.trading_session_manager.get_current_session_info()
        self.logger.debug(f'Outside trading hours: {session_info}')
        if self.trading_session_manager.should_close_positions(
            ) and self.active_positions:
            self.logger.info('Closing positions due to session end')
            for position_id, position in list(self.active_positions.items()):
                try:
                    await self._close_position(position_id)
                except Exception as e:
                    self.logger.error(
                        f'Error closing position {position_id}: {str(e)}')

    @async_with_exception_handling
    async def _close_position(self, position_id: str) ->None:
        """
        Close a specific position.
        
        Args:
            position_id: The ID of the position to close
        """
        if (position_id not in self.active_positions or not self.
            trading_gateway_client):
            return
        position = self.active_positions[position_id]
        try:
            await self.trading_gateway_client.close_position(position_id)
            self.logger.info(
                f'Closing position requested: {position_id} {position.symbol}')
        except Exception as e:
            self.logger.error(f'Error requesting position closure: {str(e)}')

    @async_with_exception_handling
    async def _get_account_balance(self) ->float:
        """
        Get the current account balance.
        
        Returns:
            Current account balance
        """
        if self.trading_gateway_client and hasattr(self.
            trading_gateway_client, 'get_account_info'):
            try:
                account_info = (await self.trading_gateway_client.
                    get_account_info())
                return account_info.get('balance', self.config.get(
                    'default_balance', 100000))
            except Exception as e:
                self.logger.warning(f'Error getting account balance: {str(e)}')
        return self.config_manager.get('default_balance', 100000)

    @async_with_exception_handling
    async def _get_market_data(self, symbol: str) ->Dict[str, Any]:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: The instrument symbol
            
        Returns:
            Dictionary with market data
        """
        if self.trading_gateway_client and hasattr(self.
            trading_gateway_client, 'get_market_data'):
            try:
                return await self.trading_gateway_client.get_market_data(symbol
                    )
            except Exception as e:
                self.logger.warning(
                    f'Error getting market data for {symbol}: {str(e)}')
        return {'symbol': symbol, 'price': 0, 'bid': 0, 'ask': 0,
            'timestamp': datetime.now().timestamp()}

    @async_with_exception_handling
    async def _get_position_updates(self, position_ids: List[str]) ->Dict[
        str, Position]:
        """
        Get updates for multiple positions.
        
        Args:
            position_ids: List of position IDs to update
            
        Returns:
            Dictionary of position_id to updated Position objects
        """
        if not self.trading_gateway_client or not position_ids:
            return {}
        try:
            return await self.trading_gateway_client.get_positions_status(
                position_ids)
        except Exception as e:
            self.logger.warning(f'Error getting position updates: {str(e)}')
            return {}

    async def _get_market_conditions(self, symbol: str) ->Dict[str, Any]:
        """
        Get current market conditions for feedback context.
        
        Args:
            symbol: The instrument symbol
            
        Returns:
            Dictionary with market condition data
        """
        market_data = await self._get_market_data(symbol)
        return {'symbol': symbol, 'timestamp': datetime.now(), 'price':
            market_data.get('price', 0), 'volatility': market_data.get(
            'volatility', 0), 'regime': market_data.get('regime', 'unknown'
            ), 'spread': market_data.get('ask', 0) - market_data.get('bid', 0)}

    def _calculate_position_pnl(self, position: Position) ->float:
        """
        Calculate profit/loss for a position.
        
        Args:
            position: The position to calculate P&L for
            
        Returns:
            Profit or loss amount
        """
        if not position.entry_price or not position.close_price:
            return 0
        price_diff = position.close_price - position.entry_price
        if position.direction == OrderDirection.SELL:
            price_diff = -price_diff
        return price_diff * position.size
