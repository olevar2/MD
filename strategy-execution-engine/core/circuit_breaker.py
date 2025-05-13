"""
Circuit Breaker Module for Forex Trading Platform

This module implements circuit breakers that can halt trading
activity based on various conditions and thresholds.
"""
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Callable
logger = logging.getLogger(__name__)


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class CircuitBreaker:
    """
    Circuit Breaker implementation for trading systems
    
    This class provides a multi-level circuit breaker system that can
    halt trading at various levels (global, instrument, strategy) based
    on predefined conditions and thresholds.
    """
    STATE_CLOSED = 'CLOSED'
    STATE_OPEN = 'OPEN'
    STATE_HALF_OPEN = 'HALF_OPEN'

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the circuit breaker
        
        Args:
            config: Configuration dictionary for circuit breaker
        """
        self.config = config or {}
        self.global_state = {'state': self.STATE_CLOSED, 'reason': None,
            'tripped_at': None, 'reset_at': None}
        self.instrument_states: Dict[str, Dict[str, Any]] = {}
        self.strategy_states: Dict[str, Dict[str, Any]] = {}
        self.default_reset_timeout = self.config.get(
            'default_reset_timeout_seconds', 300)
        self.enable_auto_reset = self.config_manager.get('enable_auto_reset', True)
        self.thresholds = self.config.get('thresholds', {'drawdown_percent':
            2.0, 'consecutive_losses': 3, 'volatility_multiple': 3.0,
            'slippage_pips': 5.0, 'error_count': 5})
        self.on_trip_callbacks: List[Callable[[str, Dict[str, Any]], None]] = [
            ]
        self.on_reset_callbacks: List[Callable[[str, Dict[str, Any]], None]
            ] = []
        self.auto_reset_thread = None
        if self.enable_auto_reset:
            self._start_auto_reset_thread()

    def register_callback(self, event_type: str, callback: Callable) ->None:
        """
        Register a callback function for circuit breaker events
        
        Args:
            event_type: "trip" or "reset"
            callback: Function to call with (circuit_id, state_info)
        """
        if event_type == 'trip':
            self.on_trip_callbacks.append(callback)
        elif event_type == 'reset':
            self.on_reset_callbacks.append(callback)
        else:
            logger.warning(f'Unknown event type: {event_type}')

    @with_exception_handling
    def trip_global(self, reason: str, reset_after: int=None) ->None:
        """
        Trip the global circuit breaker
        
        Args:
            reason: Reason for tripping the breaker
            reset_after: Seconds after which the breaker should auto-reset
        """
        self.global_state['state'] = self.STATE_OPEN
        self.global_state['reason'] = reason
        self.global_state['tripped_at'] = datetime.now()
        reset_seconds = reset_after or self.default_reset_timeout
        reset_time = datetime.now() + timedelta(seconds=reset_seconds)
        self.global_state['reset_at'] = reset_time
        logger.warning(
            f'GLOBAL CIRCUIT BREAKER TRIPPED: {reason}. Auto-reset at {reset_time}'
            )
        for callback in self.on_trip_callbacks:
            try:
                callback('global', self.global_state)
            except Exception as e:
                logger.error(f'Error in circuit breaker trip callback: {e}')

    @with_exception_handling
    def trip_instrument(self, instrument: str, reason: str, reset_after:
        int=None) ->None:
        """
        Trip the circuit breaker for a specific instrument
        
        Args:
            instrument: Instrument symbol
            reason: Reason for tripping the breaker
            reset_after: Seconds after which the breaker should auto-reset
        """
        if instrument not in self.instrument_states:
            self.instrument_states[instrument] = {'state': self.
                STATE_CLOSED, 'reason': None, 'tripped_at': None,
                'reset_at': None}
        state = self.instrument_states[instrument]
        state['state'] = self.STATE_OPEN
        state['reason'] = reason
        state['tripped_at'] = datetime.now()
        reset_seconds = reset_after or self.default_reset_timeout
        reset_time = datetime.now() + timedelta(seconds=reset_seconds)
        state['reset_at'] = reset_time
        logger.warning(
            f'CIRCUIT BREAKER TRIPPED for {instrument}: {reason}. Auto-reset at {reset_time}'
            )
        for callback in self.on_trip_callbacks:
            try:
                callback(f'instrument:{instrument}', state)
            except Exception as e:
                logger.error(f'Error in circuit breaker trip callback: {e}')

    @with_exception_handling
    def trip_strategy(self, strategy_id: str, reason: str, reset_after: int
        =None) ->None:
        """
        Trip the circuit breaker for a specific strategy
        
        Args:
            strategy_id: Strategy identifier
            reason: Reason for tripping the breaker
            reset_after: Seconds after which the breaker should auto-reset
        """
        if strategy_id not in self.strategy_states:
            self.strategy_states[strategy_id] = {'state': self.STATE_CLOSED,
                'reason': None, 'tripped_at': None, 'reset_at': None}
        state = self.strategy_states[strategy_id]
        state['state'] = self.STATE_OPEN
        state['reason'] = reason
        state['tripped_at'] = datetime.now()
        reset_seconds = reset_after or self.default_reset_timeout
        reset_time = datetime.now() + timedelta(seconds=reset_seconds)
        state['reset_at'] = reset_time
        logger.warning(
            f'CIRCUIT BREAKER TRIPPED for strategy {strategy_id}: {reason}. Auto-reset at {reset_time}'
            )
        for callback in self.on_trip_callbacks:
            try:
                callback(f'strategy:{strategy_id}', state)
            except Exception as e:
                logger.error(f'Error in circuit breaker trip callback: {e}')

    @with_exception_handling
    def reset_global(self) ->None:
        """Reset the global circuit breaker"""
        old_state = self.global_state.copy()
        self.global_state['state'] = self.STATE_CLOSED
        self.global_state['reason'] = None
        self.global_state['reset_at'] = None
        logger.info('GLOBAL CIRCUIT BREAKER RESET')
        for callback in self.on_reset_callbacks:
            try:
                callback('global', old_state)
            except Exception as e:
                logger.error(f'Error in circuit breaker reset callback: {e}')

    @with_exception_handling
    def reset_instrument(self, instrument: str) ->None:
        """
        Reset the circuit breaker for a specific instrument
        
        Args:
            instrument: Instrument symbol
        """
        if instrument not in self.instrument_states:
            return
        old_state = self.instrument_states[instrument].copy()
        self.instrument_states[instrument]['state'] = self.STATE_CLOSED
        self.instrument_states[instrument]['reason'] = None
        self.instrument_states[instrument]['reset_at'] = None
        logger.info(f'CIRCUIT BREAKER RESET for {instrument}')
        for callback in self.on_reset_callbacks:
            try:
                callback(f'instrument:{instrument}', old_state)
            except Exception as e:
                logger.error(f'Error in circuit breaker reset callback: {e}')

    @with_exception_handling
    def reset_strategy(self, strategy_id: str) ->None:
        """
        Reset the circuit breaker for a specific strategy
        
        Args:
            strategy_id: Strategy identifier
        """
        if strategy_id not in self.strategy_states:
            return
        old_state = self.strategy_states[strategy_id].copy()
        self.strategy_states[strategy_id]['state'] = self.STATE_CLOSED
        self.strategy_states[strategy_id]['reason'] = None
        self.strategy_states[strategy_id]['reset_at'] = None
        logger.info(f'CIRCUIT BREAKER RESET for strategy {strategy_id}')
        for callback in self.on_reset_callbacks:
            try:
                callback(f'strategy:{strategy_id}', old_state)
            except Exception as e:
                logger.error(f'Error in circuit breaker reset callback: {e}')

    def is_open(self, instrument: str=None, strategy_id: str=None) ->bool:
        """
        Check if a circuit breaker is open (tripped)
        
        Args:
            instrument: Instrument to check (None to ignore)
            strategy_id: Strategy to check (None to ignore)
            
        Returns:
            True if any applicable breaker is open
        """
        if self.global_state['state'] == self.STATE_OPEN:
            return True
        if instrument is not None and instrument in self.instrument_states:
            if self.instrument_states[instrument]['state'] == self.STATE_OPEN:
                return True
        if strategy_id is not None and strategy_id in self.strategy_states:
            if self.strategy_states[strategy_id]['state'] == self.STATE_OPEN:
                return True
        return False

    def get_status(self) ->Dict[str, Any]:
        """
        Get the current status of all circuit breakers
        
        Returns:
            Dictionary with circuit breaker status
        """
        return {'global': self.global_state.copy(), 'instruments': {k: v.
            copy() for k, v in self.instrument_states.items()},
            'strategies': {k: v.copy() for k, v in self.strategy_states.
            items()}}

    def evaluate_conditions(self, performance_metrics: Dict[str, Any]=None,
        market_data: Dict[str, Any]=None, execution_metrics: Dict[str, Any]
        =None, error_counts: Dict[str, int]=None) ->List[Dict[str, Any]]:
        """
        Evaluate conditions that might trigger circuit breakers
        
        Args:
            performance_metrics: Trading performance metrics
            market_data: Current market data
            execution_metrics: Order execution metrics
            error_counts: Counts of various errors
            
        Returns:
            List of triggered circuit breakers
        """
        triggered = []
        if performance_metrics and 'drawdown_percent' in performance_metrics:
            if performance_metrics['drawdown_percent'] >= self.thresholds[
                'drawdown_percent']:
                self.trip_global(
                    f"Daily drawdown exceeded: {performance_metrics['drawdown_percent']:.2f}%"
                    )
                triggered.append({'type': 'global', 'reason':
                    'drawdown_exceeded', 'threshold': self.thresholds[
                    'drawdown_percent'], 'actual': performance_metrics[
                    'drawdown_percent']})
        if performance_metrics and 'consecutive_losses' in performance_metrics:
            if performance_metrics['consecutive_losses'] >= self.thresholds[
                'consecutive_losses']:
                self.trip_global(
                    f"Consecutive losses threshold reached: {performance_metrics['consecutive_losses']}"
                    )
                triggered.append({'type': 'global', 'reason':
                    'consecutive_losses', 'threshold': self.thresholds[
                    'consecutive_losses'], 'actual': performance_metrics[
                    'consecutive_losses']})
        if market_data and 'instruments' in market_data:
            for instrument, data in market_data['instruments'].items():
                if 'atr' in data and 'atr_multiple' in data:
                    if data['atr_multiple'] >= self.thresholds[
                        'volatility_multiple']:
                        self.trip_instrument(instrument,
                            f"Excessive volatility: {data['atr_multiple']:.2f}x normal"
                            )
                        triggered.append({'type': 'instrument',
                            'instrument': instrument, 'reason':
                            'excessive_volatility', 'threshold': self.
                            thresholds['volatility_multiple'], 'actual':
                            data['atr_multiple']})
        if execution_metrics and 'slippage' in execution_metrics:
            for instrument, slippage in execution_metrics['slippage'].items():
                if slippage >= self.thresholds['slippage_pips']:
                    self.trip_instrument(instrument,
                        f'Excessive slippage: {slippage:.2f} pips')
                    triggered.append({'type': 'instrument', 'instrument':
                        instrument, 'reason': 'excessive_slippage',
                        'threshold': self.thresholds['slippage_pips'],
                        'actual': slippage})
        if error_counts:
            for error_type, count in error_counts.items():
                if count >= self.thresholds['error_count']:
                    self.trip_global(f'Excessive {error_type} errors: {count}')
                    triggered.append({'type': 'global', 'reason':
                        f'{error_type}_errors', 'threshold': self.
                        thresholds['error_count'], 'actual': count})
        return triggered

    def _start_auto_reset_thread(self) ->None:
        """Start the auto-reset thread"""
        if (self.auto_reset_thread is not None and self.auto_reset_thread.
            is_alive()):
            return
        self.auto_reset_thread = threading.Thread(target=self.
            _auto_reset_worker, daemon=True)
        self.auto_reset_thread.start()
        logger.info('Auto-reset thread started')

    @with_exception_handling
    def _auto_reset_worker(self) ->None:
        """Worker thread for automatic circuit breaker resets"""
        while True:
            try:
                now = datetime.now()
                if self.global_state['state'
                    ] == self.STATE_OPEN and self.global_state['reset_at'
                    ] is not None and now >= self.global_state['reset_at']:
                    self.reset_global()
                for instrument, state in list(self.instrument_states.items()):
                    if state['state'] == self.STATE_OPEN and state['reset_at'
                        ] is not None and now >= state['reset_at']:
                        self.reset_instrument(instrument)
                for strategy_id, state in list(self.strategy_states.items()):
                    if state['state'] == self.STATE_OPEN and state['reset_at'
                        ] is not None and now >= state['reset_at']:
                        self.reset_strategy(strategy_id)
            except Exception as e:
                logger.error(f'Error in auto-reset worker: {e}')
            time.sleep(1)
