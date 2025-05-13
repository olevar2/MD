"""
Implementation of signal flow management in the Strategy Execution Engine.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio
from uuid import uuid4
from common_lib.signal_flow.interface import ISignalFlowManager, ISignalMonitor, ISignalValidator, ISignalAggregator, ISignalExecutor
from common_lib.signal_flow.model import SignalFlow, SignalFlowState, SignalValidationResult, SignalAggregationResult, SignalCategory, SignalStrength, SignalPriority


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class StrategySignalManager:
    """
    Manages the flow of trading signals in the Strategy Execution Engine.
    
    This component:
    1. Receives signals from the Analysis Engine
    2. Validates signals against current market conditions
    3. Aggregates multiple signals when needed
    4. Manages execution state and monitoring
    5. Provides feedback to the Analysis Engine
    """

    def __init__(self, signal_flow_manager: ISignalFlowManager,
        signal_validator: ISignalValidator, signal_aggregator:
        ISignalAggregator, signal_executor: ISignalExecutor, signal_monitor:
        ISignalMonitor):
    """
      init  .
    
    Args:
        signal_flow_manager: Description of signal_flow_manager
        signal_validator: Description of signal_validator
        signal_aggregator: Description of signal_aggregator
        signal_executor: Description of signal_executor
        signal_monitor: Description of signal_monitor
    
    """

        self.flow_manager = signal_flow_manager
        self.validator = signal_validator
        self.aggregator = signal_aggregator
        self.executor = signal_executor
        self.monitor = signal_monitor
        self.logger = logging.getLogger(__name__)
        self.active_signals: Dict[str, SignalFlow] = {}
        self.signal_states: Dict[str, SignalFlowState] = {}
        self.signal_lock = asyncio.Lock()

    @async_with_exception_handling
    async def process_new_signal(self, signal: SignalFlow) ->bool:
        """
        Process a new signal received from the Analysis Engine.
        
        Args:
            signal: The trading signal to process
            
        Returns:
            bool: Whether the signal was successfully processed
        """
        try:
            if signal.signal_id in self.active_signals:
                self.logger.warning(
                    f'Duplicate signal received: {signal.signal_id}')
                return False
            validation_result = await self.validator.validate_signal(signal)
            if not validation_result.is_valid:
                await self._handle_invalid_signal(signal, validation_result)
                return False
            async with self.signal_lock:
                self.active_signals[signal.signal_id] = signal
                self.signal_states[signal.signal_id
                    ] = SignalFlowState.VALIDATED
            related_signals = await self._get_related_signals(signal)
            if related_signals:
                aggregation_result = await self.aggregator.aggregate_signals(
                    signals=[signal] + related_signals, symbol=signal.
                    symbol, timeframe=signal.timeframe)
                await self._handle_aggregation_result(signal,
                    aggregation_result)
            execution_success = await self.executor.execute_signal(signal)
            if execution_success:
                await self._update_signal_state(signal.signal_id,
                    SignalFlowState.EXECUTING)
                self.logger.info(
                    f'Successfully executed signal {signal.signal_id}')
            else:
                await self._update_signal_state(signal.signal_id,
                    SignalFlowState.REJECTED)
                self.logger.error(
                    f'Failed to execute signal {signal.signal_id}')
            return execution_success
        except Exception as e:
            self.logger.error(
                f'Error processing signal {signal.signal_id}: {str(e)}',
                exc_info=True)
            await self._update_signal_state(signal.signal_id,
                SignalFlowState.REJECTED)
            return False

    @async_with_exception_handling
    async def cancel_signal(self, signal_id: str) ->bool:
        """Cancel an active signal"""
        try:
            if signal_id not in self.active_signals:
                return False
            async with self.signal_lock:
                signal = self.active_signals[signal_id]
                current_state = self.signal_states[signal_id]
                if current_state == SignalFlowState.EXECUTING:
                    if await self.executor.cancel_signal(signal_id):
                        await self._update_signal_state(signal_id,
                            SignalFlowState.REJECTED)
                        return True
                else:
                    await self._update_signal_state(signal_id,
                        SignalFlowState.REJECTED)
                    return True
            return False
        except Exception as e:
            self.logger.error(f'Error cancelling signal {signal_id}: {str(e)}')
            return False

    @async_with_exception_handling
    async def get_signal_status(self, signal_id: str) ->Optional[Dict[str, Any]
        ]:
        """Get current status of a signal"""
        try:
            if signal_id not in self.active_signals:
                return None
            signal = self.active_signals[signal_id]
            state = self.signal_states[signal_id]
            metrics = await self.monitor.get_signal_metrics(signal_id)
            return {'signal_id': signal_id, 'symbol': signal.symbol,
                'timeframe': signal.timeframe, 'direction': signal.
                direction, 'state': state.value, 'metrics': metrics}
        except Exception as e:
            self.logger.error(f'Error getting signal status: {str(e)}')
            return None

    async def _handle_invalid_signal(self, signal: SignalFlow,
        validation_result: SignalValidationResult) ->None:
        """Handle an invalid signal"""
        self.logger.warning(
            f'Invalid signal {signal.signal_id}: Failed checks: {[k for k, v in validation_result.validation_checks.items() if not v]}'
            )
        await self._update_signal_state(signal.signal_id, SignalFlowState.
            REJECTED, metadata={'validation_checks': validation_result.
            validation_checks, 'risk_metrics': validation_result.
            risk_metrics, 'notes': validation_result.notes})

    async def _handle_aggregation_result(self, signal: SignalFlow,
        aggregation_result: SignalAggregationResult) ->None:
        """Handle signal aggregation results"""
        await self.monitor.update_signal_metrics(signal.signal_id, {
            'aggregated_direction': aggregation_result.aggregated_direction,
            'aggregated_confidence': aggregation_result.
            aggregated_confidence, 'contributing_signals':
            aggregation_result.contributing_signals, 'weights_used':
            aggregation_result.weights_used})

    async def _get_related_signals(self, signal: SignalFlow) ->List[SignalFlow
        ]:
        """Get other active signals for the same symbol/timeframe"""
        return [s for s in self.active_signals.values() if s.symbol ==
            signal.symbol and s.timeframe == signal.timeframe and s.
            signal_id != signal.signal_id and self.signal_states[s.
            signal_id] in [SignalFlowState.VALIDATED, SignalFlowState.
            AGGREGATED]]

    async def _update_signal_state(self, signal_id: str, state:
        SignalFlowState, metadata: Dict[str, Any]=None) ->None:
        """Update signal state and notify the flow manager"""
        async with self.signal_lock:
            self.signal_states[signal_id] = state
            await self.flow_manager.update_signal_state(signal_id, state, 
                metadata or {})
