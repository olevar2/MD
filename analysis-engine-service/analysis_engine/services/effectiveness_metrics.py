"""
Effectiveness Metrics Module

This module contains implementations of effectiveness metrics for technical analysis tools,
with a focus on market regime reliability and adaptive signal weighting.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Set
from collections import defaultdict
from datetime import datetime
from analysis_engine.services.tool_effectiveness import SignalEvent, SignalOutcome, TimeFrame, MarketRegime, ToolEffectivenessMetric


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ExpectedPayoffMetric(ToolEffectivenessMetric):
    """Calculates the expected payoff of signals"""

    def __init__(self):
    """
      init  .
    
    """

        super().__init__(name='expected_payoff', description=
            'Average profit/loss per signal')

    @with_exception_handling
    def calculate(self, outcomes: List[SignalOutcome], market_regime:
        Optional[MarketRegime]=None, timeframe: Optional[TimeFrame]=None,
        symbol: Optional[str]=None, start_date: Optional[datetime]=None,
        end_date: Optional[datetime]=None) ->Dict[str, Any]:
        """
        Calculate expected payoff based on signal outcomes
        
        Args:
            outcomes: List of signal outcomes to evaluate
            market_regime: Optional filter by market regime
            timeframe: Optional filter by timeframe
            symbol: Optional filter by trading symbol
            start_date: Optional filter by start date
            end_date: Optional filter by end date
            
        Returns:
            Dictionary with expected payoff value
        """
        try:
            filtered_outcomes = outcomes
            if market_regime:
                filtered_outcomes = [o for o in filtered_outcomes if o.
                    signal_event.market_context.get('regime') ==
                    market_regime.value]
            if timeframe:
                filtered_outcomes = [o for o in filtered_outcomes if o.
                    signal_event.timeframe == timeframe.value]
            if symbol:
                filtered_outcomes = [o for o in filtered_outcomes if o.
                    signal_event.symbol == symbol]
            if start_date:
                filtered_outcomes = [o for o in filtered_outcomes if o.
                    signal_event.timestamp >= start_date]
            if end_date:
                filtered_outcomes = [o for o in filtered_outcomes if o.
                    signal_event.timestamp <= end_date]
            if not filtered_outcomes:
                return {'metric': 'expected_payoff', 'value': None,
                    'sample_size': 0}
            total_pnl = sum(o.profit_loss for o in filtered_outcomes if o.
                profit_loss is not None)
            valid_count = sum(1 for o in filtered_outcomes if o.profit_loss
                 is not None)
            if valid_count == 0:
                return {'metric': 'expected_payoff', 'value': None,
                    'sample_size': 0}
            expected_payoff = total_pnl / valid_count
            return {'metric': 'expected_payoff', 'value': expected_payoff,
                'sample_size': valid_count}
        except Exception as e:
            self.logger.error(f'Error calculating expected payoff: {str(e)}')
            return {'metric': 'expected_payoff', 'value': None, 'error':
                str(e), 'sample_size': 0}


class ReliabilityByMarketRegimeMetric(ToolEffectivenessMetric):
    """Measures how consistently a tool performs across different market regimes"""

    def __init__(self):
    """
      init  .
    
    """

        super().__init__(name='reliability_by_regime', description=
            'Consistency of tool performance across different market regimes')

    @with_exception_handling
    def calculate(self, outcomes: List[SignalOutcome], timeframe: Optional[
        TimeFrame]=None, symbol: Optional[str]=None, start_date: Optional[
        datetime]=None, end_date: Optional[datetime]=None) ->Dict[str, Any]:
        """
        Calculate reliability across different market regimes
        
        Args:
            outcomes: List of signal outcomes to evaluate
            timeframe: Optional filter by timeframe
            symbol: Optional filter by trading symbol
            start_date: Optional filter by start date
            end_date: Optional filter by end date
            
        Returns:
            Dictionary with regime reliability metrics
        """
        try:
            filtered_outcomes = outcomes
            if timeframe:
                filtered_outcomes = [o for o in filtered_outcomes if o.
                    signal_event.timeframe == timeframe.value]
            if symbol:
                filtered_outcomes = [o for o in filtered_outcomes if o.
                    signal_event.symbol == symbol]
            if start_date:
                filtered_outcomes = [o for o in filtered_outcomes if o.
                    signal_event.timestamp >= start_date]
            if end_date:
                filtered_outcomes = [o for o in filtered_outcomes if o.
                    signal_event.timestamp <= end_date]
            regimes = {}
            for outcome in filtered_outcomes:
                regime = outcome.signal_event.market_context.get('regime',
                    MarketRegime.UNKNOWN.value)
                if regime not in regimes:
                    regimes[regime] = []
                regimes[regime].append(outcome)
            regime_win_rates = {}
            MIN_SAMPLES = 5
            for regime, regime_outcomes in regimes.items():
                if len(regime_outcomes) >= MIN_SAMPLES:
                    successful = sum(1 for o in regime_outcomes if o.success)
                    win_rate = successful / len(regime_outcomes)
                    regime_win_rates[regime] = {'win_rate': win_rate,
                        'sample_size': len(regime_outcomes)}
            if not regime_win_rates:
                return {'metric': 'reliability_by_regime', 'value': None,
                    'sample_size': 0, 'regime_data': {}}
            win_rates = [data['win_rate'] for data in regime_win_rates.values()
                ]
            if len(win_rates) < 2:
                reliability_score = None
            else:
                mean_win_rate = np.mean(win_rates)
                std_dev = np.std(win_rates)
                if mean_win_rate > 0:
                    coeff_of_variation = std_dev / mean_win_rate
                    reliability_score = max(0, min(1, 1 - coeff_of_variation))
                else:
                    reliability_score = 0.0
            return {'metric': 'reliability_by_regime', 'value':
                reliability_score, 'sample_size': len(filtered_outcomes),
                'regime_count': len(regime_win_rates), 'regime_data':
                regime_win_rates}
        except Exception as e:
            self.logger.error(f'Error calculating regime reliability: {str(e)}'
                )
            return {'metric': 'reliability_by_regime', 'value': None,
                'error': str(e), 'sample_size': 0, 'regime_data': {}}


class MarketRegimeSpecificPerformanceMetric(ToolEffectivenessMetric):
    """Analyzes performance for a specific market regime"""

    def __init__(self, target_regime: MarketRegime):
    """
      init  .
    
    Args:
        target_regime: Description of target_regime
    
    """

        super().__init__(name=f'regime_{target_regime.value}_performance',
            description=
            f'Performance metrics specific to {target_regime.value} market regime'
            )
        self.target_regime = target_regime

    @with_exception_handling
    def calculate(self, outcomes: List[SignalOutcome], timeframe: Optional[
        TimeFrame]=None, symbol: Optional[str]=None, start_date: Optional[
        datetime]=None, end_date: Optional[datetime]=None) ->Dict[str, Any]:
        """
        Calculate performance metrics for a specific market regime
        
        Args:
            outcomes: List of signal outcomes to evaluate
            timeframe: Optional filter by timeframe
            symbol: Optional filter by trading symbol
            start_date: Optional filter by start date
            end_date: Optional filter by end date
            
        Returns:
            Dictionary with performance metrics for the target regime
        """
        try:
            regime_outcomes = [o for o in outcomes if o.signal_event.
                market_context.get('regime') == self.target_regime.value]
            if timeframe:
                regime_outcomes = [o for o in regime_outcomes if o.
                    signal_event.timeframe == timeframe.value]
            if symbol:
                regime_outcomes = [o for o in regime_outcomes if o.
                    signal_event.symbol == symbol]
            if start_date:
                regime_outcomes = [o for o in regime_outcomes if o.
                    signal_event.timestamp >= start_date]
            if end_date:
                regime_outcomes = [o for o in regime_outcomes if o.
                    signal_event.timestamp <= end_date]
            if not regime_outcomes:
                return {'metric': self.name, 'regime': self.target_regime.
                    value, 'value': None, 'sample_size': 0, 'details': {}}
            total_count = len(regime_outcomes)
            successful = sum(1 for o in regime_outcomes if o.success)
            win_rate = successful / total_count if total_count > 0 else 0
            winning_trades = [o.profit_loss for o in regime_outcomes if o.
                success and o.profit_loss is not None]
            losing_trades = [o.profit_loss for o in regime_outcomes if not
                o.success and o.profit_loss is not None]
            total_profit = sum(winning_trades) if winning_trades else 0
            total_loss = abs(sum(losing_trades)) if losing_trades else 0
            profit_factor = (total_profit / total_loss if total_loss > 0 else
                float('inf'))
            valid_pnl_trades = [o.profit_loss for o in regime_outcomes if o
                .profit_loss is not None]
            avg_profit = sum(valid_pnl_trades) / len(valid_pnl_trades
                ) if valid_pnl_trades else 0
            return {'metric': self.name, 'regime': self.target_regime.value,
                'value': win_rate, 'sample_size': total_count, 'details': {
                'win_rate': win_rate, 'profit_factor': profit_factor,
                'avg_profit': avg_profit, 'total_profit': sum(
                valid_pnl_trades) if valid_pnl_trades else 0}}
        except Exception as e:
            self.logger.error(
                f'Error calculating regime-specific performance: {str(e)}')
            return {'metric': self.name, 'regime': self.target_regime.value,
                'value': None, 'error': str(e), 'sample_size': 0, 'details': {}
                }
