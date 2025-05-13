"""
AdvancedTAStrategy

This module provides a base class for advanced technical analysis-based trading strategies,
integrating signal confluence, harmonic patterns, and market regime awareness.

Part of Phase 4 implementation to enhance the adaptive trading capabilities.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from analysis_engine.adaptive_layer.confluence_analyzer import ConfluenceAnalyzer
from common_lib.effectiveness.interfaces import MarketRegimeEnum as MarketRegime
from analysis_engine.analysis.technical_indicators import TechnicalIndicators
from analysis_engine.services.timeframe_optimization_service import TimeframeOptimizationService, SignalOutcome


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AdvancedTAStrategy(ABC):
    """
    Base class for advanced technical analysis strategies that leverage
    confluence detection, harmonic patterns, and adaptive parameters.
    """

    def __init__(self, name: str, timeframes: List[str], primary_timeframe:
        str, symbols: List[str], risk_per_trade_pct: float=1.0,
        confluence_analyzer: Optional[ConfluenceAnalyzer]=None,
        technical_indicators: Optional[TechnicalIndicators]=None,
        timeframe_optimizer: Optional[TimeframeOptimizationService]=None):
        """
        Initialize the advanced TA strategy.

        Args:
            name: Strategy name
            timeframes: List of timeframes to analyze
            primary_timeframe: Primary timeframe for signal generation
            symbols: List of trading symbols
            risk_per_trade_pct: Risk per trade as percentage of account
            confluence_analyzer: Optional configured confluence analyzer
            technical_indicators: Optional technical indicator service
            timeframe_optimizer: Optional timeframe optimization service
        """
        self.name = name
        self.timeframes = timeframes
        self.primary_timeframe = primary_timeframe
        self.symbols = symbols
        self.risk_per_trade_pct = risk_per_trade_pct
        self.logger = logging.getLogger(f'strategy.{self.name}')
        self.confluence_analyzer = confluence_analyzer or ConfluenceAnalyzer()
        self.technical_indicators = (technical_indicators or
            TechnicalIndicators())
        self.timeframe_optimizer = (timeframe_optimizer or
            TimeframeOptimizationService(timeframes=timeframes,
            primary_timeframe=primary_timeframe))
        self.active_signals = {}
        self.market_regime = None
        self.performance_metrics = {'win_count': 0, 'loss_count': 0,
            'total_pips': 0, 'win_rate': 0.0, 'avg_win_pips': 0.0,
            'avg_loss_pips': 0.0, 'profit_factor': 0.0}
        self.config = {'min_confluence_score': 0.7, 'signal_expiry_hours': 
            24, 'confirmation_required': True, 'filter_by_market_regime': 
            True, 'adaptive_parameters': True, 'max_active_signals': 3,
            'use_timeframe_optimization': True,
            'optimize_timeframes_interval_hours': 24}
        self.adaptive_params = {}
        self.last_timeframe_optimization = datetime.now() - timedelta(hours=25)
        self._init_strategy_config()

    @abstractmethod
    def _init_strategy_config(self) ->None:
        """
        Initialize strategy-specific configuration parameters.
        This must be implemented by each concrete strategy.
        """
        pass

    def update_market_regime(self, symbol: str, regime: MarketRegime) ->None:
        """Update the market regime and adjust parameters accordingly."""
        if self.market_regime != regime:
            self.logger.info(
                f'Market regime for {symbol} changed from {self.market_regime} to {regime}'
                )
            self.market_regime = regime
            if self.config['adaptive_parameters']:
                self._adapt_parameters_to_regime(regime)

    @abstractmethod
    def _adapt_parameters_to_regime(self, regime: MarketRegime) ->None:
        """
        Adjust strategy parameters based on the current market regime.
        This must be implemented by each concrete strategy.
        """
        pass

    @with_exception_handling
    def analyze_market(self, symbol: str, price_data: Dict[str, pd.
        DataFrame], additional_data: Optional[Dict[str, Any]]=None) ->Dict[
        str, Any]:
        """
        Analyze market data and generate trading signals.

        Args:
            symbol: The trading symbol
            price_data: Dict of price data frames indexed by timeframe
            additional_data: Optional additional data for analysis

        Returns:
            Dictionary with analysis results and signals
        """
        self.logger.info(f'Analyzing market for {symbol}')
        try:
            optimized_weights = self._check_and_optimize_timeframes()
            primary_data = price_data.get(self.primary_timeframe)
            if primary_data is None or primary_data.empty:
                return {'error':
                    f'No data available for {symbol} on {self.primary_timeframe} timeframe'
                    }
            additional_timeframes = [tf for tf in self.timeframes if tf !=
                self.primary_timeframe]
            confluence_results = self.confluence_analyzer.analyze_confluence(
                symbol=symbol, primary_timeframe=self.primary_timeframe,
                price_data=primary_data, additional_timeframes=
                additional_timeframes, current_regime=self.market_regime,
                timeframe_weights=optimized_weights if self.config.get(
                'use_timeframe_optimization', True) else None)
            overall_confluence = confluence_results.get('confluence_score', 0)
            if overall_confluence < self.config['min_confluence_score']:
                self.logger.info(
                    f"Insufficient confluence for {symbol}: {overall_confluence:.2f} < {self.config['min_confluence_score']}"
                    )
                return {'symbol': symbol, 'timestamp': datetime.now().
                    isoformat(), 'confluence_score': overall_confluence,
                    'signals': [], 'message':
                    'Insufficient confluence for signal generation'}
            strategy_analysis = self._perform_strategy_analysis(symbol=
                symbol, price_data=price_data, confluence_results=
                confluence_results, additional_data=additional_data)
            if 'timeframe_scores' in strategy_analysis and self.config.get(
                'use_timeframe_optimization', True):
                strategy_analysis['original_timeframe_scores'
                    ] = strategy_analysis['timeframe_scores'].copy()
                strategy_analysis['timeframe_scores'
                    ] = self.apply_timeframe_weights(strategy_analysis[
                    'timeframe_scores'])
                strategy_analysis['timeframe_weights_applied'] = True
            signals = self._generate_signals(symbol=symbol,
                strategy_analysis=strategy_analysis, confluence_results=
                confluence_results)
            filtered_signals = self._apply_signal_filters(signals)
            self._update_active_signals(symbol, filtered_signals)
            result = {'symbol': symbol, 'timestamp': datetime.now().
                isoformat(), 'confluence_score': overall_confluence,
                'market_regime': str(self.market_regime) if self.
                market_regime else 'unknown', 'signals': filtered_signals,
                'analysis': strategy_analysis, 'confluence_details': {
                'sr_confluence': len(confluence_results.get(
                'support_resistance_confluence', [])),
                'indicator_confluence': bool(confluence_results.get(
                'indicator_confluences', [])), 'mtf_confluence': len(
                confluence_results.get('multi_timeframe_confluences', [])),
                'harmonic_patterns': len(confluence_results.get(
                'harmonic_patterns', []))}}
            if self.config_manager.get('use_timeframe_optimization', True):
                result['timeframe_optimization'] = {'weights':
                    optimized_weights, 'last_optimization': self.
                    last_timeframe_optimization.isoformat()}
            return result
        except Exception as e:
            self.logger.error(f'Error analyzing market for {symbol}: {str(e)}',
                exc_info=True)
            return {'symbol': symbol, 'error': str(e), 'signals': []}

    @abstractmethod
    def _perform_strategy_analysis(self, symbol: str, price_data: Dict[str,
        pd.DataFrame], confluence_results: Dict[str, Any], additional_data:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Perform strategy-specific market analysis.
        This must be implemented by each concrete strategy.

        Args:
            symbol: The trading symbol
            price_data: Dict of price data frames indexed by timeframe
            confluence_results: Results from confluence analysis
            additional_data: Optional additional data

        Returns:
            Dictionary with strategy-specific analysis results
        """
        pass

    @abstractmethod
    def _generate_signals(self, symbol: str, strategy_analysis: Dict[str,
        Any], confluence_results: Dict[str, Any]) ->List[Dict[str, Any]]:
        """
        Generate trading signals based on strategy analysis.
        This must be implemented by each concrete strategy.

        Args:
            symbol: The trading symbol
            strategy_analysis: Results from strategy-specific analysis
            confluence_results: Results from confluence analysis

        Returns:
            List of trading signals
        """
        pass

    def _apply_signal_filters(self, signals: List[Dict[str, Any]]) ->List[Dict
        [str, Any]]:
        """Apply configured filters to the generated signals."""
        if not signals:
            return []
        filtered_signals = signals.copy()
        if self.config['filter_by_market_regime'] and self.market_regime:
            filtered_signals = self._filter_by_market_regime(filtered_signals)
        if self.config['confirmation_required']:
            filtered_signals = self._apply_confirmation_requirements(
                filtered_signals)
        symbols_count = {}
        limited_signals = []
        for signal in filtered_signals:
            symbol = signal['symbol']
            symbols_count[symbol] = symbols_count.get(symbol, 0) + 1
            if symbols_count[symbol] <= self.config['max_active_signals']:
                limited_signals.append(signal)
        for signal in limited_signals:
            if 'expiry' not in signal:
                signal['expiry'] = (datetime.now() + timedelta(hours=self.
                    config['signal_expiry_hours'])).isoformat()
        return limited_signals

    def _filter_by_market_regime(self, signals: List[Dict[str, Any]]) ->List[
        Dict[str, Any]]:
        """Filter signals based on compatibility with current market regime."""
        if not self.market_regime:
            return signals
        filtered = []
        for signal in signals:
            if self.market_regime == MarketRegime.TRENDING:
                if signal.get('type') in ['trend_following', 'breakout']:
                    filtered.append(signal)
            elif self.market_regime == MarketRegime.RANGING:
                if signal.get('type') in ['range_bound', 'reversal']:
                    filtered.append(signal)
            elif self.market_regime == MarketRegime.VOLATILE:
                if signal.get('confidence', 0) >= 0.8:
                    filtered.append(signal)
            else:
                filtered.append(signal)
        return filtered

    def _apply_confirmation_requirements(self, signals: List[Dict[str, Any]]
        ) ->List[Dict[str, Any]]:
        """Apply additional confirmation requirements to signals."""
        return signals

    def _update_active_signals(self, symbol: str, new_signals: List[Dict[
        str, Any]]) ->None:
        """Update the set of active signals for a symbol."""
        if symbol not in self.active_signals:
            self.active_signals[symbol] = []
        current_time = datetime.now()
        self.active_signals[symbol] = [signal for signal in self.
            active_signals[symbol] if datetime.fromisoformat(signal.get(
            'expiry', '2000-01-01T00:00:00')) > current_time]
        for signal in new_signals:
            similar_exists = False
            for existing in self.active_signals[symbol]:
                if existing['direction'] == signal['direction'] and existing[
                    'type'] == signal['type']:
                    similar_exists = True
                    break
            if not similar_exists:
                self.active_signals[symbol].append(signal)

    def backtest(self, symbol: str, historical_data: Dict[str, pd.DataFrame
        ], start_date: Optional[str]=None, end_date: Optional[str]=None,
        initial_balance: float=10000.0) ->Dict[str, Any]:
        """
        Run a backtest of the strategy on historical data.

        Args:
            symbol: The symbol to backtest
            historical_data: Dict of historical price data by timeframe
            start_date: Optional start date for backtest (ISO format)
            end_date: Optional end date for backtest (ISO format)
            initial_balance: Initial balance for the backtest

        Returns:
            Dictionary with backtest results
        """
        self.logger.info(
            f'Starting backtest for {symbol} from {start_date} to {end_date}')
        return {'symbol': symbol, 'start_date': start_date, 'end_date':
            end_date, 'initial_balance': initial_balance, 'final_balance':
            initial_balance, 'total_trades': 0, 'win_rate': 0.0,
            'profit_factor': 0.0, 'max_drawdown': 0.0, 'trades': []}

    def update_performance_metrics(self, trade_result: Dict[str, Any]) ->None:
        """Update strategy performance metrics based on a completed trade."""
        if trade_result.get('result') == 'win':
            self.performance_metrics['win_count'] += 1
            pips = trade_result.get('profit_pips', 0)
        else:
            self.performance_metrics['loss_count'] += 1
            pips = -trade_result.get('loss_pips', 0)
        total_trades = self.performance_metrics['win_count'
            ] + self.performance_metrics['loss_count']
        self.performance_metrics['total_pips'] += pips
        if total_trades > 0:
            self.performance_metrics['win_rate'] = self.performance_metrics[
                'win_count'] / total_trades
        self.logger.info(
            f"Updated performance metrics: Win rate = {self.performance_metrics['win_rate']:.2f}, Total pips = {self.performance_metrics['total_pips']}"
            )

    def _check_and_optimize_timeframes(self) ->Dict[str, float]:
        """
        Check if timeframe optimization should be performed and run it if needed.

        Returns:
            Dictionary of optimized timeframe weights
        """
        if not self.config_manager.get('use_timeframe_optimization', True):
            return {tf: (1.0) for tf in self.timeframes}
        now = datetime.now()
        hours_since_optimization = (now - self.last_timeframe_optimization
            ).total_seconds() / 3600
        if hours_since_optimization >= self.config.get(
            'optimize_timeframes_interval_hours', 24):
            self.logger.info(f'Running timeframe optimization for {self.name}')
            weights = self.timeframe_optimizer.optimize_timeframe_weights()
            self.last_timeframe_optimization = now
            return weights
        else:
            return self.timeframe_optimizer.get_timeframe_weights()

    def apply_timeframe_weights(self, timeframe_scores: Dict[str, float]
        ) ->Dict[str, float]:
        """
        Apply optimized weights to timeframe scores.

        Args:
            timeframe_scores: Dictionary of scores by timeframe

        Returns:
            Dictionary of weighted scores by timeframe
        """
        if not self.config_manager.get('use_timeframe_optimization', True):
            return timeframe_scores
        weighted_avg, weighted_scores = (self.timeframe_optimizer.
            apply_weighted_score(timeframe_scores))
        self.logger.debug(
            f'Applied timeframe weights: original={timeframe_scores}, weighted={weighted_scores}'
            )
        return weighted_scores

    def record_signal_outcome(self, symbol: str, signal_id: str, outcome:
        str, pips_result: float, exit_reason: str, timeframe_data: Dict[str,
        float]=None) ->None:
        """
        Record the outcome of a signal for performance tracking and timeframe optimization.

        Args:
            symbol: The trading symbol
            signal_id: Unique identifier for the signal
            outcome: 'win', 'loss', or 'breakeven'
            pips_result: Profit/loss in pips
            exit_reason: Reason for trade exit (tp, sl, manual, etc.)
            timeframe_data: Optional dict of timeframe scores that generated this signal
        """
        signal_outcome = SignalOutcome.UNKNOWN
        if outcome.lower() == 'win':
            signal_outcome = SignalOutcome.WIN
            self.performance_metrics['win_count'] += 1
        elif outcome.lower() == 'loss':
            signal_outcome = SignalOutcome.LOSS
            self.performance_metrics['loss_count'] += 1
        elif outcome.lower() == 'breakeven':
            signal_outcome = SignalOutcome.BREAKEVEN
        self.performance_metrics['total_pips'] += pips_result
        total_trades = self.performance_metrics['win_count'
            ] + self.performance_metrics['loss_count']
        if total_trades > 0:
            self.performance_metrics['win_rate'] = self.performance_metrics[
                'win_count'] / total_trades
        if timeframe_data and self.config.get('use_timeframe_optimization',
            True):
            confidence = 0.75
            if (symbol in self.active_signals and signal_id in self.
                active_signals[symbol]):
                signal = self.active_signals[symbol][signal_id]
                confidence = signal.get('confidence', 0.75)
            for tf, score in timeframe_data.items():
                if tf in self.timeframes:
                    self.timeframe_optimizer.record_timeframe_performance(
                        timeframe=tf, outcome=signal_outcome, symbol=symbol,
                        pips_result=pips_result, confidence=confidence)
                    self.logger.debug(
                        f'Recorded {outcome} outcome for timeframe {tf} on {symbol}'
                        )
        if symbol in self.active_signals and signal_id in self.active_signals[
            symbol]:
            del self.active_signals[symbol][signal_id]
        self.logger.info(
            f'Signal {signal_id} on {symbol} completed with {outcome}, {pips_result} pips, exit: {exit_reason}'
            )
