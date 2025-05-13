"""
News Event Backtester for Forex Trading Platform.

This module provides a backtesting framework that incorporates historical news events
to evaluate trading strategies under realistic market conditions during economic releases,
central bank decisions, and other significant news events.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import json
import os
import uuid
import logging
from trading_gateway_service.simulation.news_sentiment_simulator import NewsAndSentimentSimulator, NewsEvent, NewsImpactLevel, NewsEventType, SentimentLevel
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
from trading_gateway_service.error.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class NewsEventBacktester:
    """
    Backtester that incorporates news events and their market impact.
    
    This class extends traditional backtesting by integrating historical news events 
    and their impact on market prices, volatility, slippage, and gaps.
    """

    def __init__(self, historical_data: Dict[str, pd.DataFrame],
        initial_balance: float=10000.0, commission: float=0.0,
        base_slippage: float=0.0, base_spread: float=0.0, seed: Optional[
        int]=None, backtest_id: Optional[str]=None):
        """
        Initialize the news event backtester.
        
        Args:
            historical_data: Dictionary of DataFrames with market data, keyed by instrument
            initial_balance: Starting account balance
            commission: Commission per trade (percentage)
            base_slippage: Base slippage per trade (pips)
            base_spread: Base spread in pips
            seed: Random seed for reproducibility
            backtest_id: Optional ID for the backtest, auto-generated if None
        """
        self.historical_data = historical_data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.base_slippage = base_slippage
        self.base_spread = base_spread
        self.news_simulator = NewsAndSentimentSimulator(seed=seed)
        self.backtest_id = (backtest_id or
            f"news_backtest_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        self.logger = logger
        self.positions = []
        self.closed_positions = []
        self.balance_history = []
        self.equity_history = []
        self.metrics = {}
        self.news_events = []
        self.news_impacts = {}
        self.output_dir = os.path.join('output', 'news_backtests', self.
            backtest_id)
        os.makedirs(self.output_dir, exist_ok=True)

    @with_database_resilience('load_historical_news_events')
    @with_exception_handling
    def load_historical_news_events(self, news_events_file: str) ->bool:
        """
        Load historical news events from a file.
        
        Args:
            news_events_file: Path to JSON file containing historical news events
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(news_events_file, 'r') as f:
                events_data = json.load(f)
            events = []
            for event_data in events_data:
                event = NewsEvent.from_dict(event_data)
                events.append(event)
            self.news_events = events
            self.logger.info(
                f'Loaded {len(events)} news events from {news_events_file}')
            return True
        except Exception as e:
            self.logger.error(f'Error loading news events: {str(e)}')
            return False

    def generate_synthetic_news_events(self, num_events: int=50,
        currency_pairs: Optional[List[str]]=None) ->List[NewsEvent]:
        """
        Generate synthetic news events for backtesting when historical events aren't available.
        
        Args:
            num_events: Number of news events to generate
            currency_pairs: Currency pairs to generate events for (defaults to keys in historical_data)
            
        Returns:
            List of generated NewsEvent objects
        """
        if currency_pairs is None:
            currency_pairs = list(self.historical_data.keys())
        all_dates = []
        for df in self.historical_data.values():
            if not df.empty:
                all_dates.extend([df.index[0], df.index[-1]])
        if not all_dates:
            self.logger.error('No valid dates found in historical data')
            return []
        start_date = min(all_dates)
        end_date = max(all_dates)
        events = self.news_simulator.generate_random_economic_calendar(
            start_date, end_date, currency_pairs, num_events=num_events)
        self.news_events = events
        self.logger.info(f'Generated {len(events)} synthetic news events')
        return events

    @with_broker_api_resilience('calculate_news_impacts')
    def calculate_news_impacts(self) ->None:
        """
        Calculate the impact of news events on each timepoint in the historical data.
        
        This pre-calculates news impacts for each instrument at each timestamp for efficiency.
        """
        self.news_impacts = {}
        for instrument, df in self.historical_data.items():
            self.news_impacts[instrument] = {}
            for timestamp in df.index:
                self.news_simulator.set_current_time(timestamp)
                impact = self.news_simulator.calculate_price_impact(instrument,
                    1.0, 0.0001)
                gap_prob = self.news_simulator.calculate_gap_probability(
                    instrument)
                self.news_impacts[instrument][timestamp] = {'price_change_pct':
                    impact['price_change_pct'], 'volatility_multiplier':
                    impact['volatility_multiplier'], 'spread_multiplier':
                    impact['spread_multiplier'], 'gap_probability': gap_prob}
        self.logger.info(
            'Pre-calculated news impacts for all instruments and timestamps')

    def apply_news_impact_to_prices(self) ->Dict[str, pd.DataFrame]:
        """
        Apply news impacts to historical price data.
        
        Returns:
            Dictionary of DataFrames with news-adjusted price data
        """
        adjusted_data = {}
        for instrument, df in self.historical_data.items():
            adjusted_df = df.copy()
            for i, timestamp in enumerate(adjusted_df.index):
                if i == 0:
                    continue
                if timestamp in self.news_impacts[instrument]:
                    impact = self.news_impacts[instrument][timestamp]
                    gap_occurred = False
                    if i > 0 and np.random.random() < impact['gap_probability'
                        ]:
                        gap_size = self.news_simulator.generate_gap_size(
                            instrument, adjusted_df.iloc[i - 1]['close'])
                        gap_occurred = True
                        adjusted_df.at[adjusted_df.index[i], 'open'
                            ] = adjusted_df.iloc[i - 1]['close'] + gap_size
                    price_factor = 1.0 + impact['price_change_pct']
                    if not gap_occurred:
                        adjusted_df.at[adjusted_df.index[i], 'open'
                            ] *= price_factor
                    adjusted_df.at[adjusted_df.index[i], 'high'
                        ] *= price_factor
                    adjusted_df.at[adjusted_df.index[i], 'low'] *= price_factor
                    adjusted_df.at[adjusted_df.index[i], 'close'
                        ] *= price_factor
                    if impact['volatility_multiplier'] > 1.01:
                        mid_price = (adjusted_df.at[adjusted_df.index[i],
                            'high'] + adjusted_df.at[adjusted_df.index[i],
                            'low']) / 2
                        current_range = adjusted_df.at[adjusted_df.index[i],
                            'high'] - adjusted_df.at[adjusted_df.index[i],
                            'low']
                        new_range = current_range * impact[
                            'volatility_multiplier']
                        adjusted_df.at[adjusted_df.index[i], 'high'
                            ] = mid_price + new_range / 2
                        adjusted_df.at[adjusted_df.index[i], 'low'
                            ] = mid_price - new_range / 2
                        adjusted_df.at[adjusted_df.index[i], 'high'] = max(
                            adjusted_df.at[adjusted_df.index[i], 'high'],
                            adjusted_df.at[adjusted_df.index[i], 'close'])
                        adjusted_df.at[adjusted_df.index[i], 'low'] = min(
                            adjusted_df.at[adjusted_df.index[i], 'low'],
                            adjusted_df.at[adjusted_df.index[i], 'close'])
            adjusted_data[instrument] = adjusted_df
        return adjusted_data

    @with_broker_api_resilience('get_active_news_events')
    def get_active_news_events(self, timestamp: datetime, instrument: str
        ) ->List[NewsEvent]:
        """
        Get a list of news events active at a specific time for an instrument.
        
        Args:
            timestamp: The timestamp to check
            instrument: The instrument to check
            
        Returns:
            List of active NewsEvent objects
        """
        active_events = []
        for event in self.news_events:
            if instrument in event.currencies_affected:
                event_end = event.timestamp + timedelta(minutes=event.
                    duration_minutes)
                if event.timestamp <= timestamp <= event_end:
                    active_events.append(event)
        return active_events

    @with_broker_api_resilience('calculate_slippage')
    def calculate_slippage(self, timestamp: datetime, instrument: str,
        order_size: float) ->float:
        """
        Calculate slippage for an order taking into account news impacts.
        
        Args:
            timestamp: The timestamp of the order
            instrument: The instrument being traded
            order_size: Size of the order in lots
            
        Returns:
            Slippage amount in pips
        """
        self.news_simulator.set_current_time(timestamp)
        slippage = self.news_simulator.calculate_slippage_impact(instrument,
            order_size, self.base_slippage)
        return slippage

    @with_broker_api_resilience('calculate_spread')
    def calculate_spread(self, timestamp: datetime, instrument: str) ->float:
        """
        Calculate spread for an instrument taking into account news impacts.
        
        Args:
            timestamp: The timestamp to check
            instrument: The instrument to check
            
        Returns:
            Spread in pips
        """
        if timestamp in self.news_impacts[instrument]:
            impact = self.news_impacts[instrument][timestamp]
            return self.base_spread * impact['spread_multiplier']
        return self.base_spread

    @with_exception_handling
    def run_backtest(self, strategy_func: Callable, **strategy_params) ->Dict[
        str, Any]:
        """
        Run a backtest with news event impact incorporated.
        
        Args:
            strategy_func: Strategy function to execute
            strategy_params: Additional parameters for the strategy
            
        Returns:
            Dict with backtest results
        """
        if not self.news_impacts:
            self.calculate_news_impacts()
        adjusted_data = self.apply_news_impact_to_prices()
        self.balance = self.initial_balance
        self.positions = []
        self.closed_positions = []
        self.balance_history = []
        self.equity_history = []
        try:
            strategy_results = strategy_func(data=adjusted_data, backtester
                =self, **strategy_params)
            self._calculate_metrics()
            results_path = self._save_results()
            return {'backtest_id': self.backtest_id, 'metrics': self.
                metrics, 'results_path': results_path, 'strategy_results':
                strategy_results, 'success': True}
        except Exception as e:
            self.logger.error(f'Error running strategy: {str(e)}', exc_info
                =True)
            return {'backtest_id': self.backtest_id, 'error': str(e),
                'success': False}

    def _calculate_metrics(self) ->None:
        """Calculate performance metrics for the backtest."""
        if not self.closed_positions:
            self.metrics = {'total_trades': 0, 'winning_trades': 0,
                'losing_trades': 0, 'win_rate': 0.0, 'total_profit': 0.0,
                'total_loss': 0.0, 'profit_factor': 0.0, 'average_profit': 
                0.0, 'average_loss': 0.0, 'net_profit': 0.0, 'return_pct': 
                0.0, 'max_drawdown_pct': 0.0}
            return
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p[
            'profit'] > 0])
        losing_trades = len([p for p in self.closed_positions if p['profit'
            ] < 0])
        total_profit = sum([p['profit'] for p in self.closed_positions if p
            ['profit'] > 0])
        total_loss = sum([p['profit'] for p in self.closed_positions if p[
            'profit'] < 0])
        net_profit = sum([p['profit'] for p in self.closed_positions])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(total_profit / total_loss
            ) if total_loss != 0 else float('inf')
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        return_pct = (self.balance - self.initial_balance
            ) / self.initial_balance * 100
        max_balance = self.initial_balance
        max_drawdown = 0.0
        for balance in self.balance_history:
            if balance > max_balance:
                max_balance = balance
            else:
                drawdown = (max_balance - balance) / max_balance * 100
                max_drawdown = max(max_drawdown, drawdown)
        self.metrics = {'total_trades': total_trades, 'winning_trades':
            winning_trades, 'losing_trades': losing_trades, 'win_rate':
            win_rate, 'total_profit': total_profit, 'total_loss':
            total_loss, 'profit_factor': profit_factor, 'average_profit':
            avg_profit, 'average_loss': avg_loss, 'net_profit': net_profit,
            'return_pct': return_pct, 'max_drawdown_pct': max_drawdown}

    def _save_results(self) ->str:
        """
        Save backtest results to file
        
        Returns:
            Path to saved results file
        """
        results = {'backtest_id': self.backtest_id, 'initial_balance': self
            .initial_balance, 'final_balance': self.balance, 'metrics':
            self.metrics, 'trades': self.closed_positions,
            'balance_history': self.balance_history, 'equity_history': self
            .equity_history, 'news_events': [e.to_dict() for e in self.
            news_events]}
        results_path = os.path.join(self.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f'Saved backtest results to {results_path}')
        return results_path

    def open_position(self, timestamp: datetime, instrument: str, direction:
        str, size: float, price: float=None, stop_loss: float=None,
        take_profit: float=None, metadata: Dict[str, Any]=None) ->Dict[str, Any
        ]:
        """
        Open a new position with news-aware execution modeling.
        
        Args:
            timestamp: Entry time
            instrument: Symbol to trade
            direction: Trade direction ('long' or 'short')
            size: Position size in lots
            price: Entry price (None for market orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional position metadata
            
        Returns:
            Position details dictionary
        """
        if metadata is None:
            metadata = {}
        spread = self.calculate_spread(timestamp, instrument)
        slippage_pips = self.calculate_slippage(timestamp, instrument, size)
        if price is None:
            for df in self.historical_data.values():
                if instrument in df.columns:
                    price = df.loc[timestamp]['close']
                    break
        if direction == 'long':
            execution_price = price + spread / 2 + slippage_pips
        else:
            execution_price = price - spread / 2 - slippage_pips
        position_id = f'pos_{uuid.uuid4().hex[:8]}'
        position = {'id': position_id, 'instrument': instrument,
            'direction': direction, 'size': size, 'entry_time': timestamp,
            'entry_price': execution_price, 'stop_loss': stop_loss,
            'take_profit': take_profit, 'slippage': slippage_pips,
            'metadata': metadata}
        commission = self.commission * execution_price * size
        self.balance -= commission
        self.balance_history.append(self.balance)
        active_news = self.get_active_news_events(timestamp, instrument)
        if active_news:
            news_info = []
            for event in active_news:
                news_info.append({'title': event.title, 'impact': event.
                    impact_level.value, 'price_impact': event.price_impact,
                    'volatility_impact': event.volatility_impact})
            position['active_news_at_entry'] = news_info
        self.positions.append(position)
        return position

    def close_position(self, position_id: str, timestamp: datetime, price:
        float=None, reason: str=None) ->Dict[str, Any]:
        """
        Close an existing position with news-aware execution modeling.
        
        Args:
            position_id: ID of position to close
            timestamp: Exit time
            price: Exit price (None for market orders)
            reason: Reason for closing position
            
        Returns:
            Updated position details
        """
        position = None
        for p in self.positions:
            if p['id'] == position_id:
                position = p
                break
        if position is None:
            self.logger.error(f'Position with ID {position_id} not found')
            return None
        instrument = position['instrument']
        spread = self.calculate_spread(timestamp, instrument)
        slippage_pips = self.calculate_slippage(timestamp, instrument,
            position['size'])
        if price is None:
            for df in self.historical_data.values():
                if instrument in df.columns:
                    price = df.loc[timestamp]['close']
                    break
        if position['direction'] == 'long':
            execution_price = price - spread / 2 - slippage_pips
        else:
            execution_price = price + spread / 2 + slippage_pips
        if position['direction'] == 'long':
            profit = (execution_price - position['entry_price']) * position[
                'size']
        else:
            profit = (position['entry_price'] - execution_price) * position[
                'size']
        commission = self.commission * execution_price * position['size']
        position['exit_time'] = timestamp
        position['exit_price'] = execution_price
        position['profit'] = profit
        position['commission'] = commission
        position['exit_slippage'] = slippage_pips
        position['exit_reason'] = reason
        active_news = self.get_active_news_events(timestamp, instrument)
        if active_news:
            news_info = []
            for event in active_news:
                news_info.append({'title': event.title, 'impact': event.
                    impact_level.value, 'price_impact': event.price_impact,
                    'volatility_impact': event.volatility_impact})
            position['active_news_at_exit'] = news_info
        self.positions = [p for p in self.positions if p['id'] != position_id]
        self.closed_positions.append(position)
        self.balance = self.balance + profit - commission
        self.balance_history.append(self.balance)
        return position

    @with_broker_api_resilience('analyze_performance_during_news')
    def analyze_performance_during_news(self) ->Dict[str, Any]:
        """
        Analyze trading performance during news events versus normal periods.
        
        Returns:
            Dictionary with news performance analysis
        """
        if not self.closed_positions:
            return {'error': 'No closed positions to analyze'}
        news_trades = []
        normal_trades = []
        for trade in self.closed_positions:
            if trade.get('active_news_at_entry') or trade.get(
                'active_news_at_exit'):
                news_trades.append(trade)
            else:
                normal_trades.append(trade)
        news_metrics = self._calculate_trade_group_metrics(news_trades)
        normal_metrics = self._calculate_trade_group_metrics(normal_trades)
        impact_level_trades = {'low': [], 'medium': [], 'high': [],
            'critical': []}
        for trade in news_trades:
            max_impact = 'low'
            for news_event in (trade.get('active_news_at_entry', []) +
                trade.get('active_news_at_exit', [])):
                impact = news_event.get('impact', 'low')
                if impact == 'critical':
                    max_impact = 'critical'
                    break
                elif impact == 'high' and max_impact != 'critical':
                    max_impact = 'high'
                elif impact == 'medium' and max_impact not in ['critical',
                    'high']:
                    max_impact = 'medium'
            impact_level_trades[max_impact].append(trade)
        impact_metrics = {}
        for level, trades in impact_level_trades.items():
            impact_metrics[level] = self._calculate_trade_group_metrics(trades)
        return {'news_trades_count': len(news_trades),
            'normal_trades_count': len(normal_trades),
            'news_trades_metrics': news_metrics, 'normal_trades_metrics':
            normal_metrics, 'impact_level_metrics': impact_metrics}

    def _calculate_trade_group_metrics(self, trades: List[Dict[str, Any]]
        ) ->Dict[str, Any]:
        """Calculate metrics for a group of trades."""
        if not trades:
            return {'total_trades': 0, 'winning_trades': 0, 'losing_trades':
                0, 'win_rate': 0.0, 'total_profit': 0.0, 'total_loss': 0.0,
                'profit_factor': 0.0, 'average_profit': 0.0, 'average_loss':
                0.0, 'net_profit': 0.0}
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit'] > 0])
        losing_trades = len([t for t in trades if t['profit'] < 0])
        total_profit = sum([t['profit'] for t in trades if t['profit'] > 0])
        total_loss = sum([t['profit'] for t in trades if t['profit'] < 0])
        net_profit = sum([t['profit'] for t in trades])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = abs(total_profit / total_loss
            ) if total_loss != 0 else float('inf')
        avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        return {'total_trades': total_trades, 'winning_trades':
            winning_trades, 'losing_trades': losing_trades, 'win_rate':
            win_rate, 'total_profit': total_profit, 'total_loss':
            total_loss, 'profit_factor': profit_factor, 'average_profit':
            avg_profit, 'average_loss': avg_loss, 'net_profit': net_profit}
