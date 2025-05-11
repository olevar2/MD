"""
News-Aware Strategy Demo for forex_trading_platform.

This module demonstrates a trading strategy that incorporates news event data
in the backtesting framework.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from core_foundations.utils.logger import get_logger
from trading_gateway_service.simulation.news_event_backtester import NewsEventBacktester
from trading_gateway_service.simulation.historical_news_data_collector import NewsDataCollector
from trading_gateway_service.simulation.news_sentiment_simulator import NewsImpactLevel, NewsEventType
logger = get_logger(__name__)

class NewsAwareStrategy:
    """
    Demonstration strategy that adapts to market conditions during news events.
    
    This strategy:
    1. Uses different parameters during high-impact news events
    2. Adjusts position sizing based on expected volatility
    3. Implements news-based filters for trade entry
    4. Widens stop losses during volatile news periods
    """

    def __init__(self):
        """Initialize the news-aware strategy."""
        self.fast_ma_normal = 10
        self.slow_ma_normal = 30
        self.fast_ma_news = 5
        self.slow_ma_news = 15
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.normal_size = 0.1
        self.news_size = 0.05
        self.normal_stop_pips = 20
        self.news_stop_pips = 40
        self.positions = {}
        self.ma_data = {}

    def execute(self, data: Dict[str, pd.DataFrame], backtester: NewsEventBacktester, **kwargs) -> Dict[str, Any]:
        """
        Execute the strategy on historical data.
        
        Args:
            data: Dictionary of DataFrames with market data, keyed by instrument
            backtester: NewsEventBacktester instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with strategy execution results
        """
        logger.info('Executing news-aware strategy')
        for instrument, df in data.items():
            if df.empty:
                continue
            self._calculate_indicators(instrument, df)
            for i in range(max(self.slow_ma_normal, self.rsi_period), len(df)):
                timestamp = df.index[i]
                current = {'timestamp': timestamp, 'open': df.iloc[i]['open'], 'high': df.iloc[i]['high'], 'low': df.iloc[i]['low'], 'close': df.iloc[i]['close'], 'fast_ma_normal': self.ma_data[instrument]['fast_normal'][i], 'slow_ma_normal': self.ma_data[instrument]['slow_normal'][i], 'fast_ma_news': self.ma_data[instrument]['fast_news'][i], 'slow_ma_news': self.ma_data[instrument]['slow_news'][i], 'rsi': self.ma_data[instrument]['rsi'][i], 'prev_fast_ma_normal': self.ma_data[instrument]['fast_normal'][i - 1], 'prev_slow_ma_normal': self.ma_data[instrument]['slow_normal'][i - 1], 'prev_fast_ma_news': self.ma_data[instrument]['fast_news'][i - 1], 'prev_slow_ma_news': self.ma_data[instrument]['slow_news'][i - 1], 'prev_rsi': self.ma_data[instrument]['rsi'][i - 1]}
                active_news = backtester.get_active_news_events(timestamp, instrument)
                high_impact_news = [event for event in active_news if event.impact_level in [NewsImpactLevel.HIGH, NewsImpactLevel.CRITICAL]]
                if high_impact_news:
                    news_titles = [event.title for event in high_impact_news]
                    logger.info(f'{timestamp}: High impact news active for {instrument}: {news_titles}')
                use_news_params = len(high_impact_news) > 0
                self._check_exits(instrument, timestamp, current, backtester, use_news_params)
                self._check_entries(instrument, timestamp, current, backtester, use_news_params, high_impact_news)
        return {'strategy_name': 'News-Aware MA Crossover with RSI Filter', 'parameters': {'fast_ma_normal': self.fast_ma_normal, 'slow_ma_normal': self.slow_ma_normal, 'fast_ma_news': self.fast_ma_news, 'slow_ma_news': self.slow_ma_news, 'rsi_period': self.rsi_period, 'position_sizing': {'normal': self.normal_size, 'news': self.news_size}, 'stop_loss': {'normal_pips': self.normal_stop_pips, 'news_pips': self.news_stop_pips}}}

    def _calculate_indicators(self, instrument: str, df: pd.DataFrame) -> None:
        """Calculate indicators for the given instrument."""
        data = df.copy()
        data['fast_ma_normal'] = data['close'].rolling(window=self.fast_ma_normal).mean()
        data['slow_ma_normal'] = data['close'].rolling(window=self.slow_ma_normal).mean()
        data['fast_ma_news'] = data['close'].rolling(window=self.fast_ma_news).mean()
        data['slow_ma_news'] = data['close'].rolling(window=self.slow_ma_news).mean()
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - 100 / (1 + rs)
        self.ma_data[instrument] = {'fast_normal': data['fast_ma_normal'].values, 'slow_normal': data['slow_ma_normal'].values, 'fast_news': data['fast_ma_news'].values, 'slow_news': data['slow_ma_news'].values, 'rsi': data['rsi'].values}

    def _check_entries(self, instrument: str, timestamp: datetime, current: Dict[str, Any], backtester: NewsEventBacktester, use_news_params: bool, high_impact_news: List[Any]) -> None:
        """Check for entry signals."""
        if instrument in self.positions:
            return
        if use_news_params:
            fast_ma = current['fast_ma_news']
            slow_ma = current['slow_ma_news']
            prev_fast_ma = current['prev_fast_ma_news']
            prev_slow_ma = current['prev_slow_ma_news']
            position_size = self.news_size
            stop_pips = self.news_stop_pips
        else:
            fast_ma = current['fast_ma_normal']
            slow_ma = current['slow_ma_normal']
            prev_fast_ma = current['prev_fast_ma_normal']
            prev_slow_ma = current['prev_slow_ma_normal']
            position_size = self.normal_size
            stop_pips = self.normal_stop_pips
        rsi = current['rsi']
        is_overbought = rsi > self.rsi_overbought
        is_oversold = rsi < self.rsi_oversold
        if high_impact_news:
            news_sentiment = self._determine_news_sentiment(high_impact_news)
            signal_aligned_with_news = False
            if news_sentiment > 0 and prev_fast_ma <= prev_slow_ma and (fast_ma > slow_ma):
                signal_aligned_with_news = True
            elif news_sentiment < 0 and prev_fast_ma >= prev_slow_ma and (fast_ma < slow_ma):
                signal_aligned_with_news = True
            if not signal_aligned_with_news:
                return
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma and (not is_overbought):
            entry_price = current['close']
            stop_loss = entry_price - stop_pips * 0.0001
            position = backtester.open_position(timestamp=timestamp, instrument=instrument, direction='long', size=position_size, price=entry_price, stop_loss=stop_loss, metadata={'signal_type': 'ma_crossover_long', 'during_news': use_news_params, 'news_count': len(high_impact_news), 'rsi': rsi})
            self.positions[instrument] = position
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma and (not is_oversold):
            entry_price = current['close']
            stop_loss = entry_price + stop_pips * 0.0001
            position = backtester.open_position(timestamp=timestamp, instrument=instrument, direction='short', size=position_size, price=entry_price, stop_loss=stop_loss, metadata={'signal_type': 'ma_crossover_short', 'during_news': use_news_params, 'news_count': len(high_impact_news), 'rsi': rsi})
            self.positions[instrument] = position

    def _check_exits(self, instrument: str, timestamp: datetime, current: Dict[str, Any], backtester: NewsEventBacktester, use_news_params: bool) -> None:
        """Check for exit signals."""
        if instrument not in self.positions:
            return
        position = self.positions[instrument]
        if use_news_params:
            fast_ma = current['fast_ma_news']
            slow_ma = current['slow_ma_news']
            prev_fast_ma = current['prev_fast_ma_news']
            prev_slow_ma = current['prev_slow_ma_news']
        else:
            fast_ma = current['fast_ma_normal']
            slow_ma = current['slow_ma_normal']
            prev_fast_ma = current['prev_fast_ma_normal']
            prev_slow_ma = current['prev_slow_ma_normal']
        exit_signal = False
        if position['direction'] == 'long' and prev_fast_ma >= prev_slow_ma and (fast_ma < slow_ma):
            exit_signal = True
            reason = 'bearish_crossover'
        elif position['direction'] == 'short' and prev_fast_ma <= prev_slow_ma and (fast_ma > slow_ma):
            exit_signal = True
            reason = 'bullish_crossover'
        stop_hit = False
        if position['direction'] == 'long' and current['low'] <= position['stop_loss']:
            stop_hit = True
            reason = 'stop_loss'
        elif position['direction'] == 'short' and current['high'] >= position['stop_loss']:
            stop_hit = True
            reason = 'stop_loss'
        if exit_signal or stop_hit:
            backtester.close_position(position_id=position['id'], timestamp=timestamp, reason=reason)
            del self.positions[instrument]

    def _determine_news_sentiment(self, news_events: List[Any]) -> float:
        """
        Determine the overall sentiment from news events.
        
        Returns:
            Float between -1.0 (very bearish) and 1.0 (very bullish)
        """
        if not news_events:
            return 0.0
        sentiment_sum = 0.0
        weight_sum = 0.0
        for event in news_events:
            if event.impact_level == NewsImpactLevel.CRITICAL:
                weight = 3.0
            elif event.impact_level == NewsImpactLevel.HIGH:
                weight = 2.0
            elif event.impact_level == NewsImpactLevel.MEDIUM:
                weight = 1.0
            else:
                weight = 0.5
            if event.price_impact > 0:
                sentiment = 1.0
            elif event.price_impact < 0:
                sentiment = -1.0
            else:
                sentiment = 0.0
            sentiment_sum += sentiment * weight
            weight_sum += weight
        if weight_sum == 0:
            return 0.0
        return sentiment_sum / weight_sum

def run_news_aware_backtest(start_date: datetime=None, end_date: datetime=None, instruments: List[str]=None, data_source: str='sample', initial_balance: float=10000.0) -> Dict[str, Any]:
    """
    Run a news-aware backtest.
    
    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        instruments: List of instruments to test
        data_source: Source for historical data
        initial_balance: Initial account balance
        
    Returns:
        Dictionary with backtest results
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)
    if end_date is None:
        end_date = datetime.now()
    if instruments is None:
        instruments = ['EUR/USD', 'GBP/USD', 'USD/JPY']
    logger.info(f'Running news-aware backtest from {start_date} to {end_date}')
    output_dir = os.path.join('output', 'news_backtests')
    os.makedirs(output_dir, exist_ok=True)
    price_data = _generate_sample_price_data(instruments, start_date, end_date)
    news_collector = NewsDataCollector()
    news_events = news_collector.load_or_download_news_for_period(start_date, end_date, source='sample')
    backtester = NewsEventBacktester(historical_data=price_data, initial_balance=initial_balance, commission=0.0001, base_slippage=0.2, base_spread=0.5)
    backtester.news_events = news_events
    backtester.calculate_news_impacts()
    strategy = NewsAwareStrategy()
    results = backtester.run_backtest(strategy.execute)
    if not results['success']:
        logger.error(f'Backtest failed: {results.get('error', 'Unknown error')}')
        return results
    news_analysis = backtester.analyze_performance_during_news()
    results['news_performance_analysis'] = news_analysis
    _generate_summary_report(results, news_analysis, output_dir)
    return results

def generate_sample_price_data(instruments: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
    """
    Generate sample price data for backtesting.
    
    In a real implementation, this would load actual forex data.
    """
    price_data = {}
    hours = int((end_date - start_date).total_seconds() / 3600) + 1
    dates = pd.date_range(start=start_date, periods=hours, freq='H')
    for instrument in instruments:
        if instrument == 'EUR/USD':
            base_price = 1.1
            volatility = 0.0004
        elif instrument == 'GBP/USD':
            base_price = 1.25
            volatility = 0.0005
        else:
            base_price = 110.0
            volatility = 0.04
        np.random.seed(42)
        returns = np.random.normal(0, volatility, hours)
        cumulative_returns = np.exp(np.cumsum(returns))
        prices = base_price * cumulative_returns
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        for i in range(len(df)):
            if i == 0:
                df.loc[df.index[i], 'open'] = base_price
            else:
                df.loc[df.index[i], 'open'] = df.loc[df.index[i - 1], 'close']
            daily_volatility = volatility * np.random.uniform(0.8, 1.2)
            df.loc[df.index[i], 'high'] = df.loc[df.index[i], 'close'] * (1 + np.random.uniform(0, 2.5) * daily_volatility)
            df.loc[df.index[i], 'low'] = df.loc[df.index[i], 'close'] * (1 - np.random.uniform(0, 2.5) * daily_volatility)
            df.loc[df.index[i], 'high'] = max(df.loc[df.index[i], 'high'], df.loc[df.index[i], 'open'], df.loc[df.index[i], 'close'])
            df.loc[df.index[i], 'low'] = min(df.loc[df.index[i], 'low'], df.loc[df.index[i], 'open'], df.loc[df.index[i], 'close'])
        df['volume'] = np.random.normal(1000, 200, len(df))
        price_data[instrument] = df
    return price_data

def generate_summary_report(results: Dict[str, Any], news_analysis: Dict[str, Any], output_dir: str) -> str:
    """
    Generate and save a summary report of the backtest.
    
    Args:
        results: Backtest results
        news_analysis: News performance analysis
        output_dir: Directory to save report
        
    Returns:
        Path to saved report
    """
    report_path = os.path.join(output_dir, f'{results['backtest_id']}_summary.txt')
    with open(report_path, 'w') as f:
        f.write(f'News-Aware Backtest Summary\n')
        f.write(f'==========================\n\n')
        f.write(f'Overall Performance\n')
        f.write(f'-----------------\n')
        metrics = results['metrics']
        f.write(f'Total trades: {metrics['total_trades']}\n')
        f.write(f'Win rate: {metrics['win_rate'] * 100:.2f}%\n')
        f.write(f'Net profit: ${metrics['net_profit']:.2f}\n')
        f.write(f'Return: {metrics['return_pct']:.2f}%\n')
        f.write(f'Profit factor: {metrics['profit_factor']:.2f}\n')
        f.write(f'Max drawdown: {metrics['max_drawdown_pct']:.2f}%\n\n')
        f.write(f'Performance During News vs. Normal Periods\n')
        f.write(f'---------------------------------------\n')
        f.write(f'News trades: {news_analysis['news_trades_count']}\n')
        f.write(f'Normal trades: {news_analysis['normal_trades_count']}\n\n')
        news_metrics = news_analysis['news_trades_metrics']
        normal_metrics = news_analysis['normal_trades_metrics']
        if news_metrics['total_trades'] > 0 and normal_metrics['total_trades'] > 0:
            f.write(f'Win Rates:\n')
            f.write(f'  During news: {news_metrics['win_rate'] * 100:.2f}%\n')
            f.write(f'  Normal periods: {normal_metrics['win_rate'] * 100:.2f}%\n\n')
            f.write(f'Average Profit per Trade:\n')
            news_avg = news_metrics['net_profit'] / news_metrics['total_trades'] if news_metrics['total_trades'] > 0 else 0
            normal_avg = normal_metrics['net_profit'] / normal_metrics['total_trades'] if normal_metrics['total_trades'] > 0 else 0
            f.write(f'  During news: ${news_avg:.2f}\n')
            f.write(f'  Normal periods: ${normal_avg:.2f}\n\n')
            f.write(f'Profit Factor:\n')
            f.write(f'  During news: {news_metrics['profit_factor']:.2f}\n')
            f.write(f'  Normal periods: {normal_metrics['profit_factor']:.2f}\n\n')
        f.write(f'Performance by News Impact Level\n')
        f.write(f'-----------------------------\n')
        for level, metrics in news_analysis.get('impact_level_metrics', {}).items():
            if metrics['total_trades'] > 0:
                f.write(f'{level.upper()} impact news:\n')
                f.write(f'  Trades: {metrics['total_trades']}\n')
                f.write(f'  Win rate: {metrics['win_rate'] * 100:.2f}%\n')
                f.write(f'  Net profit: ${metrics['net_profit']:.2f}\n')
                f.write(f'  Profit factor: {metrics['profit_factor']:.2f}\n\n')
        f.write(f'Conclusion\n')
        f.write(f'----------\n')
        news_roi = news_metrics['net_profit'] / news_metrics['total_trades'] if news_metrics['total_trades'] > 0 else 0
        normal_roi = normal_metrics['net_profit'] / normal_metrics['total_trades'] if normal_metrics['total_trades'] > 0 else 0
        if news_roi > normal_roi:
            f.write(f'The strategy performs BETTER during news events (ROI: ${news_roi:.2f} vs ${normal_roi:.2f} per trade).\n')
        else:
            f.write(f'The strategy performs BETTER during normal periods (ROI: ${normal_roi:.2f} vs ${news_roi:.2f} per trade).\n')
        f.write(f'\nReport generated on {datetime.now()}\n')
    logger.info(f'Saved summary report to {report_path}')
    return report_path
if __name__ == '__main__':
    start_date = datetime.now() - timedelta(days=60)
    end_date = datetime.now()
    results = run_news_aware_backtest(start_date=start_date, end_date=end_date, instruments=['EUR/USD', 'GBP/USD', 'USD/JPY'], initial_balance=10000.0)
    print(f'Backtest completed with {len(results['metrics'])} metrics calculated.')
    print(f'Final balance: ${results['metrics']['net_profit'] + 10000:.2f}')
    print(f'Return: {results['metrics']['return_pct']:.2f}%')