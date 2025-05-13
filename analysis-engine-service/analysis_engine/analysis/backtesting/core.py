"""
Base Backtesting Module for Enhanced Backtesting System.

This module provides the foundation for advanced backtesting capabilities,
including walk-forward optimization, Monte Carlo simulation, and stress testing.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from analysis_engine.utils.logger import get_logger
logger = get_logger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class DataSplit(Enum):
    """Enum representing different data splits for backtesting."""
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'
    OUT_OF_SAMPLE = 'out_of_sample'


class BacktestResult(BaseModel):
    """Model representing the results of a backtest."""
    backtest_id: str = Field(default_factory=lambda : str(uuid.uuid4()))
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_portfolio_value: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    trades_total: int
    trades_won: int
    trades_lost: int
    win_rate: float
    profit_factor: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    strategy_parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BacktestConfiguration(BaseModel):
    """Configuration settings for a backtest run."""
    strategy_name: str
    strategy_parameters: Dict[str, Any] = Field(default_factory=dict)
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    instruments: List[str]
    data_timeframe: str = '1H'
    slippage_model: Optional[str] = 'fixed'
    slippage_settings: Dict[str, Any] = Field(default_factory=dict)
    commission_model: Optional[str] = 'fixed'
    commission_settings: Dict[str, Any] = Field(default_factory=dict)
    position_sizing: Optional[str] = 'fixed'
    position_sizing_settings: Dict[str, Any] = Field(default_factory=dict)
    risk_management_settings: Dict[str, Any] = Field(default_factory=dict)
    data_settings: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BacktestSummary(BaseModel):
    """Summary of backtest results for comparison purposes."""
    backtest_id: str
    strategy_name: str
    parameter_set: Dict[str, Any]
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradeData(BaseModel):
    """Representation of a single trade for backtesting."""
    trade_id: str
    instrument: str
    direction: str
    entry_time: datetime
    entry_price: float
    entry_reason: Optional[str] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    quantity: float
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    fees: float = 0
    slippage: float = 0
    risk_reward_planned: Optional[float] = None
    strategy_name: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PositionSizingModel:
    """Base class for position sizing models."""

    def __init__(self, settings: Dict[str, Any]):
    """
      init  .
    
    Args:
        settings: Description of settings
        Any]: Description of Any]
    
    """

        self.settings = settings

    @with_analysis_resilience('calculate_position_size')
    def calculate_position_size(self, capital: float, risk_per_trade: float,
        entry_price: float, stop_price: Optional[float]=None,
        instrument_data: Optional[Dict[str, Any]]=None) ->float:
        """
        Calculate the position size for a trade.
        
        Args:
            capital: Available capital
            risk_per_trade: Percentage of capital to risk (0-1)
            entry_price: Entry price for the trade
            stop_price: Stop loss price (optional)
            instrument_data: Additional data about the instrument (optional)
            
        Returns:
            float: Number of units/contracts to trade
        """
        raise NotImplementedError('Subclasses must implement this method')


class FixedPositionSizing(PositionSizingModel):
    """Position sizing using a fixed percentage of capital."""

    @with_analysis_resilience('calculate_position_size')
    def calculate_position_size(self, capital: float, risk_per_trade: float,
        entry_price: float, stop_price: Optional[float]=None,
        instrument_data: Optional[Dict[str, Any]]=None) ->float:
        """Calculate position size using a fixed percentage of capital."""
        percentage = self.settings.get('percentage', risk_per_trade)
        value_to_risk = capital * percentage
        if entry_price > 0:
            units = value_to_risk / entry_price
        else:
            units = 0
            logger.warning('Entry price must be greater than 0')
        return units


class RiskBasedPositionSizing(PositionSizingModel):
    """Position sizing based on risk (using stop loss)."""

    @with_analysis_resilience('calculate_position_size')
    def calculate_position_size(self, capital: float, risk_per_trade: float,
        entry_price: float, stop_price: Optional[float]=None,
        instrument_data: Optional[Dict[str, Any]]=None) ->float:
        """Calculate position size based on risk and stop loss."""
        if stop_price is None:
            logger.warning(
                'Stop price is required for risk-based position sizing')
            return 0
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            logger.warning('Risk per unit must be greater than 0')
            return 0
        capital_to_risk = capital * risk_per_trade
        units = capital_to_risk / risk_per_unit
        return units


class SlippageModel:
    """Base class for slippage models."""

    def __init__(self, settings: Dict[str, Any]):
    """
      init  .
    
    Args:
        settings: Description of settings
        Any]: Description of Any]
    
    """

        self.settings = settings

    @with_analysis_resilience('calculate_slippage')
    def calculate_slippage(self, price: float, quantity: float, direction:
        str, instrument: str, order_type: str, market_data: Optional[pd.
        DataFrame]=None) ->float:
        """Calculate slippage for a trade."""
        raise NotImplementedError('Subclasses must implement this method')


class FixedSlippageModel(SlippageModel):
    """Model with fixed slippage in pips or percentage."""

    @with_analysis_resilience('calculate_slippage')
    def calculate_slippage(self, price: float, quantity: float, direction:
        str, instrument: str, order_type: str, market_data: Optional[pd.
        DataFrame]=None) ->float:
        """Calculate slippage using fixed amount."""
        slippage_amount = self.settings.get('amount', 0)
        slippage_unit = self.settings.get('unit', 'pips')
        if slippage_unit == 'pips':
            pip_value = self.settings.get('pip_value', 0.0001)
            slippage = slippage_amount * pip_value
        elif slippage_unit == 'percentage':
            slippage = price * (slippage_amount / 100)
        else:
            slippage = slippage_amount
        if direction == 'long':
            price_with_slippage = price * (1 + slippage)
        else:
            price_with_slippage = price * (1 - slippage)
        return abs(price_with_slippage - price) * quantity


class VolumeBasedSlippageModel(SlippageModel):
    """Model where slippage increases with order size and volatility."""

    @with_analysis_resilience('calculate_slippage')
    def calculate_slippage(self, price: float, quantity: float, direction:
        str, instrument: str, order_type: str, market_data: Optional[pd.
        DataFrame]=None) ->float:
        """Calculate slippage based on order size and market volatility."""
        base_slippage = self.settings.get('base_amount', 0)
        volume_factor = self.settings.get('volume_factor', 0.1)
        volatility_factor = 1.0
        if market_data is not None and len(market_data) > 1:
            if 'close' in market_data.columns:
                volatility = market_data['close'].pct_change().std()
                volatility_impact = self.settings.get('volatility_impact', 2.0)
                volatility_factor = 1.0 + volatility * volatility_impact
        slippage_percentage = base_slippage * (1 + quantity * volume_factor
            ) * volatility_factor
        slippage = price * (slippage_percentage / 100)
        if direction == 'long':
            price_with_slippage = price * (1 + slippage)
        else:
            price_with_slippage = price * (1 - slippage)
        return abs(price_with_slippage - price) * quantity


class CommissionModel:
    """Base class for commission models."""

    def __init__(self, settings: Dict[str, Any]):
    """
      init  .
    
    Args:
        settings: Description of settings
        Any]: Description of Any]
    
    """

        self.settings = settings

    @with_analysis_resilience('calculate_commission')
    def calculate_commission(self, price: float, quantity: float,
        instrument: str, direction: str) ->float:
        """Calculate commission for a trade."""
        raise NotImplementedError('Subclasses must implement this method')


class FixedCommissionModel(CommissionModel):
    """Fixed commission per trade."""

    @with_analysis_resilience('calculate_commission')
    def calculate_commission(self, price: float, quantity: float,
        instrument: str, direction: str) ->float:
        """Calculate fixed commission."""
        fixed_amount = self.settings.get('fixed_amount', 0)
        return fixed_amount


class PercentageCommissionModel(CommissionModel):
    """Commission as a percentage of trade value."""

    @with_analysis_resilience('calculate_commission')
    def calculate_commission(self, price: float, quantity: float,
        instrument: str, direction: str) ->float:
        """Calculate percentage-based commission."""
        percentage = self.settings.get('percentage', 0)
        trade_value = price * quantity
        return trade_value * (percentage / 100)


class BacktestEngine:
    """
    Base class for backtest engines.
    
    This provides core functionality for backtest execution.
    Specific backtesting strategies should subclass this and
    implement the strategy-specific logic.
    """

    def __init__(self, config: BacktestConfiguration):
        """Initialize backtest engine with configuration."""
        self.config = config
        self.initial_capital = config.initial_capital
        self.current_capital = config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.position_sizing_model = self._create_position_sizing_model()
        self.slippage_model = self._create_slippage_model()
        self.commission_model = self._create_commission_model()

    def _create_position_sizing_model(self) ->PositionSizingModel:
        """Create the position sizing model based on configuration."""
        model_name = self.config.position_sizing or 'fixed'
        settings = self.config.position_sizing_settings
        if model_name == 'fixed':
            return FixedPositionSizing(settings)
        elif model_name == 'risk_based':
            return RiskBasedPositionSizing(settings)
        else:
            logger.warning(
                f'Unknown position sizing model: {model_name}, using fixed')
            return FixedPositionSizing({})

    def _create_slippage_model(self) ->SlippageModel:
        """Create the slippage model based on configuration."""
        model_name = self.config.slippage_model or 'fixed'
        settings = self.config.slippage_settings
        if model_name == 'fixed':
            return FixedSlippageModel(settings)
        elif model_name == 'volume_based':
            return VolumeBasedSlippageModel(settings)
        else:
            logger.warning(f'Unknown slippage model: {model_name}, using fixed'
                )
            return FixedSlippageModel({})

    def _create_commission_model(self) ->CommissionModel:
        """Create the commission model based on configuration."""
        model_name = self.config.commission_model or 'fixed'
        settings = self.config.commission_settings
        if model_name == 'fixed':
            return FixedCommissionModel(settings)
        elif model_name == 'percentage':
            return PercentageCommissionModel(settings)
        else:
            logger.warning(
                f'Unknown commission model: {model_name}, using fixed')
            return FixedCommissionModel({})

    def run_backtest(self, data: Dict[str, pd.DataFrame]) ->BacktestResult:
        """
        Run the backtest.
        
        Args:
            data: Dictionary of DataFrames with market data, keyed by instrument
            
        Returns:
            BacktestResult: Results of the backtest
        """
        raise NotImplementedError('Subclasses must implement this method')

    @with_analysis_resilience('calculate_metrics')
    def calculate_metrics(self) ->Dict[str, float]:
        """Calculate performance metrics from trades and equity curve."""
        if not self.trades:
            return {}
        returns = [trade.pnl_percentage for trade in self.trades if trade.
            pnl_percentage is not None]
        profits = [trade.pnl for trade in self.trades if trade.pnl is not
            None and trade.pnl > 0]
        losses = [trade.pnl for trade in self.trades if trade.pnl is not
            None and trade.pnl < 0]
        metrics = {}
        metrics['total_trades'] = len(self.trades)
        metrics['winning_trades'] = len(profits)
        metrics['losing_trades'] = len(losses)
        if returns:
            metrics['mean_return'] = np.mean(returns)
            metrics['std_return'] = np.std(returns)
            metrics['median_return'] = np.median(returns)
            metrics['max_return'] = np.max(returns)
            metrics['min_return'] = np.min(returns)
        if profits:
            metrics['mean_profit'] = np.mean(profits)
            metrics['median_profit'] = np.median(profits)
            metrics['max_profit'] = np.max(profits)
        if losses:
            metrics['mean_loss'] = np.mean(losses)
            metrics['median_loss'] = np.median(losses)
            metrics['max_loss'] = np.min(losses)
        if metrics['winning_trades'] > 0 and metrics['losing_trades'] > 0:
            metrics['win_rate'] = metrics['winning_trades'] / metrics[
                'total_trades']
            metrics['profit_factor'] = sum(profits) / abs(sum(losses)) if sum(
                losses) != 0 else float('inf')
            metrics['expectancy'] = metrics['win_rate'] * metrics.get(
                'mean_profit', 0) - (1 - metrics['win_rate']) * abs(metrics
                .get('mean_loss', 0))
        if self.equity_curve:
            equity = pd.DataFrame(self.equity_curve)
            if 'equity' in equity.columns:
                equity['returns'] = equity['equity'].pct_change().fillna(0)
                total_days = (equity.iloc[-1]['timestamp'] - equity.iloc[0]
                    ['timestamp']).days
                if total_days > 0:
                    total_return = equity.iloc[-1]['equity'] / equity.iloc[0][
                        'equity'] - 1
                    metrics['annualized_return'] = (1 + total_return) ** (
                        365 / total_days) - 1
                equity['cummax'] = equity['equity'].cummax()
                equity['drawdown'] = (equity['equity'] - equity['cummax']
                    ) / equity['cummax']
                metrics['max_drawdown'] = abs(equity['drawdown'].min())
                annualized_std = equity['returns'].std() * np.sqrt(252)
                if annualized_std > 0:
                    metrics['sharpe_ratio'] = metrics.get('annualized_return',
                        0) / annualized_std
                downside_returns = equity['returns'][equity['returns'] < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std() * np.sqrt(252)
                    if downside_std > 0:
                        metrics['sortino_ratio'] = metrics.get(
                            'annualized_return', 0) / downside_std
        return metrics

    def _record_trade(self, trade: TradeData):
        """Record a completed trade."""
        self.trades.append(trade)

    def _update_equity_curve(self, timestamp: datetime):
        """Update the equity curve with current capital."""
        self.equity_curve.append({'timestamp': timestamp, 'equity': self.
            current_capital, 'open_positions': len(self.positions)})

    def _calculate_position_size(self, capital: float, risk_per_trade:
        float, entry_price: float, stop_price: Optional[float]=None,
        instrument_data: Optional[Dict[str, Any]]=None) ->float:
        """Calculate position size using the configured model."""
        return self.position_sizing_model.calculate_position_size(capital,
            risk_per_trade, entry_price, stop_price, instrument_data)

    def _calculate_slippage(self, price: float, quantity: float, direction:
        str, instrument: str, order_type: str, market_data: Optional[pd.
        DataFrame]=None) ->float:
        """Calculate slippage using the configured model."""
        return self.slippage_model.calculate_slippage(price, quantity,
            direction, instrument, order_type, market_data)

    def _calculate_commission(self, price: float, quantity: float,
        instrument: str, direction: str) ->float:
        """Calculate commission using the configured model."""
        return self.commission_model.calculate_commission(price, quantity,
            instrument, direction)


def calculate_drawdowns(equity_curve: pd.Series) ->pd.DataFrame:
    """
    Calculate drawdowns from an equity curve.
    
    Args:
        equity_curve: Series of equity values over time
        
    Returns:
        DataFrame with drawdown metrics
    """
    equity_peaks = equity_curve.cummax()
    drawdowns = (equity_curve - equity_peaks) / equity_peaks
    is_drawdown = drawdowns < 0
    drawdown_start = is_drawdown.ne(is_drawdown.shift()).cumsum()
    grouped = drawdowns.groupby(drawdown_start)
    results = []
    for name, group in grouped:
        if (group < 0).any():
            start_date = group.index[0]
            end_date = group.index[-1]
            max_drawdown = group.min()
            recovery_date = None
            if end_date != equity_curve.index[-1]:
                next_peak_idx = equity_curve[end_date:].idxmax()
                if equity_curve[next_peak_idx] >= equity_peaks[end_date]:
                    recovery_date = next_peak_idx
            duration = (end_date - start_date).days
            results.append({'start_date': start_date, 'end_date': end_date,
                'max_drawdown': max_drawdown, 'recovery_date':
                recovery_date, 'duration_days': duration, 'recovered': 
                recovery_date is not None})
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=['start_date', 'end_date',
            'max_drawdown', 'recovery_date', 'duration_days', 'recovered'])


def calculate_trade_statistics(trades: List[TradeData]) ->Dict[str, Any]:
    """
    Calculate detailed statistics from a list of trades.
    
    Args:
        trades: List of trade data objects
        
    Returns:
        Dictionary of trade statistics
    """
    if not trades:
        return {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'profit_factor': 0}
    trade_data = []
    for trade in trades:
        if trade.pnl is not None:
            trade_data.append({'entry_time': trade.entry_time, 'exit_time':
                trade.exit_time, 'direction': trade.direction, 'pnl': trade
                .pnl, 'pnl_percentage': trade.pnl_percentage, 'duration': (
                trade.exit_time - trade.entry_time).total_seconds() / (60 *
                60 * 24)})
    if not trade_data:
        return {'total_trades': len(trades), 'winning_trades': 0,
            'losing_trades': 0, 'win_rate': 0, 'profit_factor': 0}
    df = pd.DataFrame(trade_data)
    total_trades = len(df)
    winning_trades = len(df[df['pnl'] > 0])
    losing_trades = len(df[df['pnl'] < 0])
    break_even_trades = total_trades - winning_trades - losing_trades
    total_pnl = df['pnl'].sum()
    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = df[df['pnl'] < 0]['pnl'].sum()
    df['win'] = df['pnl'] > 0
    df['streak'] = (df['win'] != df['win'].shift()).cumsum()
    win_streaks = df[df['win']].groupby('streak').size()
    loss_streaks = df[~df['win']].groupby('streak').size()
    stats = {'total_trades': total_trades, 'winning_trades': winning_trades,
        'losing_trades': losing_trades, 'break_even_trades':
        break_even_trades, 'win_rate': winning_trades / total_trades if 
        total_trades > 0 else 0, 'total_pnl': total_pnl, 'gross_profit':
        gross_profit, 'gross_loss': gross_loss, 'profit_factor': abs(
        gross_profit / gross_loss) if gross_loss != 0 else float('inf'),
        'average_trade': total_pnl / total_trades if total_trades > 0 else 
        0, 'average_win': df[df['pnl'] > 0]['pnl'].mean() if winning_trades >
        0 else 0, 'average_loss': df[df['pnl'] < 0]['pnl'].mean() if 
        losing_trades > 0 else 0, 'largest_win': df['pnl'].max(),
        'largest_loss': df['pnl'].min(), 'max_consecutive_wins': 
        win_streaks.max() if not win_streaks.empty else 0,
        'max_consecutive_losses': loss_streaks.max() if not loss_streaks.
        empty else 0, 'average_duration_days': df['duration'].mean(),
        'pnl_by_direction': {'long': df[df['direction'] == 'long']['pnl'].
        sum(), 'short': df[df['direction'] == 'short']['pnl'].sum()}}
    if len(df) > 1:
        returns = df['pnl_percentage'].fillna(0)
        stats['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252
            ) if returns.std() > 0 else 0
        downside_returns = returns[returns < 0]
        stats['sortino_ratio'] = returns.mean() / downside_returns.std(
            ) * np.sqrt(252
            ) if not downside_returns.empty and downside_returns.std(
            ) > 0 else 0
    return stats
