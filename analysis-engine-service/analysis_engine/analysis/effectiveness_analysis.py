"""
Analysis Engine: Indicator Effectiveness Analysis

This module provides tools for evaluating and tracking the effectiveness of technical indicators,
including statistical analysis of signals and data preparation for dashboards.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import logging
import os
from datetime import datetime, timedelta
import uuid

# Configure logging
logger = logging.getLogger(__name__)


class MarketOutcome(Enum):
    """Enum for market outcomes after a signal"""
    POSITIVE = auto()  # Profitable outcome
    NEUTRAL = auto()   # Breakeven or minimal impact
    NEGATIVE = auto()  # Unprofitable outcome


@dataclass
class SignalEvaluation:
    """Evaluation of a trading signal's effectiveness"""
    signal_id: str
    indicator_name: str
    timestamp: datetime
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    timeframe: str  # Timeframe of the signal
    outcome: MarketOutcome
    profit_loss: float
    max_adverse_excursion: float  # Maximum adverse movement against signal
    max_favorable_excursion: float  # Maximum favorable movement with signal
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "signal_id": self.signal_id,
            "indicator_name": self.indicator_name,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "timeframe": self.timeframe,
            "outcome": self.outcome.name,
            "profit_loss": self.profit_loss,
            "max_adverse_excursion": self.max_adverse_excursion,
            "max_favorable_excursion": self.max_favorable_excursion,
            "metrics": self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalEvaluation':
        """Create from dictionary"""
        return cls(
            signal_id=data["signal_id"],
            indicator_name=data["indicator_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            direction=data["direction"],
            timeframe=data["timeframe"],
            outcome=MarketOutcome[data["outcome"]],
            profit_loss=data["profit_loss"],
            max_adverse_excursion=data["max_adverse_excursion"],
            max_favorable_excursion=data["max_favorable_excursion"],
            metrics=data.get("metrics", {})
        )


@dataclass
class IndicatorPerformanceMetrics:
    """Performance metrics for an indicator"""
    indicator_name: str
    timeframe: str
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    signal_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    neutral_count: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    average_mae: float = 0.0
    average_mfe: float = 0.0
    expectancy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndicatorPerformanceMetrics':
        """Create from dictionary"""
        return cls(**data)


class TimePeriod(Enum):
    """Time periods for performance analysis"""
    DAY_1 = "1D"
    WEEK_1 = "1W"
    MONTH_1 = "1M"
    MONTH_3 = "3M"
    MONTH_6 = "6M"
    YEAR_1 = "1Y"
    ALL_TIME = "ALL"
    
    def to_timedelta(self) -> Optional[timedelta]:
        """Convert to timedelta"""
        if self == self.DAY_1:
            return timedelta(days=1)
        elif self == self.WEEK_1:
            return timedelta(weeks=1)
        elif self == self.MONTH_1:
            return timedelta(days=30)
        elif self == self.MONTH_3:
            return timedelta(days=90)
        elif self == self.MONTH_6:
            return timedelta(days=180)
        elif self == self.YEAR_1:
            return timedelta(days=365)
        else:
            return None


class IndicatorPerformanceTracker:
    """Tracks the performance of indicators over time"""
    
    def __init__(self, data_dir: str = "./data/performance"):
        """
        Initialize the performance tracker
        
        Args:
            data_dir: Directory to store performance data
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Store evaluations
        self.evaluations: Dict[str, SignalEvaluation] = {}
        
        # Store performance metrics
        self.performance: Dict[str, Dict[str, IndicatorPerformanceMetrics]] = {}
        
        # Load existing data
        self._load_evaluations()
        self._load_performance()
    
    def add_evaluation(self, evaluation: SignalEvaluation) -> None:
        """
        Add a signal evaluation
        
        Args:
            evaluation: Evaluation of a signal
        """
        self.evaluations[evaluation.signal_id] = evaluation
        
        # Save evaluation to disk
        self._save_evaluation(evaluation)
        
        # Update performance metrics
        self._update_performance(evaluation)
        
        logger.debug(f"Added evaluation for {evaluation.indicator_name} signal {evaluation.signal_id}")
    
    def get_indicator_performance(self, indicator_name: str, 
                                timeframe: Optional[str] = None,
                                period: TimePeriod = TimePeriod.ALL_TIME) -> Optional[IndicatorPerformanceMetrics]:
        """
        Get performance metrics for an indicator
        
        Args:
            indicator_name: Name of the indicator
            timeframe: Timeframe of the signals
            period: Time period for performance analysis
            
        Returns:
            Performance metrics or None if no data
        """
        if indicator_name not in self.performance:
            return None
            
        if timeframe is None:
            # Return the aggregated metrics across all timeframes
            if "ALL" in self.performance[indicator_name]:
                return self.performance[indicator_name]["ALL"]
            else:
                return None
        
        if timeframe not in self.performance[indicator_name]:
            return None
            
        # Get the base metrics for the indicator and timeframe
        metrics = self.performance[indicator_name][timeframe]
        
        # If we need a specific period, recalculate
        if period != TimePeriod.ALL_TIME:
            metrics = self._calculate_period_metrics(indicator_name, timeframe, period)
            
        return metrics
    
    def get_all_indicators(self) -> Set[str]:
        """Get the set of all indicator names"""
        return set(self.performance.keys())
    
    def get_timeframes(self, indicator_name: str) -> List[str]:
        """Get available timeframes for an indicator"""
        if indicator_name not in self.performance:
            return []
            
        timeframes = list(self.performance[indicator_name].keys())
        if "ALL" in timeframes:
            timeframes.remove("ALL")
        return timeframes
    
    def _calculate_period_metrics(self, indicator_name: str, timeframe: str, 
                                period: TimePeriod) -> IndicatorPerformanceMetrics:
        """Calculate metrics for a specific time period"""
        # Get cutoff date
        cutoff = None
        if period != TimePeriod.ALL_TIME:
            delta = period.to_timedelta()
            if delta:
                cutoff = datetime.now() - delta
        
        # Filter evaluations
        filtered_evaluations = [
            e for e in self.evaluations.values()
            if e.indicator_name == indicator_name and
               (timeframe == "ALL" or e.timeframe == timeframe) and
               (cutoff is None or e.timestamp >= cutoff)
        ]
        
        # Calculate metrics
        return self._calculate_metrics(indicator_name, timeframe, filtered_evaluations)
    
    def _calculate_metrics(self, indicator_name: str, timeframe: str, 
                         evaluations: List[SignalEvaluation]) -> IndicatorPerformanceMetrics:
        """Calculate performance metrics from evaluations"""
        metrics = IndicatorPerformanceMetrics(
            indicator_name=indicator_name,
            timeframe=timeframe
        )
        
        # Check if there are any evaluations
        if not evaluations:
            return metrics
            
        # Count signals by outcome
        wins = [e for e in evaluations if e.outcome == MarketOutcome.POSITIVE]
        losses = [e for e in evaluations if e.outcome == MarketOutcome.NEGATIVE]
        neutrals = [e for e in evaluations if e.outcome == MarketOutcome.NEUTRAL]
        
        metrics.signal_count = len(evaluations)
        metrics.win_count = len(wins)
        metrics.loss_count = len(losses)
        metrics.neutral_count = len(neutrals)
        
        # Calculate win rate
        if metrics.signal_count > 0:
            metrics.win_rate = metrics.win_count / metrics.signal_count
            
        # Calculate profit metrics
        if wins:
            metrics.avg_win = sum(e.profit_loss for e in wins) / len(wins)
            metrics.max_win = max(e.profit_loss for e in wins)
            
        if losses:
            metrics.avg_loss = sum(e.profit_loss for e in losses) / len(losses)
            metrics.max_loss = min(e.profit_loss for e in losses)
            
        # Calculate profit factor
        total_profit = sum(e.profit_loss for e in wins)
        total_loss = abs(sum(e.profit_loss for e in losses)) if losses else 1
        metrics.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate expectancy
        total_trades = metrics.win_count + metrics.loss_count
        if total_trades > 0:
            metrics.expectancy = (metrics.win_rate * metrics.avg_win - 
                                (1 - metrics.win_rate) * abs(metrics.avg_loss))
        
        # Calculate MAE and MFE
        metrics.average_mae = sum(e.max_adverse_excursion for e in evaluations) / metrics.signal_count
        metrics.average_mfe = sum(e.max_favorable_excursion for e in evaluations) / metrics.signal_count
        
        # Calculate Sharpe ratio (simplified)
        if metrics.signal_count > 1:
            returns = [e.profit_loss for e in evaluations]
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            metrics.sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Calculate Sortino ratio (simplified)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_deviation = np.std(downside_returns, ddof=1)
                metrics.sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        return metrics
    
    def _update_performance(self, evaluation: SignalEvaluation) -> None:
        """Update performance metrics with a new evaluation"""
        indicator = evaluation.indicator_name
        timeframe = evaluation.timeframe
        
        # Initialize dictionaries if needed
        if indicator not in self.performance:
            self.performance[indicator] = {}
            
        if timeframe not in self.performance[indicator]:
            self.performance[indicator][timeframe] = IndicatorPerformanceMetrics(
                indicator_name=indicator,
                timeframe=timeframe
            )
            
        if "ALL" not in self.performance[indicator]:
            self.performance[indicator]["ALL"] = IndicatorPerformanceMetrics(
                indicator_name=indicator,
                timeframe="ALL"
            )
        
        # Recalculate metrics for this timeframe
        timeframe_evals = [
            e for e in self.evaluations.values()
            if e.indicator_name == indicator and e.timeframe == timeframe
        ]
        self.performance[indicator][timeframe] = self._calculate_metrics(
            indicator, timeframe, timeframe_evals)
        
        # Recalculate metrics for all timeframes
        all_evals = [
            e for e in self.evaluations.values()
            if e.indicator_name == indicator
        ]
        self.performance[indicator]["ALL"] = self._calculate_metrics(
            indicator, "ALL", all_evals)
        
        # Save updated performance
        self._save_performance()
    
    def _evaluation_path(self, signal_id: str) -> str:
        """Get the path to an evaluation file"""
        return os.path.join(self.data_dir, f"eval_{signal_id}.json")
    
    def _performance_path(self) -> str:
        """Get the path to the performance file"""
        return os.path.join(self.data_dir, "performance.json")
    
    def _save_evaluation(self, evaluation: SignalEvaluation) -> None:
        """Save a signal evaluation to disk"""
        path = self._evaluation_path(evaluation.signal_id)
        
        with open(path, 'w') as f:
            json.dump(evaluation.to_dict(), f, indent=2)
    
    def _save_performance(self) -> None:
        """Save performance metrics to disk"""
        path = self._performance_path()
        
        # Convert to serializable format
        data = {}
        for indicator, timeframes in self.performance.items():
            data[indicator] = {
                tf: metrics.to_dict() 
                for tf, metrics in timeframes.items()
            }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_evaluations(self) -> None:
        """Load signal evaluations from disk"""
        if not os.path.exists(self.data_dir):
            return
            
        # Find all evaluation files
        files = [f for f in os.listdir(self.data_dir) if f.startswith("eval_") and f.endswith(".json")]
        
        for file in files:
            path = os.path.join(self.data_dir, file)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                evaluation = SignalEvaluation.from_dict(data)
                self.evaluations[evaluation.signal_id] = evaluation
                
            except Exception as e:
                logger.error(f"Error loading evaluation from {path}: {str(e)}")
    
    def _load_performance(self) -> None:
        """Load performance metrics from disk"""
        path = self._performance_path()
        
        if not os.path.exists(path):
            return
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            for indicator, timeframes in data.items():
                self.performance[indicator] = {}
                for tf, metrics_data in timeframes.items():
                    self.performance[indicator][tf] = IndicatorPerformanceMetrics.from_dict(metrics_data)
                    
            logger.info(f"Loaded performance data for {len(self.performance)} indicators")
                
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")


class SignalAnalyzer:
    """Analyzes signals from indicators"""
    
    def __init__(self):
        """Initialize the signal analyzer"""
        pass
    
    def calculate_signal_correlation(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation between indicator signals
        
        Args:
            signals_df: DataFrame with indicator signals
            
        Returns:
            Correlation matrix
        """
        # Filter to only numeric columns for correlation
        numeric_df = signals_df.select_dtypes(include=[np.number])
        
        # Calculate correlation
        corr_matrix = numeric_df.corr()
        
        return corr_matrix
    
    def calculate_signal_lag(self, signals_df: pd.DataFrame, price_df: pd.DataFrame, 
                           max_lag: int = 10) -> Dict[str, int]:
        """
        Calculate lag between indicator signals and price movements
        
        Args:
            signals_df: DataFrame with indicator signals
            price_df: DataFrame with price data
            max_lag: Maximum lag to check
            
        Returns:
            Dictionary mapping indicator columns to their optimal lag
        """
        result = {}
        
        # Get price returns
        if 'close' in price_df.columns:
            returns = price_df['close'].pct_change()
        else:
            return result
            
        # Check each signal column
        for col in signals_df.columns:
            if col not in signals_df.select_dtypes(include=[np.number]).columns:
                continue
                
            # Calculate correlation at different lags
            corr_by_lag = {}
            for lag in range(max_lag + 1):
                # Shift returns by lag
                returns_lagged = returns.shift(-lag)
                
                # Calculate correlation
                corr = signals_df[col].corr(returns_lagged)
                corr_by_lag[lag] = corr
                
            # Find lag with highest correlation
            if corr_by_lag:
                best_lag = max(corr_by_lag.items(), key=lambda x: abs(x[1]))[0]
                result[col] = best_lag
        
        return result
    
    def find_best_signal_threshold(self, signals_df: pd.DataFrame, price_df: pd.DataFrame,
                                signal_col: str, thresholds: List[float]) -> Dict[str, Any]:
        """
        Find best threshold for a signal
        
        Args:
            signals_df: DataFrame with indicator signals
            price_df: DataFrame with price data
            signal_col: Column with signal values
            thresholds: List of thresholds to test
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        if signal_col not in signals_df.columns:
            return {}
            
        # Get price returns
        if 'close' in price_df.columns:
            returns = price_df['close'].pct_change()
        else:
            return {}
            
        results = {}
        
        # Test each threshold
        for threshold in thresholds:
            # Create binary signal based on threshold
            binary_signal = (signals_df[signal_col] > threshold).astype(int)
            
            # Calculate next-period returns when signal is active
            signal_returns = returns.shift(-1).loc[binary_signal == 1]
            
            # Calculate metrics
            if len(signal_returns) > 0:
                win_rate = (signal_returns > 0).mean()
                avg_return = signal_returns.mean()
                signal_count = len(signal_returns)
                
                results[threshold] = {
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'signal_count': signal_count,
                    'expectancy': win_rate * avg_return
                }
        
        # Find best threshold based on expectancy
        if results:
            expectancies = {t: r['expectancy'] for t, r in results.items()}
            best_threshold = max(expectancies.items(), key=lambda x: x[1])[0]
            
            return {
                'best_threshold': best_threshold,
                'metrics': results[best_threshold],
                'all_thresholds': results
            }
            
        return {}
    
    def analyze_indicator_seasonality(self, performance_data: List[SignalEvaluation]) -> Dict[str, Any]:
        """
        Analyze seasonality in indicator performance
        
        Args:
            performance_data: List of signal evaluations
            
        Returns:
            Dictionary with seasonality analysis
        """
        if not performance_data:
            return {}
            
        # Convert to DataFrame
        df = pd.DataFrame([e.to_dict() for e in performance_data])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Calculate win rate by time component
        results = {}
        
        # By hour
        hour_groups = df.groupby('hour')
        hour_stats = hour_groups.apply(lambda x: {
            'win_rate': (x['outcome'] == 'POSITIVE').mean(),
            'count': len(x),
            'avg_profit': x['profit_loss'].mean()
        }).to_dict()
        results['by_hour'] = hour_stats
        
        # By day of week
        dow_groups = df.groupby('day_of_week')
        dow_stats = dow_groups.apply(lambda x: {
            'win_rate': (x['outcome'] == 'POSITIVE').mean(),
            'count': len(x),
            'avg_profit': x['profit_loss'].mean()
        }).to_dict()
        results['by_day_of_week'] = dow_stats
        
        # By month
        month_groups = df.groupby('month')
        month_stats = month_groups.apply(lambda x: {
            'win_rate': (x['outcome'] == 'POSITIVE').mean(),
            'count': len(x),
            'avg_profit': x['profit_loss'].mean()
        }).to_dict()
        results['by_month'] = month_stats
        
        return results


class DashboardDataProvider:
    """Provides data for indicator performance dashboards"""
    
    def __init__(self, tracker: IndicatorPerformanceTracker):
        """
        Initialize the dashboard data provider
        
        Args:
            tracker: Performance tracker
        """
        self.tracker = tracker
    
    def get_overview_data(self) -> Dict[str, Any]:
        """
        Get overview data for all indicators
        
        Returns:
            Dictionary with overview data
        """
        indicators = self.tracker.get_all_indicators()
        
        result = {
            'indicators': [],
            'metrics': {},
            'performance_by_timeframe': {}
        }
        
        for indicator in indicators:
            # Get performance metrics
            metrics = self.tracker.get_indicator_performance(indicator)
            
            if metrics:
                # Add to indicators list
                result['indicators'].append(indicator)
                
                # Add key metrics
                result['metrics'][indicator] = {
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'expectancy': metrics.expectancy,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'signal_count': metrics.signal_count
                }
                
                # Add performance by timeframe
                timeframes = self.tracker.get_timeframes(indicator)
                timeframe_data = {}
                
                for tf in timeframes:
                    tf_metrics = self.tracker.get_indicator_performance(indicator, tf)
                    if tf_metrics:
                        timeframe_data[tf] = {
                            'win_rate': tf_metrics.win_rate,
                            'profit_factor': tf_metrics.profit_factor,
                            'expectancy': tf_metrics.expectancy,
                            'signal_count': tf_metrics.signal_count
                        }
                
                result['performance_by_timeframe'][indicator] = timeframe_data
        
        return result
    
    def get_indicator_detail_data(self, indicator_name: str) -> Dict[str, Any]:
        """
        Get detailed data for a specific indicator
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Dictionary with detailed data
        """
        # Get all timeframes
        timeframes = self.tracker.get_timeframes(indicator_name)
        
        # Get metrics for each timeframe and period
        result = {
            'indicator_name': indicator_name,
            'timeframes': timeframes,
            'periods': {},
            'signals': []
        }
        
        # Get metrics for each period
        for period in TimePeriod:
            period_metrics = {}
            
            # All timeframes combined
            all_metrics = self.tracker.get_indicator_performance(
                indicator_name, None, period)
                
            if all_metrics:
                period_metrics['ALL'] = {
                    'win_rate': all_metrics.win_rate,
                    'profit_factor': all_metrics.profit_factor,
                    'expectancy': all_metrics.expectancy,
                    'avg_win': all_metrics.avg_win,
                    'avg_loss': all_metrics.avg_loss,
                    'sharpe_ratio': all_metrics.sharpe_ratio,
                    'sortino_ratio': all_metrics.sortino_ratio,
                    'signal_count': all_metrics.signal_count
                }
            
            # Individual timeframes
            for tf in timeframes:
                tf_metrics = self.tracker.get_indicator_performance(
                    indicator_name, tf, period)
                    
                if tf_metrics:
                    period_metrics[tf] = {
                        'win_rate': tf_metrics.win_rate,
                        'profit_factor': tf_metrics.profit_factor,
                        'expectancy': tf_metrics.expectancy,
                        'avg_win': tf_metrics.avg_win,
                        'avg_loss': tf_metrics.avg_loss,
                        'sharpe_ratio': tf_metrics.sharpe_ratio,
                        'sortino_ratio': tf_metrics.sortino_ratio,
                        'signal_count': tf_metrics.signal_count
                    }
            
            result['periods'][period.name] = period_metrics
        
        # Get recent signal evaluations
        signal_evals = [
            e for e in self.tracker.evaluations.values()
            if e.indicator_name == indicator_name
        ]
        
        # Sort by timestamp (newest first)
        signal_evals.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Add the 50 most recent signals
        result['signals'] = [e.to_dict() for e in signal_evals[:50]]
        
        return result
    
    def get_comparison_data(self, indicators: List[str], 
                          timeframe: Optional[str] = None,
                          period: TimePeriod = TimePeriod.MONTH_3) -> Dict[str, Any]:
        """
        Get data for comparing multiple indicators
        
        Args:
            indicators: List of indicator names
            timeframe: Timeframe to compare
            period: Time period for comparison
            
        Returns:
            Dictionary with comparison data
        """
        result = {
            'indicators': indicators,
            'timeframe': timeframe or "ALL",
            'period': period.name,
            'metrics': {},
            'rankings': {}
        }
        
        # Get metrics for each indicator
        metrics_by_indicator = {}
        for indicator in indicators:
            metrics = self.tracker.get_indicator_performance(
                indicator, timeframe, period)
                
            if metrics:
                metrics_by_indicator[indicator] = {
                    'win_rate': metrics.win_rate,
                    'profit_factor': metrics.profit_factor,
                    'expectancy': metrics.expectancy,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'sortino_ratio': metrics.sortino_ratio,
                    'signal_count': metrics.signal_count
                }
        
        result['metrics'] = metrics_by_indicator
        
        # Calculate rankings
        if metrics_by_indicator:
            # Define metrics to rank and whether higher is better
            ranking_metrics = {
                'win_rate': True,
                'profit_factor': True,
                'expectancy': True,
                'sharpe_ratio': True,
                'sortino_ratio': True
            }
            
            for metric_name, higher_is_better in ranking_metrics.items():
                # Get values for this metric
                values = {ind: metrics[metric_name] 
                         for ind, metrics in metrics_by_indicator.items()
                         if not pd.isna(metrics[metric_name])}
                
                # Sort indicators by value
                sorted_indicators = sorted(
                    values.keys(),
                    key=lambda x: values[x],
                    reverse=higher_is_better
                )
                
                # Store rankings
                result['rankings'][metric_name] = {
                    ind: idx + 1 
                    for idx, ind in enumerate(sorted_indicators)
                }
        
        return result
    
    def get_export_data(self, format_type: str = 'json') -> Any:
        """
        Export all indicator performance data
        
        Args:
            format_type: Export format ('json' or 'csv')
            
        Returns:
            Exported data in requested format
        """
        # Get all indicators and timeframes
        indicators = self.tracker.get_all_indicators()
        
        # Build data structure
        data = {}
        
        for indicator in indicators:
            data[indicator] = {}
            
            # Get all timeframes
            timeframes = ['ALL'] + self.tracker.get_timeframes(indicator)
            
            for tf in timeframes:
                # Get metrics
                metrics = self.tracker.get_indicator_performance(indicator, tf)
                
                if metrics:
                    data[indicator][tf] = metrics.to_dict()
        
        # Export in requested format
        if format_type == 'json':
            return json.dumps(data, indent=2)
            
        elif format_type == 'csv':
            # Flatten data for CSV
            rows = []
            
            for indicator, timeframes in data.items():
                for tf, metrics in timeframes.items():
                    row = {
                        'indicator_name': indicator,
                        'timeframe': tf
                    }
                    row.update(metrics)
                    rows.append(row)
            
            # Convert to DataFrame
            df = pd.DataFrame(rows)
            
            # Return CSV string
            return df.to_csv(index=False)
            
        return None


def implement_effectiveness_analysis():
    """
    Implements the system to track performance of each indicator.
    - Develops statistical analysis of signals.
    - Creates dashboards for indicator comparison (integration point).
    
    Returns:
        Tuple of (performance tracker, signal analyzer, dashboard data provider)
    """
    # Initialize performance tracker
    tracker = IndicatorPerformanceTracker(data_dir="./data/performance")
    
    # Initialize signal analyzer
    analyzer = SignalAnalyzer()
    
    # Initialize dashboard data provider
    dashboard = DashboardDataProvider(tracker=tracker)
    
    # Add some sample data (for demo purposes only)
    # In a real implementation, these would come from actual signal evaluations
    for i in range(10):
        signal_id = f"sample_signal_{i}"
        
        # Create a sample evaluation
        evaluation = SignalEvaluation(
            signal_id=signal_id,
            indicator_name="MovingAverageCrossover",
            timestamp=datetime.now() - timedelta(days=i),
            direction="BULLISH" if i % 2 == 0 else "BEARISH",
            timeframe="HOUR_1",
            outcome=MarketOutcome.POSITIVE if i % 3 == 0 else 
                   (MarketOutcome.NEGATIVE if i % 3 == 1 else MarketOutcome.NEUTRAL),
            profit_loss=0.5 if i % 3 == 0 else (-0.3 if i % 3 == 1 else 0),
            max_adverse_excursion=0.2,
            max_favorable_excursion=0.7
        )
        
        tracker.add_evaluation(evaluation)
    
    logger.info("Indicator effectiveness analysis initialized")
    
    return (tracker, analyzer, dashboard)
