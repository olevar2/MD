"""
Currency Correlation Analysis Module

This module provides tools for analyzing correlations between currency pairs
to identify relationships, diversification opportunities, and risk management insights.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
import math
import logging
from datetime import datetime, timedelta
from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase, MarketDirection
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class CorrelationStrength(Enum):
    """Categories for correlation strength"""
    STRONG_POSITIVE = 'strong_positive'
    MODERATE_POSITIVE = 'moderate_positive'
    WEAK_POSITIVE = 'weak_positive'
    WEAK_NEGATIVE = 'weak_negative'
    MODERATE_NEGATIVE = 'moderate_negative'
    STRONG_NEGATIVE = 'strong_negative'
    UNCORRELATED = 'uncorrelated'


class TimeWindow(Enum):
    """Time windows for correlation analysis"""
    DAY = '1d'
    WEEK = '1w'
    MONTH = '1m'
    QUARTER = '3m'
    YEAR = '1y'


class CurrencyCorrelationAnalyzer(AdvancedAnalysisBase):
    """
    Currency Correlation analysis for forex price data
    
    This class analyzes relationships between currency pairs to identify
    correlations, potential diversification opportunities, and risk management insights.
    """

    def __init__(self, correlation_thresholds: Dict[str, float]=None,
        time_windows: List[str]=None, min_data_points: int=20,
        rolling_windows: List[int]=None):
        """
        Initialize the currency correlation analyzer
        
        Args:
            correlation_thresholds: Thresholds for different correlation strengths
            time_windows: List of time windows for analysis
            min_data_points: Minimum data points required for analysis
            rolling_windows: List of periods for rolling correlation
        """
        default_thresholds = {'strong_positive': 0.7, 'moderate_positive': 
            0.3, 'weak_positive': 0.0, 'weak_negative': -0.3,
            'moderate_negative': -0.7, 'strong_negative': -1.0}
        default_time_windows = [tw.value for tw in TimeWindow]
        default_rolling_windows = [5, 10, 20, 60]
        parameters = {'correlation_thresholds': correlation_thresholds or
            default_thresholds, 'time_windows': time_windows or
            default_time_windows, 'min_data_points': min_data_points,
            'rolling_windows': rolling_windows or default_rolling_windows,
            'use_returns': True}
        super().__init__('Currency Correlation Analysis', parameters)

    def analyze(self, price_data: Dict[str, pd.DataFrame]) ->Dict[str, Any]:
        """
        Analyze correlations between currency pairs
        
        Args:
            price_data: Dictionary mapping currency pair to price DataFrame
            
        Returns:
            Analysis results
        """
        if not price_data or len(price_data) < 2:
            return {'error': 'Insufficient price data for correlation analysis'
                }
        results = {'currency_pairs': list(price_data.keys()),
            'time_windows': {}, 'rolling_correlations': {},
            'highest_correlations': {}, 'diversification_opportunities': [],
            'correlation_clusters': []}
        for window in self.parameters['time_windows']:
            window_results = self._analyze_time_window(price_data, window)
            results['time_windows'][window] = window_results
        results['rolling_correlations'] = self._calculate_rolling_correlations(
            price_data)
        results['highest_correlations'] = self._find_highest_correlations(
            results['time_windows'])
        results['diversification_opportunities'
            ] = self._identify_diversification(results['time_windows'])
        results['correlation_clusters'] = self._identify_correlation_clusters(
            results['time_windows'])
        results['correlation_stability'] = self._analyze_correlation_stability(
            results['time_windows'], results['rolling_correlations'])
        return results

    def _preprocess_price_data(self, pair_data: pd.DataFrame, window: str=None
        ) ->pd.DataFrame:
        """
        Preprocess price data for correlation analysis
        
        Args:
            pair_data: DataFrame with price data
            window: Optional time window to filter data
            
        Returns:
            Processed DataFrame
        """
        df = pair_data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(
                'Price data does not have DatetimeIndex, correlation analysis may be inaccurate'
                )
        if window:
            current_time = datetime.now()
            if window == '1d':
                start_time = current_time - timedelta(days=1)
            elif window == '1w':
                start_time = current_time - timedelta(weeks=1)
            elif window == '1m':
                start_time = current_time - timedelta(days=30)
            elif window == '3m':
                start_time = current_time - timedelta(days=90)
            elif window == '1y':
                start_time = current_time - timedelta(days=365)
            else:
                start_time = None
            if start_time and isinstance(df.index, pd.DatetimeIndex):
                df = df[df.index >= start_time]
        if self.parameters['use_returns']:
            df['returns'] = df['close'].pct_change()
            df = df.dropna(subset=['returns'])
        return df

    def _analyze_time_window(self, price_data: Dict[str, pd.DataFrame],
        window: str) ->Dict[str, Any]:
        """
        Analyze correlations for a specific time window
        
        Args:
            price_data: Dictionary mapping currency pair to price DataFrame
            window: Time window string
            
        Returns:
            Correlation analysis results for the time window
        """
        processed_data = {}
        for pair, df in price_data.items():
            processed_df = self._preprocess_price_data(df, window)
            if len(processed_df) >= self.parameters['min_data_points']:
                if self.parameters['use_returns']:
                    processed_data[pair] = processed_df['returns']
                else:
                    processed_data[pair] = processed_df['close']
        if len(processed_data) < 2:
            return {'error':
                f'Insufficient data for correlation analysis in window {window}'
                }
        correlation_df = pd.DataFrame(processed_data)
        correlation_matrix = correlation_df.corr(method='pearson')
        categorized_correlations = self._categorize_correlations(
            correlation_matrix)
        stats = {'mean_correlation': correlation_matrix.values[np.
            triu_indices_from(correlation_matrix.values, k=1)].mean(),
            'median_correlation': np.median(correlation_matrix.values[np.
            triu_indices_from(correlation_matrix.values, k=1)]),
            'min_correlation': correlation_matrix.values[np.
            triu_indices_from(correlation_matrix.values, k=1)].min(),
            'max_correlation': correlation_matrix.values[np.
            triu_indices_from(correlation_matrix.values, k=1)].max(),
            'positive_count': sum(1 for v in correlation_matrix.values[np.
            triu_indices_from(correlation_matrix.values, k=1)] if v > 0),
            'negative_count': sum(1 for v in correlation_matrix.values[np.
            triu_indices_from(correlation_matrix.values, k=1)] if v < 0),
            'data_points': {pair: len(series) for pair, series in
            processed_data.items()}}
        return {'correlation_matrix': correlation_matrix.to_dict(),
            'categorized_correlations': categorized_correlations,
            'statistics': stats}

    def _calculate_rolling_correlations(self, price_data: Dict[str, pd.
        DataFrame]) ->Dict[str, Any]:
        """
        Calculate rolling correlations for different window sizes
        
        Args:
            price_data: Dictionary mapping currency pair to price DataFrame
            
        Returns:
            Rolling correlation analysis results
        """
        results = {}
        pairs = list(price_data.keys())
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                pair1 = pairs[i]
                pair2 = pairs[j]
                df1 = self._preprocess_price_data(price_data[pair1])
                df2 = self._preprocess_price_data(price_data[pair2])
                if self.parameters['use_returns']:
                    series1 = df1['returns']
                    series2 = df2['returns']
                else:
                    series1 = df1['close']
                    series2 = df2['close']
                combined = pd.DataFrame({pair1: series1, pair2: series2})
                combined = combined.dropna()
                pair_results = {}
                for window in self.parameters['rolling_windows']:
                    if len(combined) >= window:
                        rolling_corr = combined[pair1].rolling(window=window
                            ).corr(combined[pair2])
                        timestamps = rolling_corr.index.tolist()
                        values = rolling_corr.values.tolist()
                        pair_results[f'{window}d'] = {'timestamps':
                            timestamps, 'values': values, 'mean': np.
                            nanmean(values), 'std': np.nanstd(values),
                            'current': values[-1] if values else None,
                            'trend': self._calculate_trend(values)}
                pair_key = f'{pair1}__{pair2}'
                results[pair_key] = pair_results
        return results

    def _categorize_correlations(self, correlation_matrix: pd.DataFrame
        ) ->Dict[str, List[Tuple[str, str, float]]]:
        """
        Categorize correlations into strength categories
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            Dictionary mapping correlation categories to list of pair correlations
        """
        thresholds = self.parameters['correlation_thresholds']
        categorized = {'strong_positive': [], 'moderate_positive': [],
            'weak_positive': [], 'weak_negative': [], 'moderate_negative':
            [], 'strong_negative': [], 'uncorrelated': []}
        pairs = correlation_matrix.columns
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                pair1 = pairs[i]
                pair2 = pairs[j]
                corr_value = correlation_matrix.loc[pair1, pair2]
                if corr_value >= thresholds['strong_positive']:
                    categorized['strong_positive'].append((pair1, pair2,
                        corr_value))
                elif corr_value >= thresholds['moderate_positive']:
                    categorized['moderate_positive'].append((pair1, pair2,
                        corr_value))
                elif corr_value >= thresholds['weak_positive']:
                    categorized['weak_positive'].append((pair1, pair2,
                        corr_value))
                elif corr_value >= thresholds['weak_negative']:
                    categorized['weak_negative'].append((pair1, pair2,
                        corr_value))
                elif corr_value >= thresholds['moderate_negative']:
                    categorized['moderate_negative'].append((pair1, pair2,
                        corr_value))
                elif corr_value >= thresholds['strong_negative']:
                    categorized['strong_negative'].append((pair1, pair2,
                        corr_value))
                else:
                    categorized['uncorrelated'].append((pair1, pair2,
                        corr_value))
        return categorized

    def _find_highest_correlations(self, time_window_results: Dict[str,
        Dict[str, Any]]) ->Dict[str, List[Tuple[str, str, float]]]:
        """
        Find highest positive and negative correlations across time windows
        
        Args:
            time_window_results: Results from time window analysis
            
        Returns:
            Dictionary with highest correlations
        """
        highest = {'positive': [], 'negative': []}
        for window, results in time_window_results.items():
            if 'categorized_correlations' in results:
                if 'strong_positive' in results['categorized_correlations']:
                    for pair1, pair2, corr in results[
                        'categorized_correlations']['strong_positive']:
                        highest['positive'].append((pair1, pair2, corr, window)
                            )
                if 'strong_negative' in results['categorized_correlations']:
                    for pair1, pair2, corr in results[
                        'categorized_correlations']['strong_negative']:
                        highest['negative'].append((pair1, pair2, corr, window)
                            )
        highest['positive'] = sorted(highest['positive'], key=lambda x: x[2
            ], reverse=True)[:10]
        highest['negative'] = sorted(highest['negative'], key=lambda x: abs
            (x[2]), reverse=True)[:10]
        return highest

    def _identify_diversification(self, time_window_results: Dict[str, Dict
        [str, Any]]) ->List[Dict[str, Any]]:
        """
        Identify diversification opportunities
        
        Args:
            time_window_results: Results from time window analysis
            
        Returns:
            List of diversification opportunities
        """
        opportunities = []
        long_term_windows = ['1m', '3m', '1y']
        for window in long_term_windows:
            if window in time_window_results:
                results = time_window_results[window]
                if 'categorized_correlations' in results:
                    for category in ['weak_positive', 'weak_negative',
                        'moderate_negative', 'strong_negative']:
                        if category in results['categorized_correlations']:
                            for pair1, pair2, corr in results[
                                'categorized_correlations'][category]:
                                opportunities.append({'pair1': pair1,
                                    'pair2': pair2, 'correlation': corr,
                                    'window': window, 'category': category,
                                    'diversification_score': self.
                                    _calculate_diversification_score(corr)})
        return sorted(opportunities, key=lambda x: x[
            'diversification_score'], reverse=True)

    def _calculate_diversification_score(self, correlation: float) ->float:
        """
        Calculate diversification score based on correlation
        
        Args:
            correlation: Correlation coefficient
            
        Returns:
            Diversification score from 0.0 to 1.0
        """
        return (1 - correlation) / 2

    def _identify_correlation_clusters(self, time_window_results: Dict[str,
        Dict[str, Any]]) ->List[Dict[str, Any]]:
        """
        Identify clusters of correlated currency pairs
        
        Args:
            time_window_results: Results from time window analysis
            
        Returns:
            List of correlation clusters
        """
        medium_term = ['1m', '3m']
        selected_window = None
        for window in medium_term:
            if (window in time_window_results and 'correlation_matrix' in
                time_window_results[window]):
                selected_window = window
                break
        if not selected_window:
            return []
        corr_dict = time_window_results[selected_window]['correlation_matrix']
        corr_df = pd.DataFrame(corr_dict)
        clusters = []
        remaining_pairs = set(corr_df.columns)
        while remaining_pairs:
            current_pair = next(iter(remaining_pairs))
            cluster = [current_pair]
            remaining_pairs.remove(current_pair)
            strongly_correlated = []
            for pair in remaining_pairs:
                if abs(corr_df.loc[current_pair, pair]) >= self.parameters[
                    'correlation_thresholds']['strong_positive']:
                    strongly_correlated.append(pair)
            for pair in strongly_correlated:
                cluster.append(pair)
                if pair in remaining_pairs:
                    remaining_pairs.remove(pair)
            if len(cluster) > 1:
                avg_correlation = 0.0
                count = 0
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        avg_correlation += abs(corr_df.loc[cluster[i],
                            cluster[j]])
                        count += 1
                if count > 0:
                    avg_correlation /= count
                clusters.append({'pairs': cluster, 'size': len(cluster),
                    'avg_correlation': avg_correlation, 'window':
                    selected_window})
        return sorted(clusters, key=lambda x: (x['size'], x[
            'avg_correlation']), reverse=True)

    def _analyze_correlation_stability(self, time_window_results: Dict[str,
        Dict[str, Any]], rolling_correlations: Dict[str, Dict[str, Any]]
        ) ->Dict[str, Any]:
        """
        Analyze stability of correlations over time
        
        Args:
            time_window_results: Results from time window analysis
            rolling_correlations: Results from rolling correlation analysis
            
        Returns:
            Correlation stability analysis
        """
        stability_results = {'most_stable_pairs': [], 'least_stable_pairs':
            [], 'overall_stability_score': 0.0}
        pair_stability = {}
        for pair_key, windows in rolling_correlations.items():
            if not windows:
                continue
            longest_window = None
            max_days = 0
            for window_key, data in windows.items():
                days = int(window_key.replace('d', ''))
                if days > max_days and 'std' in data:
                    max_days = days
                    longest_window = window_key
            if not longest_window:
                continue
            std_dev = windows[longest_window]['std']
            stability_score = 1.0 - min(1.0, std_dev * 2.5)
            pair1, pair2 = pair_key.split('__')
            pair_stability[pair_key] = {'pair1': pair1, 'pair2': pair2,
                'stability_score': stability_score, 'std_dev': std_dev,
                'window': longest_window}
        sorted_stability = sorted(pair_stability.values(), key=lambda x: x[
            'stability_score'], reverse=True)
        stability_results['most_stable_pairs'] = sorted_stability[:10]
        stability_results['least_stable_pairs'] = sorted_stability[-10:
            ] if len(sorted_stability) > 10 else sorted_stability
        if pair_stability:
            stability_results['overall_stability_score'] = sum(p[
                'stability_score'] for p in pair_stability.values()) / len(
                pair_stability)
        return stability_results

    @with_exception_handling
    def _calculate_trend(self, values: List[float]) ->str:
        """
        Calculate trend direction in a time series
        
        Args:
            values: List of numerical values
            
        Returns:
            Trend direction string
        """
        if not values or len(values) < 5:
            return 'unknown'
        valid_values = [v for v in values if not (math.isnan(v) or math.
            isinf(v))]
        if not valid_values or len(valid_values) < 5:
            return 'unknown'
        x = np.arange(len(valid_values))
        y = np.array(valid_values)
        try:
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.005:
                return 'increasing'
            elif slope < -0.005:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'unknown'

    @with_resilience('update_incremental')
    def update_incremental(self, price_data: Dict[str, pd.DataFrame],
        previous_results: Dict[str, Any]) ->Dict[str, Any]:
        """
        Update correlation analysis incrementally
        
        Args:
            price_data: Dictionary mapping currency pair to price DataFrame
            previous_results: Results from previous analysis
            
        Returns:
            Updated analysis results
        """
        return self.analyze(price_data)

    @with_analysis_resilience('get_correlation_forecast')
    def get_correlation_forecast(self, pair1: str, pair2: str, price_data:
        Dict[str, pd.DataFrame]) ->Dict[str, Any]:
        """
        Get correlation forecast for a pair of currency pairs
        
        Args:
            pair1: First currency pair
            pair2: Second currency pair
            price_data: Dictionary mapping currency pair to price DataFrame
            
        Returns:
            Correlation forecast results
        """
        if pair1 not in price_data or pair2 not in price_data:
            return {'error':
                f'Missing price data for one or both pairs: {pair1}, {pair2}'}
        df1 = self._preprocess_price_data(price_data[pair1])
        df2 = self._preprocess_price_data(price_data[pair2])
        if self.parameters['use_returns']:
            series1 = df1['returns']
            series2 = df2['returns']
        else:
            series1 = df1['close']
            series2 = df2['close']
        combined = pd.DataFrame({pair1: series1, pair2: series2})
        combined = combined.dropna()
        if len(combined) < self.parameters['min_data_points']:
            return {'error':
                f'Insufficient data points for correlation forecast'}
        rolling_correlations = {}
        for window in self.parameters['rolling_windows']:
            if len(combined) >= window:
                rolling_corr = combined[pair1].rolling(window=window).corr(
                    combined[pair2])
                values = rolling_corr.values
                valid_values = values[~np.isnan(values)]
                rolling_correlations[f'{window}d'] = {'values':
                    valid_values.tolist(), 'mean': float(np.mean(
                    valid_values)), 'std': float(np.std(valid_values)),
                    'current': float(valid_values[-1]) if len(valid_values) >
                    0 else None}
        current_corr = combined[pair1].corr(combined[pair2])
        forecast = {}
        for window_key, data in rolling_correlations.items():
            if 'values' in data and len(data['values']) > 10:
                recent_values = data['values'][-10:]
                trend = self._calculate_trend(recent_values)
                if trend == 'increasing':
                    forecast_value = min(1.0, data['current'] + data['std'] *
                        0.2)
                elif trend == 'decreasing':
                    forecast_value = max(-1.0, data['current'] - data['std'
                        ] * 0.2)
                else:
                    forecast_value = data['current'] + (data['mean'] - data
                        ['current']) * 0.3
                forecast[window_key] = {'current': data['current'],
                    'forecast': forecast_value, 'trend': trend,
                    'confidence': 'medium'}
        return {'pair1': pair1, 'pair2': pair2, 'current_correlation':
            current_corr, 'rolling_correlations': rolling_correlations,
            'forecast': forecast, 'historical_stability': self.
            _calculate_correlation_stability_score(rolling_correlations)}

    def _calculate_correlation_stability_score(self, rolling_correlations:
        Dict[str, Dict[str, Any]]) ->float:
        """
        Calculate correlation stability score
        
        Args:
            rolling_correlations: Rolling correlation data
            
        Returns:
            Stability score from 0.0 to 1.0
        """
        if not rolling_correlations:
            return 0.0
        std_devs = []
        for window_key, data in rolling_correlations.items():
            if 'std' in data and not (math.isnan(data['std']) or math.isinf
                (data['std'])):
                std_devs.append(data['std'])
        if not std_devs:
            return 0.0
        avg_std = sum(std_devs) / len(std_devs)
        return 1.0 - min(1.0, avg_std * 2.5)
