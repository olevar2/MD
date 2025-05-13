"""
Correlation Tracking Service

This module implements cross-asset correlation tracking and analysis capabilities.
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from analysis_engine.multi_asset.asset_registry import AssetRegistry, AssetCorrelation, AssetClass
from analysis_engine.models.market_data import MarketData
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class CorrelationTrackingService:
    """
    Service for tracking and analyzing correlations between assets across different classes
    
    This service provides capabilities to:
    1. Calculate correlations between assets
    2. Track correlation changes over time
    3. Identify significant correlation patterns
    4. Support cross-asset correlation queries
    """

    def __init__(self, asset_registry: Optional[AssetRegistry]=None):
        """Initialize the correlation tracking service"""
        self.asset_registry = asset_registry or AssetRegistry()
        self.logger = logging.getLogger(__name__)
        self.correlation_cache = {}
        self.correlation_history: Dict[Tuple[str, str], List[Tuple[datetime,
            float]]] = {}
        self._refresh_interval = 24

    @with_analysis_resilience('calculate_correlations')
    async def calculate_correlations(self, symbols: List[str],
        lookback_days: int=30, min_periods: int=20) ->Dict[str, Dict[str,
        float]]:
        """
        Calculate correlation matrix for a list of symbols
        
        Args:
            symbols: List of symbols to calculate correlations for
            lookback_days: Number of days to look back for correlation data
            min_periods: Minimum number of overlapping periods required
            
        Returns:
            Dictionary mapping symbol pairs to correlation values
        """
        price_data = {}
        for symbol in symbols:
            price_data[symbol] = await self._get_price_data(symbol,
                lookback_days)
        df = pd.DataFrame({symbol: data['close'] for symbol, data in
            price_data.items() if data is not None and 'close' in data})
        if df.empty or len(df.columns) < 2:
            self.logger.warning('Insufficient data to calculate correlations')
            return {}
        corr_matrix = df.corr(method='pearson', min_periods=min_periods)
        result = {}
        for i, symbol1 in enumerate(corr_matrix.columns):
            result[symbol1] = {}
            for j, symbol2 in enumerate(corr_matrix.columns):
                if i != j:
                    result[symbol1][symbol2] = corr_matrix.iloc[i, j]
        return result

    @with_resilience('update_asset_correlations')
    async def update_asset_correlations(self, lookback_days: int=30,
        correlation_threshold: float=0.5) ->int:
        """
        Update correlation data for all assets in the registry
        
        Args:
            lookback_days: Days to look back for correlation calculations
            correlation_threshold: Minimum absolute correlation to save
            
        Returns:
            Number of correlation records updated
        """
        symbols_by_class = {}
        for asset_class in AssetClass:
            symbols_by_class[asset_class] = self._get_symbols_for_asset_class(
                asset_class)
        updated_count = 0
        for asset_class, symbols in symbols_by_class.items():
            if len(symbols) < 2:
                continue
            correlations = await self.calculate_correlations(symbols,
                lookback_days)
            for symbol1, corr_dict in correlations.items():
                for symbol2, corr_value in corr_dict.items():
                    if abs(corr_value) >= correlation_threshold:
                        corr = AssetCorrelation(symbol1=symbol1, symbol2=
                            symbol2, correlation=corr_value, as_of_date=
                            datetime.now(), lookback_days=lookback_days)
                        self.asset_registry.add_correlation(corr)
                        updated_count += 1
        await self._update_cross_asset_correlations(lookback_days,
            correlation_threshold)
        self.logger.info(f'Updated {updated_count} correlation records')
        return updated_count

    @with_resilience('get_correlation_matrix')
    async def get_correlation_matrix(self, symbols: List[str], use_cached:
        bool=True) ->Dict[str, Dict[str, float]]:
        """
        Get correlation matrix for a list of symbols
        
        Args:
            symbols: List of symbols to get correlations for
            use_cached: Whether to use cached correlations if available
            
        Returns:
            Dictionary mapping symbol pairs to correlation values
        """
        key = frozenset(symbols)
        now = datetime.now()
        if use_cached and key in self.correlation_cache:
            cache_time, cache_data = self.correlation_cache[key]
            if now - cache_time < timedelta(hours=self._refresh_interval):
                return cache_data
        result = {}
        for i, symbol1 in enumerate(symbols):
            result[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    result[symbol1][symbol2] = 1.0
                else:
                    corr = self.asset_registry.get_correlation(symbol1, symbol2
                        )
                    if corr is not None:
                        result[symbol1][symbol2] = corr
                    else:
                        corr_data = await self.calculate_correlations([
                            symbol1, symbol2])
                        if symbol1 in corr_data and symbol2 in corr_data[
                            symbol1]:
                            result[symbol1][symbol2] = corr_data[symbol1][
                                symbol2]
                        else:
                            result[symbol1][symbol2] = 0.0
        self.correlation_cache[key] = now, result
        for sym1, targets in result.items():
            for sym2, corr in targets.items():
                if sym1 != sym2:
                    pair = tuple(sorted([sym1, sym2]))
                    self.correlation_history.setdefault(pair, []).append((
                        now, corr))
        return result

    @with_resilience('get_correlation_stability')
    def get_correlation_stability(self, symbol1: str, symbol2: str,
        lookback_days: int=30) ->Optional[float]:
        """Calculate the stability (inverse volatility) of the correlation between two symbols over the lookback period"""
        pair = tuple(sorted([symbol1, symbol2]))
        now = datetime.now()
        history = self.correlation_history.get(pair, [])
        recent = [corr for ts, corr in history if now - ts <= timedelta(
            days=lookback_days)]
        if len(recent) < 2:
            return None
        return float(np.std(recent))

    @with_resilience('get_highest_correlations')
    async def get_highest_correlations(self, symbol: str, min_threshold:
        float=0.7, limit: int=10) ->List[Dict[str, Any]]:
        """
        Get the assets most correlated with the specified symbol
        
        Args:
            symbol: Symbol to find correlations for
            min_threshold: Minimum absolute correlation to include
            limit: Maximum number of results to return
            
        Returns:
            List of correlated assets with correlation values
        """
        correlated_assets = self.asset_registry.get_correlated_assets(symbol,
            min_threshold)
        sorted_assets = sorted(correlated_assets, key=lambda x: abs(x.get(
            'correlation', 0)), reverse=True)
        return sorted_assets[:limit]

    @with_resilience('get_cross_asset_correlations')
    async def get_cross_asset_correlations(self, symbol: str,
        other_asset_classes: Optional[List[AssetClass]]=None) ->Dict[str,
        List[Dict[str, Any]]]:
        """
        Get correlations between this symbol and symbols from other asset classes
        
        Args:
            symbol: Symbol to find correlations for
            other_asset_classes: List of other asset classes to check (all if None)
            
        Returns:
            Dictionary mapping asset classes to lists of correlated symbols
        """
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            return {}
        symbol_asset_class = asset.asset_class
        if other_asset_classes is None:
            other_asset_classes = [ac for ac in AssetClass if ac !=
                symbol_asset_class]
        else:
            other_asset_classes = [ac for ac in other_asset_classes if ac !=
                symbol_asset_class]
        result = {}
        for asset_class in other_asset_classes:
            symbols = self._get_symbols_for_asset_class(asset_class)
            correlations = []
            for other_symbol in symbols:
                corr = self.asset_registry.get_correlation(symbol, other_symbol
                    )
                if corr is not None:
                    asset_info = self.asset_registry.get_asset(other_symbol)
                    correlations.append({'symbol': other_symbol,
                        'correlation': corr, 'display_name': asset_info.
                        display_name if asset_info else other_symbol})
            correlations = sorted(correlations, key=lambda x: abs(x[
                'correlation']), reverse=True)
            result[asset_class] = correlations
        return result

    @with_resilience('get_correlation_changes')
    async def get_correlation_changes(self, symbol1: str, symbol2: str,
        lookback_periods: List[int]=[7, 30, 90]) ->Dict[str, float]:
        """
        Get correlation changes over different time periods
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            lookback_periods: List of day periods to calculate correlations for
            
        Returns:
            Dictionary mapping period names to correlation values
        """
        results = {}
        for period in lookback_periods:
            correlations = await self.calculate_correlations([symbol1,
                symbol2], lookback_days=period)
            if symbol1 in correlations and symbol2 in correlations[symbol1]:
                results[f'{period}d'] = correlations[symbol1][symbol2]
            else:
                results[f'{period}d'] = None
        return results

    async def find_correlation_regime_change(self, symbol1: str, symbol2:
        str, short_period: int=14, long_period: int=90, threshold: float=0.3
        ) ->Dict[str, Any]:
        """
        Detect if there's been a significant change in correlation regime
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            short_period: Short lookback period in days
            long_period: Long lookback period in days
            threshold: Minimum difference to consider a regime change
            
        Returns:
            Dictionary with regime change information
        """
        short_corr = await self.calculate_correlations([symbol1, symbol2],
            lookback_days=short_period)
        long_corr = await self.calculate_correlations([symbol1, symbol2],
            lookback_days=long_period)
        short_value = short_corr.get(symbol1, {}).get(symbol2)
        long_value = long_corr.get(symbol1, {}).get(symbol2)
        if short_value is None or long_value is None:
            return {'has_regime_change': False, 'short_corr': short_value,
                'long_corr': long_value, 'diff': None}
        diff = short_value - long_value
        has_regime_change = abs(diff) >= threshold
        return {'has_regime_change': has_regime_change, 'short_corr':
            short_value, 'long_corr': long_value, 'diff': diff,
            'short_period': short_period, 'long_period': long_period}

    @with_resilience('get_correlation_visualization_data')
    def get_correlation_visualization_data(self, symbols: List[str]) ->Dict[
        str, Any]:
        """
        Get data formatted for correlation visualization
        
        Args:
            symbols: List of symbols to include
            
        Returns:
            Dictionary with formatted correlation data
        """
        assets = []
        for symbol in symbols:
            asset = self.asset_registry.get_asset(symbol)
            if asset:
                assets.append({'symbol': symbol, 'display_name': asset.
                    display_name, 'asset_class': asset.asset_class, 'color':
                    self._get_color_for_asset_class(asset.asset_class)})
        loop = asyncio.get_event_loop()
        corr_matrix = loop.run_until_complete(self.get_correlation_matrix(
            symbols))
        links = []
        for symbol1 in corr_matrix:
            for symbol2, corr in corr_matrix[symbol1].items():
                if symbol1 != symbol2:
                    links.append({'source': symbol1, 'target': symbol2,
                        'value': abs(corr), 'correlation': corr, 'color':
                        self._get_color_for_correlation(corr)})
        return {'nodes': assets, 'links': links}

    async def _get_price_data(self, symbol: str, lookback_days: int) ->Dict[
        str, Any]:
        """Get historical price data for a symbol"""
        return None

    def _get_symbols_for_asset_class(self, asset_class: AssetClass) ->List[str
        ]:
        """Get all symbols for an asset class"""
        assets = self.asset_registry.list_assets(asset_class=asset_class)
        return [asset.symbol for asset in assets if asset.enabled]

    async def _update_cross_asset_correlations(self, lookback_days: int,
        correlation_threshold: float) ->int:
        """Update correlations between different asset classes"""
        updated_count = 0
        class_pairs = [(AssetClass.FOREX, AssetClass.COMMODITY), (
            AssetClass.FOREX, AssetClass.INDEX), (AssetClass.CRYPTO,
            AssetClass.STOCKS), (AssetClass.COMMODITY, AssetClass.STOCKS)]
        for class1, class2 in class_pairs:
            symbols1 = self._get_symbols_for_asset_class(class1)
            symbols2 = self._get_symbols_for_asset_class(class2)
            rep_symbols1 = self._select_representative_symbols(symbols1, 10)
            rep_symbols2 = self._select_representative_symbols(symbols2, 10)
            all_symbols = rep_symbols1 + rep_symbols2
            if len(all_symbols) >= 2:
                correlations = await self.calculate_correlations(all_symbols,
                    lookback_days)
                for symbol1 in rep_symbols1:
                    for symbol2 in rep_symbols2:
                        corr_value = correlations.get(symbol1, {}).get(symbol2)
                        if corr_value and abs(corr_value
                            ) >= correlation_threshold:
                            corr = AssetCorrelation(symbol1=symbol1,
                                symbol2=symbol2, correlation=corr_value,
                                as_of_date=datetime.now(), lookback_days=
                                lookback_days)
                            self.asset_registry.add_correlation(corr)
                            updated_count += 1
        return updated_count

    def _select_representative_symbols(self, symbols: List[str], max_count: int
        ) ->List[str]:
        """Select representative symbols from a larger list"""
        if len(symbols) <= max_count:
            return symbols
        return symbols[:max_count]

    def _get_color_for_asset_class(self, asset_class: AssetClass) ->str:
        """Get color for visualization based on asset class"""
        colors = {AssetClass.FOREX: '#4285F4', AssetClass.CRYPTO: '#FBBC05',
            AssetClass.STOCKS: '#34A853', AssetClass.COMMODITIES: '#EA4335',
            AssetClass.INDICES: '#9C27B0'}
        return colors.get(asset_class, '#757575')

    def _get_color_for_correlation(self, corr: float) ->str:
        """Get color for visualization based on correlation value"""
        if corr > 0.7:
            return '#1B5E20'
        elif corr > 0.3:
            return '#66BB6A'
        elif corr > -0.3:
            return '#BDBDBD'
        elif corr > -0.7:
            return '#EF5350'
        else:
            return '#B71C1C'
