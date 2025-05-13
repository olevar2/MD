"""
Tool Effectiveness Consumer

This module provides functionality to consume tool effectiveness metrics 
from the monitoring service and prepare them for use in the adaptive layer.
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import aiohttp
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.tool_effectiveness import MarketRegime, TimeFrame
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ToolEffectivenessConsumer:
    """
    Consumes and processes tool effectiveness metrics from the monitoring service.
    This component fetches metrics, transforms them into a usable format,
    and provides them to the AdaptiveLayer for decision-making.
    """

    def __init__(self, repository: ToolEffectivenessRepository,
        monitoring_service_url: Optional[str]=None, cache_duration: int=15):
        """
        Initialize the effectiveness consumer.
        
        Args:
            repository: Repository for tool effectiveness data
            monitoring_service_url: URL of the monitoring service API
            cache_duration: How long to cache metrics before refreshing (minutes)
        """
        self.repository = repository
        self.monitoring_service_url = monitoring_service_url
        self.logger = logging.getLogger(__name__)
        self.cache_duration = timedelta(minutes=cache_duration)
        self.metrics_cache = {}
        self.cache_expiry = {}

    @with_resilience('get_effectiveness_metrics')
    @async_with_exception_handling
    async def get_effectiveness_metrics(self, tool_id: str, timeframe:
        Optional[str]=None, instrument: Optional[str]=None, market_regime:
        Optional[MarketRegime]=None, force_refresh: bool=False) ->Dict[str, Any
        ]:
        """
        Get effectiveness metrics for a specific tool, optionally filtered by
        timeframe, instrument, and market regime.
        
        Args:
            tool_id: Identifier for the tool
            timeframe: Optional timeframe filter
            instrument: Optional instrument filter
            market_regime: Optional market regime filter
            force_refresh: Force refresh of cached data
            
        Returns:
            Dictionary containing effectiveness metrics
        """
        cache_key = (
            f"{tool_id}:{timeframe or 'all'}:{instrument or 'all'}:{market_regime.value if market_regime else 'all'}"
            )
        if (not force_refresh and cache_key in self.metrics_cache and 
            cache_key in self.cache_expiry and datetime.now() < self.
            cache_expiry[cache_key]):
            return self.metrics_cache[cache_key]
        try:
            metrics = await self._fetch_metrics_from_repository(tool_id,
                timeframe, instrument, market_regime)
            if (not metrics or force_refresh) and self.monitoring_service_url:
                external_metrics = (await self.
                    _fetch_metrics_from_monitoring_service(tool_id,
                    timeframe, instrument, market_regime))
                if external_metrics:
                    metrics = external_metrics
            self.metrics_cache[cache_key] = metrics
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
            return metrics
        except Exception as e:
            self.logger.error(
                f'Error fetching effectiveness metrics for {tool_id}: {e}')
            return self._get_default_metrics()

    async def _fetch_metrics_from_repository(self, tool_id: str, timeframe:
        Optional[str]=None, instrument: Optional[str]=None, market_regime:
        Optional[MarketRegime]=None) ->Dict[str, Any]:
        """
        Fetch effectiveness metrics from the internal repository.
        
        Args:
            tool_id: Identifier for the tool
            timeframe: Optional timeframe filter
            instrument: Optional instrument filter
            market_regime: Optional market regime filter
            
        Returns:
            Dictionary containing effectiveness metrics
        """
        win_rate, signal_count, outcome_count = (await self.repository.
            get_tool_win_rate_async(tool_id=tool_id, timeframe=timeframe,
            instrument=instrument, market_regime=market_regime))
        latest_metrics = await self.repository.get_latest_tool_metrics_async(
            tool_id)
        regime_metrics = {}
        if market_regime:
            effectiveness_metrics = (await self.repository.
                get_effectiveness_metrics_async(tool_id=tool_id,
                metric_type='reliability_by_regime'))
            if effectiveness_metrics:
                latest_metric = max(effectiveness_metrics, key=lambda x: x.
                    created_at)
                if hasattr(latest_metric, 'details') and latest_metric.details:
                    details = json.loads(latest_metric.details) if isinstance(
                        latest_metric.details, str) else latest_metric.details
                    regime_metrics = details.get('by_regime', {})
        metrics = {'win_rate': win_rate, 'signal_count': signal_count,
            'outcome_count': outcome_count, 'profit_factor': latest_metrics
            .get('profit_factor', 1.0), 'expected_payoff': latest_metrics.
            get('expected_payoff', 0.0), 'max_drawdown': latest_metrics.get
            ('max_drawdown', 0.0), 'recovery_factor': latest_metrics.get(
            'recovery_factor', 0.0), 'regime_metrics': regime_metrics,
            'fetched_at': datetime.now().isoformat(), 'source': 'repository'}
        return metrics

    @async_with_exception_handling
    async def _fetch_metrics_from_monitoring_service(self, tool_id: str,
        timeframe: Optional[str]=None, instrument: Optional[str]=None,
        market_regime: Optional[MarketRegime]=None) ->Dict[str, Any]:
        """
        Fetch effectiveness metrics from the external monitoring service API.
        
        Args:
            tool_id: Identifier for the tool
            timeframe: Optional timeframe filter
            instrument: Optional instrument filter
            market_regime: Optional market regime filter
            
        Returns:
            Dictionary containing effectiveness metrics, or None if failed
        """
        if not self.monitoring_service_url:
            return None
        try:
            params = {'tool_id': tool_id, 'days': 30}
            if timeframe:
                params['timeframe'] = timeframe
            if instrument:
                params['instrument'] = instrument
            if market_regime:
                params['market_regime'] = market_regime.value
            async with aiohttp.ClientSession() as session:
                url = (
                    f'{self.monitoring_service_url}/api/v1/tool-effectiveness/metrics'
                    )
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        metrics = {'win_rate': data.get('win_rate', 50.0),
                            'signal_count': data.get('signal_count', 0),
                            'outcome_count': data.get('outcome_count', 0),
                            'profit_factor': data.get('profit_factor', 1.0),
                            'expected_payoff': data.get('expected_payoff', 
                            0.0), 'max_drawdown': data.get('max_drawdown', 
                            0.0), 'recovery_factor': data.get(
                            'recovery_factor', 0.0), 'regime_metrics': data
                            .get('regime_metrics', {}), 'fetched_at':
                            datetime.now().isoformat(), 'source':
                            'monitoring_service'}
                        return metrics
                    else:
                        self.logger.warning(
                            f'Failed to fetch metrics from monitoring service: {response.status}'
                            )
                        return None
        except Exception as e:
            self.logger.error(f'Error connecting to monitoring service: {e}')
            return None

    def _get_default_metrics(self) ->Dict[str, Any]:
        """
        Get default metrics when no real metrics are available.
        
        Returns:
            Dictionary containing default effectiveness metrics
        """
        return {'win_rate': 50.0, 'signal_count': 0, 'outcome_count': 0,
            'profit_factor': 1.0, 'expected_payoff': 0.0, 'max_drawdown': 
            0.0, 'recovery_factor': 0.0, 'regime_metrics': {}, 'fetched_at':
            datetime.now().isoformat(), 'source': 'default'}

    @with_resilience('get_aggregated_effectiveness')
    @async_with_exception_handling
    async def get_aggregated_effectiveness(self, tools: List[str],
        market_regime: Optional[MarketRegime]=None, timeframe: Optional[str
        ]=None, instrument: Optional[str]=None) ->Dict[str, Any]:
        """
        Get aggregated effectiveness metrics for multiple tools.
        
        Args:
            tools: List of tool IDs
            market_regime: Optional market regime filter
            timeframe: Optional timeframe filter
            instrument: Optional instrument filter
            
        Returns:
            Dictionary containing aggregated metrics
        """
        all_metrics = {}
        tasks = []
        for tool_id in tools:
            task = self.get_effectiveness_metrics(tool_id, timeframe,
                instrument, market_regime)
            tasks.append((tool_id, task))
        for tool_id, task in tasks:
            try:
                metrics = await task
                all_metrics[tool_id] = metrics
            except Exception as e:
                self.logger.error(f'Error fetching metrics for {tool_id}: {e}')
                all_metrics[tool_id] = self._get_default_metrics()
        if not all_metrics:
            return {'tools': [], 'avg_win_rate': 50.0, 'avg_profit_factor':
                1.0, 'total_signals': 0, 'best_tool': None, 'worst_tool': None}
        win_rates = [m.get('win_rate', 50.0) for m in all_metrics.values()]
        profit_factors = [m.get('profit_factor', 1.0) for m in all_metrics.
            values() if m.get('profit_factor')]
        signal_counts = [m.get('signal_count', 0) for m in all_metrics.values()
            ]
        best_tool = max(all_metrics.items(), key=lambda x: x[1].get(
            'win_rate', 0))
        worst_tool = min(all_metrics.items(), key=lambda x: x[1].get(
            'win_rate', 100))
        return {'tools': list(all_metrics.keys()), 'avg_win_rate': sum(
            win_rates) / len(win_rates) if win_rates else 50.0,
            'avg_profit_factor': sum(profit_factors) / len(profit_factors) if
            profit_factors else 1.0, 'total_signals': sum(signal_counts),
            'best_tool': {'id': best_tool[0], 'win_rate': best_tool[1].get(
            'win_rate')} if all_metrics else None, 'worst_tool': {'id':
            worst_tool[0], 'win_rate': worst_tool[1].get('win_rate')} if
            all_metrics else None}
