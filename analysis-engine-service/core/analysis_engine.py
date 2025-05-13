"""
Core Analysis Engine service coordinating market analysis operations with performance monitoring.
"""
import logging
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
from analysis_engine.monitoring.performance_monitoring import AnalysisEngineMonitoring
from analysis_engine.services.market_regime_analysis import MarketRegimeAnalyzer
from analysis_engine.services.signal_quality_evaluator import SignalQualityEvaluator
from analysis_engine.services.tool_effectiveness_service import ToolEffectivenessService
from analysis_engine.adapters.multi_asset_adapter import MultiAssetServiceAdapter
from analysis_engine.adapters.feature_store_client import FeatureStoreClient
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AnalysisEngine:
    """Coordinates analysis operations and integrates performance monitoring.

    This class acts as the central orchestrator for various analysis components
    like market regime detection, signal evaluation, and multi-asset analysis.
    It utilizes the AnalysisEngineMonitoring class to track the performance
    of key operations.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the engine.
        monitoring (AnalysisEngineMonitoring): Instance for performance monitoring.
        market_regime_analyzer (MarketRegimeAnalyzer): Component for regime analysis.
        signal_evaluator (SignalQualityEvaluator): Component for evaluating signals.
        tool_effectiveness (ToolEffectivenessService): Service for tracking tool effectiveness.
        multi_asset_analyzer (MultiAssetAnalyzer): Component for multi-asset analysis.
        is_running (bool): Flag indicating if the service is currently running.
        _health_check_task (Optional[asyncio.Task]): Background task for health checks.
    """
    """
    Core Analysis Engine service coordinating market analysis operations.
    Integrates performance monitoring for critical operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]]=None,
        monitoring_dir: str='monitoring/analysis_engine'):
        """Initializes the AnalysisEngine.

        Args:
            config: Optional dictionary containing configuration settings.
            monitoring_dir: Directory path for storing monitoring data.
        """
        self.config = config or {}
        self.monitoring = AnalysisEngineMonitoring(base_dir=monitoring_dir)
        self.market_regime_analyzer = MarketRegimeAnalyzer()
        self.signal_evaluator = SignalQualityEvaluator()
        self.tool_effectiveness = ToolEffectivenessService()
        self.multi_asset_analyzer = MultiAssetServiceAdapter()
        self.feature_store_client = FeatureStoreClient()
        self.is_running = False
        self._health_check_task = None

    async def start(self) ->None:
        """Starts the Analysis Engine service and its background tasks.

        Initializes the health check loop if the service is not already running.
        """
        """Start the analysis engine service."""
        if self.is_running:
            return
        self.is_running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop()
            )
        logger.info('Analysis Engine service started')

    async def stop(self) ->None:
        """Stops the Analysis Engine service and cancels background tasks.

        Sets the running flag to False and cancels the health check task.
        """
        """Stop the analysis engine service."""
        self.is_running = False
        if self._health_check_task:
            self._health_check_task.cancel()
        logger.info('Analysis Engine service stopped')

    @with_analysis_resilience('analyze_market')
    async def analyze_market(self, symbol: str, timeframe: str, data: Dict[
        str, Any]) ->Dict[str, Any]:
        """
        Perform comprehensive market analysis with performance tracking.

        Args:
            symbol: The market symbol to analyze
            timeframe: The timeframe for analysis
            data: Market data for analysis

        Returns:
            Analysis results including patterns, regime, and signals
        """

        @self.monitoring.track_market_analysis
        async def _analyze():
            """Internal helper function to perform analysis with monitoring."""
            regime_result = await self.analyze_regime(symbol, timeframe, data)
            patterns = await self.detect_patterns(symbol, data)
            indicators = await self.calculate_indicators(symbol, data)
            signals = await self.signal_evaluator.evaluate_signals(symbol=
                symbol, patterns=patterns, regime=regime_result, indicators
                =indicators)
            return {'regime': regime_result, 'patterns': patterns,
                'indicators': indicators, 'signals': signals, 'timestamp':
                datetime.utcnow().isoformat()}
        return await _analyze()

    @with_analysis_resilience('analyze_regime')
    async def analyze_regime(self, symbol: str, timeframe: str, data: Dict[
        str, Any]) ->Dict[str, Any]:
        """Analyze market regime with performance tracking."""

        @self.monitoring.track_regime_analysis
        async def _analyze_regime():
            """Internal helper function for regime analysis with monitoring."""
            return await self.market_regime_analyzer.analyze(symbol=symbol,
                timeframe=timeframe, data=data)
        return await _analyze_regime()

    async def detect_patterns(self, symbol: str, data: Dict[str, Any]) ->Dict[
        str, Any]:
        """Detect market patterns with performance tracking."""

        @self.monitoring.track_pattern_detection
        async def _detect_patterns():
            """Internal helper function for pattern detection with monitoring."""
            return await self.multi_asset_analyzer.detect_patterns(symbol=
                symbol, data=data)
        return await _detect_patterns()

    @with_analysis_resilience('calculate_indicators')
    async def calculate_indicators(self, symbol: str, data: Dict[str, Any]
        ) ->Dict[str, Any]:
        """Calculate technical indicators with performance tracking."""

        @self.monitoring.track_indicator_calculation
        async def _calculate_indicators():
            """Internal helper function for indicator calculation with monitoring."""
            indicator_configs = [{'type': 'sma', 'name': 'sma_14', 'window':
                14}, {'type': 'ema', 'name': 'ema_50', 'window': 50}, {
                'type': 'rsi', 'name': 'rsi_14', 'window': 14}]
            if isinstance(data, dict) and 'close' in data:
                df_data = pd.DataFrame(data)
                if 'timestamp' in df_data.columns:
                    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
            else:
                logger.warning(
                    f'Cannot process data for indicator calculation for {symbol}. Unexpected format: {type(data)}'
                    )
                return {}
            return await self.feature_store_client.calculate_indicators(symbol
                =symbol, data=df_data, indicator_configs=indicator_configs)
        return await _calculate_indicators()

    @async_with_exception_handling
    async def _health_check_loop(self) ->None:
        """Periodically monitors service health and performance metrics.

        Runs in the background, logging health status and performance data
        at regular intervals.
        """
        """Monitor service health including performance metrics."""
        while self.is_running:
            try:
                health_status = self.monitoring.get_health_status()
                if not health_status['healthy']:
                    for issue in health_status['issues']:
                        logger.warning(f'Health check issue: {issue}')
                metrics = self.monitoring.get_metrics()
                logger.info(f'Performance metrics: {metrics}')
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f'Health check error: {e}')
                await asyncio.sleep(60)
