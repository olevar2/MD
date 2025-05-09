"""
Tool Effectiveness Adapter Module

This module provides adapters for tool effectiveness functionality from analysis-engine-service,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uuid
import os
import httpx
import asyncio
import json

from common_lib.effectiveness.interfaces import (
    MarketRegimeEnum as MarketRegime,
    TimeFrameEnum as TimeFrame,
    SignalEvent,
    SignalOutcome,
    IToolEffectivenessTracker
)

logger = logging.getLogger(__name__)


class ToolEffectivenessTrackerAdapter(IToolEffectivenessTracker):
    """
    Adapter for ToolEffectivenessTracker that implements the common interface.
    
    This adapter can either wrap an actual tracker instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, tracker_instance=None, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            tracker_instance: Optional tracker instance to wrap
            config: Configuration parameters
        """
        self.tracker = tracker_instance
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get analysis engine URL from config or environment
        analysis_engine_base_url = self.config.get(
            "analysis_engine_base_url", 
            os.environ.get("ANALYSIS_ENGINE_BASE_URL", "http://analysis-engine-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{analysis_engine_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
        
        # Local cache for signals and outcomes
        self.signals_cache = {}
        self.outcomes_cache = {}
        
        # Cache for effectiveness metrics
        self.effectiveness_cache = {}
        self.effectiveness_cache_expiry = {}
        self.cache_ttl = self.config.get("cache_ttl_minutes", 15)  # Cache TTL in minutes
    
    def record_signal(self, signal: SignalEvent) -> str:
        """Record a new trading signal."""
        if self.tracker:
            try:
                return self.tracker.record_signal(signal)
            except Exception as e:
                self.logger.warning(f"Error recording signal with tracker: {str(e)}")
        
        # Generate a signal ID
        signal_id = str(uuid.uuid4())
        
        # Store in local cache
        self.signals_cache[signal_id] = signal
        
        # Try to send to API asynchronously
        asyncio.create_task(self._send_signal_to_api(signal_id, signal))
        
        return signal_id
    
    def record_outcome(self, outcome: SignalOutcome) -> bool:
        """Record the outcome of a signal."""
        if self.tracker:
            try:
                return self.tracker.record_outcome(outcome)
            except Exception as e:
                self.logger.warning(f"Error recording outcome with tracker: {str(e)}")
        
        # Store in local cache
        self.outcomes_cache[outcome.signal_id] = outcome
        
        # Try to send to API asynchronously
        asyncio.create_task(self._send_outcome_to_api(outcome))
        
        return True
    
    def get_tool_effectiveness(
        self,
        tool_id: str,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get effectiveness metrics for a tool."""
        if self.tracker:
            try:
                return self.tracker.get_tool_effectiveness(
                    tool_id, market_regime, timeframe, symbol, start_date, end_date
                )
            except Exception as e:
                self.logger.warning(f"Error getting tool effectiveness from tracker: {str(e)}")
        
        # Check cache first
        cache_key = f"{tool_id}_{market_regime}_{timeframe}_{symbol}"
        if cache_key in self.effectiveness_cache:
            cache_time = self.effectiveness_cache_expiry.get(cache_key, datetime.min)
            if datetime.now() < cache_time:
                return self.effectiveness_cache[cache_key]
        
        # Try to get from API
        try:
            loop = asyncio.get_event_loop()
            effectiveness = loop.run_until_complete(
                self._get_tool_effectiveness_from_api(
                    tool_id, market_regime, timeframe, symbol, start_date, end_date
                )
            )
            
            # Cache the result
            self.effectiveness_cache[cache_key] = effectiveness
            self.effectiveness_cache_expiry[cache_key] = datetime.now() + timedelta(minutes=self.cache_ttl)
            
            return effectiveness
            
        except Exception as e:
            self.logger.error(f"Error getting tool effectiveness from API: {str(e)}")
            
            # Fallback to local calculation
            return self._calculate_effectiveness_locally(
                tool_id, market_regime, timeframe, symbol, start_date, end_date
            )
    
    def get_best_tools(
        self,
        market_regime: str,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        min_signals: int = 10,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get best performing tools for a market regime."""
        if self.tracker:
            try:
                return self.tracker.get_best_tools(
                    market_regime, timeframe, symbol, min_signals, top_n
                )
            except Exception as e:
                self.logger.warning(f"Error getting best tools from tracker: {str(e)}")
        
        # Try to get from API
        try:
            loop = asyncio.get_event_loop()
            best_tools = loop.run_until_complete(
                self._get_best_tools_from_api(
                    market_regime, timeframe, symbol, min_signals, top_n
                )
            )
            
            return best_tools
            
        except Exception as e:
            self.logger.error(f"Error getting best tools from API: {str(e)}")
            
            # Fallback to empty list
            return []
    
    def get_tool_history(
        self,
        tool_id: str,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get signal history for a tool."""
        if self.tracker:
            try:
                return self.tracker.get_tool_history(tool_id, limit, start_date, end_date)
            except Exception as e:
                self.logger.warning(f"Error getting tool history from tracker: {str(e)}")
        
        # Try to get from API
        try:
            loop = asyncio.get_event_loop()
            history = loop.run_until_complete(
                self._get_tool_history_from_api(tool_id, limit, start_date, end_date)
            )
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting tool history from API: {str(e)}")
            
            # Fallback to empty list
            return []
    
    def get_performance_summary(
        self,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get overall performance summary for all tools."""
        if self.tracker:
            try:
                return self.tracker.get_performance_summary(timeframe, symbol, start_date, end_date)
            except Exception as e:
                self.logger.warning(f"Error getting performance summary from tracker: {str(e)}")
        
        # Try to get from API
        try:
            loop = asyncio.get_event_loop()
            summary = loop.run_until_complete(
                self._get_performance_summary_from_api(timeframe, symbol, start_date, end_date)
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary from API: {str(e)}")
            
            # Fallback to empty summary
            return {
                "total_tools": 0,
                "total_signals": 0,
                "regime_performance": {},
                "tool_rankings": {}
            }
    
    async def _send_signal_to_api(self, signal_id: str, signal: SignalEvent) -> bool:
        """Send a signal to the API."""
        try:
            payload = {
                "signal_id": signal_id,
                "tool_id": signal.tool_id,
                "symbol": signal.symbol,
                "timestamp": signal.timestamp.isoformat(),
                "direction": signal.direction,
                "timeframe": signal.timeframe,
                "confidence": signal.confidence,
                "market_regime": signal.market_regime,
                "metadata": signal.metadata or {}
            }
            
            response = await self.client.post("/tool-effectiveness/signals", json=payload)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending signal to API: {str(e)}")
            return False
    
    async def _send_outcome_to_api(self, outcome: SignalOutcome) -> bool:
        """Send an outcome to the API."""
        try:
            payload = {
                "signal_id": outcome.signal_id,
                "result": outcome.result,
                "pips": outcome.pips,
                "profit_loss": outcome.profit_loss,
                "exit_timestamp": outcome.exit_timestamp.isoformat(),
                "metadata": outcome.metadata or {}
            }
            
            response = await self.client.post("/tool-effectiveness/outcomes", json=payload)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending outcome to API: {str(e)}")
            return False
    
    async def _get_tool_effectiveness_from_api(
        self,
        tool_id: str,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get tool effectiveness from the API."""
        try:
            params = {"tool_id": tool_id}
            
            if market_regime:
                params["market_regime"] = market_regime
            if timeframe:
                params["timeframe"] = timeframe
            if symbol:
                params["symbol"] = symbol
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
            
            response = await self.client.get("/tool-effectiveness/metrics", params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting tool effectiveness from API: {str(e)}")
            raise
    
    async def _get_best_tools_from_api(
        self,
        market_regime: str,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        min_signals: int = 10,
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get best tools from the API."""
        try:
            params = {"market_regime": market_regime, "min_signals": min_signals}
            
            if timeframe:
                params["timeframe"] = timeframe
            if symbol:
                params["symbol"] = symbol
            if top_n:
                params["top_n"] = top_n
            
            response = await self.client.get("/tool-effectiveness/best-tools", params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting best tools from API: {str(e)}")
            raise
    
    async def _get_tool_history_from_api(
        self,
        tool_id: str,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get tool history from the API."""
        try:
            params = {"tool_id": tool_id}
            
            if limit:
                params["limit"] = limit
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
            
            response = await self.client.get("/tool-effectiveness/history", params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting tool history from API: {str(e)}")
            raise
    
    async def _get_performance_summary_from_api(
        self,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get performance summary from the API."""
        try:
            params = {}
            
            if timeframe:
                params["timeframe"] = timeframe
            if symbol:
                params["symbol"] = symbol
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
            
            response = await self.client.get("/tool-effectiveness/summary", params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary from API: {str(e)}")
            raise
    
    def _calculate_effectiveness_locally(
        self,
        tool_id: str,
        market_regime: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Calculate effectiveness metrics locally."""
        # Filter signals
        signals = [
            signal for signal_id, signal in self.signals_cache.items()
            if signal.tool_id == tool_id
            and (market_regime is None or signal.market_regime == market_regime)
            and (timeframe is None or signal.timeframe == timeframe)
            and (symbol is None or signal.symbol == symbol)
            and (start_date is None or signal.timestamp >= start_date)
            and (end_date is None or signal.timestamp <= end_date)
        ]
        
        # Get outcomes for these signals
        signal_ids = [signal_id for signal_id, signal in self.signals_cache.items() 
                     if signal in signals]
        outcomes = [
            outcome for signal_id, outcome in self.outcomes_cache.items()
            if signal_id in signal_ids
        ]
        
        # Calculate metrics
        if not outcomes:
            return {
                "tool_id": tool_id,
                "signal_count": 0,
                "win_rate": 0.0,
                "average_pips": 0.0,
                "average_profit_loss": 0.0,
                "regime_performance": {}
            }
        
        win_count = sum(1 for outcome in outcomes if outcome.result == "win")
        win_rate = win_count / len(outcomes) if outcomes else 0.0
        
        average_pips = sum(outcome.pips for outcome in outcomes) / len(outcomes)
        average_profit_loss = sum(outcome.profit_loss for outcome in outcomes) / len(outcomes)
        
        # Calculate regime performance
        regime_performance = {}
        if market_regime:
            regime_signals = [signal for signal in signals 
                             if signal.market_regime == market_regime]
            regime_signal_ids = [signal_id for signal_id, signal in self.signals_cache.items() 
                               if signal in regime_signals]
            regime_outcomes = [outcome for signal_id, outcome in self.outcomes_cache.items()
                              if signal_id in regime_signal_ids]
            
            if regime_outcomes:
                regime_win_count = sum(1 for outcome in regime_outcomes 
                                      if outcome.result == "win")
                regime_win_rate = regime_win_count / len(regime_outcomes)
                
                regime_performance[market_regime] = regime_win_rate
        
        return {
            "tool_id": tool_id,
            "signal_count": len(outcomes),
            "win_rate": win_rate,
            "average_pips": average_pips,
            "average_profit_loss": average_profit_loss,
            "regime_performance": regime_performance
        }
