"""
Implementation of signal flow management in the Analysis Engine Service.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio
from uuid import uuid4

from common_lib.signal_flow.interface import ISignalFlowManager, ISignalMonitor
from common_lib.signal_flow.model import (
    SignalFlow,
    SignalFlowState,
    SignalCategory,
    SignalSource,
    SignalStrength,
    SignalPriority
)

class AnalysisSignalManager:
    """
    Manages the generation and publication of trading signals from analysis components.
    """
    
    def __init__(
        self,
        signal_flow_manager: ISignalFlowManager,
        signal_monitor: ISignalMonitor
    ):
        self.flow_manager = signal_flow_manager
        self.signal_monitor = signal_monitor
        self.logger = logging.getLogger(__name__)
        
    async def create_signal_from_analysis(
        self,
        analysis_results: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> SignalFlow:
        """
        Create a unified signal from analysis results.
        
        Args:
            analysis_results: Results from technical/ML analysis
            market_context: Current market context
            
        Returns:
            A SignalFlow object representing the analysis signal
        """
        try:
            # Extract core signal components
            signal = SignalFlow(
                signal_id=str(uuid4()),
                generated_at=datetime.utcnow(),
                symbol=analysis_results["symbol"],
                timeframe=analysis_results["timeframe"],
                category=SignalCategory.TECHNICAL_ANALYSIS,
                source=SignalSource.INDICATOR,
                direction=analysis_results["direction"],
                strength=self._determine_strength(analysis_results),
                confidence=analysis_results.get("confidence", 0.5),
                priority=self._determine_priority(analysis_results, market_context),
                expiry=self._calculate_expiry(analysis_results),
                
                market_context=market_context,
                technical_context=self._extract_technical_context(analysis_results),
                model_context=self._extract_model_context(analysis_results),
                
                quality_metrics=self._calculate_quality_metrics(analysis_results),
                confluence_score=analysis_results.get("confluence_score", 0.0),
                
                risk_parameters=self._calculate_risk_parameters(analysis_results),
                suggested_entry=analysis_results.get("suggested_entry"),
                suggested_stop=analysis_results.get("suggested_stop"),
                suggested_target=analysis_results.get("suggested_target"),
                position_size_factor=analysis_results.get("position_size_factor", 1.0),
                
                metadata={
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "analysis_type": analysis_results.get("analysis_type"),
                    "indicators_used": analysis_results.get("indicators_used", []),
                    "market_regime": market_context.get("market_regime")
                }
            )
            
            # Publish signal to flow
            success = await self.flow_manager.publish_signal(signal)
            if success:
                # Start monitoring the signal
                await self.signal_monitor.track_signal(signal)
                self.logger.info(f"Published signal {signal.signal_id} for {signal.symbol}")
            else:
                self.logger.error(f"Failed to publish signal for {signal.symbol}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating signal from analysis: {str(e)}", exc_info=True)
            raise
            
    def _determine_strength(self, analysis_results: Dict[str, Any]) -> SignalStrength:
        """Determine signal strength based on analysis results"""
        confidence = analysis_results.get("confidence", 0.5)
        if confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.6:
            return SignalStrength.MODERATE
        elif confidence >= 0.4:
            return SignalStrength.WEAK
        else:
            return SignalStrength.NEUTRAL
            
    def _determine_priority(
        self,
        analysis_results: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> SignalPriority:
        """Determine signal priority based on analysis and market context"""
        # High priority if strong confluence or key level breakout
        if (analysis_results.get("confluence_score", 0) > 0.8 or
            analysis_results.get("is_key_level_breakout", False)):
            return SignalPriority.HIGH
            
        # Medium priority for standard technical signals in favorable regimes
        elif (analysis_results.get("confluence_score", 0) > 0.6 and
              market_context.get("regime_aligned", False)):
            return SignalPriority.MEDIUM
            
        # Low priority for everything else
        else:
            return SignalPriority.LOW
            
    def _calculate_expiry(self, analysis_results: Dict[str, Any]) -> Optional[datetime]:
        """Calculate signal expiry time based on timeframe"""
        timeframe = analysis_results.get("timeframe", "").lower()
        if not timeframe:
            return None
            
        now = datetime.utcnow()
        if "m" in timeframe:
            minutes = int(timeframe.replace("m", ""))
            return now + timedelta(minutes=minutes*3)
        elif "h" in timeframe:
            hours = int(timeframe.replace("h", ""))
            return now + timedelta(hours=hours*3)
        elif "d" in timeframe:
            days = int(timeframe.replace("d", ""))
            return now + timedelta(days=days*3)
        return None
        
    def _extract_technical_context(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical analysis context from results"""
        return {
            "trend_aligned": analysis_results.get("trend_aligned", False),
            "trend_strength": analysis_results.get("trend_strength", 0.0),
            "key_levels": analysis_results.get("key_levels", []),
            "patterns_detected": analysis_results.get("patterns", []),
            "indicator_signals": analysis_results.get("indicator_signals", {}),
            "momentum_signals": analysis_results.get("momentum_signals", {}),
            "volatility_signals": analysis_results.get("volatility_signals", {})
        }
        
    def _extract_model_context(self, analysis_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract ML model context if available"""
        if not analysis_results.get("model_predictions"):
            return None
            
        return {
            "model_id": analysis_results.get("model_id"),
            "predictions": analysis_results.get("model_predictions", {}),
            "model_confidence": analysis_results.get("model_confidence", 0.0),
            "features_used": analysis_results.get("features_used", [])
        }
        
    def _calculate_quality_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate signal quality metrics"""
        return {
            "trend_alignment": analysis_results.get("trend_alignment_score", 0.0),
            "momentum_alignment": analysis_results.get("momentum_alignment_score", 0.0),
            "volatility_suitability": analysis_results.get("volatility_suitability_score", 0.0),
            "pattern_reliability": analysis_results.get("pattern_reliability_score", 0.0),
            "model_accuracy": analysis_results.get("model_accuracy_score", 0.0),
            "confluence_level": analysis_results.get("confluence_score", 0.0)
        }
        
    def _calculate_risk_parameters(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk parameters for the signal"""
        return {
            "risk_reward_ratio": analysis_results.get("risk_reward_ratio", 0.0),
            "position_size_factor": analysis_results.get("position_size_factor", 1.0),
            "max_risk_percent": analysis_results.get("max_risk_percent", 1.0),
            "volatility_adjustment": analysis_results.get("volatility_adjustment", 1.0),
            "market_risk_level": analysis_results.get("market_risk_level", "medium"),
            "stop_loss_buffer": analysis_results.get("stop_loss_buffer", 0.0)
        }
