"""
Market analysis workflow.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, UTC
import logging

from analysis_coordinator_service.workflows.base_workflow import BaseWorkflow
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskResult,
    AnalysisTaskStatusEnum,
    AnalysisServiceType
)
from analysis_coordinator_service.repositories.task_repository import TaskRepository
from analysis_coordinator_service.adapters.market_analysis_adapter import MarketAnalysisAdapter

logger = logging.getLogger(__name__)


class MarketAnalysisWorkflow(BaseWorkflow):
    """
    Workflow for market analysis.
    """

    def __init__(
        self,
        task_repository: TaskRepository,
        market_analysis_adapter: MarketAnalysisAdapter
    ):
        """
        Initialize the workflow.

        Args:
            task_repository: Task repository for storing and retrieving tasks
            market_analysis_adapter: Adapter for the market analysis service
        """
        super().__init__(task_repository)
        self.market_analysis_adapter = market_analysis_adapter

    async def execute(
        self,
        task_id: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> AnalysisTaskResult:
        """
        Execute the market analysis workflow.

        Args:
            task_id: Task ID
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis

        Returns:
            Analysis task result
        """
        if parameters is None:
            parameters = {}

        try:
            # Update task status to running
            await self.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.RUNNING,
                progress=0.1,
                message=f"Starting market analysis workflow for {symbol}"
            )

            # Step 1: Get market data
            await self.update_task_status(
                task_id=task_id,
                progress=0.2,
                message=f"Getting market data for {symbol}"
            )

            # Step 2: Perform pattern recognition
            await self.update_task_status(
                task_id=task_id,
                progress=0.4,
                message=f"Performing pattern recognition for {symbol}"
            )

            pattern_params = parameters.get("pattern_recognition", {})
            patterns_result = await self.market_analysis_adapter.recognize_patterns(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=pattern_params
            )

            # Step 3: Detect market regime
            await self.update_task_status(
                task_id=task_id,
                progress=0.6,
                message=f"Detecting market regime for {symbol}"
            )

            regime_params = parameters.get("market_regime", {})
            regime_result = await self.market_analysis_adapter.detect_market_regime(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=regime_params
            )

            # Step 4: Identify support and resistance levels
            await self.update_task_status(
                task_id=task_id,
                progress=0.8,
                message=f"Identifying support and resistance levels for {symbol}"
            )

            sr_params = parameters.get("support_resistance", {})
            sr_result = await self.market_analysis_adapter.detect_support_resistance(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=sr_params
            )

            # Aggregate results
            results = {
                "patterns": patterns_result,
                "market_regime": regime_result,
                "support_resistance": sr_result
            }

            aggregated_results = self.aggregate_results(results)

            # Update task status to completed
            await self.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.COMPLETED,
                progress=1.0,
                message=f"Completed market analysis workflow for {symbol}",
                result=aggregated_results
            )

            # Return the task result
            return AnalysisTaskResult(
                task_id=task_id,
                service_type=AnalysisServiceType.MARKET_ANALYSIS,
                status=AnalysisTaskStatusEnum.COMPLETED,
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                result=aggregated_results
            )

        except Exception as e:
            logger.error(f"Error executing market analysis workflow: {str(e)}")

            # Update task status to failed
            await self.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.FAILED,
                message=f"Failed to execute market analysis workflow: {str(e)}",
                error=str(e)
            )

            # Return the task result
            return AnalysisTaskResult(
                task_id=task_id,
                service_type=AnalysisServiceType.MARKET_ANALYSIS,
                status=AnalysisTaskStatusEnum.FAILED,
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                error=str(e)
            )

    def aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results from multiple analysis steps.

        Args:
            results: Dictionary of results from different steps

        Returns:
            Aggregated results
        """
        aggregated = {
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": {},
            "details": results
        }

        # Extract patterns
        if "patterns" in results and results["patterns"]:
            patterns = results["patterns"]
            pattern_summary = {
                "count": len(patterns),
                "types": {},
                "confidence": 0.0
            }

            for pattern in patterns:
                pattern_type = pattern.get("pattern_type", "unknown")
                if pattern_type not in pattern_summary["types"]:
                    pattern_summary["types"][pattern_type] = 0
                pattern_summary["types"][pattern_type] += 1
                pattern_summary["confidence"] += pattern.get("confidence", 0.0)

            if pattern_summary["count"] > 0:
                pattern_summary["confidence"] /= pattern_summary["count"]

            aggregated["summary"]["patterns"] = pattern_summary

        # Extract market regime
        if "market_regime" in results and results["market_regime"]:
            regime = results["market_regime"]
            regime_summary = {
                "type": regime.get("regime_type", "unknown"),
                "confidence": regime.get("confidence", 0.0),
                "volatility": regime.get("volatility", "medium")
            }
            aggregated["summary"]["market_regime"] = regime_summary

        # Extract support and resistance levels
        if "support_resistance" in results and results["support_resistance"]:
            sr = results["support_resistance"]
            sr_summary = {
                "support_count": len(sr.get("support_levels", [])),
                "resistance_count": len(sr.get("resistance_levels", [])),
                "key_levels": []
            }

            # Add key levels (strongest support and resistance)
            support_levels = sorted(
                sr.get("support_levels", []),
                key=lambda x: x.get("strength", 0.0),
                reverse=True
            )
            resistance_levels = sorted(
                sr.get("resistance_levels", []),
                key=lambda x: x.get("strength", 0.0),
                reverse=True
            )

            if support_levels:
                sr_summary["key_levels"].append({
                    "type": "support",
                    "price": support_levels[0].get("price", 0.0),
                    "strength": support_levels[0].get("strength", 0.0)
                })

            if resistance_levels:
                sr_summary["key_levels"].append({
                    "type": "resistance",
                    "price": resistance_levels[0].get("price", 0.0),
                    "strength": resistance_levels[0].get("strength", 0.0)
                })

            aggregated["summary"]["support_resistance"] = sr_summary

        # Generate insights
        insights = []

        # Pattern insights
        if "patterns" in aggregated["summary"]:
            pattern_summary = aggregated["summary"]["patterns"]
            if pattern_summary["count"] > 0:
                pattern_types = list(pattern_summary["types"].keys())
                if pattern_types:
                    insights.append(f"Detected {pattern_summary['count']} patterns, including {', '.join(pattern_types[:3])}")

        # Market regime insights
        if "market_regime" in aggregated["summary"]:
            regime_summary = aggregated["summary"]["market_regime"]
            insights.append(f"Market is in a {regime_summary['type']} regime with {regime_summary['confidence']*100:.1f}% confidence")
            insights.append(f"Volatility is {regime_summary['volatility']}")

        # Support and resistance insights
        if "support_resistance" in aggregated["summary"]:
            sr_summary = aggregated["summary"]["support_resistance"]
            if sr_summary["key_levels"]:
                level_insights = []
                for level in sr_summary["key_levels"]:
                    level_insights.append(f"{level['type']} at {level['price']:.4f} (strength: {level['strength']*100:.1f}%)")
                insights.append(f"Key levels: {', '.join(level_insights)}")

        aggregated["insights"] = insights

        return aggregated