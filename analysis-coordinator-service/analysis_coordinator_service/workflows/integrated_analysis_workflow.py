"""
Integrated analysis workflow.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, UTC
import logging
import asyncio

from analysis_coordinator_service.workflows.base_workflow import BaseWorkflow
from analysis_coordinator_service.workflows.market_analysis_workflow import MarketAnalysisWorkflow
from analysis_coordinator_service.workflows.causal_analysis_workflow import CausalAnalysisWorkflow
from analysis_coordinator_service.workflows.backtesting_workflow import BacktestingWorkflow
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskResult,
    AnalysisTaskStatusEnum,
    AnalysisServiceType
)
from analysis_coordinator_service.repositories.task_repository import TaskRepository
from analysis_coordinator_service.adapters.market_analysis_adapter import MarketAnalysisAdapter
from analysis_coordinator_service.adapters.causal_analysis_adapter import CausalAnalysisAdapter
from analysis_coordinator_service.adapters.backtesting_adapter import BacktestingAdapter

logger = logging.getLogger(__name__)


class IntegratedAnalysisWorkflow(BaseWorkflow):
    """
    Workflow for integrated analysis across multiple services.
    """

    def __init__(
        self,
        task_repository: TaskRepository,
        market_analysis_adapter: MarketAnalysisAdapter,
        causal_analysis_adapter: CausalAnalysisAdapter,
        backtesting_adapter: BacktestingAdapter
    ):
        """
        Initialize the workflow.

        Args:
            task_repository: Task repository for storing and retrieving tasks
            market_analysis_adapter: Adapter for the market analysis service
            causal_analysis_adapter: Adapter for the causal analysis service
            backtesting_adapter: Adapter for the backtesting service
        """
        super().__init__(task_repository)
        self.market_analysis_adapter = market_analysis_adapter
        self.causal_analysis_adapter = causal_analysis_adapter
        self.backtesting_adapter = backtesting_adapter

        # Create sub-workflows
        self.market_analysis_workflow = MarketAnalysisWorkflow(
            task_repository=task_repository,
            market_analysis_adapter=market_analysis_adapter
        )
        self.causal_analysis_workflow = CausalAnalysisWorkflow(
            task_repository=task_repository,
            causal_analysis_adapter=causal_analysis_adapter
        )
        self.backtesting_workflow = BacktestingWorkflow(
            task_repository=task_repository,
            backtesting_adapter=backtesting_adapter
        )

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
        Execute the integrated analysis workflow.

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
                message=f"Starting integrated analysis workflow for {symbol}"
            )

            # Determine which services to run
            services = parameters.get("services", ["market_analysis", "causal_analysis"])
            
            # Create subtasks for each service
            subtasks = []
            results = {}
            
            # Market Analysis
            if "market_analysis" in services:
                await self.update_task_status(
                    task_id=task_id,
                    progress=0.2,
                    message=f"Running market analysis for {symbol}"
                )
                
                market_params = parameters.get("market_analysis", {})
                market_task_id = f"{task_id}_market"
                
                # Create a task for market analysis
                await self.task_repository.create_task(
                    task_id=market_task_id,
                    service_type=AnalysisServiceType.MARKET_ANALYSIS,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=market_params
                )
                
                # Add to subtasks
                subtasks.append(self.market_analysis_workflow.execute(
                    task_id=market_task_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=market_params
                ))
            
            # Causal Analysis
            if "causal_analysis" in services:
                await self.update_task_status(
                    task_id=task_id,
                    progress=0.3,
                    message=f"Running causal analysis for {symbol}"
                )
                
                causal_params = parameters.get("causal_analysis", {})
                causal_task_id = f"{task_id}_causal"
                
                # Create a task for causal analysis
                await self.task_repository.create_task(
                    task_id=causal_task_id,
                    service_type=AnalysisServiceType.CAUSAL_ANALYSIS,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=causal_params
                )
                
                # Add to subtasks
                subtasks.append(self.causal_analysis_workflow.execute(
                    task_id=causal_task_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=causal_params
                ))
            
            # Backtesting
            if "backtesting" in services:
                await self.update_task_status(
                    task_id=task_id,
                    progress=0.4,
                    message=f"Running backtesting for {symbol}"
                )
                
                backtest_params = parameters.get("backtesting", {})
                backtest_task_id = f"{task_id}_backtest"
                
                # Create a task for backtesting
                await self.task_repository.create_task(
                    task_id=backtest_task_id,
                    service_type=AnalysisServiceType.BACKTESTING,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=backtest_params
                )
                
                # Add to subtasks
                subtasks.append(self.backtesting_workflow.execute(
                    task_id=backtest_task_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=backtest_params
                ))
            
            # Execute all subtasks concurrently
            await self.update_task_status(
                task_id=task_id,
                progress=0.5,
                message=f"Executing analysis tasks for {symbol}"
            )
            
            subtask_results = await asyncio.gather(*subtasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(subtask_results):
                if isinstance(result, Exception):
                    logger.error(f"Subtask failed: {str(result)}")
                    continue
                
                if result.service_type == AnalysisServiceType.MARKET_ANALYSIS:
                    results["market_analysis"] = result.result
                elif result.service_type == AnalysisServiceType.CAUSAL_ANALYSIS:
                    results["causal_analysis"] = result.result
                elif result.service_type == AnalysisServiceType.BACKTESTING:
                    results["backtesting"] = result.result
            
            # Integrate results
            await self.update_task_status(
                task_id=task_id,
                progress=0.8,
                message=f"Integrating analysis results for {symbol}"
            )
            
            integrated_results = self.aggregate_results(results)
            
            # Update task status to completed
            await self.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.COMPLETED,
                progress=1.0,
                message=f"Completed integrated analysis workflow for {symbol}",
                result=integrated_results
            )

            # Return the task result
            return AnalysisTaskResult(
                task_id=task_id,
                service_type="integrated_analysis",
                status=AnalysisTaskStatusEnum.COMPLETED,
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                result=integrated_results
            )

        except Exception as e:
            logger.error(f"Error executing integrated analysis workflow: {str(e)}")

            # Update task status to failed
            await self.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.FAILED,
                message=f"Failed to execute integrated analysis workflow: {str(e)}",
                error=str(e)
            )

            # Return the task result
            return AnalysisTaskResult(
                task_id=task_id,
                service_type="integrated_analysis",
                status=AnalysisTaskStatusEnum.FAILED,
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                error=str(e)
            )

    def aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results from multiple analysis services.

        Args:
            results: Dictionary of results from different services

        Returns:
            Aggregated results
        """
        aggregated = {
            "timestamp": datetime.now(UTC).isoformat(),
            "services_used": list(results.keys()),
            "summary": {},
            "details": results,
            "correlations": [],
            "recommendations": []
        }

        # Extract summaries from each service
        for service, result in results.items():
            if "summary" in result:
                aggregated["summary"][service] = result["summary"]

        # Generate cross-service correlations
        correlations = []

        # Market regime and performance correlation
        if "market_analysis" in results and "backtesting" in results:
            market_result = results["market_analysis"]
            backtest_result = results["backtesting"]
            
            if "summary" in market_result and "market_regime" in market_result["summary"]:
                regime = market_result["summary"]["market_regime"]["type"]
                
                if "summary" in backtest_result and "performance" in backtest_result["summary"]:
                    performance = backtest_result["summary"]["performance"]
                    
                    correlations.append({
                        "type": "regime_performance",
                        "description": f"Strategy performance in {regime} market regime",
                        "regime": regime,
                        "sharpe_ratio": performance.get("sharpe_ratio", 0.0),
                        "win_rate": backtest_result["summary"]["backtest"].get("win_rate", 0.0) if "backtest" in backtest_result["summary"] else 0.0
                    })

        # Pattern and causal factor correlation
        if "market_analysis" in results and "causal_analysis" in results:
            market_result = results["market_analysis"]
            causal_result = results["causal_analysis"]
            
            if "summary" in market_result and "patterns" in market_result["summary"]:
                patterns = market_result["summary"]["patterns"].get("types", {})
                
                if "summary" in causal_result and "causal_graph" in causal_result["summary"]:
                    factors = causal_result["summary"]["causal_graph"].get("key_factors", [])
                    
                    if patterns and factors:
                        correlations.append({
                            "type": "pattern_causal",
                            "description": "Correlation between detected patterns and causal factors",
                            "patterns": list(patterns.keys()),
                            "factors": [f["name"] for f in factors]
                        })

        aggregated["correlations"] = correlations

        # Generate recommendations
        recommendations = []

        # Trading recommendations based on market analysis
        if "market_analysis" in results:
            market_result = results["market_analysis"]
            
            if "summary" in market_result:
                summary = market_result["summary"]
                
                # Check for patterns
                if "patterns" in summary and summary["patterns"]["count"] > 0:
                    pattern_types = list(summary["patterns"]["types"].keys())
                    if pattern_types:
                        pattern = pattern_types[0]
                        if "head_and_shoulders" in pattern.lower() or "double_top" in pattern.lower():
                            recommendations.append({
                                "type": "trading",
                                "action": "sell",
                                "confidence": summary["patterns"]["confidence"],
                                "reason": f"Detected bearish {pattern} pattern"
                            })
                        elif "inverse_head_and_shoulders" in pattern.lower() or "double_bottom" in pattern.lower():
                            recommendations.append({
                                "type": "trading",
                                "action": "buy",
                                "confidence": summary["patterns"]["confidence"],
                                "reason": f"Detected bullish {pattern} pattern"
                            })
                
                # Check for market regime
                if "market_regime" in summary:
                    regime = summary["market_regime"]
                    if regime["type"] == "trending":
                        recommendations.append({
                            "type": "strategy",
                            "strategy": "trend_following",
                            "confidence": regime["confidence"],
                            "reason": "Market is in a trending regime"
                        })
                    elif regime["type"] == "ranging":
                        recommendations.append({
                            "type": "strategy",
                            "strategy": "mean_reversion",
                            "confidence": regime["confidence"],
                            "reason": "Market is in a ranging regime"
                        })
                    elif regime["type"] == "volatile":
                        recommendations.append({
                            "type": "strategy",
                            "strategy": "options_strategy",
                            "confidence": regime["confidence"],
                            "reason": "Market is in a volatile regime"
                        })

        # Strategy recommendations based on backtesting
        if "backtesting" in results:
            backtest_result = results["backtesting"]
            
            if "summary" in backtest_result:
                summary = backtest_result["summary"]
                
                if "backtest" in summary and "performance" in summary:
                    backtest = summary["backtest"]
                    performance = summary["performance"]
                    
                    if backtest["profit_factor"] > 1.5 and performance["sharpe_ratio"] > 1.0:
                        recommendations.append({
                            "type": "strategy",
                            "action": "deploy",
                            "confidence": min(backtest["profit_factor"] / 3, 1.0),
                            "reason": f"Strategy has good profit factor ({backtest['profit_factor']:.2f}) and Sharpe ratio ({performance['sharpe_ratio']:.2f})"
                        })
                    elif backtest["profit_factor"] < 1.0 or performance["sharpe_ratio"] < 0.5:
                        recommendations.append({
                            "type": "strategy",
                            "action": "optimize",
                            "confidence": 0.8,
                            "reason": f"Strategy has poor profit factor ({backtest['profit_factor']:.2f}) or Sharpe ratio ({performance['sharpe_ratio']:.2f})"
                        })

        # Risk management recommendations based on causal analysis
        if "causal_analysis" in results:
            causal_result = results["causal_analysis"]
            
            if "summary" in causal_result and "intervention_effect" in causal_result["summary"]:
                effects = causal_result["summary"]["intervention_effect"].get("top_effects", [])
                
                for effect in effects:
                    if "price" in effect["target"].lower() and effect["effect_size"] < -0.1:
                        recommendations.append({
                            "type": "risk",
                            "action": "hedge",
                            "confidence": min(abs(effect["effect_size"]) * 2, 1.0),
                            "reason": f"{effect['intervention']} has a significant negative effect on {effect['target']}"
                        })

        aggregated["recommendations"] = recommendations

        # Generate insights
        insights = []

        # Collect insights from each service
        for service, result in results.items():
            if "insights" in result:
                for insight in result["insights"]:
                    insights.append(f"[{service}] {insight}")

        # Add cross-service insights
        for correlation in correlations:
            insights.append(f"[correlation] {correlation['description']}")

        for recommendation in recommendations:
            insights.append(f"[recommendation] {recommendation['reason']}")

        aggregated["insights"] = insights

        return aggregated