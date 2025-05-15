"""
Backtesting workflow.
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
from analysis_coordinator_service.adapters.backtesting_adapter import BacktestingAdapter

logger = logging.getLogger(__name__)


class BacktestingWorkflow(BaseWorkflow):
    """
    Workflow for backtesting.
    """

    def __init__(
        self,
        task_repository: TaskRepository,
        backtesting_adapter: BacktestingAdapter
    ):
        """
        Initialize the workflow.

        Args:
            task_repository: Task repository for storing and retrieving tasks
            backtesting_adapter: Adapter for the backtesting service
        """
        super().__init__(task_repository)
        self.backtesting_adapter = backtesting_adapter

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
        Execute the backtesting workflow.

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
                message=f"Starting backtesting workflow for {symbol}"
            )

            # Step 1: Run backtest
            await self.update_task_status(
                task_id=task_id,
                progress=0.3,
                message=f"Running backtest for {symbol}"
            )

            backtest_params = parameters.get("backtest", {})
            backtest_result = await self.backtesting_adapter.run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=backtest_params
            )

            # Step 2: Analyze performance
            await self.update_task_status(
                task_id=task_id,
                progress=0.6,
                message=f"Analyzing performance for {symbol}"
            )

            performance_params = parameters.get("performance", {})
            performance_result = await self.backtesting_adapter.analyze_performance(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=performance_params
            )

            # Step 3: Run optimization (if requested)
            optimization_result = None
            if parameters.get("run_optimization", False):
                await self.update_task_status(
                    task_id=task_id,
                    progress=0.8,
                    message=f"Running optimization for {symbol}"
                )

                optimization_params = parameters.get("optimization", {})
                optimization_result = await self.backtesting_adapter.run_optimization(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=optimization_params
                )

            # Aggregate results
            results = {
                "backtest": backtest_result,
                "performance": performance_result
            }

            if optimization_result:
                results["optimization"] = optimization_result

            aggregated_results = self.aggregate_results(results)

            # Update task status to completed
            await self.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.COMPLETED,
                progress=1.0,
                message=f"Completed backtesting workflow for {symbol}",
                result=aggregated_results
            )

            # Return the task result
            return AnalysisTaskResult(
                task_id=task_id,
                service_type=AnalysisServiceType.BACKTESTING,
                status=AnalysisTaskStatusEnum.COMPLETED,
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                result=aggregated_results
            )

        except Exception as e:
            logger.error(f"Error executing backtesting workflow: {str(e)}")

            # Update task status to failed
            await self.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.FAILED,
                message=f"Failed to execute backtesting workflow: {str(e)}",
                error=str(e)
            )

            # Return the task result
            return AnalysisTaskResult(
                task_id=task_id,
                service_type=AnalysisServiceType.BACKTESTING,
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

        # Extract backtest information
        if "backtest" in results and results["backtest"]:
            backtest = results["backtest"]
            backtest_summary = {
                "total_trades": backtest.get("total_trades", 0),
                "win_rate": backtest.get("win_rate", 0.0),
                "profit_factor": backtest.get("profit_factor", 0.0),
                "net_profit": backtest.get("net_profit", 0.0),
                "max_drawdown": backtest.get("max_drawdown", 0.0)
            }
            aggregated["summary"]["backtest"] = backtest_summary

        # Extract performance information
        if "performance" in results and results["performance"]:
            performance = results["performance"]
            performance_summary = {
                "sharpe_ratio": performance.get("sharpe_ratio", 0.0),
                "sortino_ratio": performance.get("sortino_ratio", 0.0),
                "calmar_ratio": performance.get("calmar_ratio", 0.0),
                "annual_return": performance.get("annual_return", 0.0),
                "volatility": performance.get("volatility", 0.0)
            }
            aggregated["summary"]["performance"] = performance_summary

        # Extract optimization information
        if "optimization" in results and results["optimization"]:
            optimization = results["optimization"]
            optimization_summary = {
                "iterations": optimization.get("iterations", 0),
                "best_parameters": optimization.get("best_parameters", {}),
                "best_score": optimization.get("best_score", 0.0)
            }
            aggregated["summary"]["optimization"] = optimization_summary

        # Generate insights
        insights = []

        # Backtest insights
        if "backtest" in aggregated["summary"]:
            backtest_summary = aggregated["summary"]["backtest"]
            insights.append(f"Strategy executed {backtest_summary['total_trades']} trades with a win rate of {backtest_summary['win_rate']*100:.1f}%")
            insights.append(f"Net profit: {backtest_summary['net_profit']:.2f} with a profit factor of {backtest_summary['profit_factor']:.2f}")
            insights.append(f"Maximum drawdown: {backtest_summary['max_drawdown']*100:.1f}%")

        # Performance insights
        if "performance" in aggregated["summary"]:
            performance_summary = aggregated["summary"]["performance"]
            insights.append(f"Sharpe ratio: {performance_summary['sharpe_ratio']:.2f}, Sortino ratio: {performance_summary['sortino_ratio']:.2f}")
            insights.append(f"Annual return: {performance_summary['annual_return']*100:.1f}% with volatility of {performance_summary['volatility']*100:.1f}%")

        # Optimization insights
        if "optimization" in aggregated["summary"]:
            optimization_summary = aggregated["summary"]["optimization"]
            insights.append(f"Optimization ran {optimization_summary['iterations']} iterations")
            if optimization_summary["best_parameters"]:
                params = [f"{k}={v}" for k, v in optimization_summary["best_parameters"].items()]
                insights.append(f"Best parameters: {', '.join(params)} with score {optimization_summary['best_score']:.2f}")

        aggregated["insights"] = insights

        return aggregated