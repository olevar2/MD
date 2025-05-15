"""
Causal analysis workflow.
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
from analysis_coordinator_service.adapters.causal_analysis_adapter import CausalAnalysisAdapter

logger = logging.getLogger(__name__)


class CausalAnalysisWorkflow(BaseWorkflow):
    """
    Workflow for causal analysis.
    """

    def __init__(
        self,
        task_repository: TaskRepository,
        causal_analysis_adapter: CausalAnalysisAdapter
    ):
        """
        Initialize the workflow.

        Args:
            task_repository: Task repository for storing and retrieving tasks
            causal_analysis_adapter: Adapter for the causal analysis service
        """
        super().__init__(task_repository)
        self.causal_analysis_adapter = causal_analysis_adapter

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
        Execute the causal analysis workflow.

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
                message=f"Starting causal analysis workflow for {symbol}"
            )

            # Step 1: Generate causal graph
            await self.update_task_status(
                task_id=task_id,
                progress=0.3,
                message=f"Generating causal graph for {symbol}"
            )

            graph_params = parameters.get("causal_graph", {})
            graph_result = await self.causal_analysis_adapter.generate_causal_graph(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=graph_params
            )

            # Step 2: Analyze intervention effects
            await self.update_task_status(
                task_id=task_id,
                progress=0.6,
                message=f"Analyzing intervention effects for {symbol}"
            )

            intervention_params = parameters.get("intervention_effect", {})
            intervention_result = await self.causal_analysis_adapter.analyze_intervention_effect(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=intervention_params
            )

            # Step 3: Generate counterfactual scenarios
            await self.update_task_status(
                task_id=task_id,
                progress=0.8,
                message=f"Generating counterfactual scenarios for {symbol}"
            )

            counterfactual_params = parameters.get("counterfactual", {})
            counterfactual_result = await self.causal_analysis_adapter.generate_counterfactual(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=counterfactual_params
            )

            # Aggregate results
            results = {
                "causal_graph": graph_result,
                "intervention_effect": intervention_result,
                "counterfactual": counterfactual_result
            }

            aggregated_results = self.aggregate_results(results)

            # Update task status to completed
            await self.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.COMPLETED,
                progress=1.0,
                message=f"Completed causal analysis workflow for {symbol}",
                result=aggregated_results
            )

            # Return the task result
            return AnalysisTaskResult(
                task_id=task_id,
                service_type=AnalysisServiceType.CAUSAL_ANALYSIS,
                status=AnalysisTaskStatusEnum.COMPLETED,
                created_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                result=aggregated_results
            )

        except Exception as e:
            logger.error(f"Error executing causal analysis workflow: {str(e)}")

            # Update task status to failed
            await self.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.FAILED,
                message=f"Failed to execute causal analysis workflow: {str(e)}",
                error=str(e)
            )

            # Return the task result
            return AnalysisTaskResult(
                task_id=task_id,
                service_type=AnalysisServiceType.CAUSAL_ANALYSIS,
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

        # Extract causal graph information
        if "causal_graph" in results and results["causal_graph"]:
            graph = results["causal_graph"]
            graph_summary = {
                "node_count": len(graph.get("nodes", [])),
                "edge_count": len(graph.get("edges", [])),
                "key_factors": []
            }

            # Extract key factors (nodes with highest centrality)
            nodes = graph.get("nodes", [])
            if nodes:
                sorted_nodes = sorted(
                    nodes,
                    key=lambda x: x.get("centrality", 0.0),
                    reverse=True
                )
                for node in sorted_nodes[:3]:
                    graph_summary["key_factors"].append({
                        "name": node.get("name", "unknown"),
                        "centrality": node.get("centrality", 0.0)
                    })

            aggregated["summary"]["causal_graph"] = graph_summary

        # Extract intervention effect information
        if "intervention_effect" in results and results["intervention_effect"]:
            effects = results["intervention_effect"]
            effect_summary = {
                "count": len(effects.get("effects", [])),
                "top_effects": []
            }

            # Extract top effects
            effect_list = effects.get("effects", [])
            if effect_list:
                sorted_effects = sorted(
                    effect_list,
                    key=lambda x: abs(x.get("effect_size", 0.0)),
                    reverse=True
                )
                for effect in sorted_effects[:3]:
                    effect_summary["top_effects"].append({
                        "intervention": effect.get("intervention", "unknown"),
                        "target": effect.get("target", "unknown"),
                        "effect_size": effect.get("effect_size", 0.0)
                    })

            aggregated["summary"]["intervention_effect"] = effect_summary

        # Extract counterfactual information
        if "counterfactual" in results and results["counterfactual"]:
            counterfactuals = results["counterfactual"]
            cf_summary = {
                "count": len(counterfactuals.get("scenarios", [])),
                "top_scenarios": []
            }

            # Extract top scenarios
            scenarios = counterfactuals.get("scenarios", [])
            if scenarios:
                sorted_scenarios = sorted(
                    scenarios,
                    key=lambda x: abs(x.get("impact", 0.0)),
                    reverse=True
                )
                for scenario in sorted_scenarios[:3]:
                    cf_summary["top_scenarios"].append({
                        "description": scenario.get("description", "unknown"),
                        "impact": scenario.get("impact", 0.0)
                    })

            aggregated["summary"]["counterfactual"] = cf_summary

        # Generate insights
        insights = []

        # Causal graph insights
        if "causal_graph" in aggregated["summary"]:
            graph_summary = aggregated["summary"]["causal_graph"]
            if graph_summary["key_factors"]:
                factors = [f"{factor['name']} (centrality: {factor['centrality']:.2f})" for factor in graph_summary["key_factors"]]
                insights.append(f"Key causal factors: {', '.join(factors)}")

        # Intervention effect insights
        if "intervention_effect" in aggregated["summary"]:
            effect_summary = aggregated["summary"]["intervention_effect"]
            if effect_summary["top_effects"]:
                for effect in effect_summary["top_effects"]:
                    direction = "positive" if effect["effect_size"] > 0 else "negative"
                    insights.append(f"{effect['intervention']} has a {direction} effect on {effect['target']} (size: {abs(effect['effect_size']):.2f})")

        # Counterfactual insights
        if "counterfactual" in aggregated["summary"]:
            cf_summary = aggregated["summary"]["counterfactual"]
            if cf_summary["top_scenarios"]:
                for scenario in cf_summary["top_scenarios"]:
                    direction = "positive" if scenario["impact"] > 0 else "negative"
                    insights.append(f"Scenario: {scenario['description']} would have a {direction} impact of {abs(scenario['impact']):.2f}")

        aggregated["insights"] = insights

        return aggregated