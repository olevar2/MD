"""
NLP Integration Module

This module provides integration between the NLP components and the rest of
the analysis engine, ensuring that NLP-derived signals and insights are
properly incorporated into the trading decision process.
"""

from typing import Dict, List, Any, Union, Optional
import logging
import asyncio
from datetime import datetime

from analysis_engine.analysis.nlp.news_analyzer import NewsAnalyzer
from analysis_engine.analysis.nlp.economic_report_parser import EconomicReportParser
from analysis_engine.models.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

class NLPIntegration:
    """
    Integration layer between NLP components and the decision engine.
    
    This class handles the aggregation of NLP insights, integration with
    other analysis components, and preparation of NLP-derived signals for
    the decision engine.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the NLP integration
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.news_analyzer = NewsAnalyzer(self.config.get("news_analyzer_params"))
        self.economic_report_parser = EconomicReportParser(self.config.get("economic_report_parser_params"))
        
    async def process_news_data(self, news_data: Dict[str, Any]) -> AnalysisResult:
        """
        Process news data using the news analyzer
        
        Args:
            news_data: Dictionary containing news items
            
        Returns:
            AnalysisResult with news analysis results
        """
        return self.news_analyzer.analyze(news_data)
    
    async def process_economic_report(self, report_data: Dict[str, Any]) -> AnalysisResult:
        """
        Process economic report data
        
        Args:
            report_data: Dictionary containing economic report data
            
        Returns:
            AnalysisResult with economic report analysis results
        """
        return self.economic_report_parser.analyze(report_data)
    
    async def generate_nlp_insights(self, 
                              news_data: Optional[Dict[str, Any]] = None,
                              economic_reports: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive NLP insights by combining news and economic report analysis
        
        Args:
            news_data: Dictionary containing news items
            economic_reports: List of economic report data
            
        Returns:
            Dictionary with combined NLP insights
        """
        tasks = []
        results = {}
        
        # Process news data if available
        if news_data:
            news_task = self.process_news_data(news_data)
            tasks.append(news_task)
        
        # Process economic reports if available
        report_tasks = []
        if economic_reports:
            for report in economic_reports:
                report_task = self.process_economic_report({"report": report})
                report_tasks.append(report_task)
                tasks.append(report_task)
        
        # Run all analysis tasks in parallel
        if tasks:
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Extract news results
            if news_data:
                news_result = completed_tasks[0]
                if isinstance(news_result, Exception):
                    logger.error(f"Error processing news data: {news_result}")
                    results["news_analysis"] = {"error": str(news_result)}
                else:
                    results["news_analysis"] = news_result.result_data
                
                # Remove news result from completed tasks
                completed_tasks = completed_tasks[1:]
            
            # Extract economic report results
            if economic_reports:
                report_results = []
                for i, result in enumerate(completed_tasks):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing economic report: {result}")
                        report_results.append({"error": str(result)})
                    else:
                        report_results.append(result.result_data)
                
                results["economic_report_analysis"] = report_results
        
        # Aggregate insights across all analyses
        currency_pair_insights = self._aggregate_pair_insights(results)
        
        # Add aggregate results
        results["aggregate_insights"] = {
            "currency_pair_insights": currency_pair_insights,
            "generated_at": datetime.now().isoformat(),
        }
        
        return results
    
    def _aggregate_pair_insights(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate insights for each currency pair from all analyses
        
        Args:
            results: Dictionary with analysis results
            
        Returns:
            Dictionary mapping currency pairs to aggregated insights
        """
        pair_insights = {}
        
        # Process news analysis results
        news_analysis = results.get("news_analysis", {})
        if "pair_summary" in news_analysis:
            for pair, summary in news_analysis["pair_summary"].items():
                if pair not in pair_insights:
                    pair_insights[pair] = {
                        "news_sentiment": 0,
                        "news_impact": 0,
                        "news_count": 0,
                        "economic_impact": 0,
                        "economic_reports": [],
                        "combined_sentiment": 0,
                        "combined_impact": 0
                    }
                
                pair_insights[pair]["news_sentiment"] = summary.get("avg_direction", 0)
                pair_insights[pair]["news_impact"] = summary.get("avg_impact", 0)
                pair_insights[pair]["news_count"] = summary.get("count", 0)
        
        # Process economic report results
        econ_reports = results.get("economic_report_analysis", [])
        for report in econ_reports:
            if "pair_impacts" in report:
                for pair, impact in report["pair_impacts"].items():
                    if pair not in pair_insights:
                        pair_insights[pair] = {
                            "news_sentiment": 0,
                            "news_impact": 0,
                            "news_count": 0,
                            "economic_impact": 0,
                            "economic_reports": [],
                            "combined_sentiment": 0,
                            "combined_impact": 0
                        }
                    
                    # Add this report's impact to the pair
                    impact_value = impact.get("impact_value", 0)
                    pair_insights[pair]["economic_impact"] += impact_value
                    
                    # Add report reference to the pair
                    if "report_info" in report:
                        report_info = {
                            "type": report["report_info"].get("type", ""),
                            "timestamp": report["report_info"].get("timestamp", ""),
                            "impact_value": impact_value,
                            "impact_strength": impact.get("impact_strength", "")
                        }
                        pair_insights[pair]["economic_reports"].append(report_info)
        
        # Calculate combined metrics for each pair
        for pair, insights in pair_insights.items():
            # Balance news impact and economic impact
            news_weight = min(insights["news_count"] / 5, 1.0) if insights["news_count"] > 0 else 0
            econ_weight = min(len(insights["economic_reports"]), 1.0)
            
            # If both sources are available, combine them; otherwise use whatever is available
            if news_weight > 0 and econ_weight > 0:
                insights["combined_sentiment"] = (insights["news_sentiment"] * news_weight + 
                                                insights["economic_impact"] * econ_weight) / (news_weight + econ_weight)
                insights["combined_impact"] = (insights["news_impact"] * news_weight + 
                                             abs(insights["economic_impact"]) * econ_weight) / (news_weight + econ_weight)
            elif news_weight > 0:
                insights["combined_sentiment"] = insights["news_sentiment"]
                insights["combined_impact"] = insights["news_impact"]
            elif econ_weight > 0:
                insights["combined_sentiment"] = insights["economic_impact"]
                insights["combined_impact"] = abs(insights["economic_impact"])
            
        return pair_insights
