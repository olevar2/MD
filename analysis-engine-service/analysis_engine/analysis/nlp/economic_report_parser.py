"""
Economic Report Parser Module

This module provides functionality for parsing and analyzing economic reports
such as NFP, GDP, CPI, and other market-moving economic data releases.
"""

from typing import Dict, List, Any, Union, Optional
import logging
import re
from datetime import datetime
import json

from analysis_engine.analysis.nlp.base_nlp_analyzer import BaseNLPAnalyzer
from analysis_engine.models.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

class EconomicReportParser(BaseNLPAnalyzer):
    """
    Parser for economic reports and data releases.
    
    This analyzer extracts key metrics from economic reports, compares them
    to expectations, and assesses potential market impact on forex pairs.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the economic report parser
        
        Args:
            parameters: Configuration parameters for the analyzer
        """
        default_params = {
            "spacy_model": "en_core_web_sm",
            "significant_deviation_threshold": 0.1,  # 10% deviation is considered significant
            "impact_mapping": {  # Default impact mapping for economic reports
                "NFP": {"high": ["USD"]},
                "GDP": {"high": ["USD", "EUR", "GBP", "AUD", "CAD", "JPY"]},
                "CPI": {"high": ["USD", "EUR", "GBP", "AUD", "CAD"], "medium": ["JPY", "CHF"]},
                "Retail Sales": {"high": ["USD", "GBP", "AUD"], "medium": ["EUR", "CAD"]},
                "PMI": {"high": ["EUR"], "medium": ["USD", "GBP"]},
                "Interest Rate Decision": {"high": ["USD", "EUR", "GBP", "AUD", "CAD", "JPY", "CHF", "NZD"]},
                "Unemployment Rate": {"high": ["USD", "EUR", "GBP", "AUD"], "medium": ["CAD", "JPY"]},
                "Trade Balance": {"medium": ["USD", "EUR", "GBP", "AUD", "CAD", "JPY"]},
                "Industrial Production": {"medium": ["USD", "EUR", "GBP"], "low": ["AUD", "CAD"]}
            },
            "currency_weight": {  # Default weight of each currency in major pairs
                "USD": 1.0,
                "EUR": 0.9,
                "JPY": 0.8,
                "GBP": 0.7,
                "AUD": 0.6,
                "CAD": 0.6,
                "CHF": 0.6,
                "NZD": 0.5
            }
        }
        
        # Merge default parameters with provided parameters
        merged_params = {**default_params, **(parameters or {})}
        super().__init__("economic_report_parser", merged_params)
        
    def _extract_metrics(self, report_text: str) -> Dict[str, Any]:
        """
        Extract key metrics from report text using regex patterns
        
        Args:
            report_text: Text of the economic report
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        # Extract values using regex patterns
        # Pattern for numbers with optional decimals, +/- signs, and % symbols
        value_pattern = r'(-?\d+\.?\d*%?)'
        
        # Common patterns in economic reports
        patterns = {
            "current": [
                r'actual:?\s*' + value_pattern,
                r'reported:?\s*' + value_pattern,
                r'came in at:?\s*' + value_pattern,
                r'released at:?\s*' + value_pattern,
                r'current:?\s*' + value_pattern,
                r'reading:?\s*' + value_pattern
            ],
            "previous": [
                r'previous:?\s*' + value_pattern,
                r'prior:?\s*' + value_pattern,
                r'last:?\s*' + value_pattern,
                r'earlier:?\s*' + value_pattern
            ],
            "forecast": [
                r'forecast:?\s*' + value_pattern,
                r'expected:?\s*' + value_pattern,
                r'estimate:?\s*' + value_pattern,
                r'consensus:?\s*' + value_pattern,
                r'expectation:?\s*' + value_pattern,
                r'projection:?\s*' + value_pattern,
                r'anticipated:?\s*' + value_pattern
            ],
            "revised": [
                r'revised:?\s*' + value_pattern,
                r'adjusted:?\s*' + value_pattern,
                r'updated:?\s*' + value_pattern
            ]
        }
        
        # Try to extract values for each metric type
        for metric_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, report_text, re.IGNORECASE)
                if matches:
                    # Take the first match
                    value_str = matches[0]
                    
                    # Convert to numeric if possible
                    try:
                        if '%' in value_str:
                            # Handle percentages
                            value = float(value_str.replace('%', '')) / 100
                            metrics[metric_type] = {
                                "value": value,
                                "display": value_str,
                                "is_percent": True
                            }
                        else:
                            # Handle regular numbers
                            value = float(value_str)
                            metrics[metric_type] = {
                                "value": value,
                                "display": value_str,
                                "is_percent": False
                            }
                    except ValueError:
                        # Keep as string if conversion fails
                        metrics[metric_type] = {
                            "value": value_str,
                            "display": value_str,
                            "is_percent": '%' in value_str
                        }
                    
                    # Stop after finding the first match for this metric type
                    break
        
        return metrics
        
    def _calculate_deviations(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate deviations from forecast and previous values
        
        Args:
            metrics: Extracted metrics
            
        Returns:
            Dictionary of calculated deviations
        """
        deviations = {}
        
        # Ensure we have numeric values for calculations
        current = metrics.get("current", {}).get("value")
        previous = metrics.get("previous", {}).get("value")
        forecast = metrics.get("forecast", {}).get("value")
        
        # Only calculate if values are numeric
        if isinstance(current, (int, float)):
            # Deviation from forecast
            if isinstance(forecast, (int, float)) and forecast != 0:
                deviations["from_forecast"] = (current - forecast) / abs(forecast)
                deviations["from_forecast_significant"] = abs(deviations["from_forecast"]) >= self.parameters["significant_deviation_threshold"]
            
            # Deviation from previous
            if isinstance(previous, (int, float)) and previous != 0:
                deviations["from_previous"] = (current - previous) / abs(previous)
                deviations["from_previous_significant"] = abs(deviations["from_previous"]) >= self.parameters["significant_deviation_threshold"]
        
        return deviations
    
    def _determine_report_type(self, report_data: Dict[str, Any]) -> str:
        """
        Determine the type of economic report based on title and content
        
        Args:
            report_data: Report data including title and content
            
        Returns:
            Type of report (e.g., "NFP", "GDP", "CPI", etc.)
        """
        title = report_data.get("title", "").lower()
        content = report_data.get("content", "").lower()
        
        # Check title and content for keywords
        text = title + " " + content
        
        # Common economic report types
        report_keywords = {
            "NFP": ["nonfarm payroll", "nonfarm payrolls", "non-farm payroll", "non farm payroll", "jobs report"],
            "GDP": ["gross domestic product", "gdp", "economic growth"],
            "CPI": ["consumer price index", "cpi", "inflation", "consumer prices"],
            "PPI": ["producer price index", "ppi", "producer prices"],
            "Retail Sales": ["retail sales", "consumer spending"],
            "PMI": ["purchasing managers index", "pmi", "manufacturing index", "services index"],
            "Interest Rate Decision": ["interest rate", "rate decision", "fomc", "central bank", "fed funds", "monetary policy"],
            "Unemployment Rate": ["unemployment rate", "jobless", "unemployment"],
            "Trade Balance": ["trade balance", "trade deficit", "trade surplus", "imports", "exports"],
            "Industrial Production": ["industrial production", "manufacturing output"]
        }
        
        for report_type, keywords in report_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return report_type
        
        # Default to generic "Economic Report" if no specific type is found
        return "Economic Report"
    
    def _determine_sentiment(self, deviations: Dict[str, float], report_type: str) -> Dict[str, Any]:
        """
        Determine sentiment based on deviations and report type
        
        Args:
            deviations: Calculated deviations
            report_type: Type of economic report
            
        Returns:
            Dictionary with sentiment information
        """
        sentiment = {
            "value": 0,  # -1 to 1 scale
            "label": "neutral",
            "confidence": 0.5
        }
        
        # Some economic indicators have inverse correlation (higher is negative)
        inverse_indicators = ["Unemployment Rate", "CPI", "PPI"]
        invert_reading = report_type in inverse_indicators
        
        # Calculate sentiment from deviations
        if "from_forecast" in deviations:
            deviation = deviations["from_forecast"]
            
            # Invert the deviation if necessary
            if invert_reading:
                deviation = -deviation
                
            # Calculate sentiment value (-1 to 1)
            # Limit to range [-1, 1]
            sentiment["value"] = max(-1, min(1, deviation * 3))  # Amplify deviation
            
            # Determine confidence based on deviation magnitude
            if abs(deviation) >= 0.2:  # Significant deviation
                sentiment["confidence"] = 0.9
            elif abs(deviation) >= 0.1:  # Moderate deviation
                sentiment["confidence"] = 0.7
            else:  # Minor deviation
                sentiment["confidence"] = 0.5
                
            # Assign sentiment label
            if sentiment["value"] > 0.3:
                sentiment["label"] = "bullish"
            elif sentiment["value"] > 0.1:
                sentiment["label"] = "slightly bullish"
            elif sentiment["value"] < -0.3:
                sentiment["label"] = "bearish"
            elif sentiment["value"] < -0.1:
                sentiment["label"] = "slightly bearish"
            else:
                sentiment["label"] = "neutral"
        
        return sentiment
    
    def _assess_currency_impacts(self, report_type: str, sentiment: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess impact on individual currencies
        
        Args:
            report_type: Type of economic report
            sentiment: Determined sentiment
            
        Returns:
            Dictionary mapping currencies to impact values
        """
        impact_mapping = self.parameters["impact_mapping"]
        currency_impacts = {}
        
        # Get currencies affected by this report type
        report_impacts = impact_mapping.get(report_type, {})
        
        # Calculate impact for each currency
        for impact_level, currencies in report_impacts.items():
            impact_multiplier = 1.0
            if impact_level == "high":
                impact_multiplier = 1.0
            elif impact_level == "medium":
                impact_multiplier = 0.7
            elif impact_level == "low":
                impact_multiplier = 0.4
            
            # Apply sentiment to each currency
            for currency in currencies:
                currency_weight = self.parameters["currency_weight"].get(currency, 0.5)
                
                # Calculate final impact value
                impact = sentiment["value"] * impact_multiplier * currency_weight * sentiment["confidence"]
                currency_impacts[currency] = impact
        
        return currency_impacts
    
    def _assess_pair_impacts(self, currency_impacts: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Assess impact on currency pairs based on individual currency impacts
        
        Args:
            currency_impacts: Impact on individual currencies
            
        Returns:
            Dictionary mapping currency pairs to impact information
        """
        pair_impacts = {}
        
        # Major currency pairs to assess
        major_pairs = [
            ("EUR", "USD"), ("USD", "JPY"), ("GBP", "USD"), ("USD", "CHF"), 
            ("USD", "CAD"), ("AUD", "USD"), ("NZD", "USD")
        ]
        
        # Calculate impact for each pair
        for base, quote in major_pairs:
            base_impact = currency_impacts.get(base, 0)
            quote_impact = currency_impacts.get(quote, 0)
            
            # Skip if neither currency is affected
            if base_impact == 0 and quote_impact == 0:
                continue
                
            # Calculate pair impact (positive means base strengthens against quote)
            # For USD/JPY and similar, positive means first currency strengthens
            if base == "USD" and quote in ["JPY", "CHF", "CAD"]:
                net_impact = base_impact - quote_impact
            else:
                # For EUR/USD and similar, positive means first currency strengthens
                net_impact = base_impact - quote_impact
            
            # Create pair name
            pair_name = f"{base}/{quote}"
            
            # Determine impact strength
            abs_impact = abs(net_impact)
            if abs_impact >= 0.5:
                strength = "strong"
            elif abs_impact >= 0.2:
                strength = "moderate"
            else:
                strength = "mild"
                
            # Determine direction label
            if net_impact > 0.1:
                direction = f"{pair_name} likely to rise"
            elif net_impact < -0.1:
                direction = f"{pair_name} likely to fall"
            else:
                direction = f"Limited impact on {pair_name}"
            
            # Store impact information
            pair_impacts[pair_name] = {
                "impact_value": net_impact,
                "impact_strength": strength,
                "direction": direction,
                "base_currency_impact": base_impact,
                "quote_currency_impact": quote_impact
            }
        
        return pair_impacts
                
    def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Analyze economic report data
        
        Args:
            data: Dictionary containing economic report data
            
        Returns:
            AnalysisResult containing analysis results
        """
        if not data or "report" not in data:
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"error": "No economic report data provided"},
                is_valid=False
            )
            
        report_data = data["report"]
        if not isinstance(report_data, dict):
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"error": "Invalid report data format"},
                is_valid=False
            )
            
        # Extract report content
        title = report_data.get("title", "")
        content = report_data.get("content", "")
        timestamp = report_data.get("timestamp", datetime.now().isoformat())
        source = report_data.get("source", "")
        
        if not content:
            return AnalysisResult(
                analyzer_name=self.name,
                result_data={"error": "Report content is empty"},
                is_valid=False
            )
            
        # Determine report type
        report_type = report_data.get("type") or self._determine_report_type(report_data)
        
        # Extract metrics from the report
        metrics = self._extract_metrics(content)
        
        # Calculate deviations
        deviations = self._calculate_deviations(metrics)
        
        # Determine sentiment
        sentiment = self._determine_sentiment(deviations, report_type)
        
        # Assess impact on individual currencies
        currency_impacts = self._assess_currency_impacts(report_type, sentiment)
        
        # Assess impact on currency pairs
        pair_impacts = self._assess_pair_impacts(currency_impacts)
        
        # Compile final results
        result = {
            "report_info": {
                "title": title,
                "type": report_type,
                "timestamp": timestamp,
                "source": source
            },
            "metrics": metrics,
            "deviations": deviations,
            "sentiment": sentiment,
            "currency_impacts": currency_impacts,
            "pair_impacts": pair_impacts,
            "trading_signals": []
        }
        
        # Generate trading signals for significant impacts
        for pair, impact in pair_impacts.items():
            if abs(impact["impact_value"]) >= 0.2:  # Only significant impacts
                signal = {
                    "pair": pair,
                    "action": "buy" if impact["impact_value"] > 0 else "sell",
                    "strength": impact["impact_strength"],
                    "rationale": f"{report_type} report indicates {impact['direction']}",
                    "timeframe": "short_term"  # Economic reports typically have short-term impact
                }
                result["trading_signals"].append(signal)
        
        return AnalysisResult(
            analyzer_name=self.name,
            result_data=result,
            is_valid=True
        )
