"""
Trading Incident Runbooks for forex trading emergencies.

This module contains detailed runbooks for handling various trading incidents
that may occur during forex trading operations.
"""

import json
import os
from typing import Dict, Any, Optional, List

# Default runbooks for common forex trading incidents
FOREX_TRADING_RUNBOOKS = {
    # Market data incidents
    "market_data": {
        "title": "Market Data Incident Response",
        "description": "Procedures for handling market data issues in forex trading",
        "incidents": [
            {
                "title": "Price Feed Interruption",
                "severity": "high",
                "description": "Loss of price feed from primary data source",
                "immediate_actions": [
                    "Identify affected currency pairs and timeframes",
                    "Check connectivity with data provider",
                    "Switch to backup data source if available",
                    "Suspend automated trading on affected pairs"
                ],
                "root_cause_investigation": [
                    "Check provider status pages and announcements",
                    "Verify network connectivity and DNS resolution",
                    "Review API rate limits and usage",
                    "Check system logs for connection errors"
                ],
                "resolution_steps": [
                    "Restore primary data connection",
                    "Verify data consistency before resuming trading",
                    "Gradually re-enable automated trading",
                    "Document incident and update monitoring if needed"
                ],
                "prevention": [
                    "Implement redundant data sources",
                    "Add data quality monitoring alerts",
                    "Regular connectivity tests with providers"
                ]
            },
            {
                "title": "Price Spikes or Anomalies",
                "severity": "medium",
                "description": "Unusual price movements that may be data errors",
                "immediate_actions": [
                    "Identify affected currency pairs",
                    "Compare data across multiple sources",
                    "Flag suspicious movements for investigation",
                    "Apply spike filters to affected feeds"
                ],
                "root_cause_investigation": [
                    "Verify if movement is reflected across providers",
                    "Check for major news events that might explain volatility",
                    "Look for patterns in anomalies (specific time periods, etc.)",
                    "Review historical data for similar patterns"
                ],
                "resolution_steps": [
                    "Correct or filter anomalous data if confirmed as errors",
                    "Update spike detection parameters if needed",
                    "Re-process affected signals with correct data",
                    "Document findings for future reference"
                ],
                "prevention": [
                    "Enhance data validation rules",
                    "Implement cross-source verification",
                    "Improve anomaly detection algorithms"
                ]
            },
            {
                "title": "Delayed Market Data",
                "severity": "medium",
                "description": "Market data arriving with significant delay",
                "immediate_actions": [
                    "Measure and document the delay",
                    "Check if delay affects all currency pairs or specific ones",
                    "Throttle trading frequency to match data delay",
                    "Consider switching to alternative data source"
                ],
                "root_cause_investigation": [
                    "Check network latency between system and data providers",
                    "Review system resource utilization",
                    "Investigate potential bottlenecks in data processing pipeline",
                    "Contact provider to check for known issues"
                ],
                "resolution_steps": [
                    "Optimize data processing pipeline",
                    "Increase resource allocation if needed",
                    "Implement better queuing mechanisms for data processing",
                    "Update latency monitoring thresholds"
                ],
                "prevention": [
                    "Regular performance testing of data pipeline",
                    "Implement redundant data paths",
                    "Add detailed latency monitoring by currency pair"
                ]
            }
        ]
    },
    
    # Execution incidents
    "execution": {
        "title": "Trade Execution Incident Response",
        "description": "Procedures for handling trade execution issues in forex trading",
        "incidents": [
            {
                "title": "Order Rejection Pattern",
                "severity": "high",
                "description": "Multiple orders being rejected by broker",
                "immediate_actions": [
                    "Pause automated order submission",
                    "Collect rejection reasons and error codes",
                    "Check account status and margin levels",
                    "Verify that trading parameters are within broker limits"
                ],
                "root_cause_investigation": [
                    "Analyze patterns in rejected orders",
                    "Check for recent changes to order parameters",
                    "Review broker documentation for rejection codes",
                    "Verify compatibility with recent broker API changes"
                ],
                "resolution_steps": [
                    "Adjust order parameters to comply with broker requirements",
                    "Implement gradual order submission with verification",
                    "Test with small orders before resuming full trading",
                    "Add specific handling for identified rejection scenarios"
                ],
                "prevention": [
                    "Regular review of broker requirements and limits",
                    "Pre-submission validation for order parameters",
                    "Monitoring of rejection rates with alerting"
                ]
            },
            {
                "title": "Severe Slippage Events",
                "severity": "high",
                "description": "Orders executing with significantly higher slippage than expected",
                "immediate_actions": [
                    "Identify affected currency pairs and order types",
                    "Compare execution prices with market data across sources",
                    "Increase slippage tolerance temporarily if required",
                    "Consider switching to limit orders instead of market orders"
                ],
                "root_cause_investigation": [
                    "Check market conditions for unusual volatility",
                    "Review recent economic announcements and news",
                    "Analyze liquidity levels at execution times",
                    "Compare execution quality across different brokers (if available)"
                ],
                "resolution_steps": [
                    "Adjust execution algorithms for current market conditions",
                    "Implement dynamic slippage tolerance based on volatility",
                    "Consider alternative execution venues if available",
                    "Update risk parameters to account for higher execution uncertainty"
                ],
                "prevention": [
                    "Implement pre-trade liquidity analysis",
                    "Add volatility-based execution rules",
                    "Include economic calendar awareness in trading system"
                ]
            },
            {
                "title": "Order Stuck in Pending",
                "severity": "medium",
                "description": "Orders remain in pending state without execution or rejection",
                "immediate_actions": [
                    "Identify affected orders and their details",
                    "Check connectivity with broker execution system",
                    "Attempt to cancel stuck orders manually",
                    "Pause new order submission until resolved"
                ],
                "root_cause_investigation": [
                    "Check order status via alternative interfaces (web portal, etc.)",
                    "Review system logs for communication errors",
                    "Check for broker system status announcements",
                    "Verify order parameters are still valid (price, expiry, etc.)"
                ],
                "resolution_steps": [
                    "Cancel and replace stuck orders if needed",
                    "Re-establish broker connectivity",
                    "Verify order book consistency after resolution",
                    "Implement order timeout and automatic cancellation"
                ],
                "prevention": [
                    "Add order state monitoring with timeouts",
                    "Implement automatic cancellation for orders pending too long",
                    "Add redundant order status checking mechanisms"
                ]
            }
        ]
    },
    
    # Connectivity incidents
    "connectivity": {
        "title": "Connectivity Incident Response",
        "description": "Procedures for handling connectivity issues with brokers and data providers",
        "incidents": [
            {
                "title": "Complete Broker Disconnection",
                "severity": "critical",
                "description": "Total loss of connectivity with broker",
                "immediate_actions": [
                    "Verify the nature of the disconnection (local vs. broker-side)",
                    "Pause all trading activities",
                    "Document current open positions and orders",
                    "Attempt to access broker through alternative channels"
                ],
                "root_cause_investigation": [
                    "Check network connectivity and routes to broker",
                    "Verify broker system status through official channels",
                    "Check for API authentication or certificate issues",
                    "Review recent changes to connection parameters"
                ],
                "resolution_steps": [
                    "Establish connection through backup channels if available",
                    "Reconnect using primary channel when available",
                    "Reconcile positions and orders after reconnection",
                    "Resume trading gradually, starting with monitoring only"
                ],
                "prevention": [
                    "Implement redundant connectivity paths",
                    "Regular connection health checks and heartbeat monitoring",
                    "Establish emergency communication protocols with broker",
                    "Maintain updated broker contact information for emergencies"
                ]
            },
            {
                "title": "Intermittent Connection Drops",
                "severity": "high",
                "description": "Periodic loss of connectivity affecting reliability",
                "immediate_actions": [
                    "Document pattern and frequency of disconnections",
                    "Enable more aggressive reconnection logic",
                    "Reduce trading frequency to match reliable connection windows",
                    "Consider switching to more robust order types"
                ],
                "root_cause_investigation": [
                    "Monitor network quality metrics (latency, packet loss)",
                    "Check for patterns related to market activity or time of day",
                    "Review system resource utilization during drops",
                    "Test alternative network paths if available"
                ],
                "resolution_steps": [
                    "Optimize connection parameters (timeout, keepalive)",
                    "Implement connection pooling if supported",
                    "Consider infrastructure improvements if network-related",
                    "Update reconnection logic based on findings"
                ],
                "prevention": [
                    "Implement connection quality monitoring",
                    "Add circuit breakers triggered by connection instability",
                    "Develop robust state recovery mechanisms",
                    "Regular stress testing of reconnection capabilities"
                ]
            }
        ]
    },
    
    # Risk management incidents
    "risk": {
        "title": "Risk Management Incident Response",
        "description": "Procedures for handling risk-related incidents in forex trading",
        "incidents": [
            {
                "title": "Margin Call or Closeout",
                "severity": "critical",
                "description": "Account approaching or experiencing margin call situation",
                "immediate_actions": [
                    "Pause all new position opening",
                    "Document current margin level and requirements",
                    "Identify positions contributing most to margin usage",
                    "Prepare for controlled position reduction"
                ],
                "root_cause_investigation": [
                    "Analyze recent trading activity and position growth",
                    "Check for sudden market movements affecting equity",
                    "Review position sizing parameters and risk settings",
                    "Verify margin calculation is functioning correctly"
                ],
                "resolution_steps": [
                    "Reduce positions in order of risk contribution",
                    "Add funds to account if appropriate and available",
                    "Adjust position sizing algorithms to prevent recurrence",
                    "Document lessons learned and update risk limits"
                ],
                "prevention": [
                    "Implement proactive margin monitoring with early warnings",
                    "Add circuit breakers based on margin utilization",
                    "Regular stress testing of portfolio under extreme conditions",
                    "Diversify position concentration risks"
                ]
            },
            {
                "title": "Drawdown Limit Breach",
                "severity": "high",
                "description": "Trading strategy exceeding maximum allowable drawdown",
                "immediate_actions": [
                    "Pause trading for the affected strategy",
                    "Document current drawdown level and circumstances",
                    "Check for correlation with market conditions",
                    "Verify drawdown calculation accuracy"
                ],
                "root_cause_investigation": [
                    "Analyze performance across different market conditions",
                    "Review recent trades for pattern of losses",
                    "Check for strategy parameter drift",
                    "Test strategy on recent data with different parameters"
                ],
                "resolution_steps": [
                    "Adjust strategy parameters if issues identified",
                    "Consider reducing position size upon restart",
                    "Implement gradual recovery mode with lower risk",
                    "Add additional risk filters before re-enabling"
                ],
                "prevention": [
                    "Implement progressive risk reduction as drawdown increases",
                    "Regular strategy performance review across regimes",
                    "Add correlation analysis with market conditions",
                    "Multiple drawdown measures (time-based windows)"
                ]
            },
            {
                "title": "Unexpected Exposure Increase",
                "severity": "high",
                "description": "Sudden increase in exposure to specific currency or risk factor",
                "immediate_actions": [
                    "Identify source of increased exposure",
                    "Temporarily cap further exposure to affected currencies",
                    "Document current exposure levels across portfolio",
                    "Validate that position reporting is accurate"
                ],
                "root_cause_investigation": [
                    "Check for changes in correlation between positions",
                    "Review recent trading signals and decision factors",
                    "Verify exposure calculation methodology",
                    "Check for market structure changes affecting correlations"
                ],
                "resolution_steps": [
                    "Rebalance portfolio to reduce concentrated exposure",
                    "Implement temporary correlation-aware position limits",
                    "Update exposure calculation to capture identified risks",
                    "Add monitoring for similar patterns in the future"
                ],
                "prevention": [
                    "Enhance exposure monitoring across currency pairs",
                    "Implement correlation-aware position sizing",
                    "Regular stress testing for concentration risks",
                    "Add currency exposure limits and alerts"
                ]
            }
        ]
    },
    
    # System incidents
    "system": {
        "title": "System Incident Response",
        "description": "Procedures for handling system-related incidents in forex trading",
        "incidents": [
            {
                "title": "Performance Degradation",
                "severity": "high",
                "description": "System showing significant latency or processing delays",
                "immediate_actions": [
                    "Identify components showing degraded performance",
                    "Reduce processing load if possible",
                    "Monitor resource utilization (CPU, memory, disk, network)",
                    "Scale resources if possible and needed"
                ],
                "root_cause_investigation": [
                    "Analyze system metrics to identify bottlenecks",
                    "Check for recent changes in data volume or traffic",
                    "Review recent deployments or configuration changes",
                    "Examine database performance and query patterns"
                ],
                "resolution_steps": [
                    "Optimize identified bottlenecks",
                    "Increase resources for affected components",
                    "Consider architectural changes if structural issue",
                    "Implement performance testing to validate improvements"
                ],
                "prevention": [
                    "Regular performance testing and benchmarking",
                    "Resource utilization monitoring and alerts",
                    "Capacity planning and scaling policies",
                    "Performance regression testing for deployments"
                ]
            },
            {
                "title": "Critical Service Failure",
                "severity": "critical",
                "description": "Failure of a critical system component affecting trading",
                "immediate_actions": [
                    "Identify failed component and affected dependencies",
                    "Pause trading activities dependent on the component",
                    "Activate relevant backup systems if available",
                    "Notify all stakeholders according to severity"
                ],
                "root_cause_investigation": [
                    "Review system logs for error messages and patterns",
                    "Check for recent changes that might have contributed",
                    "Verify external dependencies and their status",
                    "Inspect resource utilization leading up to failure"
                ],
                "resolution_steps": [
                    "Restore service using established recovery procedures",
                    "Verify data integrity after recovery",
                    "Validate system consistency across components",
                    "Resume operations with careful monitoring"
                ],
                "prevention": [
                    "Implement robust service monitoring and health checks",
                    "Regular disaster recovery testing",
                    "Component redundancy for critical services",
                    "Automated recovery procedures where possible"
                ]
            },
            {
                "title": "Data Consistency Issue",
                "severity": "high",
                "description": "Inconsistencies detected between system components",
                "immediate_actions": [
                    "Document the nature and extent of inconsistencies",
                    "Restrict operations that might worsen the problem",
                    "Create point-in-time backup of current state if possible",
                    "Identify transactions or updates that might be affected"
                ],
                "root_cause_investigation": [
                    "Trace data flow between affected systems",
                    "Review recent schema or data model changes",
                    "Check for race conditions or timing issues",
                    "Verify transaction integrity mechanisms"
                ],
                "resolution_steps": [
                    "Reconcile inconsistent data using reliable sources",
                    "Implement data validation and correction procedures",
                    "Verify consistency after corrections",
                    "Document recovery process for future reference"
                ],
                "prevention": [
                    "Implement data consistency checks and alerts",
                    "Add transaction integrity guarantees where needed",
                    "Regular data reconciliation processes",
                    "Improve isolation between system components"
                ]
            }
        ]
    }
}


def load_runbooks(custom_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load runbooks from file or use defaults.
    
    Args:
        custom_path: Optional path to custom runbooks file
        
    Returns:
        Dictionary of runbooks
    """
    if custom_path and os.path.exists(custom_path):
        try:
            with open(custom_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading custom runbooks: {str(e)}")
            print("Falling back to default runbooks")
            
    return FOREX_TRADING_RUNBOOKS


def get_runbook_for_incident(
    category: str,
    title: Optional[str] = None,
    severity: Optional[str] = None,
    runbooks: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Get a specific runbook for an incident.
    
    Args:
        category: Incident category
        title: Specific incident title to look for
        severity: Incident severity level
        runbooks: Runbooks dictionary (loads defaults if None)
        
    Returns:
        Matching runbook or None if not found
    """
    # Load default runbooks if none provided
    if runbooks is None:
        runbooks = FOREX_TRADING_RUNBOOKS
        
    # Check if category exists
    if category not in runbooks:
        return None
        
    category_runbooks = runbooks[category]
    incidents = category_runbooks.get("incidents", [])
    
    # If no specific title or severity, return category overview
    if not title and not severity:
        # Return category-level information without incidents details
        return {
            "title": category_runbooks.get("title", f"{category.capitalize()} Incidents"),
            "description": category_runbooks.get("description", "")
        }
    
    # Search for matching incident
    for incident in incidents:
        # Match by exact title if provided
        if title and incident.get("title") == title:
            return incident
            
        # Match by severity if title not provided or not found
        if not title and severity and incident.get("severity") == severity:
            return incident
            
    # If we get here with a severity but no match, return the highest severity incident
    if severity and not title:
        severity_levels = ["critical", "high", "medium", "low"]
        
        # Try matching from highest to lowest severity
        for sev in severity_levels:
            for incident in incidents:
                if incident.get("severity") == sev:
                    return incident
                    
    return None
