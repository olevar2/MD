"""
Feature Store Dashboard Module

This module provides a dashboard for monitoring the feature store client.
"""
import time
import threading
import logging
import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio

from core_foundations.utils.logger import get_logger
from strategy_execution_engine.monitoring.feature_store_metrics import feature_store_metrics

# Try to import Dash for dashboard
try:
    import dash
    from dash import dcc, html
    import dash_bootstrap_components as dbc
    from dash.dependencies import Input, Output
    import plotly.graph_objs as go
    has_dash = True
except ImportError:
    has_dash = False


class FeatureStoreDashboard:
    """
    Dashboard for monitoring the feature store client.
    
    This class provides a web dashboard for monitoring the feature store client,
    including API calls, cache hits/misses, errors, and performance metrics.
    
    Attributes:
        logger: Logger instance
        app: Dash application
        update_interval: Interval for updating the dashboard in seconds
    """
    
    def __init__(self, update_interval: int = 5, port: int = 8050):
        """
        Initialize the dashboard.
        
        Args:
            update_interval: Interval for updating the dashboard in seconds
            port: Port to run the dashboard on
        """
        self.logger = get_logger("feature_store_dashboard")
        self.update_interval = update_interval
        self.port = port
        
        if not has_dash:
            self.logger.warning("Dash not installed, dashboard not available")
            return
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="Feature Store Monitoring"
        )
        
        # Set up layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
        
        self.logger.info("Feature store dashboard initialized")
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        if not has_dash:
            return
        
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Feature Store Monitoring Dashboard", className="text-center my-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("API Calls"),
                    dcc.Graph(id="api-calls-graph"),
                ], width=6),
                dbc.Col([
                    html.H3("Cache Performance"),
                    dcc.Graph(id="cache-performance-graph"),
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Errors"),
                    dcc.Graph(id="errors-graph"),
                ], width=6),
                dbc.Col([
                    html.H3("Response Time"),
                    dcc.Graph(id="response-time-graph"),
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Fallbacks"),
                    dcc.Graph(id="fallbacks-graph"),
                ], width=6),
                dbc.Col([
                    html.H3("Summary"),
                    html.Div(id="summary-div", className="p-3 bg-light rounded"),
                ], width=6)
            ]),
            
            dcc.Interval(
                id="interval-component",
                interval=self.update_interval * 1000,  # in milliseconds
                n_intervals=0
            )
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        if not has_dash:
            return
        
        @self.app.callback(
            Output("api-calls-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_api_calls_graph(n):
            metrics = feature_store_metrics.get_metrics()
            
            # Create bar chart for API calls
            labels = list(metrics["api_calls"].keys())
            values = list(metrics["api_calls"].values())
            
            return {
                "data": [
                    go.Bar(
                        x=labels,
                        y=values,
                        marker=dict(color="royalblue")
                    )
                ],
                "layout": go.Layout(
                    title="API Calls by Method",
                    xaxis=dict(title="Method"),
                    yaxis=dict(title="Count"),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
            }
        
        @self.app.callback(
            Output("cache-performance-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_cache_performance_graph(n):
            metrics = feature_store_metrics.get_metrics()
            
            # Create pie chart for cache performance
            labels = ["Hits", "Misses"]
            values = [metrics["cache"]["hits"], metrics["cache"]["misses"]]
            
            return {
                "data": [
                    go.Pie(
                        labels=labels,
                        values=values,
                        marker=dict(colors=["green", "red"]),
                        hole=0.4
                    )
                ],
                "layout": go.Layout(
                    title=f"Cache Performance (Hit Rate: {metrics['cache']['hit_rate']:.2f})",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
            }
        
        @self.app.callback(
            Output("errors-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_errors_graph(n):
            metrics = feature_store_metrics.get_metrics()
            
            # Create bar chart for errors
            labels = list(metrics["errors"].keys())
            values = list(metrics["errors"].values())
            
            return {
                "data": [
                    go.Bar(
                        x=labels,
                        y=values,
                        marker=dict(color="crimson")
                    )
                ],
                "layout": go.Layout(
                    title="Errors by Type",
                    xaxis=dict(title="Error Type"),
                    yaxis=dict(title="Count"),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
            }
        
        @self.app.callback(
            Output("response-time-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_response_time_graph(n):
            metrics = feature_store_metrics.get_metrics()
            
            # Create gauge for average response time
            avg_response_time = metrics["performance"]["avg_response_time_ms"]
            
            return {
                "data": [
                    go.Indicator(
                        mode="gauge+number",
                        value=avg_response_time,
                        title={"text": "Avg Response Time (ms)"},
                        gauge={
                            "axis": {"range": [0, max(1000, avg_response_time * 2)]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 100], "color": "green"},
                                {"range": [100, 500], "color": "yellow"},
                                {"range": [500, 1000], "color": "orange"},
                                {"range": [1000, max(1000, avg_response_time * 2)], "color": "red"}
                            ]
                        }
                    )
                ],
                "layout": go.Layout(
                    margin=dict(l=40, r=40, t=40, b=40)
                )
            }
        
        @self.app.callback(
            Output("fallbacks-graph", "figure"),
            Input("interval-component", "n_intervals")
        )
        def update_fallbacks_graph(n):
            metrics = feature_store_metrics.get_metrics()
            
            # Create bar chart for fallbacks
            labels = list(metrics["fallbacks"].keys())
            values = list(metrics["fallbacks"].values())
            
            return {
                "data": [
                    go.Bar(
                        x=labels,
                        y=values,
                        marker=dict(color="orange")
                    )
                ],
                "layout": go.Layout(
                    title="Fallbacks by Method",
                    xaxis=dict(title="Method"),
                    yaxis=dict(title="Count"),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
            }
        
        @self.app.callback(
            Output("summary-div", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_summary(n):
            metrics = feature_store_metrics.get_metrics()
            
            # Create summary table
            return html.Table([
                html.Thead(
                    html.Tr([html.Th("Metric"), html.Th("Value")])
                ),
                html.Tbody([
                    html.Tr([html.Td("Total API Calls"), html.Td(metrics["api_calls"]["total"])]),
                    html.Tr([html.Td("Cache Hit Rate"), html.Td(f"{metrics['cache']['hit_rate']:.2f}")]),
                    html.Tr([html.Td("Total Errors"), html.Td(metrics["errors"]["total"])]),
                    html.Tr([html.Td("Avg Response Time"), html.Td(f"{metrics['performance']['avg_response_time_ms']:.2f} ms")]),
                    html.Tr([html.Td("Max Response Time"), html.Td(f"{metrics['performance']['max_response_time_ms']:.2f} ms")]),
                    html.Tr([html.Td("Total Fallbacks"), html.Td(metrics["fallbacks"]["total"])]),
                    html.Tr([html.Td("Last Updated"), html.Td(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))])
                ])
            ], className="table table-striped")
    
    def run(self, debug: bool = False):
        """
        Run the dashboard.
        
        Args:
            debug: Whether to run in debug mode
        """
        if not has_dash:
            self.logger.warning("Dash not installed, dashboard not available")
            return
        
        self.logger.info(f"Starting feature store dashboard on port {self.port}")
        self.app.run_server(debug=debug, port=self.port)


# Singleton instance
feature_store_dashboard = FeatureStoreDashboard()


def run_dashboard(debug: bool = False):
    """
    Run the feature store dashboard.
    
    Args:
        debug: Whether to run in debug mode
    """
    feature_store_dashboard.run(debug=debug)


if __name__ == "__main__":
    run_dashboard(debug=True)
