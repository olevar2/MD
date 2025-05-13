"""
Interactive Model Performance Visualization Components

This module provides visualization tools for model performance analysis,
feature importance, prediction confidence, and regime-specific metrics.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np

class ModelPerformanceVisualizer:
    """
    Creates interactive visualizations for model performance analysis.
    """
    
    def create_performance_dashboard(
        self,
        performance_data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Create an interactive performance dashboard with multiple views.
        
        Args:
            performance_data: DataFrame with model performance metrics
            regime_data: Optional DataFrame with market regime information
            
        Returns:
            Dictionary containing Plotly figure objects for each visualization
        """
        figures = {}
        
        # Main performance metrics over time
        figures["metrics_timeline"] = self._create_metrics_timeline(performance_data)
        
        # Regime-specific performance if regime data is available
        if regime_data is not None:
            figures["regime_performance"] = self._create_regime_performance(
                performance_data, regime_data
            )
            
        # Prediction confidence distribution
        figures["confidence_dist"] = self._create_confidence_distribution(performance_data)
        
        return figures
    
    def create_feature_importance_view(
        self,
        feature_importance: pd.DataFrame,
        confidence_intervals: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create interactive feature importance visualization.
        
        Args:
            feature_importance: DataFrame with feature importance scores
            confidence_intervals: Optional confidence intervals for importance scores
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Sort features by importance
        feature_importance = feature_importance.sort_values(
            "importance_score", ascending=True
        )
        
        # Base bar chart
        fig.add_trace(
            go.Bar(
                x=feature_importance["importance_score"],
                y=feature_importance["feature_name"],
                orientation="h",
                name="Importance Score"
            )
        )
        
        # Add confidence intervals if available
        if confidence_intervals is not None:
            fig.add_trace(
                go.Scatter(
                    x=confidence_intervals["lower_bound"],
                    y=feature_importance["feature_name"],
                    mode="lines",
                    name="95% CI",
                    line=dict(color="rgba(68, 68, 68, 0.5)", width=2),
                    showlegend=False
                )
            )
            
        fig.update_layout(
            title="Feature Importance Analysis",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=max(400, len(feature_importance) * 25),
            showlegend=True
        )
        
        return fig
    
    def create_prediction_confidence_view(
        self,
        predictions: pd.DataFrame,
        actuals: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create visualization for prediction confidence analysis.
        
        Args:
            predictions: DataFrame with model predictions and confidence scores
            actuals: Optional DataFrame with actual values for comparison
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Predictions with Confidence", "Confidence Distribution")
        )
        
        # Predictions with confidence bands
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["prediction"],
                mode="lines",
                name="Prediction",
                line=dict(color="blue")
            ),
            row=1, col=1
        )
        
        # Add confidence bands
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["prediction"] + predictions["confidence"],
                mode="lines",
                name="Upper Bound",
                line=dict(color="gray", dash="dash"),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["prediction"] - predictions["confidence"],
                mode="lines",
                name="Lower Bound",
                line=dict(color="gray", dash="dash"),
                fill="tonexty",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add actuals if available
        if actuals is not None:
            fig.add_trace(
                go.Scatter(
                    x=actuals.index,
                    y=actuals["value"],
                    mode="markers",
                    name="Actual",
                    marker=dict(color="red", size=6)
                ),
                row=1, col=1
            )
            
        # Confidence distribution
        fig.add_trace(
            go.Histogram(
                x=predictions["confidence"],
                nbinsx=30,
                name="Confidence Distribution"
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title="Prediction Confidence Analysis",
            showlegend=True
        )
        
        return fig
    
    def _create_metrics_timeline(self, performance_data: pd.DataFrame) -> go.Figure:
        """Create timeline view of performance metrics."""
        fig = go.Figure()
        
        for metric in performance_data.columns:
            if metric != "timestamp":
                fig.add_trace(
                    go.Scatter(
                        x=performance_data["timestamp"],
                        y=performance_data[metric],
                        name=metric,
                        mode="lines"
                    )
                )
                
        fig.update_layout(
            title="Performance Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Metric Value",
            height=400
        )
        
        return fig
    
    def _create_regime_performance(
        self,
        performance_data: pd.DataFrame,
        regime_data: pd.DataFrame
    ) -> go.Figure:
        """Create regime-specific performance visualization."""
        fig = go.Figure()
        
        # Merge performance and regime data
        merged_data = pd.merge(
            performance_data,
            regime_data,
            on="timestamp",
            how="inner"
        )
        
        # Calculate regime-specific metrics
        regime_metrics = merged_data.groupby("regime").mean()
        
        # Create radar chart for each regime
        for regime in regime_metrics.index:
            fig.add_trace(
                go.Scatterpolar(
                    r=regime_metrics.loc[regime],
                    theta=regime_metrics.columns,
                    name=f"Regime {regime}"
                )
            )
            
        fig.update_layout(
            title="Performance by Market Regime",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True
        )
        
        return fig
    
    def _create_confidence_distribution(self, performance_data: pd.DataFrame) -> go.Figure:
        """Create confidence distribution visualization."""
        fig = go.Figure()
        
        if "confidence" in performance_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=performance_data["confidence"],
                    nbinsx=30,
                    name="Confidence Distribution"
                )
            )
            
            fig.update_layout(
                title="Prediction Confidence Distribution",
                xaxis_title="Confidence Score",
                yaxis_title="Count",
                height=400
            )
            
        return fig
