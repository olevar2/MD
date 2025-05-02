"""
Causal Relationship Visualization Module

Provides interactive visualization tools for causal relationships, graphs,
and performance impact analysis.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class CausalVisualizer:
    """Generates interactive visualizations for causal relationships and insights."""
    
    def __init__(self):
        self.color_scale = px.colors.sequential.Viridis
        self.edge_color_scale = px.colors.sequential.Reds
        
    def create_causal_graph_visualization(
        self,
        graph: nx.DiGraph,
        confidence_scores: Optional[Dict[Tuple[str, str], float]] = None
    ) -> go.Figure:
        """
        Creates an interactive visualization of the causal graph.
        
        Args:
            graph: NetworkX DiGraph representing causal relationships
            confidence_scores: Optional dict mapping edge tuples to confidence scores
            
        Returns:
            Plotly figure object
        """
        # Use Kamada-Kawai layout for graph visualization
        pos = nx.kamada_kawai_layout(graph)
        
        # Create edges
        edge_traces = []
        annotations = []
        
        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            weight = edge[2].get('weight', 0.5)
            confidence = confidence_scores.get((edge[0], edge[1]), weight) if confidence_scores else weight
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(
                    width=2,
                    color=px.colors.sample_colorscale(self.edge_color_scale, confidence)[0]
                ),
                hoverinfo='text',
                text=f"{edge[0]} → {edge[1]}<br>Confidence: {confidence:.2f}",
                mode='lines'
            )
            edge_traces.append(edge_trace)
            
            # Add edge label
            annotations.append(
                dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2,
                    text=f"{confidence:.2f}",
                    showarrow=False,
                    font=dict(size=10)
                )
            )
        
        # Create nodes
        node_trace = go.Scatter(
            x=[pos[node][0] for node in graph.nodes()],
            y=[pos[node][1] for node in graph.nodes()],
            mode='markers+text',
            text=list(graph.nodes()),
            textposition='bottom center',
            marker=dict(
                size=20,
                color=px.colors.sample_colorscale(
                    self.color_scale,
                    [graph.degree(node) / max(1, graph.number_of_nodes())
                     for node in graph.nodes()]
                )
            ),
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title='Causal Relationship Graph',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=annotations,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    def create_confidence_evolution_plot(
        self,
        confidence_history: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Creates a plot showing how confidence scores evolved over time.
        
        Args:
            confidence_history: List of confidence score records with timestamps
            
        Returns:
            Plotly figure object
        """
        df = pd.DataFrame(confidence_history)
        
        fig = go.Figure()
        
        # Add overall confidence line
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['score'],
                mode='lines+markers',
                name='Overall Confidence',
                line=dict(width=3)
            )
        )
        
        # Add component confidence lines
        for component in ['statistical_significance', 'temporal_stability',
                        'predictive_power', 'graph_consistency']:
            if f'components.{component}' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[f'components.{component}'],
                        mode='lines',
                        name=component.replace('_', ' ').title(),
                        line=dict(width=1, dash='dash')
                    )
                )
        
        fig.update_layout(
            title='Confidence Score Evolution',
            xaxis_title='Time',
            yaxis_title='Confidence Score',
            hovermode='x unified',
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_performance_impact_dashboard(
        self,
        impact_analysis: Dict[str, Any]
    ) -> go.Figure:
        """
        Creates a dashboard showing the performance impact of causal insights.
        
        Args:
            impact_analysis: Dictionary containing performance impact metrics
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Success Rate by Confidence Level',
                'Insight Value Analysis',
                'Confidence-Performance Correlation',
                'Overall Impact'
            )
        )
        
        # 1. Success Rate by Confidence Level
        success_rates = [
            impact_analysis['validation_summary']['low_confidence_success_rate'],
            impact_analysis['validation_summary']['high_confidence_success_rate']
        ]
        
        fig.add_trace(
            go.Bar(
                x=['Low Confidence', 'High Confidence'],
                y=success_rates,
                marker_color=['lightgray', self.color_scale[5]],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Insight Value Analysis
        insight_values = impact_analysis.get('insight_value', {})
        factors = list(insight_values.keys())
        success_rates = [insight_values[f]['success_rate'] for f in factors]
        avg_pnls = [insight_values[f]['avg_pnl'] for f in factors]
        
        fig.add_trace(
            go.Bar(
                x=factors,
                y=success_rates,
                name='Success Rate',
                marker_color=self.color_scale[3]
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=factors,
                y=avg_pnls,
                name='Avg PnL',
                yaxis='y2',
                line=dict(color=self.color_scale[7])
            ),
            row=1, col=2
        )
        
        # 3. Confidence-Performance Correlation
        correlation = impact_analysis.get('confidence_correlation', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=correlation,
                gauge={
                    'axis': {'range': [-1, 1]},
                    'steps': [
                        {'range': [-1, 0], 'color': 'lightgray'},
                        {'range': [0, 1], 'color': self.color_scale[5]}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # 4. Overall Impact
        improvement = impact_analysis.get('overall_improvement', 0)
        fig.add_trace(
            go.Indicator(
                mode="delta+number",
                value=improvement + 0.5,  # Center at 0.5 for better visualization
                delta={'reference': 0.5},
                title={'text': "Performance Improvement"}
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Causal Analysis Performance Impact",
            showlegend=True
        )
        
        # Update axes
        fig.update_yaxes(title_text="Success Rate", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Success Rate", range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="Avg PnL", overlaying='y', side='right', row=1, col=2)
        
        return fig
