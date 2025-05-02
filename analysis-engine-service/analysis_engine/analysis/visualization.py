"""
Dashboard and Visualization Components for Indicator Analysis

This module provides visualization components for rendering indicators, signals, and
other trading analysis data in interactive dashboards and charts.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import uuid
from datetime import datetime, timedelta
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Types of charts for visualizations"""
    LINE = "line"               # Line chart
    CANDLESTICK = "candlestick" # Candlestick chart
    BAR = "bar"                 # Bar chart
    AREA = "area"               # Area chart
    SCATTER = "scatter"         # Scatter plot
    HEATMAP = "heatmap"         # Heatmap
    HISTOGRAM = "histogram"     # Histogram
    OHLC = "ohlc"               # OHLC chart
    CUSTOM = "custom"           # Custom chart type


class IndicatorRenderMode(Enum):
    """Rendering modes for indicators"""
    OVERLAY = "overlay"         # Render on top of price chart
    PANEL = "panel"             # Render in separate panel
    STANDALONE = "standalone"   # Render as standalone chart
    TOOLTIP = "tooltip"         # Render in tooltip only


@dataclass
class VisualizationConfig:
    """Configuration for visualizing an indicator or data series"""
    name: str                           # Name of the visualization
    data_source: str                    # Indicator name or data source
    chart_type: ChartType               # Type of chart to render
    render_mode: IndicatorRenderMode    # How to render the indicator
    columns: List[str]                  # Columns to visualize
    colors: Dict[str, str] = field(default_factory=dict)  # Colors for each column
    labels: Dict[str, str] = field(default_factory=dict)  # Labels for each column
    panel_height: int = 200             # Height of panel in pixels
    visible: bool = True                # Whether visualization is visible
    params: Dict[str, Any] = field(default_factory=dict)  # Additional parameters
    
    def __post_init__(self):
        """Initialize with defaults if needed"""
        # Set default colors if not provided
        default_colors = {
            'close': '#000000',
            'open': '#1f77b4',
            'high': '#2ca02c',
            'low': '#d62728',
            'sma': '#ff7f0e',
            'ema': '#9467bd',
            'upper': '#2ca02c',
            'lower': '#d62728',
            'middle': '#1f77b4',
            'signal': '#ff7f0e',
            'histogram': '#9467bd',
            'buy': '#2ca02c',
            'sell': '#d62728'
        }
        
        for col in self.columns:
            if col not in self.colors:
                if col in default_colors:
                    self.colors[col] = default_colors[col]
                else:
                    # Generate a pseudorandom but consistent color
                    hue = abs(hash(col)) % 360
                    self.colors[col] = f"hsl({hue}, 70%, 50%)"
            
            if col not in self.labels:
                self.labels[col] = col.replace('_', ' ').title()


@dataclass
class Dashboard:
    """Configuration for a dashboard of visualizations"""
    id: str                             # Dashboard ID
    name: str                           # Dashboard name
    description: str                    # Dashboard description
    visualizations: List[VisualizationConfig] = field(default_factory=list)  # Visualizations in dashboard
    layout: Dict[str, Any] = field(default_factory=dict)  # Layout configuration
    created_at: datetime = field(default_factory=datetime.now)  # Creation timestamp
    updated_at: datetime = field(default_factory=datetime.now)  # Last update timestamp
    created_by: str = "system"          # Creator username
    
    def __post_init__(self):
        """Initialize with defaults if needed"""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        if not self.layout:
            # Default to a vertical stack layout
            self.layout = {
                "type": "vertical",
                "options": {"spacing": 10}
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "visualizations": [
                {
                    "name": v.name,
                    "data_source": v.data_source,
                    "chart_type": v.chart_type.value,
                    "render_mode": v.render_mode.value,
                    "columns": v.columns,
                    "colors": v.colors,
                    "labels": v.labels,
                    "panel_height": v.panel_height,
                    "visible": v.visible,
                    "params": v.params
                } for v in self.visualizations
            ],
            "layout": self.layout,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dashboard':
        """Create from dictionary"""
        visualizations = []
        for v_data in data.get("visualizations", []):
            viz = VisualizationConfig(
                name=v_data["name"],
                data_source=v_data["data_source"],
                chart_type=ChartType(v_data["chart_type"]),
                render_mode=IndicatorRenderMode(v_data["render_mode"]),
                columns=v_data["columns"],
                colors=v_data.get("colors", {}),
                labels=v_data.get("labels", {}),
                panel_height=v_data.get("panel_height", 200),
                visible=v_data.get("visible", True),
                params=v_data.get("params", {})
            )
            visualizations.append(viz)
            
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            visualizations=visualizations,
            layout=data.get("layout", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            created_by=data.get("created_by", "system")
        )


class ChartRenderer:
    """Base class for chart renderers"""
    
    def __init__(self, config: VisualizationConfig):
        """
        Initialize the chart renderer
        
        Args:
            config: Visualization configuration
        """
        self.config = config
    
    def render_html(self, data: pd.DataFrame) -> str:
        """
        Render chart as HTML
        
        Args:
            data: Data to visualize
            
        Returns:
            HTML string with chart
        """
        raise NotImplementedError("Subclasses must implement render_html")
    
    def render_json(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Render chart data as JSON
        
        Args:
            data: Data to visualize
            
        Returns:
            JSON-compatible dictionary with chart data
        """
        raise NotImplementedError("Subclasses must implement render_json")


class PlotlyChartRenderer(ChartRenderer):
    """Chart renderer using Plotly"""
    
    def render_html(self, data: pd.DataFrame) -> str:
        """
        Render chart as HTML using Plotly
        
        Args:
            data: Data to visualize
            
        Returns:
            HTML string with chart
        """
        try:
            import plotly.graph_objs as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
            
            # Create appropriate figure based on chart type
            if self.config.chart_type == ChartType.CANDLESTICK:
                fig = go.Figure()
                
                # Add candlestick trace
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['open'] if 'open' in data.columns else None,
                    high=data['high'] if 'high' in data.columns else None,
                    low=data['low'] if 'low' in data.columns else None,
                    close=data['close'] if 'close' in data.columns else None,
                    name="Price"
                ))
                
                # Add overlay indicators
                for col in self.config.columns:
                    if col not in ['open', 'high', 'low', 'close'] and col in data.columns:
                        color = self.config.colors.get(col, '#1f77b4')
                        label = self.config.labels.get(col, col)
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[col],
                            mode='lines',
                            name=label,
                            line=dict(color=color)
                        ))
                
                # Update layout
                fig.update_layout(
                    title=self.config.name,
                    yaxis_title='Price',
                    xaxis_title='Date',
                    height=600
                )
                
                # Set y-axis to match well with price scale
                fig.update_yaxes(fixedrange=False)
                
            elif self.config.chart_type == ChartType.LINE:
                fig = go.Figure()
                
                # Add each column as a line trace
                for col in self.config.columns:
                    if col in data.columns:
                        color = self.config.colors.get(col, '#1f77b4')
                        label = self.config.labels.get(col, col)
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[col],
                            mode='lines',
                            name=label,
                            line=dict(color=color)
                        ))
                
                # Update layout
                fig.update_layout(
                    title=self.config.name,
                    yaxis_title='Value',
                    xaxis_title='Date',
                    height=self.config.panel_height
                )
                
            elif self.config.chart_type == ChartType.BAR:
                fig = go.Figure()
                
                # Add each column as a bar trace
                for col in self.config.columns:
                    if col in data.columns:
                        color = self.config.colors.get(col, '#1f77b4')
                        label = self.config.labels.get(col, col)
                        
                        fig.add_trace(go.Bar(
                            x=data.index,
                            y=data[col],
                            name=label,
                            marker_color=color
                        ))
                
                # Update layout
                fig.update_layout(
                    title=self.config.name,
                    yaxis_title='Value',
                    xaxis_title='Date',
                    height=self.config.panel_height
                )
                
            elif self.config.chart_type == ChartType.HISTOGRAM:
                fig = go.Figure()
                
                # Add each column as a histogram
                for col in self.config.columns:
                    if col in data.columns:
                        color = self.config.colors.get(col, '#1f77b4')
                        label = self.config.labels.get(col, col)
                        
                        fig.add_trace(go.Histogram(
                            x=data[col],
                            name=label,
                            marker_color=color,
                            opacity=0.7
                        ))
                
                # Update layout
                fig.update_layout(
                    title=self.config.name,
                    yaxis_title='Frequency',
                    xaxis_title='Value',
                    barmode='overlay',
                    height=self.config.panel_height
                )
                
            else:
                # Default to line chart for other types
                fig = go.Figure()
                
                for col in self.config.columns:
                    if col in data.columns:
                        color = self.config.colors.get(col, '#1f77b4')
                        label = self.config.labels.get(col, col)
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[col],
                            mode='lines',
                            name=label,
                            line=dict(color=color)
                        ))
                
                # Update layout
                fig.update_layout(
                    title=self.config.name,
                    yaxis_title='Value',
                    xaxis_title='Date',
                    height=self.config.panel_height
                )
            
            # Convert to HTML
            html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            return html
            
        except ImportError:
            return f"<div>Error: Plotly not installed. Please install with 'pip install plotly'.</div>"
        except Exception as e:
            logger.error(f"Error rendering chart: {str(e)}")
            return f"<div>Error rendering chart: {str(e)}</div>"
    
    def render_json(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Render chart data as JSON for frontend rendering
        
        Args:
            data: Data to visualize
            
        Returns:
            JSON-compatible dictionary with chart data
        """
        result = {
            "config": {
                "name": self.config.name,
                "type": self.config.chart_type.value,
                "mode": self.config.render_mode.value,
                "colors": self.config.colors,
                "labels": self.config.labels,
                "height": self.config.panel_height
            },
            "data": {}
        }
        
        # Add timestamps
        result["data"]["timestamps"] = data.index.astype(str).tolist()
        
        # Add each data series
        for col in self.config.columns:
            if col in data.columns:
                # Convert to Python native types for JSON serialization
                if data[col].dtype == 'datetime64[ns]':
                    result["data"][col] = data[col].astype(str).tolist()
                elif data[col].dtype == 'bool':
                    result["data"][col] = data[col].astype(int).tolist()
                else:
                    # Replace NaN with None for JSON compatibility
                    values = data[col].tolist()
                    result["data"][col] = [None if pd.isna(v) else v for v in values]
                    
        return result


class MatplotlibChartRenderer(ChartRenderer):
    """Chart renderer using Matplotlib"""
    
    def render_html(self, data: pd.DataFrame) -> str:
        """
        Render chart as HTML using Matplotlib
        
        Args:
            data: Data to visualize
            
        Returns:
            HTML string with chart as embedded image
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from io import BytesIO
            import base64
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if self.config.chart_type == ChartType.CANDLESTICK:
                # Need to implement candlestick rendering using matplotlib's mpl_finance
                # or mplfinance package
                pass
                
            elif self.config.chart_type == ChartType.LINE:
                for col in self.config.columns:
                    if col in data.columns:
                        color = self.config.colors.get(col, None)
                        label = self.config.labels.get(col, col)
                        ax.plot(data.index, data[col], label=label, color=color)
                
                ax.legend()
                ax.set_title(self.config.name)
                ax.set_ylabel('Value')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                
            elif self.config.chart_type == ChartType.BAR:
                # For simplicity, just use the first column for bar chart
                if self.config.columns and self.config.columns[0] in data.columns:
                    col = self.config.columns[0]
                    color = self.config.colors.get(col, None)
                    ax.bar(data.index, data[col], label=col, color=color)
                    
                ax.set_title(self.config.name)
                ax.set_ylabel('Value')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                
            # Convert plot to HTML
            buf = BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            
            # Encode image to base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Return HTML with embedded image
            return f'<img src="data:image/png;base64,{img_str}" />'
            
        except ImportError:
            return "<div>Error: Matplotlib not installed. Please install with 'pip install matplotlib'.</div>"
        except Exception as e:
            logger.error(f"Error rendering chart: {str(e)}")
            return f"<div>Error rendering chart: {str(e)}</div>"
    
    def render_json(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Render chart data as JSON
        
        Args:
            data: Data to visualize
            
        Returns:
            JSON-compatible dictionary with chart data
        """
        # Similar to PlotlyChartRenderer.render_json
        result = {
            "config": {
                "name": self.config.name,
                "type": self.config.chart_type.value,
                "mode": self.config.render_mode.value,
                "colors": self.config.colors,
                "labels": self.config.labels,
                "height": self.config.panel_height
            },
            "data": {}
        }
        
        # Add timestamps
        result["data"]["timestamps"] = data.index.astype(str).tolist()
        
        # Add each data series
        for col in self.config.columns:
            if col in data.columns:
                # Convert to Python native types for JSON serialization
                values = data[col].tolist()
                result["data"][col] = [None if pd.isna(v) else v for v in values]
                    
        return result


class BokehChartRenderer(ChartRenderer):
    """Chart renderer using Bokeh"""
    
    def render_html(self, data: pd.DataFrame) -> str:
        """
        Render chart as HTML using Bokeh
        
        Args:
            data: Data to visualize
            
        Returns:
            HTML string with chart
        """
        try:
            from bokeh.plotting import figure
            from bokeh.embed import components
            from bokeh.models import ColumnDataSource, HoverTool
            
            # Convert data for Bokeh
            source = ColumnDataSource(data)
            
            # Create figure
            p = figure(
                title=self.config.name,
                x_axis_type='datetime',
                width=800,
                height=self.config.panel_height,
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )
            
            # Add hover tool
            hover = HoverTool(
                tooltips=[
                    ("Date", "@index{%F}"),
                ] + [
                    (self.config.labels.get(col, col), f"@{col}{{0.000}}") 
                    for col in self.config.columns if col in data.columns
                ],
                formatters={"@index": "datetime"}
            )
            p.add_tools(hover)
            
            # Render based on chart type
            if self.config.chart_type == ChartType.CANDLESTICK:
                # Need additional processing for candlestick charts
                pass
                
            elif self.config.chart_type == ChartType.LINE:
                for col in self.config.columns:
                    if col in data.columns:
                        color = self.config.colors.get(col, None)
                        label = self.config.labels.get(col, col)
                        p.line('index', col, source=source, line_width=2, 
                               color=color, legend_label=label)
                
                p.legend.location = "top_left"
                p.legend.click_policy = "hide"
                
            elif self.config.chart_type == ChartType.BAR:
                # For simplicity, just use the first column for bar chart
                if self.config.columns and self.config.columns[0] in data.columns:
                    col = self.config.columns[0]
                    color = self.config.colors.get(col, None)
                    p.vbar(x='index', top=col, source=source, width=0.5, 
                           color=color, legend_label=col)
            
            # Get components
            script, div = components(p)
            
            # Return HTML
            return script + div
            
        except ImportError:
            return "<div>Error: Bokeh not installed. Please install with 'pip install bokeh'.</div>"
        except Exception as e:
            logger.error(f"Error rendering chart: {str(e)}")
            return f"<div>Error rendering chart: {str(e)}</div>"
    
    def render_json(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Render chart data as JSON
        
        Args:
            data: Data to visualize
            
        Returns:
            JSON-compatible dictionary with chart data
        """
        # Similar implementation to PlotlyChartRenderer.render_json
        result = {
            "config": {
                "name": self.config.name,
                "type": self.config.chart_type.value,
                "mode": self.config.render_mode.value,
                "colors": self.config.colors,
                "labels": self.config.labels,
                "height": self.config.panel_height
            },
            "data": {}
        }
        
        # Add timestamps
        result["data"]["timestamps"] = data.index.astype(str).tolist()
        
        # Add each data series
        for col in self.config.columns:
            if col in data.columns:
                values = data[col].tolist()
                result["data"][col] = [None if pd.isna(v) else v for v in values]
                    
        return result


class DashboardManager:
    """Manager for creating and rendering dashboards"""
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize dashboard manager
        
        Args:
            storage_dir: Directory to store dashboard configurations
        """
        self._dashboards: Dict[str, Dashboard] = {}
        self._renderers = {
            "plotly": PlotlyChartRenderer,
            "matplotlib": MatplotlibChartRenderer,
            "bokeh": BokehChartRenderer
        }
        self._default_renderer = "plotly"
        
        # Set storage directory
        if storage_dir:
            self.storage_dir = storage_dir
        else:
            # Default to a dashboards directory in the current module's directory
            module_dir = os.path.dirname(os.path.abspath(__file__))
            self.storage_dir = os.path.join(module_dir, "dashboards")
            
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load existing dashboards
        self.load_dashboards()
    
    def create_dashboard(self, 
                       name: str, 
                       description: str = "",
                       created_by: str = "system") -> Dashboard:
        """
        Create a new dashboard
        
        Args:
            name: Dashboard name
            description: Dashboard description
            created_by: Creator username
            
        Returns:
            The created dashboard
        """
        dashboard = Dashboard(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_by=created_by
        )
        
        self._dashboards[dashboard.id] = dashboard
        return dashboard
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """
        Get a dashboard by ID
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            The dashboard or None if not found
        """
        return self._dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """
        List all dashboards
        
        Returns:
            List of dashboard summaries
        """
        return [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "created_at": d.created_at.isoformat(),
                "updated_at": d.updated_at.isoformat(),
                "created_by": d.created_by,
                "visualization_count": len(d.visualizations)
            }
            for d in self._dashboards.values()
        ]
    
    def add_visualization(self, 
                        dashboard_id: str, 
                        config: VisualizationConfig) -> bool:
        """
        Add a visualization to a dashboard
        
        Args:
            dashboard_id: Dashboard ID
            config: Visualization configuration
            
        Returns:
            True if successful, False otherwise
        """
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return False
            
        dashboard.visualizations.append(config)
        dashboard.updated_at = datetime.now()
        
        return True
    
    def remove_visualization(self, 
                           dashboard_id: str, 
                           visualization_name: str) -> bool:
        """
        Remove a visualization from a dashboard
        
        Args:
            dashboard_id: Dashboard ID
            visualization_name: Name of visualization to remove
            
        Returns:
            True if successful, False otherwise
        """
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return False
            
        before_count = len(dashboard.visualizations)
        dashboard.visualizations = [
            v for v in dashboard.visualizations if v.name != visualization_name
        ]
        
        if len(dashboard.visualizations) < before_count:
            dashboard.updated_at = datetime.now()
            return True
            
        return False
    
    def update_dashboard(self, dashboard: Dashboard) -> bool:
        """
        Update a dashboard
        
        Args:
            dashboard: Updated dashboard
            
        Returns:
            True if successful, False otherwise
        """
        if dashboard.id not in self._dashboards:
            return False
            
        dashboard.updated_at = datetime.now()
        self._dashboards[dashboard.id] = dashboard
        return True
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """
        Delete a dashboard
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            True if successful, False otherwise
        """
        if dashboard_id in self._dashboards:
            del self._dashboards[dashboard_id]
            
            # Also delete the file if it exists
            file_path = os.path.join(self.storage_dir, f"{dashboard_id}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error deleting dashboard file: {str(e)}")
            
            return True
            
        return False
    
    def save_dashboards(self) -> int:
        """
        Save all dashboards to files
        
        Returns:
            Number of dashboards saved
        """
        count = 0
        
        for dashboard_id, dashboard in self._dashboards.items():
            try:
                file_path = os.path.join(self.storage_dir, f"{dashboard_id}.json")
                with open(file_path, 'w') as f:
                    json.dump(dashboard.to_dict(), f, indent=2)
                count += 1
            except Exception as e:
                logger.error(f"Error saving dashboard {dashboard_id}: {str(e)}")
                continue
                
        return count
    
    def load_dashboards(self) -> int:
        """
        Load dashboards from files
        
        Returns:
            Number of dashboards loaded
        """
        self._dashboards = {}
        count = 0
        
        try:
            # Find all JSON files in storage directory
            for file_path in Path(self.storage_dir).glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    dashboard = Dashboard.from_dict(data)
                    self._dashboards[dashboard.id] = dashboard
                    count += 1
                except Exception as e:
                    logger.error(f"Error loading dashboard from {file_path}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error loading dashboards: {str(e)}")
            
        return count
    
    def render_dashboard(self, 
                       dashboard_id: str, 
                       data: Dict[str, pd.DataFrame],
                       renderer: str = None) -> str:
        """
        Render a dashboard as HTML
        
        Args:
            dashboard_id: Dashboard ID
            data: Dictionary mapping data sources to DataFrames
            renderer: Renderer to use (default: plotly)
            
        Returns:
            HTML string with rendered dashboard
        """
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return "<div>Dashboard not found</div>"
            
        if renderer is None:
            renderer = self._default_renderer
            
        renderer_class = self._renderers.get(renderer, PlotlyChartRenderer)
        
        html_parts = [
            f"<h1>{dashboard.name}</h1>",
            f"<p>{dashboard.description}</p>",
            "<div class='dashboard-container'>"
        ]
        
        for viz_config in dashboard.visualizations:
            if not viz_config.visible:
                continue
                
            # Get data for this visualization
            viz_data = data.get(viz_config.data_source)
            if viz_data is None:
                html_parts.append(f"<div>Data source '{viz_config.data_source}' not found</div>")
                continue
                
            # Create renderer
            viz_renderer = renderer_class(viz_config)
            
            # Add to HTML
            html_parts.append(f"<div class='visualization' id='{viz_config.name}'>")
            html_parts.append(f"<h3>{viz_config.name}</h3>")
            html_parts.append(viz_renderer.render_html(viz_data))
            html_parts.append("</div>")
            
        html_parts.append("</div>")
        
        # Add basic CSS
        css = """
        <style>
        .dashboard-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .visualization {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #fff;
        }
        </style>
        """
        
        return css + "\n".join(html_parts)
    
    def render_dashboard_json(self, 
                            dashboard_id: str, 
                            data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Render a dashboard as JSON for frontend rendering
        
        Args:
            dashboard_id: Dashboard ID
            data: Dictionary mapping data sources to DataFrames
            
        Returns:
            JSON-compatible dictionary with dashboard data
        """
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return {"error": "Dashboard not found"}
            
        result = {
            "id": dashboard.id,
            "name": dashboard.name,
            "description": dashboard.description,
            "created_at": dashboard.created_at.isoformat(),
            "updated_at": dashboard.updated_at.isoformat(),
            "created_by": dashboard.created_by,
            "layout": dashboard.layout,
            "visualizations": []
        }
        
        for viz_config in dashboard.visualizations:
            # Get data for this visualization
            viz_data = data.get(viz_config.data_source)
            if viz_data is None:
                # Skip visualizations with missing data
                continue
                
            # Create renderer and render as JSON
            viz_renderer = PlotlyChartRenderer(viz_config)  # Always use Plotly for JSON rendering
            viz_json = viz_renderer.render_json(viz_data)
            
            result["visualizations"].append({
                "name": viz_config.name,
                "visible": viz_config.visible,
                "render_mode": viz_config.render_mode.value,
                "data": viz_json
            })
            
        return result


class IndicatorDashboardFactory:
    """Factory for creating common indicator dashboards"""
    
    @staticmethod
    def create_price_dashboard(dashboard_manager: DashboardManager, name: str = "Price Analysis") -> str:
        """
        Create a basic price analysis dashboard
        
        Args:
            dashboard_manager: Dashboard manager
            name: Dashboard name
            
        Returns:
            ID of the created dashboard
        """
        dashboard = dashboard_manager.create_dashboard(
            name=name,
            description="Basic price analysis with moving averages and volume"
        )
        
        # Add price chart
        price_chart = VisualizationConfig(
            name="Price Chart",
            data_source="market_data",
            chart_type=ChartType.CANDLESTICK,
            render_mode=IndicatorRenderMode.STANDALONE,
            columns=["open", "high", "low", "close"],
            panel_height=400
        )
        dashboard_manager.add_visualization(dashboard.id, price_chart)
        
        # Add SMA indicators
        sma_chart = VisualizationConfig(
            name="Moving Averages",
            data_source="sma_data",
            chart_type=ChartType.LINE,
            render_mode=IndicatorRenderMode.OVERLAY,
            columns=["close", "sma20", "sma50", "sma200"],
            colors={
                "close": "#000000",
                "sma20": "#1f77b4",
                "sma50": "#ff7f0e",
                "sma200": "#2ca02c"
            },
            labels={
                "close": "Price",
                "sma20": "20-day SMA",
                "sma50": "50-day SMA",
                "sma200": "200-day SMA"
            },
            panel_height=400
        )
        dashboard_manager.add_visualization(dashboard.id, sma_chart)
        
        # Add volume chart
        volume_chart = VisualizationConfig(
            name="Volume",
            data_source="market_data",
            chart_type=ChartType.BAR,
            render_mode=IndicatorRenderMode.PANEL,
            columns=["volume"],
            colors={"volume": "#1f77b4"},
            panel_height=200
        )
        dashboard_manager.add_visualization(dashboard.id, volume_chart)
        
        # Save the dashboard
        dashboard_manager.save_dashboards()
        
        return dashboard.id
    
    @staticmethod
    def create_technical_dashboard(dashboard_manager: DashboardManager, name: str = "Technical Indicators") -> str:
        """
        Create a technical indicator dashboard
        
        Args:
            dashboard_manager: Dashboard manager
            name: Dashboard name
            
        Returns:
            ID of the created dashboard
        """
        dashboard = dashboard_manager.create_dashboard(
            name=name,
            description="Technical indicator analysis dashboard"
        )
        
        # Add price chart
        price_chart = VisualizationConfig(
            name="Price Chart",
            data_source="market_data",
            chart_type=ChartType.CANDLESTICK,
            render_mode=IndicatorRenderMode.STANDALONE,
            columns=["open", "high", "low", "close"],
            panel_height=300
        )
        dashboard_manager.add_visualization(dashboard.id, price_chart)
        
        # Add RSI indicator
        rsi_chart = VisualizationConfig(
            name="RSI (14)",
            data_source="rsi_data",
            chart_type=ChartType.LINE,
            render_mode=IndicatorRenderMode.PANEL,
            columns=["rsi"],
            colors={"rsi": "#ff7f0e"},
            panel_height=200,
            params={
                "period": 14,
                "overbought": 70,
                "oversold": 30
            }
        )
        dashboard_manager.add_visualization(dashboard.id, rsi_chart)
        
        # Add MACD indicator
        macd_chart = VisualizationConfig(
            name="MACD",
            data_source="macd_data",
            chart_type=ChartType.LINE,
            render_mode=IndicatorRenderMode.PANEL,
            columns=["macd", "signal", "histogram"],
            colors={
                "macd": "#1f77b4",
                "signal": "#ff7f0e",
                "histogram": "#2ca02c"
            },
            panel_height=200
        )
        dashboard_manager.add_visualization(dashboard.id, macd_chart)
        
        # Add Bollinger Bands
        bb_chart = VisualizationConfig(
            name="Bollinger Bands",
            data_source="bollinger_data",
            chart_type=ChartType.LINE,
            render_mode=IndicatorRenderMode.OVERLAY,
            columns=["close", "upper", "middle", "lower"],
            colors={
                "close": "#000000",
                "upper": "#2ca02c",
                "middle": "#1f77b4",
                "lower": "#d62728"
            },
            panel_height=300
        )
        dashboard_manager.add_visualization(dashboard.id, bb_chart)
        
        # Save the dashboard
        dashboard_manager.save_dashboards()
        
        return dashboard.id
    
    @staticmethod
    def create_ml_prediction_dashboard(dashboard_manager: DashboardManager, name: str = "ML Predictions") -> str:
        """
        Create a dashboard for ML predictions
        
        Args:
            dashboard_manager: Dashboard manager
            name: Dashboard name
            
        Returns:
            ID of the created dashboard
        """
        dashboard = dashboard_manager.create_dashboard(
            name=name,
            description="Machine learning prediction analysis"
        )
        
        # Add price chart with predictions
        price_chart = VisualizationConfig(
            name="Price with Predictions",
            data_source="prediction_data",
            chart_type=ChartType.LINE,
            render_mode=IndicatorRenderMode.STANDALONE,
            columns=["close", "predicted_close", "upper_bound", "lower_bound"],
            colors={
                "close": "#000000",
                "predicted_close": "#ff7f0e",
                "upper_bound": "#2ca02c",
                "lower_bound": "#d62728"
            },
            labels={
                "close": "Actual Price",
                "predicted_close": "Predicted Price",
                "upper_bound": "Upper Bound (95%)",
                "lower_bound": "Lower Bound (95%)"
            },
            panel_height=400
        )
        dashboard_manager.add_visualization(dashboard.id, price_chart)
        
        # Add prediction error chart
        error_chart = VisualizationConfig(
            name="Prediction Error",
            data_source="prediction_data",
            chart_type=ChartType.LINE,
            render_mode=IndicatorRenderMode.PANEL,
            columns=["error", "error_pct"],
            colors={
                "error": "#1f77b4",
                "error_pct": "#ff7f0e"
            },
            labels={
                "error": "Absolute Error",
                "error_pct": "% Error"
            },
            panel_height=200
        )
        dashboard_manager.add_visualization(dashboard.id, error_chart)
        
        # Add feature importance chart
        importance_chart = VisualizationConfig(
            name="Feature Importance",
            data_source="feature_importance",
            chart_type=ChartType.BAR,
            render_mode=IndicatorRenderMode.PANEL,
            columns=["importance"],
            panel_height=300
        )
        dashboard_manager.add_visualization(dashboard.id, importance_chart)
        
        # Save the dashboard
        dashboard_manager.save_dashboards()
        
        return dashboard.id


# Create a global dashboard manager instance
dashboard_manager = DashboardManager()

# Create some default dashboards
try:
    IndicatorDashboardFactory.create_price_dashboard(dashboard_manager)
    IndicatorDashboardFactory.create_technical_dashboard(dashboard_manager)
    IndicatorDashboardFactory.create_ml_prediction_dashboard(dashboard_manager)
except Exception as e:
    logger.error(f"Error creating default dashboards: {str(e)}")
