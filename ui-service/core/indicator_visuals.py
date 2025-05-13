"""
UI Service: Indicator Visualization Components

This module provides visualization components for displaying technical indicators,
including charts, heatmaps, and interactive analysis tools.
"""

import json
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Note: In a real implementation, these Python classes would generate React/Vue/Angular
# component definitions or use a Python visualization library like Plotly or Bokeh.
# For this implementation, we'll define the component structures and API interfaces.

class BaseIndicatorChart:
    """Base class for indicator chart components"""
    
    def __init__(self, id: str, title: str = "", height: int = 300, width: int = "100%"):
        """
        Initialize the base chart
        
        Args:
            id: Unique ID for the chart
            title: Chart title
            height: Chart height in pixels
            width: Chart width (pixels or percentage)
        """
        self.id = id
        self.title = title
        self.height = height
        self.width = width
        self.options = {
            "chart": {
                "id": id,
                "height": height,
                "width": width,
                "animations": {
                    "enabled": True,
                    "easing": "easeinout"
                },
                "toolbar": {
                    "show": True,
                    "tools": {
                        "download": True,
                        "selection": True,
                        "zoom": True,
                        "zoomin": True,
                        "zoomout": True,
                        "pan": True,
                        "reset": True
                    }
                },
                "zoom": {
                    "enabled": True
                }
            },
            "title": {
                "text": title,
                "align": "center"
            },
            "theme": {
                "mode": "light",
                "palette": "palette1"
            },
            "tooltip": {
                "enabled": True,
                "shared": True,
                "intersect": False
            }
        }
        self.series = []
    
    def add_series(self, name: str, data: List[Any], type: str = "line") -> None:
        """
        Add a data series to the chart
        
        Args:
            name: Name of the series
            data: Series data
            type: Series type (line, bar, etc.)
        """
        self.series.append({
            "name": name,
            "data": data,
            "type": type
        })
    
    def set_x_axis(self, title: str, categories: Optional[List[Any]] = None) -> None:
        """
        Set X-axis properties
        
        Args:
            title: X-axis title
            categories: X-axis categories
        """
        if "xaxis" not in self.options:
            self.options["xaxis"] = {}
            
        self.options["xaxis"]["title"] = {
            "text": title
        }
        
        if categories:
            self.options["xaxis"]["categories"] = categories
    
    def set_y_axis(self, title: str, decimals: int = 2) -> None:
        """
        Set Y-axis properties
        
        Args:
            title: Y-axis title
            decimals: Number of decimal places
        """
        if "yaxis" not in self.options:
            self.options["yaxis"] = {}
            
        self.options["yaxis"]["title"] = {
            "text": title
        }
        
        self.options["yaxis"]["decimalsInFloat"] = decimals
    
    def set_colors(self, colors: List[str]) -> None:
        """
        Set chart colors
        
        Args:
            colors: List of color codes
        """
        self.options["colors"] = colors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chart to dictionary for rendering"""
        return {
            "options": self.options,
            "series": self.series
        }
    
    def to_json(self) -> str:
        """Convert chart to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class PriceChart(BaseIndicatorChart):
    """Price chart with indicators overlay"""
    
    def __init__(self, id: str, title: str = "Price Chart", height: int = 400):
        """Initialize the price chart"""
        super().__init__(id, title, height)
        
        # Configure for candlestick chart
        self.options["chart"]["type"] = "candlestick"
        self.options["plotOptions"] = {
            "candlestick": {
                "colors": {
                    "upward": "#3C90EB",
                    "downward": "#DF7D46"
                }
            }
        }
    
    def set_ohlc_data(self, data: List[Dict[str, Union[str, float]]]) -> None:
        """
        Set OHLC price data
        
        Args:
            data: List of OHLC data points with x, o, h, l, c values
        """
        self.series = [{
            "name": "Price",
            "data": data
        }]
    
    def add_indicator_overlay(self, name: str, data: List[Union[float, Dict[str, float]]], 
                            type: str = "line", color: str = None) -> None:
        """
        Add an indicator overlay to the price chart
        
        Args:
            name: Indicator name
            data: Indicator data
            type: Chart type for the indicator
            color: Line/fill color
        """
        series = {
            "name": name,
            "type": type,
            "data": data
        }
        
        if color:
            series["color"] = color
            
        self.series.append(series)
    
    def add_volume(self, data: List[float], colors: List[str] = ["#26C281", "#ED3419"]) -> None:
        """
        Add volume bars to the chart
        
        Args:
            data: Volume data
            colors: Up/down colors
        """
        # Add volume as a series with separate y-axis
        self.series.append({
            "name": "Volume",
            "type": "bar",
            "data": data,
            "yaxis": 2
        })
        
        # Configure second y-axis for volume
        self.options["yaxis"] = [
            # Primary y-axis (price)
            {
                "title": {
                    "text": "Price"
                },
                "decimalsInFloat": 4
            },
            # Secondary y-axis (volume)
            {
                "opposite": True,
                "title": {
                    "text": "Volume"
                },
                "decimalsInFloat": 0
            }
        ]
        
        # Set volume colors
        self.options["plotOptions"]["bar"] = {
            "colors": {
                "ranges": [
                    {
                        "from": -1000000000,
                        "to": 0,
                        "color": colors[1]
                    },
                    {
                        "from": 0,
                        "to": 1000000000,
                        "color": colors[0]
                    }
                ]
            }
        }


class IndicatorPanel(BaseIndicatorChart):
    """Panel for displaying a single indicator"""
    
    def __init__(self, id: str, indicator_name: str, height: int = 200):
        """Initialize the indicator panel"""
        super().__init__(id, indicator_name, height)
        self.indicator_name = indicator_name
    
    def set_signal_zones(self, overbought: float = None, oversold: float = None, 
                       colors: List[str] = ["#FFCCBC", "#C8E6C9"]) -> None:
        """
        Set overbought/oversold zones for the indicator
        
        Args:
            overbought: Overbought level
            oversold: Oversold level
            colors: Colors for overbought/oversold zones
        """
        annotations = []
        
        # Add overbought zone
        if overbought is not None:
            annotations.append({
                "y": overbought,
                "borderColor": "#FF5722",
                "label": {
                    "borderColor": "#FF5722",
                    "text": "Overbought"
                }
            })
            
            # Add fill above overbought level
            self.options["fill"] = {
                "type": "gradient",
                "gradient": {
                    "shade": "light",
                    "type": "vertical",
                    "shadeIntensity": 0.5,
                    "opacityFrom": 0.8,
                    "opacityTo": 0,
                    "stops": [0, 100],
                    "colorStops": [
                        {
                            "offset": 0,
                            "color": colors[0],
                            "opacity": 0.4
                        },
                        {
                            "offset": overbought,
                            "color": colors[0],
                            "opacity": 0
                        }
                    ]
                }
            }
        
        # Add oversold zone
        if oversold is not None:
            annotations.append({
                "y": oversold,
                "borderColor": "#4CAF50",
                "label": {
                    "borderColor": "#4CAF50",
                    "text": "Oversold"
                }
            })
            
            # Add fill below oversold level
            self.options["fill"] = {
                "type": "gradient",
                "gradient": {
                    "shade": "light",
                    "type": "vertical",
                    "shadeIntensity": 0.5,
                    "opacityFrom": 0,
                    "opacityTo": 0.8,
                    "stops": [0, 100],
                    "colorStops": [
                        {
                            "offset": oversold,
                            "color": colors[1],
                            "opacity": 0
                        },
                        {
                            "offset": 100,
                            "color": colors[1],
                            "opacity": 0.4
                        }
                    ]
                }
            }
        
        # Add annotations to chart
        if annotations:
            self.options["annotations"] = {
                "yaxis": annotations
            }


class MultiIndicatorChart(BaseIndicatorChart):
    """Chart for comparing multiple indicators"""
    
    def __init__(self, id: str, title: str = "Indicator Comparison", height: int = 350):
        """Initialize the multi-indicator chart"""
        super().__init__(id, title, height)
        
        # Configure for comparison
        self.options["chart"]["type"] = "line"
        self.options["stroke"] = {
            "curve": "smooth",
            "width": 2
        }
        self.options["legend"] = {
            "show": True,
            "position": "top"
        }
    
    def add_indicator(self, name: str, data: List[float], color: str = None) -> None:
        """
        Add an indicator to the comparison chart
        
        Args:
            name: Indicator name
            data: Indicator data
            color: Line color
        """
        series = {
            "name": name,
            "data": data
        }
        
        if color:
            series["color"] = color
            
        self.series.append(series)
    
    def set_normalized(self, normalized: bool = True, min_value: float = 0, 
                     max_value: float = 100) -> None:
        """
        Set whether indicators should be normalized
        
        Args:
            normalized: Whether to normalize indicators
            min_value: Minimum value after normalization
            max_value: Maximum value after normalization
        """
        if normalized:
            self.options["yaxis"]["min"] = min_value
            self.options["yaxis"]["max"] = max_value
            self.options["tooltip"]["y"] = {
                "formatter": f"function(value) {{ return value.toFixed(2) + '%'; }}"
            }


class HeatmapChart(BaseIndicatorChart):
    """Heatmap for visualizing indicator values over time"""
    
    def __init__(self, id: str, title: str = "Indicator Heatmap", height: int = 350):
        """Initialize the heatmap chart"""
        super().__init__(id, title, height)
        
        # Configure for heatmap
        self.options["chart"]["type"] = "heatmap"
        self.options["dataLabels"] = {
            "enabled": False
        }
        self.options["colors"] = ["#008FFB"]
        self.options["plotOptions"] = {
            "heatmap": {
                "radius": 0,
                "enableShades": True,
                "shadeIntensity": 0.5,
                "colorScale": {
                    "ranges": [
                        {
                            "from": -30,
                            "to": 0,
                            "name": "low",
                            "color": "#FF5733"
                        },
                        {
                            "from": 0,
                            "to": 30,
                            "name": "medium",
                            "color": "#FFC300"
                        },
                        {
                            "from": 30,
                            "to": 60,
                            "name": "high",
                            "color": "#DAF7A6"
                        },
                        {
                            "from": 60,
                            "to": 100,
                            "name": "very high",
                            "color": "#33FF57"
                        }
                    ]
                }
            }
        }
    
    def set_data(self, data: List[Dict[str, Any]], x_categories: List[str], 
               y_categories: List[str], min_value: float = None, 
               max_value: float = None) -> None:
        """
        Set heatmap data
        
        Args:
            data: Heatmap data (list of {x, y, value} objects)
            x_categories: X-axis categories
            y_categories: Y-axis categories
            min_value: Minimum value for color scale
            max_value: Maximum value for color scale
        """
        # Convert data to series format
        series_data = []
        for y_idx, y_cat in enumerate(y_categories):
            series = {
                "name": y_cat,
                "data": []
            }
            
            for x_idx, x_cat in enumerate(x_categories):
                # Find value for this cell
                cell_value = next((item["value"] for item in data 
                                if item["x"] == x_cat and item["y"] == y_cat), None)
                
                series["data"].append(cell_value)
            
            series_data.append(series)
        
        self.series = series_data
        
        # Set axis categories
        self.options["xaxis"]["categories"] = x_categories
        
        # Set color scale range if provided
        if min_value is not None and max_value is not None:
            self.options["plotOptions"]["heatmap"]["colorScale"]["min"] = min_value
            self.options["plotOptions"]["heatmap"]["colorScale"]["max"] = max_value


class CorrelationMatrix(HeatmapChart):
    """Correlation matrix chart for indicators"""
    
    def __init__(self, id: str, title: str = "Indicator Correlation Matrix"):
        """Initialize the correlation matrix chart"""
        super().__init__(id, title)
        
        # Configure for correlation matrix
        self.options["tooltip"] = {
            "custom": """
            function({series, seriesIndex, dataPointIndex, w}) {
                const value = w.globals.series[seriesIndex][dataPointIndex];
                const xLabel = w.globals.labels[dataPointIndex];
                const yLabel = w.globals.seriesNames[seriesIndex];
                return '<div class="correlation-tooltip">' +
                       '<span>' + xLabel + ' / ' + yLabel + '</span><br>' +
                       '<span>Correlation: ' + value.toFixed(2) + '</span>' +
                       '</div>';
            }
            """
        }
        
        # Set color scale for correlation (-1 to 1)
        self.options["plotOptions"]["heatmap"]["colorScale"] = {
            "ranges": [
                {
                    "from": -1,
                    "to": -0.5,
                    "name": "strong negative",
                    "color": "#7B1FA2"
                },
                {
                    "from": -0.5,
                    "to": -0.25,
                    "name": "negative",
                    "color": "#C2185B"
                },
                {
                    "from": -0.25,
                    "to": 0.25,
                    "name": "neutral",
                    "color": "#BDBDBD"
                },
                {
                    "from": 0.25,
                    "to": 0.5,
                    "name": "positive",
                    "color": "#00ACC1"
                },
                {
                    "from": 0.5,
                    "to": 1,
                    "name": "strong positive",
                    "color": "#00897B"
                }
            ]
        }
    
    def set_correlation_data(self, correlation_matrix: Dict[str, Dict[str, float]]) -> None:
        """
        Set correlation matrix data
        
        Args:
            correlation_matrix: Dictionary mapping indicator pairs to correlation values
        """
        # Extract unique indicator names
        indicators = set()
        for pair, value in correlation_matrix.items():
            ind1, ind2 = pair.split('/')
            indicators.add(ind1)
            indicators.add(ind2)
        
        indicators = sorted(list(indicators))
        
        # Create heatmap data
        data = []
        for ind1 in indicators:
            for ind2 in indicators:
                # Find correlation value
                pair_key = f"{ind1}/{ind2}"
                rev_pair_key = f"{ind2}/{ind1}"
                
                if pair_key in correlation_matrix:
                    value = correlation_matrix[pair_key]
                elif rev_pair_key in correlation_matrix:
                    value = correlation_matrix[rev_pair_key]
                elif ind1 == ind2:
                    value = 1.0  # Self-correlation
                else:
                    value = None
                
                if value is not None:
                    data.append({
                        "x": ind1,
                        "y": ind2,
                        "value": value
                    })
        
        # Set the data
        self.set_data(data, indicators, indicators, -1, 1)


class SignalDistributionChart(BaseIndicatorChart):
    """Chart for displaying signal distribution and outcomes"""
    
    def __init__(self, id: str, title: str = "Signal Distribution", height: int = 300):
        """Initialize the signal distribution chart"""
        super().__init__(id, title, height)
        
        # Configure for bar chart
        self.options["chart"]["type"] = "bar"
        self.options["plotOptions"] = {
            "bar": {
                "horizontal": False,
                "columnWidth": "55%",
                "endingShape": "rounded"
            }
        }
        self.options["dataLabels"] = {
            "enabled": False
        }
        self.options["stroke"] = {
            "show": True,
            "width": 2,
            "colors": ["transparent"]
        }
        self.options["legend"] = {
            "show": True,
            "position": "top"
        }
        self.options["fill"] = {
            "opacity": 1
        }
    
    def set_distribution_data(self, categories: List[str], 
                           positive_counts: List[int],
                           negative_counts: List[int],
                           neutral_counts: List[int]) -> None:
        """
        Set signal distribution data
        
        Args:
            categories: Categories (e.g., timeframes, months)
            positive_counts: Counts of positive signals
            negative_counts: Counts of negative signals
            neutral_counts: Counts of neutral signals
        """
        self.series = [
            {
                "name": "Positive",
                "data": positive_counts,
                "color": "#00C853"
            },
            {
                "name": "Negative",
                "data": negative_counts,
                "color": "#FF5252"
            },
            {
                "name": "Neutral",
                "data": neutral_counts,
                "color": "#90CAF9"
            }
        ]
        
        # Set categories
        self.set_x_axis("Categories", categories)
        self.set_y_axis("Count")


class PerformanceComparisonChart(BaseIndicatorChart):
    """Chart for comparing indicator performance metrics"""
    
    def __init__(self, id: str, title: str = "Performance Comparison", height: int = 350):
        """Initialize the performance comparison chart"""
        super().__init__(id, title, height)
        
        # Configure for radar chart
        self.options["chart"]["type"] = "radar"
        self.options["dataLabels"] = {
            "enabled": True
        }
        self.options["plotOptions"] = {
            "radar": {
                "size": 140,
                "polygons": {
                    "strokeColors": "#e9e9e9",
                    "fill": {
                        "colors": ["#f8f8f8", "#fff"]
                    }
                }
            }
        }
        self.options["legend"] = {
            "show": True,
            "position": "top"
        }
    
    def set_comparison_data(self, indicators: List[str], metrics: List[str], 
                         values: List[List[float]]) -> None:
        """
        Set performance comparison data
        
        Args:
            indicators: List of indicator names
            metrics: List of metric names
            values: 2D list of values [indicator_idx][metric_idx]
        """
        # Create series for each indicator
        series_data = []
        for idx, indicator in enumerate(indicators):
            series_data.append({
                "name": indicator,
                "data": values[idx]
            })
        
        self.series = series_data
        
        # Set categories as metrics
        self.options["xaxis"] = {
            "categories": metrics
        }


def implement_visualization_components():
    """
    Implements consistent visual display components for all indicators.
    - Develops advanced charts and heat maps.
    - Adds interactive analysis tools.
    
    Returns:
        Dictionary with chart component classes
    """
    # Create component classes
    chart_components = {
        "price_chart": PriceChart,
        "indicator_panel": IndicatorPanel,
        "multi_indicator": MultiIndicatorChart,
        "heatmap": HeatmapChart,
        "correlation_matrix": CorrelationMatrix,
        "signal_distribution": SignalDistributionChart,
        "performance_comparison": PerformanceComparisonChart
    }
    
    # Create example chart for demonstration
    example_chart = PriceChart("demo-chart", "Example Price Chart")
    
    # Add sample OHLC data
    example_data = [
        {"x": "2023-01-01", "o": 1.2340, "h": 1.2390, "l": 1.2310, "c": 1.2350},
        {"x": "2023-01-02", "o": 1.2350, "h": 1.2410, "l": 1.2320, "c": 1.2370},
        {"x": "2023-01-03", "o": 1.2370, "h": 1.2420, "l": 1.2330, "c": 1.2340},
        {"x": "2023-01-04", "o": 1.2340, "h": 1.2390, "l": 1.2310, "c": 1.2380},
        {"x": "2023-01-05", "o": 1.2380, "h": 1.2430, "l": 1.2350, "c": 1.2400}
    ]
    example_chart.set_ohlc_data(example_data)
    
    # Add a sample indicator overlay
    ma_data = [1.2330, 1.2340, 1.2350, 1.2355, 1.2370]
    example_chart.add_indicator_overlay("MA(20)", ma_data, color="#8E24AA")
    
    logger.info(f"Initialized {len(chart_components)} visualization components")
    
    return chart_components
