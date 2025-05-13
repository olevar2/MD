"""
Customizable Dashboards Module for Forex Trading Platform

This module provides components for creating and saving custom dashboards,
configurable dashboard components, and a drag-and-drop interface.
"""
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import uuid
from datetime import datetime
logger = logging.getLogger(__name__)


from ui_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DashboardComponent:
    """Base class for all dashboard components"""

    def __init__(self, component_id: str=None, title: str='', position:
        Dict[str, Any]=None):
        """
        Initialize a dashboard component
        
        Args:
            component_id: Unique ID for the component
            title: Title of the component
            position: Position and size of component on dashboard
        """
        self.component_id = component_id or str(uuid.uuid4())
        self.title = title
        self.position = position or {'x': 0, 'y': 0, 'w': 6, 'h': 4, 'minW':
            2, 'minH': 2}
        self.component_type = 'base'
        self.config = {}

    def to_dict(self) ->Dict[str, Any]:
        """Convert the component to a dictionary for serialization"""
        return {'component_id': self.component_id, 'title': self.title,
            'position': self.position, 'component_type': self.
            component_type, 'config': self.config}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->'DashboardComponent':
        """Create a component from a dictionary"""
        component = cls(component_id=data.get('component_id', str(uuid.
            uuid4())), title=data.get('title', ''), position=data.get(
            'position', {}))
        component.config = data.get('config', {})
        return component


class IndicatorComponent(DashboardComponent):
    """Component for displaying a technical indicator"""

    def __init__(self, indicator_id: str, indicator_type: str, symbol: str=
        'EURUSD', timeframe: str='1h', component_id: str=None, title: str=
        '', position: Dict[str, Any]=None):
        """
        Initialize an indicator component
        
        Args:
            indicator_id: ID of the indicator
            indicator_type: Type of indicator
            symbol: Trading symbol
            timeframe: Timeframe for the indicator
            component_id: Unique ID for the component
            title: Title of the component
            position: Position and size of component on dashboard
        """
        super().__init__(component_id, title or
            f'{indicator_type.capitalize()} ({symbol} {timeframe})', position)
        self.component_type = 'indicator'
        self.config = {'indicator_id': indicator_id, 'indicator_type':
            indicator_type, 'symbol': symbol, 'timeframe': timeframe,
            'display_options': {'show_values': True, 'show_levels': True,
            'color_theme': 'default'}}


class PriceChartComponent(DashboardComponent):
    """Component for displaying a price chart"""

    def __init__(self, symbol: str='EURUSD', timeframe: str='1h',
        chart_type: str='candlestick', component_id: str=None, title: str=
        '', position: Dict[str, Any]=None):
        """
        Initialize a price chart component
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            chart_type: Type of chart (candlestick, line, area)
            component_id: Unique ID for the component
            title: Title of the component
            position: Position and size of component on dashboard
        """
        super().__init__(component_id, title or f'{symbol} {timeframe}',
            position)
        self.component_type = 'price_chart'
        self.config = {'symbol': symbol, 'timeframe': timeframe,
            'chart_type': chart_type, 'overlay_indicators': [],
            'display_options': {'show_volume': True, 'show_grid': True,
            'color_theme': 'default'}}


class CorrelationMatrixComponent(DashboardComponent):
    """Component for displaying a correlation matrix"""

    def __init__(self, symbols: List[str]=None, timeframe: str='1d',
        component_id: str=None, title: str='', position: Dict[str, Any]=None):
        """
        Initialize a correlation matrix component
        
        Args:
            symbols: List of symbols to include in the matrix
            timeframe: Timeframe for correlation calculation
            component_id: Unique ID for the component
            title: Title of the component
            position: Position and size of component on dashboard
        """
        super().__init__(component_id, title or 'Correlation Matrix', position)
        self.component_type = 'correlation_matrix'
        self.config = {'symbols': symbols or ['EURUSD', 'GBPUSD', 'USDJPY',
            'AUDUSD', 'USDCAD'], 'timeframe': timeframe, 'display_options':
            {'show_values': True, 'color_gradient': 'heatmap',
            'time_period': 30}}


class NewsComponent(DashboardComponent):
    """Component for displaying market news"""

    def __init__(self, sources: List[str]=None, max_items: int=10,
        component_id: str=None, title: str='', position: Dict[str, Any]=None):
        """
        Initialize a news component
        
        Args:
            sources: List of news sources
            max_items: Maximum number of news items to display
            component_id: Unique ID for the component
            title: Title of the component
            position: Position and size of component on dashboard
        """
        super().__init__(component_id, title or 'Market News', position)
        self.component_type = 'news'
        self.config = {'sources': sources or ['economic_calendar',
            'market_news', 'company_news'], 'max_items': max_items,
            'filter_keywords': [], 'display_options': {'show_timestamps': 
            True, 'show_source': True, 'show_summary': True}}


class AlertsComponent(DashboardComponent):
    """Component for displaying alerts"""

    def __init__(self, max_items: int=10, show_history: bool=True,
        component_id: str=None, title: str='', position: Dict[str, Any]=None):
        """
        Initialize an alerts component
        
        Args:
            max_items: Maximum number of alerts to display
            show_history: Whether to show historical alerts
            component_id: Unique ID for the component
            title: Title of the component
            position: Position and size of component on dashboard
        """
        super().__init__(component_id, title or 'Alert Center', position)
        self.component_type = 'alerts'
        self.config = {'max_items': max_items, 'show_history': show_history,
            'filter_severity': ['critical', 'warning', 'info'],
            'display_options': {'show_timestamps': True, 'show_severity': 
            True, 'group_by_symbol': False}}


class PerformanceComponent(DashboardComponent):
    """Component for displaying trading performance"""

    def __init__(self, time_period: str='1M', metrics: List[str]=None,
        component_id: str=None, title: str='', position: Dict[str, Any]=None):
        """
        Initialize a performance component
        
        Args:
            time_period: Time period for performance calculation
            metrics: List of metrics to display
            component_id: Unique ID for the component
            title: Title of the component
            position: Position and size of component on dashboard
        """
        super().__init__(component_id, title or
            f'Trading Performance ({time_period})', position)
        self.component_type = 'performance'
        self.config = {'time_period': time_period, 'metrics': metrics or [
            'profit_loss', 'win_rate', 'drawdown', 'sharpe_ratio'],
            'display_options': {'chart_type': 'bar', 'show_summary': True,
            'color_theme': 'default'}}


class Dashboard:
    """Represents a dashboard configuration"""

    def __init__(self, name: str, description: str='', components: List[
        DashboardComponent]=None):
        """
        Initialize a dashboard
        
        Args:
            name: Name of the dashboard
            description: Description of the dashboard
            components: List of dashboard components
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.components = components or []
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.layout = 'grid'
        self.config = {'grid_columns': 12, 'row_height': 50, 'background':
            '#f5f5f5', 'auto_refresh': True, 'refresh_interval': 60}

    def add_component(self, component: DashboardComponent) ->str:
        """
        Add a component to the dashboard
        
        Args:
            component: The component to add
            
        Returns:
            The component ID
        """
        self.components.append(component)
        self.updated_at = datetime.now()
        return component.component_id

    def remove_component(self, component_id: str) ->bool:
        """
        Remove a component from the dashboard
        
        Args:
            component_id: ID of the component to remove
            
        Returns:
            True if component was removed, False otherwise
        """
        for i, component in enumerate(self.components):
            if component.component_id == component_id:
                self.components.pop(i)
                self.updated_at = datetime.now()
                return True
        return False

    def update_component(self, component_id: str, updates: Dict[str, Any]
        ) ->bool:
        """
        Update a component's configuration
        
        Args:
            component_id: ID of the component to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if component was updated, False otherwise
        """
        for component in self.components:
            if component.component_id == component_id:
                if 'title' in updates:
                    component.title = updates['title']
                if 'position' in updates:
                    component.position.update(updates['position'])
                if 'config' in updates:
                    component.config.update(updates['config'])
                self.updated_at = datetime.now()
                return True
        return False

    def to_dict(self) ->Dict[str, Any]:
        """Convert the dashboard to a dictionary for serialization"""
        return {'id': self.id, 'name': self.name, 'description': self.
            description, 'components': [c.to_dict() for c in self.
            components], 'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(), 'layout': self.
            layout, 'config': self.config}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->'Dashboard':
        """Create a dashboard from a dictionary"""
        dashboard = cls(name=data.get('name', ''), description=data.get(
            'description', ''))
        dashboard.id = data.get('id', str(uuid.uuid4()))
        for component_data in data.get('components', []):
            component_type = component_data.get('component_type', 'base')
            if component_type == 'indicator':
                config = component_data.get('config', {})
                component = IndicatorComponent(indicator_id=config.get(
                    'indicator_id', ''), indicator_type=config.get(
                    'indicator_type', ''), symbol=config.get('symbol',
                    'EURUSD'), timeframe=config_manager.get('timeframe', '1h'),
                    component_id=component_data.get('component_id'), title=
                    component_data.get('title', ''), position=
                    component_data.get('position', {}))
                component.config = config
                dashboard.components.append(component)
            elif component_type == 'price_chart':
                config = component_data.get('config', {})
                component = PriceChartComponent(symbol=config.get('symbol',
                    'EURUSD'), timeframe=config_manager.get('timeframe', '1h'),
                    chart_type=config_manager.get('chart_type', 'candlestick'),
                    component_id=component_data.get('component_id'), title=
                    component_data.get('title', ''), position=
                    component_data.get('position', {}))
                component.config = config
                dashboard.components.append(component)
            elif component_type == 'correlation_matrix':
                config = component_data.get('config', {})
                component = CorrelationMatrixComponent(symbols=config.get(
                    'symbols', []), timeframe=config_manager.get('timeframe', '1d'),
                    component_id=component_data.get('component_id'), title=
                    component_data.get('title', ''), position=
                    component_data.get('position', {}))
                component.config = config
                dashboard.components.append(component)
            elif component_type == 'news':
                config = component_data.get('config', {})
                component = NewsComponent(sources=config_manager.get('sources', []),
                    max_items=config_manager.get('max_items', 10), component_id=
                    component_data.get('component_id'), title=
                    component_data.get('title', ''), position=
                    component_data.get('position', {}))
                component.config = config
                dashboard.components.append(component)
            elif component_type == 'alerts':
                config = component_data.get('config', {})
                component = AlertsComponent(max_items=config.get(
                    'max_items', 10), show_history=config.get(
                    'show_history', True), component_id=component_data.get(
                    'component_id'), title=component_data.get('title', ''),
                    position=component_data.get('position', {}))
                component.config = config
                dashboard.components.append(component)
            elif component_type == 'performance':
                config = component_data.get('config', {})
                component = PerformanceComponent(time_period=config.get(
                    'time_period', '1M'), metrics=config_manager.get('metrics', []),
                    component_id=component_data.get('component_id'), title=
                    component_data.get('title', ''), position=
                    component_data.get('position', {}))
                component.config = config
                dashboard.components.append(component)
            else:
                component = DashboardComponent.from_dict(component_data)
                dashboard.components.append(component)
        if 'created_at' in data:
            dashboard.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            dashboard.updated_at = datetime.fromisoformat(data['updated_at'])
        dashboard.layout = data.get('layout', 'grid')
        dashboard.config = data.get('config', dashboard.config)
        return dashboard


class DashboardManager:
    """
    Manages creating, loading, saving, and applying dashboard configurations
    """

    def __init__(self, storage_path: str=None):
        """
        Initialize the dashboard manager
        
        Args:
            storage_path: Path to store dashboards
        """
        self.storage_path = storage_path or 'dashboards'
        self.dashboards: Dict[str, Dashboard] = {}
        self.active_dashboard_id = None

    def create_dashboard(self, name: str, description: str='', components:
        List[DashboardComponent]=None) ->Dashboard:
        """
        Create a new dashboard
        
        Args:
            name: Name of the dashboard
            description: Description of the dashboard
            components: Initial list of components
            
        Returns:
            The created dashboard
        """
        dashboard = Dashboard(name, description, components)
        self.dashboards[dashboard.id] = dashboard
        return dashboard

    def get_dashboard(self, dashboard_id: str) ->Optional[Dashboard]:
        """Get a dashboard by ID"""
        return self.dashboards.get(dashboard_id)

    def get_all_dashboards(self) ->List[Dict[str, Any]]:
        """
        Get all dashboards as summary dictionaries
        
        Returns:
            List of dashboard summaries
        """
        return [{'id': dashboard.id, 'name': dashboard.name, 'description':
            dashboard.description, 'component_count': len(dashboard.
            components), 'created_at': dashboard.created_at.isoformat(),
            'updated_at': dashboard.updated_at.isoformat()} for dashboard in
            self.dashboards.values()]

    def set_active_dashboard(self, dashboard_id: str) ->bool:
        """
        Set the active dashboard
        
        Args:
            dashboard_id: ID of the dashboard to activate
            
        Returns:
            True if successful, False otherwise
        """
        if dashboard_id in self.dashboards:
            self.active_dashboard_id = dashboard_id
            return True
        return False

    def get_active_dashboard(self) ->Optional[Dashboard]:
        """Get the active dashboard"""
        if self.active_dashboard_id:
            return self.dashboards.get(self.active_dashboard_id)
        return None

    @with_exception_handling
    def save_dashboard(self, dashboard: Dashboard) ->bool:
        """
        Save a dashboard to storage
        
        Args:
            dashboard: The dashboard to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.storage_path:
            logger.error('No storage path configured')
            return False
        try:
            import os
            os.makedirs(self.storage_path, exist_ok=True)
            filepath = os.path.join(self.storage_path, f'{dashboard.id}.json')
            with open(filepath, 'w') as f:
                json.dump(dashboard.to_dict(), f, indent=2)
            logger.info(f'Dashboard saved: {dashboard.name}')
            return True
        except Exception as e:
            logger.error(f'Failed to save dashboard: {str(e)}')
            return False

    @with_exception_handling
    def load_dashboards(self) ->int:
        """
        Load all dashboards from storage
        
        Returns:
            Number of dashboards loaded
        """
        if not self.storage_path:
            logger.error('No storage path configured')
            return 0
        try:
            import os
            if not os.path.exists(self.storage_path):
                logger.warning(
                    f'Storage path does not exist: {self.storage_path}')
                return 0
            count = 0
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.storage_path, filename)
                    try:
                        with open(filepath, 'r') as f:
                            dashboard_data = json.load(f)
                            dashboard = Dashboard.from_dict(dashboard_data)
                            self.dashboards[dashboard.id] = dashboard
                            count += 1
                    except Exception as e:
                        logger.error(
                            f'Failed to load dashboard {filename}: {str(e)}')
            logger.info(f'Loaded {count} dashboards')
            return count
        except Exception as e:
            logger.error(f'Failed to load dashboards: {str(e)}')
            return 0

    @with_exception_handling
    def delete_dashboard(self, dashboard_id: str) ->bool:
        """
        Delete a dashboard
        
        Args:
            dashboard_id: ID of the dashboard to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if dashboard_id not in self.dashboards:
            return False
        del self.dashboards[dashboard_id]
        if self.active_dashboard_id == dashboard_id:
            self.active_dashboard_id = None
        if self.storage_path:
            try:
                import os
                filepath = os.path.join(self.storage_path,
                    f'{dashboard_id}.json')
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.error(f'Failed to delete dashboard file: {str(e)}')
        return True


class DashboardComponentFactory:
    """
    Factory for creating dashboard components
    """

    @staticmethod
    def create_indicator_component(indicator_id: str, indicator_type: str,
        symbol: str='EURUSD', timeframe: str='1h', title: str=''
        ) ->IndicatorComponent:
        """Create an indicator component"""
        return IndicatorComponent(indicator_id=indicator_id, indicator_type
            =indicator_type, symbol=symbol, timeframe=timeframe, title=title)

    @staticmethod
    def create_price_chart(symbol: str='EURUSD', timeframe: str='1h',
        chart_type: str='candlestick', title: str='') ->PriceChartComponent:
        """Create a price chart component"""
        return PriceChartComponent(symbol=symbol, timeframe=timeframe,
            chart_type=chart_type, title=title)

    @staticmethod
    def create_correlation_matrix(symbols: List[str]=None, timeframe: str=
        '1d', title: str='') ->CorrelationMatrixComponent:
        """Create a correlation matrix component"""
        return CorrelationMatrixComponent(symbols=symbols, timeframe=
            timeframe, title=title)

    @staticmethod
    def create_news_component(sources: List[str]=None, max_items: int=10,
        title: str='') ->NewsComponent:
        """Create a news component"""
        return NewsComponent(sources=sources, max_items=max_items, title=title)

    @staticmethod
    def create_alerts_component(max_items: int=10, show_history: bool=True,
        title: str='') ->AlertsComponent:
        """Create an alerts component"""
        return AlertsComponent(max_items=max_items, show_history=
            show_history, title=title)

    @staticmethod
    def create_performance_component(time_period: str='1M', metrics: List[
        str]=None, title: str='') ->PerformanceComponent:
        """Create a performance component"""
        return PerformanceComponent(time_period=time_period, metrics=
            metrics, title=title)


""""""
