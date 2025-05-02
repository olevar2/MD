"""
Real-time Incremental Data Processor

This module handles real-time processing of tick data and incremental
calculation of technical indicators for low-latency applications.
"""
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Callable
from ..indicators.incremental_indicators import (
    IncrementalIndicatorManager, SMAState, EMAState,
    RSIState, MACDState, BollingerBandsState, ATRState
)


class RealTimeFeatureProcessor:
    """
    Process tick data in real-time and maintain incrementally calculated indicators
    
    This class provides low-latency processing of market data ticks and
    incremental calculation of technical indicators for reducing latency
    in the critical decision-making path.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real-time feature processor
        
        Args:
            config: Optional configuration dictionary
                   Can include 'indicators' list with indicator configurations
        """
        self.indicator_manager = IncrementalIndicatorManager()
        self.latest_ticks: Dict[str, Dict[str, Any]] = {}  # Symbol -> tick data
        self.listeners: Dict[str, List[Callable]] = {}  # Symbol -> callback functions
        self.logger = logging.getLogger(__name__)
        
        # Configure indicators from config if provided
        if config and 'indicators' in config:
            self._configure_indicators(config['indicators'])
    
    def _configure_indicators(self, indicator_configs: List[Dict[str, Any]]) -> None:
        """
        Configure indicators based on provided configurations
        
        Args:
            indicator_configs: List of indicator configurations
        """
        for config in indicator_configs:
            indicator_type = config.get('type', '').lower()
            
            if indicator_type == 'sma':
                self.add_sma_indicator(
                    window_size=config.get('window_size', 20),
                    price_type=config.get('price_type', 'close')
                )
            
            elif indicator_type == 'ema':
                self.add_ema_indicator(
                    window_size=config.get('window_size', 20),
                    price_type=config.get('price_type', 'close'),
                    smoothing=config.get('smoothing', 2.0)
                )
            
            elif indicator_type == 'rsi':
                self.add_rsi_indicator(
                    window_size=config.get('window_size', 14),
                    price_type=config.get('price_type', 'close')
                )
                
            elif indicator_type == 'macd':
                self.add_macd_indicator(
                    fast_period=config.get('fast_period', 12),
                    slow_period=config.get('slow_period', 26),
                    signal_period=config.get('signal_period', 9),
                    price_type=config.get('price_type', 'close')
                )
                
            elif indicator_type == 'bollinger_bands' or indicator_type == 'bbands':
                self.add_bollinger_bands_indicator(
                    window_size=config.get('window_size', 20),
                    num_std=config.get('num_std', 2.0),
                    price_type=config.get('price_type', 'close')
                )
                
            elif indicator_type == 'atr':
                self.add_atr_indicator(
                    window_size=config.get('window_size', 14)
                )
            
            else:
                self.logger.warning(f"Unknown indicator type: {indicator_type}")
    
    def add_sma_indicator(self, window_size: int, price_type: str = 'close') -> None:
        """
        Add a Simple Moving Average indicator
        
        Args:
            window_size: Window size for SMA calculation
            price_type: Price type to use (open, high, low, close, etc.)
        """
        indicator = SMAState(window_size=window_size, price_type=price_type)
        self.indicator_manager.add_indicator(indicator)
    
    def add_ema_indicator(
        self, window_size: int, price_type: str = 'close', smoothing: float = 2.0
    ) -> None:
        """
        Add an Exponential Moving Average indicator
        
        Args:
            window_size: Window size for EMA calculation
            price_type: Price type to use
            smoothing: Smoothing factor for EMA
        """
        indicator = EMAState(window_size=window_size, price_type=price_type, smoothing=smoothing)
        self.indicator_manager.add_indicator(indicator)
    
    def add_rsi_indicator(self, window_size: int = 14, price_type: str = 'close') -> None:
        """
        Add a Relative Strength Index indicator
        
        Args:
            window_size: Window size for RSI calculation
            price_type: Price type to use
        """
        indicator = RSIState(window_size=window_size, price_type=price_type)
        self.indicator_manager.add_indicator(indicator)
    
    def add_macd_indicator(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_type: str = 'close'
    ) -> None:
        """
        Add a MACD indicator
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            price_type: Price type to use
        """
        indicator = MACDState(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            price_type=price_type
        )
        self.indicator_manager.add_indicator(indicator)
        
    def add_bollinger_bands_indicator(
        self, 
        window_size: int = 20, 
        num_std: float = 2.0, 
        price_type: str = 'close'
    ) -> None:
        """
        Add a Bollinger Bands indicator
        
        Args:
            window_size: Window size for calculation
            num_std: Number of standard deviations for bands
            price_type: Price type to use
        """
        indicator = BollingerBandsState(
            window_size=window_size,
            num_std=num_std,
            price_type=price_type
        )
        self.indicator_manager.add_indicator(indicator)
        
    def add_atr_indicator(self, window_size: int = 14) -> None:
        """
        Add an Average True Range indicator
        
        Args:
            window_size: Window size for ATR calculation
        """
        indicator = ATRState(window_size=window_size)
        self.indicator_manager.add_indicator(indicator)
    
    def process_tick(self, symbol: str, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a new tick and update indicators incrementally
        
        Args:
            symbol: Trading symbol (e.g., 'EUR_USD')
            tick_data: Dictionary with tick data
                      Must include at least 'timestamp' and price fields
        
        Returns:
            Dictionary with updated indicator values
        """
        if 'timestamp' not in tick_data:
            tick_data['timestamp'] = datetime.now()
        
        # Store latest tick for the symbol
        self.latest_ticks[symbol] = tick_data
        
        # Update indicators with new tick data
        updated_indicators = self.indicator_manager.update_all(tick_data)
        
        # Combine tick data with indicator values
        result = {
            'symbol': symbol,
            'timestamp': tick_data['timestamp'],
            **tick_data,
            'indicators': updated_indicators
        }
        
        # Notify listeners if any
        if symbol in self.listeners:
            for callback in self.listeners[symbol]:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Error in listener callback: {str(e)}")
        
        return result
    
    def register_listener(self, symbol: str, callback: Callable) -> None:
        """
        Register a callback function to be called when new tick data arrives
        
        Args:
            symbol: Symbol to listen for
            callback: Function to call with processed tick data
        """
        if symbol not in self.listeners:
            self.listeners[symbol] = []
        
        self.listeners[symbol].append(callback)
        self.logger.info(f"Registered listener for {symbol}")
    
    def unregister_listener(self, symbol: str, callback: Callable) -> bool:
        """
        Unregister a previously registered callback
        
        Args:
            symbol: Symbol to unregister listener for
            callback: Callback function to unregister
            
        Returns:
            True if successfully unregistered, False otherwise
        """
        if symbol not in self.listeners:
            return False
        
        try:
            self.listeners[symbol].remove(callback)
            self.logger.info(f"Unregistered listener for {symbol}")
            return True
        except ValueError:
            return False
    
    def get_latest_indicators(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the latest indicator values
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Dictionary with latest indicator values
        """
        indicators = self.indicator_manager.get_latest_values()
        
        if symbol:
            if symbol in self.latest_ticks:
                return {
                    'symbol': symbol,
                    'timestamp': self.latest_ticks[symbol].get('timestamp', datetime.now()),
                    'indicators': indicators
                }
            else:
                return {
                    'symbol': symbol,
                    'indicators': indicators
                }
        else:
            return {
                'indicators': indicators,
                'symbols': list(self.latest_ticks.keys())
            }
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset indicator state
        
        Args:
            symbol: Optional symbol to reset state for
                   If None, resets all indicator state
        """
        if symbol:
            if symbol in self.latest_ticks:
                del self.latest_ticks[symbol]
        else:
            self.latest_ticks.clear()
        
        # Reset all indicator state
        self.indicator_manager.reset_all()
        self.logger.info(f"Reset indicator state for {symbol if symbol else 'all symbols'}")


class WebSocketTickProcessor:
    """
    Process tick data from WebSocket connections
    
    This class manages WebSocket connections to market data providers
    and processes incoming tick data in real-time.
    """
    
    def __init__(self, processor: RealTimeFeatureProcessor):
        """
        Initialize WebSocket tick processor
        
        Args:
            processor: Real-time feature processor instance
        """
        self.processor = processor
        self.websocket_tasks = {}
        self.running = False
        self.connection_info = {}  # Store URL and API key for reconnection
        self.logger = logging.getLogger(__name__)
    
    async def connect_to_feed(self, url: str, symbol: str, api_key: Optional[str] = None) -> None:
        """
        Connect to a WebSocket feed and begin processing ticks
        
        Args:
            url: WebSocket URL
            symbol: Symbol to subscribe to
            api_key: Optional API key for authentication
        """
        try:
            import websockets
        except ImportError:
            self.logger.error("websockets package not installed. Run 'pip install websockets'")
            return
        
        # If already connected to this symbol, disconnect first
        if symbol in self.websocket_tasks and not self.websocket_tasks[symbol].done():
            self.logger.info(f"Already connected to {symbol}, disconnecting first")
            self.websocket_tasks[symbol].cancel()
        
        # Store connection info for reconnection
        self.connection_info[symbol] = {
            'url': url,
            'api_key': api_key
        }
        
        # Create headers if API key is provided
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        
        self.running = True
        
        async def _process_feed():
            reconnect_delay = 1
            max_reconnect_delay = 60
            
            while self.running:
                try:
                    self.logger.info(f"Connecting to WebSocket feed for {symbol}: {url}")
                    
                    async with websockets.connect(url, extra_headers=headers) as websocket:
                        # Subscribe to the symbol
                        subscribe_msg = {
                            "type": "subscribe",
                            "symbol": symbol
                        }
                        await websocket.send(json.dumps(subscribe_msg))
                        
                        # Reset reconnect delay on successful connection
                        reconnect_delay = 1
                        
                        # Process incoming messages
                        while self.running:
                            try:
                                message = await websocket.recv()
                                tick_data = json.loads(message)
                                
                                # Process tick data
                                self.processor.process_tick(symbol, tick_data)
                            except websockets.exceptions.ConnectionClosed:
                                self.logger.warning(f"WebSocket connection closed for {symbol}")
                                break
                            except json.JSONDecodeError:
                                self.logger.error(f"Invalid JSON received: {message}")
                                continue
                            except Exception as e:
                                self.logger.error(f"Error processing WebSocket message: {str(e)}")
                                continue
                
                except Exception as e:
                    self.logger.error(f"WebSocket connection error for {symbol}: {str(e)}")
                
                # Only try to reconnect if still running
                if self.running:
                    self.logger.info(f"Reconnecting in {reconnect_delay} seconds...")
                    await asyncio.sleep(reconnect_delay)
                    
                    # Exponential backoff with max delay
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
        
        # Start processing task
        self.websocket_tasks[symbol] = asyncio.create_task(_process_feed())
        self.logger.info(f"Started WebSocket processing task for {symbol}")
    
    async def disconnect(self, symbol: Optional[str] = None) -> None:
        """
        Disconnect from WebSocket feed(s)
        
        Args:
            symbol: Optional symbol to disconnect from
                   If None, disconnects from all feeds
        """
        if symbol:
            if symbol in self.websocket_tasks:
                self.websocket_tasks[symbol].cancel()
                self.logger.info(f"Disconnected from {symbol} feed")
        else:
            self.running = False
            for symbol, task in self.websocket_tasks.items():
                task.cancel()
            self.logger.info("Disconnected from all WebSocket feeds")
    
    async def reconnect_all(self) -> None:
        """Reconnect all active WebSocket feeds"""
        for symbol, info in self.connection_info.items():
            self.logger.info(f"Reconnecting to {symbol} feed")
            await self.connect_to_feed(
                url=info['url'],
                symbol=symbol,
                api_key=info['api_key']
            )


class LatencyMonitor:
    """
    Monitor and report latency statistics for real-time processing
    
    This class tracks processing time for various components of the 
    real-time data pipeline and generates latency reports.
    """
    
    def __init__(self):
        """Initialize the latency monitor"""
        self.latency_stats = {
            'tick_processing': [],  # Processing time for each tick
            'indicator_updates': {},  # Processing time for each indicator type
            'total_pipeline': []  # End-to-end latency
        }
        self.max_samples = 1000  # Maximum number of samples to store
        self.logger = logging.getLogger(__name__)
    
    def record_tick_latency(self, processing_time_ms: float) -> None:
        """
        Record tick processing latency
        
        Args:
            processing_time_ms: Time in milliseconds to process a tick
        """
        self._add_sample('tick_processing', processing_time_ms)
    
    def record_indicator_latency(self, indicator_type: str, processing_time_ms: float) -> None:
        """
        Record indicator update latency
        
        Args:
            indicator_type: Type of indicator being updated
            processing_time_ms: Time in milliseconds to update the indicator
        """
        if indicator_type not in self.latency_stats['indicator_updates']:
            self.latency_stats['indicator_updates'][indicator_type] = []
        
        samples = self.latency_stats['indicator_updates'][indicator_type]
        if len(samples) >= self.max_samples:
            samples.pop(0)
        
        samples.append(processing_time_ms)
    
    def record_pipeline_latency(self, processing_time_ms: float) -> None:
        """
        Record end-to-end pipeline latency
        
        Args:
            processing_time_ms: Time in milliseconds for the entire pipeline
        """
        self._add_sample('total_pipeline', processing_time_ms)
    
    def _add_sample(self, category: str, value: float) -> None:
        """Helper to add a sample to a category with bounds checking"""
        samples = self.latency_stats[category]
        if len(samples) >= self.max_samples:
            samples.pop(0)
        
        samples.append(value)
    
    def get_latency_report(self) -> Dict[str, Any]:
        """
        Generate a latency report with statistics
        
        Returns:
            Dictionary with latency statistics
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'tick_processing': self._calculate_stats(self.latency_stats['tick_processing']),
            'indicator_updates': {},
            'total_pipeline': self._calculate_stats(self.latency_stats['total_pipeline'])
        }
        
        # Calculate stats for each indicator type
        for indicator_type, samples in self.latency_stats['indicator_updates'].items():
            report['indicator_updates'][indicator_type] = self._calculate_stats(samples)
        
        return report
    
    def _calculate_stats(self, samples: List[float]) -> Dict[str, float]:
        """Calculate common statistics from a sample list"""
        if not samples:
            return {
                'min': None,
                'max': None,
                'avg': None,
                'median': None,
                'p95': None,
                'p99': None,
                'samples': 0
            }
        
        import numpy as np
        
        return {
            'min': float(np.min(samples)),
            'max': float(np.max(samples)),
            'avg': float(np.mean(samples)),
            'median': float(np.median(samples)),
            'p95': float(np.percentile(samples, 95)),
            'p99': float(np.percentile(samples, 99)),
            'samples': len(samples)
        }
    
    def reset_stats(self) -> None:
        """Clear all latency statistics"""
        self.latency_stats = {
            'tick_processing': [],
            'indicator_updates': {},
            'total_pipeline': []
        }
        self.logger.info("Reset all latency statistics")