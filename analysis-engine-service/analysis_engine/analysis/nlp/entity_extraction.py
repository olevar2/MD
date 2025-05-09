"""
Entity Extraction Module

This module provides advanced entity extraction capabilities for the chat interface,
identifying domain-specific entities like currency pairs, timeframes, indicators, etc.
"""

from typing import Dict, List, Any, Optional, Union
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    Advanced entity extraction for chat messages.
    
    This class provides sophisticated entity extraction capabilities,
    identifying domain-specific entities in user messages.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the entity extractor.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize entity extractors
        self.extractors = {
            "CURRENCY_PAIR": self._extract_currency_pairs,
            "TIMEFRAME": self._extract_timeframes,
            "INDICATOR": self._extract_indicators,
            "AMOUNT": self._extract_amounts,
            "PRICE": self._extract_prices,
            "DATE": self._extract_dates,
            "TIME": self._extract_times,
            "PERCENTAGE": self._extract_percentages,
            "TERM": self._extract_trading_terms,
            "PATTERN": self._extract_chart_patterns,
        }
        
        # Initialize entity data
        self._initialize_entity_data()
    
    def _initialize_entity_data(self):
        """Initialize data for entity extraction."""
        # Currency data
        self.currencies = {
            "USD": ["dollar", "usd", "$", "us dollar", "greenback"],
            "EUR": ["euro", "eur", "€", "single currency"],
            "GBP": ["pound", "gbp", "£", "sterling", "cable"],
            "JPY": ["yen", "jpy", "¥", "japanese yen"],
            "AUD": ["aussie", "aud", "australian dollar"],
            "CAD": ["loonie", "cad", "canadian dollar"],
            "CHF": ["swissie", "chf", "swiss franc"],
            "NZD": ["kiwi", "nzd", "new zealand dollar"],
        }
        
        # Common currency pairs
        self.currency_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", 
            "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
            "AUDJPY", "EURAUD", "EURCHF", "EURNZD", "GBPAUD",
            "GBPCAD", "GBPCHF", "AUDCAD", "AUDCHF", "AUDNZD",
            "CADCHF", "CADJPY", "CHFJPY", "NZDCAD", "NZDCHF",
            "NZDJPY"
        ]
        
        # Timeframes with aliases
        self.timeframes = {
            "1m": ["1m", "1 minute", "1min", "one minute", "1-minute", "m1"],
            "5m": ["5m", "5 minute", "5min", "five minute", "5-minute", "m5"],
            "15m": ["15m", "15 minute", "15min", "fifteen minute", "15-minute", "m15"],
            "30m": ["30m", "30 minute", "30min", "thirty minute", "30-minute", "m30"],
            "1h": ["1h", "1 hour", "1hr", "hourly", "one hour", "1-hour", "h1"],
            "4h": ["4h", "4 hour", "4hr", "four hour", "4-hour", "h4"],
            "1d": ["1d", "daily", "day", "one day", "1-day", "d1", "daily chart"],
            "1w": ["1w", "weekly", "week", "one week", "1-week", "w1", "weekly chart"],
            "1M": ["1M", "monthly", "month", "one month", "1-month", "M1", "monthly chart"]
        }
        
        # Technical indicators
        self.indicators = {
            "RSI": ["rsi", "relative strength index", "relative strength"],
            "MACD": ["macd", "moving average convergence divergence"],
            "Bollinger Bands": ["bollinger", "bollinger bands", "bands"],
            "Moving Average": ["ma", "moving average", "average", "ema", "sma", "exponential moving average", "simple moving average"],
            "Stochastic": ["stochastic", "stoch", "stochastic oscillator"],
            "Fibonacci": ["fibonacci", "fib", "fibonacci retracement", "fibonacci levels"],
            "Ichimoku": ["ichimoku", "ichimoku cloud", "cloud"],
            "ATR": ["atr", "average true range"],
            "ADX": ["adx", "average directional index", "directional index"],
            "OBV": ["obv", "on balance volume"],
            "VWAP": ["vwap", "volume weighted average price"]
        }
        
        # Chart patterns
        self.chart_patterns = {
            "Head and Shoulders": ["head and shoulders", "h&s", "head & shoulders", "inverse head and shoulders"],
            "Double Top": ["double top", "double bottom", "double tops", "double bottoms"],
            "Triangle": ["triangle", "ascending triangle", "descending triangle", "symmetrical triangle"],
            "Flag": ["flag", "pennant", "flag pattern", "pennant pattern"],
            "Channel": ["channel", "ascending channel", "descending channel", "horizontal channel"],
            "Wedge": ["wedge", "rising wedge", "falling wedge"],
            "Cup and Handle": ["cup and handle", "cup & handle", "cup with handle"],
            "Engulfing": ["engulfing", "bullish engulfing", "bearish engulfing"],
            "Doji": ["doji", "doji star", "long-legged doji", "dragonfly doji", "gravestone doji"]
        }
        
        # Trading terms
        self.trading_terms = {
            "Pip": ["pip", "pips", "point", "points"],
            "Lot": ["lot", "lots", "micro lot", "mini lot", "standard lot"],
            "Spread": ["spread", "spreads", "bid-ask spread"],
            "Leverage": ["leverage", "margin", "leveraged"],
            "Stop Loss": ["stop loss", "sl", "stop", "stops"],
            "Take Profit": ["take profit", "tp", "profit target", "target"],
            "Trend": ["trend", "trending", "uptrend", "downtrend", "sideways"],
            "Support": ["support", "support level", "support area", "support zone"],
            "Resistance": ["resistance", "resistance level", "resistance area", "resistance zone"],
            "Breakout": ["breakout", "break out", "breaking out", "break"],
            "Reversal": ["reversal", "reverse", "reversing", "trend reversal"],
            "Consolidation": ["consolidation", "consolidating", "consolidate", "range", "ranging"]
        }
    
    def extract_entities(self, message: str) -> List[Dict[str, Any]]:
        """
        Extract entities from a message.
        
        Args:
            message: User message
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Apply each extractor
        for entity_type, extractor in self.extractors.items():
            extracted = extractor(message)
            entities.extend(extracted)
        
        # Sort entities by start position
        entities.sort(key=lambda x: x["start"])
        
        return entities
    
    def _extract_currency_pairs(self, message: str) -> List[Dict[str, Any]]:
        """Extract currency pairs from message."""
        entities = []
        message_upper = message.upper()
        
        # First, check for exact currency pair matches
        for pair in self.currency_pairs:
            if pair in message_upper:
                start = message_upper.find(pair)
                entities.append({
                    "text": pair,
                    "label": "CURRENCY_PAIR",
                    "start": start,
                    "end": start + len(pair),
                    "value": pair
                })
        
        # Then, check for currency pair descriptions
        for pair in self.currency_pairs:
            base = pair[:3]
            quote = pair[3:]
            
            # Check for "EUR/USD", "EUR-USD", "EUR USD" formats
            for separator in ["/", "-", " "]:
                pair_with_separator = f"{base}{separator}{quote}"
                if pair_with_separator in message_upper:
                    start = message_upper.find(pair_with_separator)
                    entities.append({
                        "text": pair_with_separator,
                        "label": "CURRENCY_PAIR",
                        "start": start,
                        "end": start + len(pair_with_separator),
                        "value": pair
                    })
        
        # Check for currency names
        message_lower = message.lower()
        for pair in self.currency_pairs:
            base = pair[:3]
            quote = pair[3:]
            
            # Check if both currencies are mentioned close to each other
            if base in self.currencies and quote in self.currencies:
                base_aliases = self.currencies[base]
                quote_aliases = self.currencies[quote]
                
                for base_alias in base_aliases:
                    if base_alias in message_lower:
                        base_pos = message_lower.find(base_alias)
                        
                        for quote_alias in quote_aliases:
                            if quote_alias in message_lower:
                                quote_pos = message_lower.find(quote_alias)
                                
                                # If currencies are within 20 characters of each other
                                if abs(base_pos - quote_pos) < 20:
                                    start = min(base_pos, quote_pos)
                                    end = max(base_pos + len(base_alias), quote_pos + len(quote_alias))
                                    
                                    entities.append({
                                        "text": message[start:end],
                                        "label": "CURRENCY_PAIR",
                                        "start": start,
                                        "end": end,
                                        "value": pair
                                    })
        
        return entities
    
    def _extract_timeframes(self, message: str) -> List[Dict[str, Any]]:
        """Extract timeframes from message."""
        entities = []
        message_lower = message.lower()
        
        # Check for timeframe aliases
        for tf, aliases in self.timeframes.items():
            for alias in aliases:
                if alias in message_lower:
                    start = message_lower.find(alias)
                    entities.append({
                        "text": alias,
                        "label": "TIMEFRAME",
                        "start": start,
                        "end": start + len(alias),
                        "value": tf
                    })
        
        # Check for numeric timeframes (e.g., "15 minute chart")
        numeric_timeframe_patterns = [
            r'(\d+)\s*(?:minute|min|m)s?',
            r'(\d+)\s*(?:hour|hr|h)s?',
            r'(\d+)\s*(?:day|d)s?',
            r'(\d+)\s*(?:week|w)s?',
            r'(\d+)\s*(?:month|M)s?'
        ]
        
        for pattern in numeric_timeframe_patterns:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                value = match.group(1)
                unit = match.group(0)[len(value):].strip()
                
                # Determine the timeframe value
                if "minute" in unit or "min" in unit or unit.endswith("m"):
                    if value == "1":
                        tf_value = "1m"
                    elif value == "5":
                        tf_value = "5m"
                    elif value == "15":
                        tf_value = "15m"
                    elif value == "30":
                        tf_value = "30m"
                    else:
                        tf_value = f"{value}m"
                elif "hour" in unit or "hr" in unit or unit.endswith("h"):
                    if value == "1":
                        tf_value = "1h"
                    elif value == "4":
                        tf_value = "4h"
                    else:
                        tf_value = f"{value}h"
                elif "day" in unit or unit.endswith("d"):
                    tf_value = "1d" if value == "1" else f"{value}d"
                elif "week" in unit or unit.endswith("w"):
                    tf_value = "1w" if value == "1" else f"{value}w"
                elif "month" in unit or unit.endswith("M"):
                    tf_value = "1M" if value == "1" else f"{value}M"
                else:
                    continue
                
                entities.append({
                    "text": match.group(0),
                    "label": "TIMEFRAME",
                    "start": match.start(),
                    "end": match.end(),
                    "value": tf_value
                })
        
        return entities
    
    def _extract_indicators(self, message: str) -> List[Dict[str, Any]]:
        """Extract technical indicators from message."""
        entities = []
        message_lower = message.lower()
        
        # Check for indicator aliases
        for indicator, aliases in self.indicators.items():
            for alias in aliases:
                if alias in message_lower:
                    start = message_lower.find(alias)
                    entities.append({
                        "text": alias,
                        "label": "INDICATOR",
                        "start": start,
                        "end": start + len(alias),
                        "value": indicator
                    })
        
        # Check for indicator parameters
        indicator_param_patterns = [
            r'(\d+)(?:-|\s)?(?:period|day|bar)s?\s+(?:rsi|macd|ma|ema|sma|stochastic)',
            r'(?:rsi|macd|ma|ema|sma|stochastic)\s+(?:with\s+)?(\d+)(?:-|\s)?(?:period|day|bar)s?'
        ]
        
        for pattern in indicator_param_patterns:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                period = match.group(1)
                indicator_text = match.group(0)
                
                # Determine the indicator type
                indicator_type = None
                for ind, aliases in self.indicators.items():
                    if any(alias in indicator_text for alias in aliases):
                        indicator_type = ind
                        break
                
                if indicator_type:
                    entities.append({
                        "text": indicator_text,
                        "label": "INDICATOR",
                        "start": match.start(),
                        "end": match.end(),
                        "value": indicator_type,
                        "parameters": {
                            "period": int(period)
                        }
                    })
        
        return entities
    
    def _extract_amounts(self, message: str) -> List[Dict[str, Any]]:
        """Extract trading amounts from message."""
        entities = []
        
        # Look for patterns like "0.1 lot", "2 lots", "5 micro lots", etc.
        amount_patterns = [
            r'(\d+\.?\d*)\s*(lot|lots)',
            r'(\d+\.?\d*)\s*(micro|mini|standard)\s*(lot|lots)',
            r'(\d+\.?\d*)\s*k',
            r'(\d+\.?\d*)\s*units'
        ]
        
        for pattern in amount_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                value = float(match.group(1))
                
                # Adjust value based on lot type
                if "micro" in match.group(0):
                    value *= 0.01
                elif "mini" in match.group(0):
                    value *= 0.1
                
                entities.append({
                    "text": match.group(0),
                    "label": "AMOUNT",
                    "start": match.start(),
                    "end": match.end(),
                    "value": value
                })
        
        return entities
    
    def _extract_prices(self, message: str) -> List[Dict[str, Any]]:
        """Extract price levels from message."""
        entities = []
        
        # Look for patterns like "at 1.2345", "price of 1.2345", etc.
        price_patterns = [
            r'at\s+(\d+\.?\d*)',
            r'price\s+of\s+(\d+\.?\d*)',
            r'level\s+of\s+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s+level',
            r'(\d+\.?\d*)\s+price',
            r'(\d+\.?\d*)\s+resistance',
            r'(\d+\.?\d*)\s+support',
            r'resistance\s+at\s+(\d+\.?\d*)',
            r'support\s+at\s+(\d+\.?\d*)'
        ]
        
        for pattern in price_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                price_value = float(match.group(1))
                
                # Determine if it's a support or resistance level
                level_type = None
                if "resistance" in match.group(0):
                    level_type = "resistance"
                elif "support" in match.group(0):
                    level_type = "support"
                
                entity = {
                    "text": match.group(0),
                    "label": "PRICE",
                    "start": match.start(),
                    "end": match.end(),
                    "value": price_value
                }
                
                if level_type:
                    entity["level_type"] = level_type
                
                entities.append(entity)
        
        return entities
    
    def _extract_dates(self, message: str) -> List[Dict[str, Any]]:
        """Extract dates from message."""
        entities = []
        
        # Look for absolute dates (YYYY-MM-DD, MM/DD/YYYY, etc.)
        date_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY or DD/MM/YYYY
            r'(\d{1,2})-(\d{1,2})-(\d{4})'   # MM-DD-YYYY or DD-MM-YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, message)
            for match in matches:
                try:
                    # Try to parse the date
                    if "-" in match.group(0) and len(match.group(1)) == 4:
                        # YYYY-MM-DD
                        year = int(match.group(1))
                        month = int(match.group(2))
                        day = int(match.group(3))
                    else:
                        # MM/DD/YYYY or DD/MM/YYYY (assume MM/DD/YYYY for simplicity)
                        month = int(match.group(1))
                        day = int(match.group(2))
                        year = int(match.group(3))
                    
                    date_value = datetime(year, month, day).strftime("%Y-%m-%d")
                    
                    entities.append({
                        "text": match.group(0),
                        "label": "DATE",
                        "start": match.start(),
                        "end": match.end(),
                        "value": date_value
                    })
                except ValueError:
                    # Invalid date, skip
                    pass
        
        # Look for relative dates (today, yesterday, tomorrow, next week, etc.)
        relative_date_patterns = {
            r'\btoday\b': lambda: datetime.now().strftime("%Y-%m-%d"),
            r'\byesterday\b': lambda: (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            r'\btomorrow\b': lambda: (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            r'\bnext\s+week\b': lambda: (datetime.now() + timedelta(weeks=1)).strftime("%Y-%m-%d"),
            r'\blast\s+week\b': lambda: (datetime.now() - timedelta(weeks=1)).strftime("%Y-%m-%d"),
            r'\bnext\s+month\b': lambda: (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            r'\blast\s+month\b': lambda: (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        }
        
        for pattern, date_func in relative_date_patterns.items():
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                entities.append({
                    "text": match.group(0),
                    "label": "DATE",
                    "start": match.start(),
                    "end": match.end(),
                    "value": date_func()
                })
        
        return entities
    
    def _extract_times(self, message: str) -> List[Dict[str, Any]]:
        """Extract times from message."""
        entities = []
        
        # Look for time patterns (HH:MM, HH:MM:SS, etc.)
        time_patterns = [
            r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?',  # HH:MM(:SS) (AM/PM)
            r'(\d{1,2})\s*(am|pm)'  # HH AM/PM
        ]
        
        for pattern in time_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                try:
                    if ":" in match.group(0):
                        # HH:MM(:SS) format
                        hour = int(match.group(1))
                        minute = int(match.group(2))
                        second = int(match.group(3)) if match.group(3) else 0
                        
                        # Adjust for AM/PM
                        if match.group(4):
                            if match.group(4).lower() == "pm" and hour < 12:
                                hour += 12
                            elif match.group(4).lower() == "am" and hour == 12:
                                hour = 0
                    else:
                        # HH AM/PM format
                        hour = int(match.group(1))
                        minute = 0
                        second = 0
                        
                        # Adjust for AM/PM
                        if match.group(2).lower() == "pm" and hour < 12:
                            hour += 12
                        elif match.group(2).lower() == "am" and hour == 12:
                            hour = 0
                    
                    # Format time as HH:MM:SS
                    time_value = f"{hour:02d}:{minute:02d}:{second:02d}"
                    
                    entities.append({
                        "text": match.group(0),
                        "label": "TIME",
                        "start": match.start(),
                        "end": match.end(),
                        "value": time_value
                    })
                except (ValueError, IndexError):
                    # Invalid time, skip
                    pass
        
        return entities
    
    def _extract_percentages(self, message: str) -> List[Dict[str, Any]]:
        """Extract percentages from message."""
        entities = []
        
        # Look for percentage patterns
        percentage_patterns = [
            r'(\d+\.?\d*)%',  # 50% or 50.5%
            r'(\d+\.?\d*)\s+percent',  # 50 percent or 50.5 percent
            r'(\d+\.?\d*)\s+pct'  # 50 pct or 50.5 pct
        ]
        
        for pattern in percentage_patterns:
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                value = float(match.group(1))
                
                entities.append({
                    "text": match.group(0),
                    "label": "PERCENTAGE",
                    "start": match.start(),
                    "end": match.end(),
                    "value": value
                })
        
        return entities
    
    def _extract_trading_terms(self, message: str) -> List[Dict[str, Any]]:
        """Extract trading terms from message."""
        entities = []
        message_lower = message.lower()
        
        # Check for trading term aliases
        for term, aliases in self.trading_terms.items():
            for alias in aliases:
                if alias in message_lower:
                    start = message_lower.find(alias)
                    entities.append({
                        "text": alias,
                        "label": "TERM",
                        "start": start,
                        "end": start + len(alias),
                        "value": term
                    })
        
        return entities
    
    def _extract_chart_patterns(self, message: str) -> List[Dict[str, Any]]:
        """Extract chart patterns from message."""
        entities = []
        message_lower = message.lower()
        
        # Check for chart pattern aliases
        for pattern, aliases in self.chart_patterns.items():
            for alias in aliases:
                if alias in message_lower:
                    start = message_lower.find(alias)
                    entities.append({
                        "text": alias,
                        "label": "PATTERN",
                        "start": start,
                        "end": start + len(alias),
                        "value": pattern
                    })
        
        return entities
