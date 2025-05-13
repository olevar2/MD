"""
Trading Session Manager for handling market sessions, trading hours, and time-based constraints.

This component manages trading hours, identifies market sessions (Asian, European, US),
and provides functionality to determine if trading is allowed at the current time.
"""
import logging
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
import pytz
from core_foundations.utils.logger import get_logger


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MarketSession(Enum):
    """Different market sessions in forex trading."""
    ASIAN = 'asian'
    EUROPEAN = 'european'
    NORTH_AMERICAN = 'north_american'
    OVERLAPPING_EUROPEAN_US = 'overlapping_european_us'
    OVERLAPPING_ASIAN_EUROPEAN = 'overlapping_asian_european'
    WEEKEND = 'weekend'
    OFF_HOURS = 'off_hours'


class TradingSessionManager:
    """
    Manages trading sessions, hours, and time-based constraints for forex trading.
    
    This component keeps track of trading hours, market sessions, and provides
    functionality to determine if trading is currently allowed based on configured
    rules and preferences.
    """

    def __init__(self, config: Optional[Dict[str, Any]]=None):
        """
        Initialize the TradingSessionManager.
        
        Args:
            config: Configuration for trading hours and sessions
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or {}
        self.timezone = pytz.timezone(self.config_manager.get('timezone', 'UTC'))
        self.session_times = self.config.get('session_times', {
            MarketSession.ASIAN: {'start': time(22, 0), 'end': time(7, 0)},
            MarketSession.EUROPEAN: {'start': time(7, 0), 'end': time(16, 0
            )}, MarketSession.NORTH_AMERICAN: {'start': time(12, 0), 'end':
            time(21, 0)}, MarketSession.OVERLAPPING_EUROPEAN_US: {'start':
            time(12, 0), 'end': time(16, 0)}, MarketSession.
            OVERLAPPING_ASIAN_EUROPEAN: {'start': time(7, 0), 'end': time(8,
            0)}})
        self.trading_days = self.config_manager.get('trading_days', [0, 1, 2, 3, 4])
        self.allow_off_hours_trading = self.config.get(
            'allow_off_hours_trading', True)
        self.close_positions_at_session_end = self.config.get(
            'close_positions_at_session_end', False)
        self.close_positions_at_week_end = self.config.get(
            'close_positions_at_week_end', True)
        self.restrict_trading_before_high_impact_news = self.config.get(
            'restrict_trading_before_high_impact_news', True)
        self.high_impact_news_restriction_minutes = self.config.get(
            'high_impact_news_restriction_minutes', 15)
        self.high_impact_news_events = self.config.get(
            'high_impact_news_events', [])
        self._current_session = None
        self._session_end_time = None
        self._should_close_positions = False
        self._is_running = False
        self.logger.info('TradingSessionManager initialized')

    async def start(self) ->None:
        """Start the trading session manager."""
        if self._is_running:
            self.logger.warning('Trading session manager is already running')
            return
        self._is_running = True
        self._update_current_session()
        self.logger.info(
            f'Trading session manager started. Current session: {self._current_session}'
            )

    async def stop(self) ->None:
        """Stop the trading session manager."""
        if not self._is_running:
            return
        self._is_running = False
        self.logger.info('Trading session manager stopped')

    def is_trading_allowed(self, instrument: Optional[str]=None) ->bool:
        """
        Determine if trading is allowed at the current time for the given instrument.
        
        Args:
            instrument: Optional instrument symbol to check specific restrictions
            
        Returns:
            True if trading is allowed, False otherwise
        """
        if not self._is_running:
            self._update_current_session()
        current_time = datetime.now(self.timezone)
        if current_time.weekday() not in self.trading_days:
            return False
        if (self._current_session == MarketSession.WEEKEND or self.
            _current_session == MarketSession.OFF_HOURS):
            return self.allow_off_hours_trading
        if (self.restrict_trading_before_high_impact_news and self.
            _is_near_high_impact_news(instrument)):
            return False
        if instrument and not self._is_instrument_allowed_in_session(instrument
            , self._current_session):
            return False
        return True

    def get_current_session_info(self) ->Dict[str, Any]:
        """
        Get information about the current trading session.
        
        Returns:
            Dictionary with session information
        """
        if not self._is_running:
            self._update_current_session()
        current_time = datetime.now(self.timezone)
        return {'session': self._current_session.value if self.
            _current_session else None, 'current_time': current_time,
            'session_end_time': self._session_end_time,
            'time_until_session_end': self._session_end_time - current_time if
            self._session_end_time else None, 'trading_allowed': self.
            is_trading_allowed(), 'should_close_positions': self.
            _should_close_positions, 'day_of_week': current_time.strftime(
            '%A'), 'is_trading_day': current_time.weekday() in self.
            trading_days}

    def should_close_positions(self) ->bool:
        """
        Determine if positions should be closed based on session rules.
        
        Returns:
            True if positions should be closed, False otherwise
        """
        if not self._is_running:
            self._update_current_session()
        return self._should_close_positions

    def _update_current_session(self) ->None:
        """Update the current session based on current time."""
        current_time = datetime.now(self.timezone)
        current_weekday = current_time.weekday()
        current_time_obj = current_time.time()
        self._should_close_positions = False
        if current_weekday not in self.trading_days:
            self._current_session = MarketSession.WEEKEND
            days_until_next_session = min((day - current_weekday) % 7 for
                day in self.trading_days)
            next_trading_day = current_time + timedelta(days=
                days_until_next_session)
            next_trading_day = next_trading_day.replace(hour=0, minute=0,
                second=0, microsecond=0)
            first_session_start = None
            for session, times in self.session_times.items():
                session_start = datetime.combine(next_trading_day.date(),
                    times['start'])
                session_start = self.timezone.localize(session_start)
                if (first_session_start is None or session_start <
                    first_session_start):
                    first_session_start = session_start
            self._session_end_time = first_session_start
            return
        found_session = False
        self._session_end_time = None
        if self._is_time_in_session(current_time_obj, MarketSession.
            OVERLAPPING_EUROPEAN_US):
            self._current_session = MarketSession.OVERLAPPING_EUROPEAN_US
            self._session_end_time = self._get_session_end_time(current_time,
                MarketSession.OVERLAPPING_EUROPEAN_US)
            found_session = True
        elif self._is_time_in_session(current_time_obj, MarketSession.
            OVERLAPPING_ASIAN_EUROPEAN):
            self._current_session = MarketSession.OVERLAPPING_ASIAN_EUROPEAN
            self._session_end_time = self._get_session_end_time(current_time,
                MarketSession.OVERLAPPING_ASIAN_EUROPEAN)
            found_session = True
        if not found_session:
            for session in [MarketSession.ASIAN, MarketSession.EUROPEAN,
                MarketSession.NORTH_AMERICAN]:
                if self._is_time_in_session(current_time_obj, session):
                    self._current_session = session
                    self._session_end_time = self._get_session_end_time(
                        current_time, session)
                    found_session = True
                    break
        if not found_session:
            self._current_session = MarketSession.OFF_HOURS
            next_session_start = None
            next_session = None
            for session, times in self.session_times.items():
                if session in [MarketSession.OVERLAPPING_EUROPEAN_US,
                    MarketSession.OVERLAPPING_ASIAN_EUROPEAN]:
                    continue
                session_start = times['start']
                if session_start > current_time_obj:
                    session_start_dt = datetime.combine(current_time.date(),
                        session_start)
                    session_start_dt = self.timezone.localize(session_start_dt)
                    if (next_session_start is None or session_start_dt <
                        next_session_start):
                        next_session_start = session_start_dt
                        next_session = session
            if next_session_start is None:
                next_day = current_time + timedelta(days=1)
                next_weekday = next_day.weekday()
                if next_weekday in self.trading_days:
                    earliest_session = None
                    earliest_time = None
                    for session, times in self.session_times.items():
                        if session in [MarketSession.
                            OVERLAPPING_EUROPEAN_US, MarketSession.
                            OVERLAPPING_ASIAN_EUROPEAN]:
                            continue
                        session_start = times['start']
                        if (earliest_time is None or session_start <
                            earliest_time):
                            earliest_time = session_start
                            earliest_session = session
                    if earliest_session:
                        next_session = earliest_session
                        next_session_start = datetime.combine(next_day.date
                            (), earliest_time)
                        next_session_start = self.timezone.localize(
                            next_session_start)
                else:
                    days_until_next_session = min((day - next_weekday) % 7 for
                        day in self.trading_days)
                    next_trading_day = next_day + timedelta(days=
                        days_until_next_session)
                    earliest_session = None
                    earliest_time = None
                    for session, times in self.session_times.items():
                        if session in [MarketSession.
                            OVERLAPPING_EUROPEAN_US, MarketSession.
                            OVERLAPPING_ASIAN_EUROPEAN]:
                            continue
                        session_start = times['start']
                        if (earliest_time is None or session_start <
                            earliest_time):
                            earliest_time = session_start
                            earliest_session = session
                    if earliest_session:
                        next_session = earliest_session
                        next_session_start = datetime.combine(next_trading_day
                            .date(), earliest_time)
                        next_session_start = self.timezone.localize(
                            next_session_start)
            self._session_end_time = next_session_start
        if (self.close_positions_at_week_end and current_weekday == self.
            trading_days[-1]):
            last_session = MarketSession.NORTH_AMERICAN
            last_session_end = self.session_times[last_session]['end']
            if (current_time_obj >= last_session_end or last_session_end >
                current_time_obj and (datetime.combine(current_time.date(),
                last_session_end) - current_time).total_seconds() / 60 < 30):
                self._should_close_positions = True
        elif self.close_positions_at_session_end and self._session_end_time:
            minutes_to_end = (self._session_end_time - current_time
                ).total_seconds() / 60
            if minutes_to_end < 10:
                self._should_close_positions = True

    def _is_time_in_session(self, time_obj: time, session: MarketSession
        ) ->bool:
        """
        Check if a given time falls within a specific market session.
        
        Args:
            time_obj: Time object to check
            session: Market session to check against
            
        Returns:
            True if time is in session, False otherwise
        """
        if session not in self.session_times:
            return False
        start_time = self.session_times[session]['start']
        end_time = self.session_times[session]['end']
        if start_time < end_time:
            return start_time <= time_obj < end_time
        else:
            return time_obj >= start_time or time_obj < end_time

    def _get_session_end_time(self, current_dt: datetime, session:
        MarketSession) ->datetime:
        """
        Get the end time of a session as a datetime object.
        
        Args:
            current_dt: Current datetime
            session: Market session
            
        Returns:
            Datetime representing the end of the session
        """
        if session not in self.session_times:
            return None
        session_end = self.session_times[session]['end']
        session_start = self.session_times[session]['start']
        if session_start > session_end and current_dt.time() >= session_start:
            next_day = current_dt.date() + timedelta(days=1)
            end_datetime = datetime.combine(next_day, session_end)
        else:
            end_datetime = datetime.combine(current_dt.date(), session_end)
        return self.timezone.localize(end_datetime)

    @with_exception_handling
    def _is_near_high_impact_news(self, instrument: Optional[str]=None) ->bool:
        """
        Check if current time is near high-impact news event.
        
        Args:
            instrument: Optional instrument to check specific news
            
        Returns:
            True if near high-impact news, False otherwise
        """
        if not self.high_impact_news_events:
            return False
        current_time = datetime.now(self.timezone)
        restriction_window = timedelta(minutes=self.
            high_impact_news_restriction_minutes)
        for event in self.high_impact_news_events:
            event_time = event.get('time')
            if not event_time:
                continue
            if isinstance(event_time, str):
                try:
                    event_time = datetime.fromisoformat(event_time)
                    if event_time.tzinfo is None:
                        event_time = self.timezone.localize(event_time)
                except ValueError:
                    continue
            time_diff = abs((event_time - current_time).total_seconds())
            if time_diff <= restriction_window.total_seconds():
                if instrument:
                    event_currencies = event.get('currencies', [])
                    if any(currency in instrument for currency in
                        event_currencies):
                        return True
                else:
                    return True
        return False

    def _is_instrument_allowed_in_session(self, instrument: str, session:
        MarketSession) ->bool:
        """
        Check if trading a specific instrument is allowed in the given session.
        
        Args:
            instrument: Instrument symbol
            session: Market session
            
        Returns:
            True if instrument trading is allowed, False otherwise
        """
        if 'instrument_session_restrictions' not in self.config:
            return True
        restrictions = self.config_manager.get('instrument_session_restrictions', {})
        if instrument in restrictions:
            allowed_sessions = restrictions[instrument]
            return session.value in allowed_sessions
        for currency_pair, allowed_sessions in restrictions.items():
            if (currency_pair in instrument and session.value not in
                allowed_sessions):
                return False
        return True
