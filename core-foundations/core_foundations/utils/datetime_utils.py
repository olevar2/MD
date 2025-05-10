"""
Datetime Utilities Module.

Provides standard datetime handling functions with market calendar awareness.
"""

import datetime
import functools
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pandas_market_calendars as mcal
import pytz


class MarketCalendarType(str, Enum):
    """Enum for market calendar types."""
    FOREX = "forex"
    NYSE = "nyse"
    LSE = "lse"
    JPX = "jpx"
    SSE = "sse"
    HKEX = "hkex"
    EUREX = "eurex"


def get_current_utc_time() -> datetime.datetime:
    """
    Get the current UTC time with timezone information.

    Returns:
        Current UTC datetime with timezone information
    """
    return datetime.datetime.now(datetime.timezone.utc)


def convert_to_utc(dt: datetime.datetime) -> datetime.datetime:
    """
    Convert a datetime to UTC.

    Args:
        dt: Input datetime (timezone-aware or naive)

    Returns:
        UTC datetime with timezone information

    Note:
        If input datetime is naive (no timezone), it's assumed to be in UTC
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def format_iso8601(dt: datetime.datetime) -> str:
    """
    Format datetime as ISO8601 string (UTC).

    Args:
        dt: Input datetime

    Returns:
        ISO8601 formatted string
    """
    utc_dt = convert_to_utc(dt)
    return utc_dt.isoformat().replace('+00:00', 'Z')


def parse_iso8601(dt_str: str) -> datetime.datetime:
    """
    Parse ISO8601 datetime string to datetime object.

    Args:
        dt_str: ISO8601 datetime string

    Returns:
        Datetime object with timezone information (UTC)

    Raises:
        ValueError: If the string cannot be parsed
    """
    try:
        # Try parsing with the Z suffix for UTC
        if dt_str.endswith('Z'):
            dt = datetime.datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        else:
            dt = datetime.datetime.fromisoformat(dt_str)
        return convert_to_utc(dt)
    except ValueError as e:
        raise ValueError(f"Could not parse ISO8601 datetime string: {dt_str}") from e


def get_market_calendar(calendar_type: MarketCalendarType) -> mcal.MarketCalendar:
    """
    Get market calendar by type.

    Args:
        calendar_type: Type of market calendar

    Returns:
        MarketCalendar instance
    """
    calendar_map = {
        MarketCalendarType.FOREX: "forex",
        MarketCalendarType.NYSE: "NYSE",
        MarketCalendarType.LSE: "LSE",
        MarketCalendarType.JPX: "JPX",
        MarketCalendarType.SSE: "SSE",
        MarketCalendarType.HKEX: "HKEX",
        MarketCalendarType.EUREX: "EUREX"
    }

    return mcal.get_calendar(calendar_map[calendar_type])


def is_market_open(calendar_type: MarketCalendarType, dt: Optional[datetime.datetime] = None) -> bool:
    """
    Check if a market is currently open.

    Args:
        calendar_type: Type of market calendar
        dt: Datetime to check (default: current UTC time)

    Returns:
        True if market is open, False otherwise
    """
    dt = dt or get_current_utc_time()
    utc_dt = convert_to_utc(dt)

    calendar = get_market_calendar(calendar_type)

    # Get schedule for the day
    schedule = calendar.schedule(
        start_date=utc_dt.date(),
        end_date=utc_dt.date()
    )

    # No trading day
    if len(schedule) == 0:
        return False

    # Check if current time is within market hours
    market_open = schedule.iloc[0]["market_open"].to_pydatetime()
    market_close = schedule.iloc[0]["market_close"].to_pydatetime()

    return market_open <= utc_dt <= market_close


def get_next_market_open(calendar_type: MarketCalendarType, dt: Optional[datetime.datetime] = None) -> datetime.datetime:
    """
    Get the next market open time.

    Args:
        calendar_type: Type of market calendar
        dt: Reference datetime (default: current UTC time)

    Returns:
        Next market open datetime
    """
    dt = dt or get_current_utc_time()
    utc_dt = convert_to_utc(dt)

    calendar = get_market_calendar(calendar_type)

    # Get schedule for next 7 days
    schedule = calendar.schedule(
        start_date=utc_dt.date(),
        end_date=utc_dt.date() + datetime.timedelta(days=7)
    )

    # Find next market open after the reference time
    for _, row in schedule.iterrows():
        market_open = row["market_open"].to_pydatetime()
        if market_open > utc_dt:
            return market_open

    # If no open found in next 7 days, look further
    schedule = calendar.schedule(
        start_date=utc_dt.date() + datetime.timedelta(days=8),
        end_date=utc_dt.date() + datetime.timedelta(days=30)
    )

    if len(schedule) > 0:
        return schedule.iloc[0]["market_open"].to_pydatetime()

    # Fallback
    return utc_dt + datetime.timedelta(days=1)


def get_timeframe_start_end(
    timeframe: str,
    reference_dt: Optional[datetime.datetime] = None
) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Calculate start and end times for a given timeframe.

    Args:
        timeframe: Timeframe code (e.g., "1d", "4h", "15m")
        reference_dt: Reference datetime (default: current UTC time)

    Returns:
        Tuple of (start_time, end_time) for the timeframe

    Raises:
        ValueError: If timeframe format is invalid
    """
    reference_dt = reference_dt or get_current_utc_time()
    utc_dt = convert_to_utc(reference_dt)

    # Parse timeframe
    if len(timeframe) < 2:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    try:
        value = int(timeframe[:-1])
        unit = timeframe[-1].lower()
    except ValueError:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    if unit == 'm':  # Minutes
        # Calculate current period start
        minutes = utc_dt.minute
        period_minute = (minutes // value) * value
        start_time = utc_dt.replace(minute=period_minute, second=0, microsecond=0)
        end_time = start_time + datetime.timedelta(minutes=value)

    elif unit == 'h':  # Hours
        # Calculate current period start
        start_time = utc_dt.replace(minute=0, second=0, microsecond=0)
        hour = start_time.hour
        period_hour = (hour // value) * value
        start_time = start_time.replace(hour=period_hour)
        end_time = start_time + datetime.timedelta(hours=value)

    elif unit == 'd':  # Days
        # Start at midnight UTC
        start_time = utc_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if value > 1:
            # For multi-day periods, calculate the period start
            days_since_epoch = (start_time.date() - datetime.date(1970, 1, 1)).days
            period_start_days = (days_since_epoch // value) * value
            period_start_date = datetime.date(1970, 1, 1) + datetime.timedelta(days=period_start_days)
            start_time = datetime.datetime.combine(
                period_start_date,
                datetime.time(0, 0),
                tzinfo=datetime.timezone.utc
            )
        end_time = start_time + datetime.timedelta(days=value)

    elif unit == 'w':  # Weeks
        # Week starts on Monday (weekday=0)
        current_weekday = utc_dt.weekday()
        days_to_subtract = current_weekday
        start_time = (utc_dt - datetime.timedelta(days=days_to_subtract)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if value > 1:
            # For multi-week periods, calculate the period start
            days_since_epoch = (start_time.date() - datetime.date(1970, 1, 1)).days
            days_since_epoch = days_since_epoch - (days_since_epoch % 7)  # Snap to week start
            period_start_days = (days_since_epoch // (7 * value)) * (7 * value)
            period_start_date = datetime.date(1970, 1, 1) + datetime.timedelta(days=period_start_days)
            start_time = datetime.datetime.combine(
                period_start_date,
                datetime.time(0, 0),
                tzinfo=datetime.timezone.utc
            )
        end_time = start_time + datetime.timedelta(weeks=value)

    elif unit == 'M':  # Months
        # Start at first day of month
        start_time = utc_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if value > 1:
            # For multi-month periods, calculate the period start
            month_since_epoch = (start_time.year - 1970) * 12 + start_time.month - 1
            period_start_month = (month_since_epoch // value) * value
            period_start_year = 1970 + (period_start_month // 12)
            period_start_month = (period_start_month % 12) + 1
            start_time = start_time.replace(year=period_start_year, month=period_start_month)

        # Calculate end time (adding months requires calendar awareness)
        month = start_time.month - 1 + value
        year = start_time.year + month // 12
        month = month % 12 + 1
        end_time = start_time.replace(year=year, month=month)

    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")

    return start_time, end_time


@functools.lru_cache(maxsize=128)
def align_time_to_timeframe(dt: datetime.datetime, timeframe: str) -> datetime.datetime:
    """
    Align a time to the start of its timeframe period with caching for performance.

    Args:
        dt: Input datetime
        timeframe: Timeframe code (e.g., "1d", "4h", "15m")

    Returns:
        Datetime aligned to timeframe start
    """
    # Parse timeframe for direct calculation without calling get_timeframe_start_end
    # This is an optimization to avoid the overhead of the full function call
    if len(timeframe) < 2:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    try:
        value = int(timeframe[:-1])
        unit = timeframe[-1].lower()
    except ValueError:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    # Ensure datetime is timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    # Fast path for common timeframes
    if unit == 'm':  # Minutes
        minutes = dt.minute
        period_minute = (minutes // value) * value
        return dt.replace(minute=period_minute, second=0, microsecond=0)

    elif unit == 'h':  # Hours
        start_time = dt.replace(minute=0, second=0, microsecond=0)
        hour = start_time.hour
        period_hour = (hour // value) * value
        return start_time.replace(hour=period_hour)

    elif unit == 'd':  # Days
        # Start at midnight UTC
        start_time = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if value > 1:
            # For multi-day periods, calculate the period start
            days_since_epoch = (start_time.date() - datetime.date(1970, 1, 1)).days
            period_start_days = (days_since_epoch // value) * value
            period_start_date = datetime.date(1970, 1, 1) + datetime.timedelta(days=period_start_days)
            return datetime.datetime.combine(
                period_start_date,
                datetime.time(0, 0),
                tzinfo=dt.tzinfo
            )
        return start_time

    # For less common timeframes, use the full function
    start_time, _ = get_timeframe_start_end(timeframe, dt)
    return start_time


def generate_timeframe_sequence(
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    timeframe: str
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Generate a sequence of timeframe periods between start and end times.

    Args:
        start_dt: Start datetime
        end_dt: End datetime
        timeframe: Timeframe code (e.g., "1d", "4h", "15m")

    Returns:
        List of (period_start, period_end) tuples
    """
    periods = []

    # Align start to timeframe
    current_start, current_end = get_timeframe_start_end(timeframe, start_dt)

    while current_start < end_dt:
        if current_end > start_dt:  # Only include periods that overlap with the requested range
            periods.append((current_start, current_end))

        # Move to next period
        current_start, current_end = get_timeframe_start_end(timeframe, current_end)

    return periods


def resample_to_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    timestamp_column: str = 'timestamp'
) -> pd.DataFrame:
    """
    Resample a dataframe to the specified timeframe.
    Assumes OHLCV structure with timestamp column.

    Args:
        df: Input dataframe with OHLCV data
        timeframe: Target timeframe code (e.g., "1d", "4h", "15m")
        timestamp_column: Name of the timestamp column

    Returns:
        Resampled dataframe

    Note:
        Expects columns: timestamp_column, 'open', 'high', 'low', 'close', 'volume'
    """
    # Convert timeframe to pandas offset alias
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])

    offset_map = {
        'm': f'{value}min',
        'h': f'{value}H',
        'd': f'{value}D',
        'w': f'{value}W',
        'M': f'{value}M'
    }

    if unit not in offset_map:
        raise ValueError(f"Unsupported timeframe unit: {unit}")

    offset = offset_map[unit]

    # Ensure timestamp is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_dtype(df[timestamp_column]):
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Set timestamp as index for resampling
    df = df.set_index(timestamp_column)

    # Define aggregation functions for OHLCV data
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Filter aggregation to only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    # Add any non-OHLCV columns with 'last' aggregation
    for col in df.columns:
        if col not in agg_dict:
            agg_dict[col] = 'last'

    # Resample and aggregate
    resampled = df.resample(offset).agg(agg_dict)

    # Reset index to convert timestamp back to column
    resampled = resampled.reset_index()

    return resampled