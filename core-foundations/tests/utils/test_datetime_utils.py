"""
Unit tests for datetime utilities with market calendars.
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest import mock

import pandas as pd
import pandas_market_calendars as mcal
import pytz

from core_foundations.utils.datetime_utils import (
    MarketCalendarType,
    align_time_to_timeframe,
    convert_to_utc,
    format_iso8601,
    generate_timeframe_sequence,
    get_current_utc_time,
    get_market_calendar,
    get_next_market_open,
    get_timeframe_start_end,
    is_market_open,
    parse_iso8601,
    resample_to_timeframe,
)


class TestDatetimeUtils(unittest.TestCase):
    """Tests for datetime utilities."""
    
    def test_get_current_utc_time(self):
        """Test getting current UTC time."""
        # Test that the returned time has timezone info
        dt = get_current_utc_time()
        self.assertEqual(dt.tzinfo, timezone.utc)
        
        # Test that the time is close to now
        now = datetime.now(timezone.utc)
        self.assertLess((now - dt).total_seconds(), 1.0)
    
    def test_convert_to_utc(self):
        """Test converting datetime to UTC."""
        # Test with timezone-aware datetime
        est = pytz.timezone("US/Eastern")
        dt_est = datetime(2023, 1, 1, 12, 0, 0, tzinfo=est)
        dt_utc = convert_to_utc(dt_est)
        
        self.assertEqual(dt_utc.tzinfo, timezone.utc)
        self.assertEqual(dt_utc.hour, 17)  # EST is UTC-5
        
        # Test with naive datetime (assumed to be UTC)
        dt_naive = datetime(2023, 1, 1, 12, 0, 0)
        dt_utc = convert_to_utc(dt_naive)
        
        self.assertEqual(dt_utc.tzinfo, timezone.utc)
        self.assertEqual(dt_utc.hour, 12)  # Time should be unchanged
    
    def test_format_iso8601(self):
        """Test ISO8601 formatting."""
        dt = datetime(2023, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
        iso_str = format_iso8601(dt)
        
        self.assertEqual(iso_str, "2023-01-01T12:30:45Z")
        
        # Test with non-UTC timezone
        est = pytz.timezone("US/Eastern")
        dt_est = datetime(2023, 1, 1, 7, 30, 45, tzinfo=est)
        iso_str = format_iso8601(dt_est)
        
        self.assertEqual(iso_str, "2023-01-01T12:30:45Z")
    
    def test_parse_iso8601(self):
        """Test parsing ISO8601 strings."""
        # Test with Z suffix
        dt = parse_iso8601("2023-01-01T12:30:45Z")
        
        self.assertEqual(dt.year, 2023)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.day, 1)
        self.assertEqual(dt.hour, 12)
        self.assertEqual(dt.minute, 30)
        self.assertEqual(dt.second, 45)
        self.assertEqual(dt.tzinfo, timezone.utc)
        
        # Test with explicit timezone offset
        dt = parse_iso8601("2023-01-01T07:30:45-05:00")
        
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(dt.hour, 12)  # Should be converted to UTC
        
        # Test with invalid string
        with self.assertRaises(ValueError):
            parse_iso8601("not-a-date")
    
    def test_get_market_calendar(self):
        """Test getting market calendar by type."""
        forex_cal = get_market_calendar(MarketCalendarType.FOREX)
        self.assertIsInstance(forex_cal, mcal.MarketCalendar)
        
        nyse_cal = get_market_calendar(MarketCalendarType.NYSE)
        self.assertIsInstance(nyse_cal, mcal.MarketCalendar)
    
    @mock.patch('pandas_market_calendars.get_calendar')
    def test_is_market_open(self, mock_get_calendar):
        """Test checking if a market is open."""
        # Mock the schedule returned by the calendar
        mock_calendar = mock.MagicMock()
        mock_get_calendar.return_value = mock_calendar
        
        # Case 1: Market is open
        mock_schedule = pd.DataFrame({
            'market_open': [pd.Timestamp('2023-01-01 09:00:00', tz='UTC')],
            'market_close': [pd.Timestamp('2023-01-01 17:00:00', tz='UTC')]
        })
        mock_calendar.schedule.return_value = mock_schedule
        
        test_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self.assertTrue(is_market_open(MarketCalendarType.FOREX, test_dt))
        
        # Case 2: Market is closed
        mock_schedule = pd.DataFrame({
            'market_open': [pd.Timestamp('2023-01-02 09:00:00', tz='UTC')],
            'market_close': [pd.Timestamp('2023-01-02 17:00:00', tz='UTC')]
        })
        mock_calendar.schedule.return_value = mock_schedule
        
        test_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self.assertFalse(is_market_open(MarketCalendarType.FOREX, test_dt))
        
        # Case 3: No trading day
        mock_calendar.schedule.return_value = pd.DataFrame()
        
        test_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self.assertFalse(is_market_open(MarketCalendarType.FOREX, test_dt))
    
    @mock.patch('pandas_market_calendars.get_calendar')
    def test_get_next_market_open(self, mock_get_calendar):
        """Test getting the next market open time."""
        # Mock the schedule returned by the calendar
        mock_calendar = mock.MagicMock()
        mock_get_calendar.return_value = mock_calendar
        
        # Setup mock schedule data
        current_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        next_open = current_dt + timedelta(hours=20)
        
        mock_schedule = pd.DataFrame({
            'market_open': [pd.Timestamp(next_open)],
            'market_close': [pd.Timestamp(next_open + timedelta(hours=8))]
        })
        mock_calendar.schedule.return_value = mock_schedule
        
        # Test getting next open
        result = get_next_market_open(MarketCalendarType.FOREX, current_dt)
        self.assertEqual(result, next_open)
        
        # Test when no schedule is returned (fallback)
        mock_calendar.schedule.return_value = pd.DataFrame()
        result = get_next_market_open(MarketCalendarType.FOREX, current_dt)
        self.assertEqual(result, current_dt + timedelta(days=1))
    
    def test_get_timeframe_start_end(self):
        """Test calculating timeframe start and end times."""
        # Test minute timeframe
        dt = datetime(2023, 1, 1, 14, 27, 35, tzinfo=timezone.utc)
        start, end = get_timeframe_start_end("15m", dt)
        
        self.assertEqual(start, datetime(2023, 1, 1, 14, 15, 0, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2023, 1, 1, 14, 30, 0, tzinfo=timezone.utc))
        
        # Test hourly timeframe
        start, end = get_timeframe_start_end("4h", dt)
        
        self.assertEqual(start, datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2023, 1, 1, 16, 0, 0, tzinfo=timezone.utc))
        
        # Test daily timeframe
        start, end = get_timeframe_start_end("1d", dt)
        
        self.assertEqual(start, datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc))
        
        # Test weekly timeframe
        dt = datetime(2023, 1, 4, 14, 27, 35, tzinfo=timezone.utc)  # Wednesday
        start, end = get_timeframe_start_end("1w", dt)
        
        self.assertEqual(start, datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc))  # Monday
        self.assertEqual(end, datetime(2023, 1, 9, 0, 0, 0, tzinfo=timezone.utc))  # Next Monday
        
        # Test invalid timeframe
        with self.assertRaises(ValueError):
            get_timeframe_start_end("invalid", dt)
    
    def test_align_time_to_timeframe(self):
        """Test aligning time to timeframe start."""
        dt = datetime(2023, 1, 1, 14, 27, 35, tzinfo=timezone.utc)
        aligned = align_time_to_timeframe(dt, "15m")
        
        self.assertEqual(aligned, datetime(2023, 1, 1, 14, 15, 0, tzinfo=timezone.utc))
    
    def test_generate_timeframe_sequence(self):
        """Test generating a sequence of timeframe periods."""
        start = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2023, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
        
        # Test 1-hour timeframes
        periods = generate_timeframe_sequence(start, end, "1h")
        
        self.assertEqual(len(periods), 5)
        self.assertEqual(periods[0][0], datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(periods[0][1], datetime(2023, 1, 1, 11, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(periods[-1][0], datetime(2023, 1, 1, 14, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(periods[-1][1], datetime(2023, 1, 1, 15, 0, 0, tzinfo=timezone.utc))
    
    def test_resample_to_timeframe(self):
        """Test resampling OHLCV data to different timeframes."""
        # Create test data (1-minute bars)
        data = []
        base_time = pd.Timestamp('2023-01-01 10:00:00', tz='UTC')
        
        for i in range(120):  # 2 hours of 1-minute data
            data.append({
                'timestamp': base_time + pd.Timedelta(minutes=i),
                'open': 100 + i * 0.1,
                'high': 100 + i * 0.1 + 0.05,
                'low': 100 + i * 0.1 - 0.05,
                'close': 100 + i * 0.1 + 0.02,
                'volume': 1000 + i
            })
        
        df = pd.DataFrame(data)
        
        # Resample to 15-minute bars
        resampled = resample_to_timeframe(df, "15m")
        
        # Check result
        self.assertEqual(len(resampled), 8)  # 120 minutes / 15 = 8 bars
        self.assertEqual(resampled.iloc[0]['open'], data[0]['open'])
        self.assertGreaterEqual(resampled.iloc[0]['high'], data[0]['high'])
        self.assertLessEqual(resampled.iloc[0]['low'], data[0]['low'])
        self.assertEqual(resampled.iloc[0]['close'], data[14]['close'])
        
        # Sum of volumes for first 15 minutes
        expected_volume = sum(d['volume'] for d in data[:15])
        self.assertEqual(resampled.iloc[0]['volume'], expected_volume)


if __name__ == "__main__":
    unittest.main()