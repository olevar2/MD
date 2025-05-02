"""
Date utility functions for the forex trading platform.
"""
from datetime import datetime, timezone

def ensure_timezone(dt):
    """Ensures a datetime has timezone info, defaulting to UTC if naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt