"""
Update datetime module.

This module provides functionality for...
"""

import os
import re


from trading_gateway_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@with_exception_handling
def update_datetime_utcnow():
    """Update all occurrences of datetime.utcnow() to datetime.now(timezone.utc)."""
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    continue
                if 'datetime.utcnow()' in content:
                    print(f'Updating {file_path}')
                    if ('from datetime import datetime' in content and 
                        'timezone' not in content):
                        content = content.replace(
                            'from datetime import datetime',
                            'from datetime import datetime, timezone')
                    content = content.replace('datetime.utcnow()',
                        'datetime.now(timezone.utc)')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)


if __name__ == '__main__':
    update_datetime_utcnow()
