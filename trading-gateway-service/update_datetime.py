import os
import re

def update_datetime_utcnow():
    """Update all occurrences of datetime.utcnow() to datetime.now(timezone.utc)."""
    # First, add the timezone import to all files that use datetime
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    continue
                
                # Check if the file uses datetime.utcnow()
                if 'datetime.utcnow()' in content:
                    print(f"Updating {file_path}")
                    
                    # Add timezone import if not already present
                    if 'from datetime import datetime' in content and 'timezone' not in content:
                        content = content.replace('from datetime import datetime', 'from datetime import datetime, timezone')
                    
                    # Replace all occurrences of datetime.utcnow()
                    content = content.replace('datetime.utcnow()', 'datetime.now(timezone.utc)')
                    
                    # Write the updated content back to the file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)

if __name__ == "__main__":
    update_datetime_utcnow()