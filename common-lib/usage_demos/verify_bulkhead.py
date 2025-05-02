"""
Very basic verification of Bulkhead functionality.
"""

import os
import sys

# Add parent directory to path to include common_lib
common_lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, common_lib_dir)

# Explicitly add core-foundations directory to path
core_foundations_dir = os.path.abspath(os.path.join(common_lib_dir, '../core-foundations'))
sys.path.insert(0, core_foundations_dir)

print(f"Added to sys.path: {common_lib_dir}")
print(f"Added to sys.path: {core_foundations_dir}")

# Try to import directly from the bulkhead module
print("Attempting to import Bulkhead class...")
try:
    from common_lib.resilience.bulkhead import Bulkhead
    print("✅ Bulkhead class imported successfully")
    
    # Try to create an instance
    bulkhead = Bulkhead(name="test", max_concurrent=5)
    print(f"✅ Bulkhead instance created with name: {bulkhead.name}")
    print(f"✅ Max concurrent: {bulkhead.max_concurrent}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
    # Try to inspect the module
    print("\nTrying to inspect module...")
    try:
        import common_lib.resilience.bulkhead as bh_mod
        print(f"Module attributes: {dir(bh_mod)}")
        print(f"Module __all__: {getattr(bh_mod, '__all__', 'Not defined')}")
    except Exception as e2:
        print(f"❌ Module inspection error: {e2}")
except Exception as e:
    print(f"❌ Other error: {e}")
