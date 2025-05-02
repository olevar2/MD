"""
Script to inspect the bulkhead module structure.
"""

import os
import sys
import inspect

# Add parent directory to path to include common_lib
common_lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, common_lib_dir)

# Explicitly add core-foundations directory to path
core_foundations_dir = os.path.abspath(os.path.join(common_lib_dir, '../core-foundations'))
sys.path.insert(0, core_foundations_dir)

print(f"Added to sys.path: {common_lib_dir}")
print(f"Added to sys.path: {core_foundations_dir}")

try:
    # Import the module but not any specific objects from it
    import common_lib.resilience.bulkhead as bh_mod
    
    # Print module attributes
    print("\nModule attributes:")
    module_attrs = dir(bh_mod)
    for attr in module_attrs:
        print(f"- {attr}")
        
    # Check if Bulkhead is defined in the module
    if 'Bulkhead' in module_attrs:
        print("\n'Bulkhead' is defined in the module")
        
        # Inspect the Bulkhead class
        bulkhead_class = getattr(bh_mod, 'Bulkhead')
        print(f"Type: {type(bulkhead_class)}")
        
        if inspect.isclass(bulkhead_class):
            print("'Bulkhead' is a class")
            print(f"Methods: {[m for m in dir(bulkhead_class) if not m.startswith('_')]}")
        else:
            print("'Bulkhead' is not a class")
    else:
        print("\n'Bulkhead' is NOT defined in the module")
        
    # Print the module's __all__ attribute
    if hasattr(bh_mod, '__all__'):
        print(f"\nModule __all__: {bh_mod.__all__}")
    else:
        print("\nModule does not have __all__ attribute")
    
except Exception as e:
    print(f"Error inspecting module: {e}")

# Also try to import directly to see error
try:
    # This is the import that's failing in other scripts
    from common_lib.resilience.bulkhead import Bulkhead
    print("\nSuccessfully imported Bulkhead directly")
except Exception as e:
    print(f"\nError importing Bulkhead directly: {e}")
