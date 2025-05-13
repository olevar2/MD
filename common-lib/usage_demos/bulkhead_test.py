"""
Simplified bulkhead test to verify it works correctly.
"""

import asyncio
import sys
import os

# Add parent directory to path to include common_lib
common_lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, common_lib_dir)

# Explicitly add core-foundations directory to path
core_foundations_dir = os.path.abspath(os.path.join(common_lib_dir, '../core-foundations'))
sys.path.insert(0, core_foundations_dir)

print(f"Added to sys.path: {common_lib_dir}")
print(f"Added to sys.path: {core_foundations_dir}")

from common_lib.resilience.bulkhead import (
    bulkhead, BulkheadFullException
)

async def test_bulkhead_pattern():
    """Test that bulkhead limits concurrent executions and waiting queue."""
    print("\n=== Testing Bulkhead Pattern ===")
    
    execution_count = 0
    max_concurrent_observed = 0
    
    @bulkhead(name="test-bulkhead", max_concurrent=2, max_waiting=1)
    async def guarded_operation(id, duration):
    """
    Guarded operation.
    
    Args:
        id: Description of id
        duration: Description of duration
    
    """

        nonlocal execution_count, max_concurrent_observed
        execution_count += 1
        max_concurrent_observed = max(max_concurrent_observed, execution_count)
        print(f"Operation {id} started, current executions: {execution_count}")
        await asyncio.sleep(duration)
        execution_count -= 1
        print(f"Operation {id} completed")
        return f"operation {id} completed"
    
    # Start first two operations - should execute immediately
    print("Starting first two operations (should run immediately)")
    task1 = asyncio.create_task(guarded_operation(1, 0.3))
    task2 = asyncio.create_task(guarded_operation(2, 0.3))
    
    # Small delay to ensure the first two operations start
    await asyncio.sleep(0.1)
    
    # Start third operation - should wait in the queue
    print("Starting third operation (should wait in queue)")
    task3 = asyncio.create_task(guarded_operation(3, 0.1))
    
    # Small delay to ensure the third operation is queued
    await asyncio.sleep(0.05)
    
    # Start fourth operation - should be rejected (queue full)
    print("Starting fourth operation (should be rejected)")
    try:
        await guarded_operation(4, 0.1)
        print("ERROR: Fourth operation should have been rejected")
    except BulkheadFullException:
        print("Fourth operation rejected as expected (bulkhead full)")
    
    # Wait for all tasks to complete
    results = await asyncio.gather(task1, task2, task3, return_exceptions=True)
    print("All tasks completed with results:", results)
    
    # Check that max concurrent was limited to 2
    assert max_concurrent_observed == 2, f"Max concurrent should be 2, got {max_concurrent_observed}"
    
    print("Bulkhead pattern test PASSED")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_bulkhead_pattern())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error running test: {e}")
        sys.exit(1)
