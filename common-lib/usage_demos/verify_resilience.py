"""
Basic verification script for resilience module components.
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

# Verify imports - Circuit Breaker
print("Testing CircuitBreaker imports...")
from common_lib.resilience.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, create_circuit_breaker, CircuitBreakerOpen
)
print("√ CircuitBreaker imports successful")

# Verify imports - Retry Policy
print("\nTesting RetryPolicy imports...")
from common_lib.resilience.retry_policy import (
    retry_with_policy, RetryExhaustedException
)
print("√ RetryPolicy imports successful")

# Verify imports - Timeout Handler
print("\nTesting TimeoutHandler imports...")
from common_lib.resilience.timeout_handler import (
    timeout_handler, TimeoutError
)
print("√ TimeoutHandler imports successful")

# Verify imports - Bulkhead
print("\nTesting Bulkhead imports...")
from common_lib.resilience.bulkhead import (
    bulkhead, BulkheadFullException, Bulkhead
)
print("√ Bulkhead imports successful")

# Verify basic instantiation
print("\nVerifying basic instantiation...")

# Circuit Breaker
cb = create_circuit_breaker(
    service_name="test-service",
    resource_name="test-resource"
)
print(f"√ CircuitBreaker created, state: {cb.state.name}")

# Bulkhead
bh = Bulkhead(
    name="test-bulkhead",
    max_concurrent=5
)
print(f"√ Bulkhead created, name: {bh.name}, max_concurrent: {bh.max_concurrent}")

print("\n✅ All basic verification tests PASSED")
