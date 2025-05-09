import sys
import os
import unittest

# Add the feature-store-service directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the adapter interfaces from common-lib
from common_lib.indicators.fibonacci_interfaces import (
    IFibonacciRetracement,
    IFibonacciExtension,
    IFibonacciFan,
    IFibonacciTimeZones,
    IFibonacciCircles,
    IFibonacciClusters
)

# Import the adapter implementations from feature-store-service
from feature_store_service.adapters import (
    FibonacciRetracementAdapter,
    FibonacciExtensionAdapter,
    FibonacciFanAdapter,
    FibonacciTimeZonesAdapter,
    FibonacciCirclesAdapter,
    FibonacciClustersAdapter
)

# Import the test suite factory from tests
from tests import create_fibonacci_test_suite

if __name__ == '__main__':
    # Create adapter instances
    retracement_adapter = FibonacciRetracementAdapter()
    extension_adapter = FibonacciExtensionAdapter()
    fan_adapter = FibonacciFanAdapter()
    time_zones_adapter = FibonacciTimeZonesAdapter()
    circles_adapter = FibonacciCirclesAdapter()
    clusters_adapter = FibonacciClustersAdapter()

    # Create a test suite using the adapters
    suite = create_fibonacci_test_suite(
        retracement_adapter=retracement_adapter,
        extension_adapter=extension_adapter,
        fan_adapter=fan_adapter,
        time_zones_adapter=time_zones_adapter,
        circles_adapter=circles_adapter,
        clusters_adapter=clusters_adapter
    )

    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())