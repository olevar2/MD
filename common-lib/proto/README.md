# Protocol Buffer Definitions for Forex Trading Platform

This directory contains Protocol Buffer definitions for the Forex Trading Platform's gRPC services.

## Directory Structure

```
proto/
  ├── common/                    # Common message types used across services
  │   ├── common_types.proto     # Common data types (Symbol, Timeframe, etc.)
  │   └── error_types.proto      # Error response types
  ├── causal_analysis/           # Causal Analysis Service definitions
  │   └── causal_analysis_service.proto
  ├── backtesting/               # Backtesting Service definitions
  │   └── backtesting_service.proto
  ├── market_analysis/           # Market Analysis Service definitions
  │   └── market_analysis_service.proto
  ├── analysis_coordinator/      # Analysis Coordinator Service definitions
  │   └── analysis_coordinator_service.proto
  ├── compile_protos.py          # Script to compile .proto files
  ├── Makefile                   # Makefile for common tasks
  └── README.md                  # This file
```

## Compiling Protocol Buffer Files

To compile the Protocol Buffer files, you can use the provided Makefile:

```bash
# Install dependencies and compile all .proto files
make

# Only compile .proto files (if dependencies are already installed)
make compile-protos

# Clean generated files
make clean
```

Alternatively, you can run the compilation script directly:

```bash
# Install dependencies
pip install grpcio grpcio-tools

# Compile .proto files
python compile_protos.py
```

## Generated Code

The compiled Python code will be generated in the `common_lib/grpc` directory, with the following structure:

```
common_lib/grpc/
  ├── common/
  ├── causal_analysis/
  ├── backtesting/
  ├── market_analysis/
  ├── analysis_coordinator/
  └── __init__.py
```

## Usage

To use the generated gRPC code in your Python code:

```python
# Import service stubs
from common_lib.grpc.causal_analysis import causal_analysis_service_pb2, causal_analysis_service_pb2_grpc
from common_lib.grpc.backtesting import backtesting_service_pb2, backtesting_service_pb2_grpc
from common_lib.grpc.market_analysis import market_analysis_service_pb2, market_analysis_service_pb2_grpc
from common_lib.grpc.analysis_coordinator import analysis_coordinator_service_pb2, analysis_coordinator_service_pb2_grpc

# Import common types
from common_lib.grpc.common import common_types_pb2, error_types_pb2
```

## Design Principles

The Protocol Buffer definitions follow these design principles:

1. **Clear and Concise**: Message names are clear, concise, and self-descriptive.
2. **Future Evolution**: Designed for future evolution with reserved fields and avoiding field renaming.
3. **Appropriate Data Types**: Using appropriate Protobuf data types for each field.
4. **Documentation**: All messages and fields are documented with comments.
5. **Consistency**: Consistent naming and structure across all services.
6. **Error Handling**: Standardized error responses across all services.
7. **Versioning**: Support for versioning through package naming and field numbering.

## Adding New Services

To add a new service:

1. Create a new directory for the service in the `proto` directory.
2. Create a `.proto` file for the service with the appropriate package name and imports.
3. Define the service interface and message types.
4. Update the `compile_protos.py` script to include the new service.
5. Compile the Protocol Buffer files using the Makefile or script.