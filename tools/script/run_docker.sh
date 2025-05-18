#!/bin/bash
# Script to run individual services in Docker for isolated testing

# Root directory of the forex trading platform
ROOT_DIR="D:/MD/forex_trading_platform"

# Available services
AVAILABLE_SERVICES=(
    "causal-analysis-service"
    "backtesting-service"
    "market-analysis-service"
    "analysis-coordinator-service"
    "data-management-service"
    "monitoring-alerting-service"
    "strategy-execution-engine"
)

# Function to print usage
print_usage() {
    echo "Usage: $0 --service SERVICE_NAME [--build] [--stop] [--logs] [--detach]"
    echo ""
    echo "Options:"
    echo "  --service SERVICE_NAME  Service to run in Docker"
    echo "  --build                 Build the Docker image before running"
    echo "  --stop                  Stop the running service"
    echo "  --logs                  View logs for the service"
    echo "  --detach                Run in detached mode"
    echo ""
    echo "Available services:"
    for service in "${AVAILABLE_SERVICES[@]}"; do
        echo "  - $service"
    done
    exit 1
}

# Function to check if service is valid
is_valid_service() {
    local service=$1
    for valid_service in "${AVAILABLE_SERVICES[@]}"; do
        if [ "$service" == "$valid_service" ]; then
            return 0
        fi
    done
    return 1
}

# Function to create docker-compose.yml if it doesn't exist
create_docker_compose() {
    local service_dir=$1
    local service_name=$2
    
    if [ -f "$service_dir/docker-compose.yml" ]; then
        echo "Using existing docker-compose.yml in $service_name"
        return
    fi
    
    if [ ! -f "$service_dir/Dockerfile" ]; then
        echo "Error: Dockerfile not found in $service_name"
        exit 1
    fi
    
    echo "Creating docker-compose.yml for $service_name"
    
    # Determine port based on service
    local port=8000
    case "$service_name" in
        "causal-analysis-service") port=8000 ;;
        "backtesting-service") port=8002 ;;
        "market-analysis-service") port=8001 ;;
        "analysis-coordinator-service") port=8003 ;;
        "data-management-service") port=8004 ;;
        "monitoring-alerting-service") port=8005 ;;
        "strategy-execution-engine") port=8006 ;;
    esac
    
    cat > "$service_dir/docker-compose.yml" << EOFINNER
version: '3.8'

services:
  $service_name:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "$port:$port"
    environment:
      - DEBUG_MODE=true
      - LOG_LEVEL=DEBUG
      - HOST=0.0.0.0
      - PORT=$port
    volumes:
      - ./:/app
    networks:
      - forex-platform-network

networks:
  forex-platform-network:
    driver: bridge
EOFINNER
    
    echo "Created docker-compose.yml for $service_name"
}

# Parse command line arguments
SERVICE=""
BUILD=false
STOP=false
LOGS=false
DETACH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --service)
            SERVICE="$2"
            shift 2
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --stop)
            STOP=true
            shift
            ;;
        --logs)
            LOGS=true
            shift
            ;;
        --detach)
            DETACH=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Check if service is provided
if [ -z "$SERVICE" ]; then
    echo "Error: Service name is required"
    print_usage
fi

# Check if service is valid
if ! is_valid_service "$SERVICE"; then
    echo "Error: Invalid service name: $SERVICE"
    print_usage
fi

# Get service directory
SERVICE_DIR="$ROOT_DIR/$SERVICE"
if [ ! -d "$SERVICE_DIR" ]; then
    echo "Error: Service directory not found: $SERVICE_DIR"
    exit 1
fi

# Change to service directory
cd "$SERVICE_DIR" || exit 1

# Create docker-compose.yml if it doesn't exist
create_docker_compose "$SERVICE_DIR" "$SERVICE"

# Stop the service
if [ "$STOP" = true ]; then
    echo "Stopping $SERVICE"
    docker-compose down
    exit 0
fi

# Show logs
if [ "$LOGS" = true ]; then
    echo "Showing logs for $SERVICE"
    docker-compose logs -f
    exit 0
fi

# Build the Docker image
if [ "$BUILD" = true ]; then
    echo "Building Docker image for $SERVICE"
    docker-compose build
fi

# Run the service
echo "Running $SERVICE"
if [ "$DETACH" = true ]; then
    docker-compose up -d
else
    docker-compose up
fi
