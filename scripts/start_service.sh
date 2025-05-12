#!/bin/bash
# Service Startup Script for Forex Trading Platform
# This script is a wrapper for start_service.py that handles environment setup and error handling

# Set strict error handling
set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Error function
error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

# Warning function
warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Success function
success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed. Please install Python 3 and try again."
        exit 1
    fi
    
    log "Python 3 is installed: $(python3 --version)"
}

# Check if pip is installed
check_pip() {
    if ! command -v pip3 &> /dev/null; then
        error "pip3 is not installed. Please install pip3 and try again."
        exit 1
    fi
    
    log "pip3 is installed: $(pip3 --version)"
}

# Install required Python packages
install_packages() {
    log "Installing required Python packages..."
    pip3 install requests
    success "Required Python packages installed successfully"
}

# Run the service startup script
run_startup_script() {
    log "Starting the service..."
    
    # Parse command line arguments
    SERVICE=""
    ENV_FILE=""
    PORT=""
    HOST=""
    TIMEOUT=30
    SKIP_DEPS=false
    SKIP_HEALTH_CHECK=false
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --service)
                SERVICE="$2"
                shift 2
                ;;
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --host)
                HOST="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --skip-health-check)
                SKIP_HEALTH_CHECK=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check if service is specified
    if [ -z "$SERVICE" ]; then
        error "Service not specified. Please specify a service with --service."
        exit 1
    fi
    
    # Build command
    CMD="python3 $SCRIPT_DIR/start_service.py --service $SERVICE --timeout $TIMEOUT"
    
    if [ -n "$ENV_FILE" ]; then
        CMD="$CMD --env-file $ENV_FILE"
    fi
    
    if [ -n "$PORT" ]; then
        CMD="$CMD --port $PORT"
    fi
    
    if [ -n "$HOST" ]; then
        CMD="$CMD --host $HOST"
    fi
    
    if [ "$SKIP_DEPS" = true ]; then
        CMD="$CMD --skip-deps"
    fi
    
    if [ "$SKIP_HEALTH_CHECK" = true ]; then
        CMD="$CMD --skip-health-check"
    fi
    
    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi
    
    # Run the script
    log "Executing: $CMD"
    eval "$CMD"
    
    # Check exit code
    if [ $? -eq 0 ]; then
        success "Service $SERVICE started successfully"
    else
        error "Failed to start service $SERVICE"
        exit 1
    fi
}

# Main function
main() {
    log "Starting service..."
    
    # Check requirements
    check_python
    check_pip
    
    # Install required packages
    install_packages
    
    # Run startup script
    run_startup_script "$@"
    
    success "Service startup process completed"
}

# Run main function with all arguments
main "$@"
