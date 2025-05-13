#!/bin/bash
# Platform Health Check Script for Forex Trading Platform
# This script is a wrapper for check_platform_health.py that handles environment setup and error handling

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

# Run the platform health check script
run_health_check_script() {
    log "Checking Forex Trading Platform health..."
    
    # Parse command line arguments
    TIMEOUT=5
    SERVICES=""
    CHECK_DB=false
    CHECK_KAFKA=false
    CHECK_REDIS=false
    VERBOSE=false
    OUTPUT_FORMAT="text"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --services)
                SERVICES="$2"
                shift 2
                ;;
            --check-db)
                CHECK_DB=true
                shift
                ;;
            --check-kafka)
                CHECK_KAFKA=true
                shift
                ;;
            --check-redis)
                CHECK_REDIS=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --output-format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Build command
    CMD="python3 $SCRIPT_DIR/check_platform_health.py --timeout $TIMEOUT --output-format $OUTPUT_FORMAT"
    
    if [ -n "$SERVICES" ]; then
        CMD="$CMD --services $SERVICES"
    fi
    
    if [ "$CHECK_DB" = true ]; then
        CMD="$CMD --check-db"
    fi
    
    if [ "$CHECK_KAFKA" = true ]; then
        CMD="$CMD --check-kafka"
    fi
    
    if [ "$CHECK_REDIS" = true ]; then
        CMD="$CMD --check-redis"
    fi
    
    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi
    
    # Run the script
    log "Executing: $CMD"
    eval "$CMD"
    
    # Check exit code
    if [ $? -eq 0 ]; then
        success "Forex Trading Platform is healthy"
        return 0
    else
        error "Forex Trading Platform is not healthy"
        return 1
    fi
}

# Main function
main() {
    log "Starting platform health check..."
    
    # Check requirements
    check_python
    check_pip
    
    # Install required packages
    install_packages
    
    # Run health check script
    run_health_check_script "$@"
    
    return $?
}

# Run main function with all arguments
main "$@"
