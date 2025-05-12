#!/bin/bash
# Platform Shutdown Script for Forex Trading Platform
# This script is a wrapper for stop_platform.py that handles environment setup and error handling

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

# Run the platform shutdown script
run_shutdown_script() {
    log "Stopping the Forex Trading Platform..."
    
    # Parse command line arguments
    TIMEOUT=10
    SERVICES=""
    FORCE=false
    VERBOSE=false
    
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
            --force)
                FORCE=true
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
    
    # Build command
    CMD="python3 $SCRIPT_DIR/stop_platform.py --timeout $TIMEOUT"
    
    if [ -n "$SERVICES" ]; then
        CMD="$CMD --services $SERVICES"
    fi
    
    if [ "$FORCE" = true ]; then
        CMD="$CMD --force"
    fi
    
    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi
    
    # Run the script
    log "Executing: $CMD"
    eval "$CMD"
    
    # Check exit code
    if [ $? -eq 0 ]; then
        success "Forex Trading Platform stopped successfully"
    else
        error "Failed to stop Forex Trading Platform"
        exit 1
    fi
}

# Main function
main() {
    log "Stopping Forex Trading Platform..."
    
    # Check requirements
    check_python
    
    # Run shutdown script
    run_shutdown_script "$@"
    
    success "Forex Trading Platform shutdown process completed"
}

# Run main function with all arguments
main "$@"
