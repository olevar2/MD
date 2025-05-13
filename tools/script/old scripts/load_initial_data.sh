#!/bin/bash
# Initial Data Loading Script for Forex Trading Platform
# This script is a wrapper for load_initial_data.py that handles environment setup and error handling

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
    pip3 install requests pandas numpy
    success "Required Python packages installed successfully"
}

# Check if services are running
check_services() {
    log "Checking if services are running..."
    
    # Check data-pipeline-service
    if ! curl -s http://localhost:8001/health > /dev/null; then
        warning "data-pipeline-service is not running. Some data loading operations may fail."
    else
        success "data-pipeline-service is running"
    fi
    
    # Check feature-store-service
    if ! curl -s http://localhost:8002/health > /dev/null; then
        warning "feature-store-service is not running. Some data loading operations may fail."
    else
        success "feature-store-service is running"
    fi
    
    # Check portfolio-management-service
    if ! curl -s http://localhost:8006/health > /dev/null; then
        warning "portfolio-management-service is not running. Some data loading operations may fail."
    else
        success "portfolio-management-service is running"
    fi
    
    # Check ml-integration-service
    if ! curl -s http://localhost:8004/health > /dev/null; then
        warning "ml-integration-service is not running. Some data loading operations may fail."
    else
        success "ml-integration-service is running"
    fi
}

# Run the data loading script
run_data_loading_script() {
    log "Loading initial data into the Forex Trading Platform..."
    
    # Parse command line arguments
    DATA_DIR="data/sample"
    SKIP_TYPES=""
    ONLY_TYPES=""
    VERIFY=false
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --skip-types)
                SKIP_TYPES="$2"
                shift 2
                ;;
            --only-types)
                ONLY_TYPES="$2"
                shift 2
                ;;
            --verify)
                VERIFY=true
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
    CMD="python3 $SCRIPT_DIR/load_initial_data.py --data-dir $DATA_DIR"
    
    if [ -n "$SKIP_TYPES" ]; then
        CMD="$CMD --skip-types $SKIP_TYPES"
    fi
    
    if [ -n "$ONLY_TYPES" ]; then
        CMD="$CMD --only-types $ONLY_TYPES"
    fi
    
    if [ "$VERIFY" = true ]; then
        CMD="$CMD --verify"
    fi
    
    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi
    
    # Run the script
    log "Executing: $CMD"
    eval "$CMD"
    
    # Check exit code
    if [ $? -eq 0 ]; then
        success "Initial data loaded successfully"
    else
        error "Failed to load initial data"
        exit 1
    fi
}

# Main function
main() {
    log "Starting initial data loading process..."
    
    # Check requirements
    check_python
    check_pip
    
    # Install required packages
    install_packages
    
    # Check if services are running
    check_services
    
    # Run data loading script
    run_data_loading_script "$@"
    
    success "Initial data loading process completed"
}

# Run main function with all arguments
main "$@"
