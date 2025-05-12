#!/bin/bash
# Platform Startup Script for Forex Trading Platform
# This script is a wrapper for start_platform.py that handles environment setup and error handling

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
    pip3 install requests psycopg2-binary pyyaml
    success "Required Python packages installed successfully"
}

# Check if database is running
check_database() {
    log "Checking if PostgreSQL is running..."
    
    if ! command -v pg_isready &> /dev/null; then
        warning "pg_isready command not found, skipping database check"
        return 0
    fi
    
    if pg_isready -h localhost -p 5432 -U postgres > /dev/null 2>&1; then
        success "PostgreSQL is running"
        return 0
    else
        error "PostgreSQL is not running. Please start PostgreSQL and try again."
        error "You can start PostgreSQL using the following command:"
        error "  sudo systemctl start postgresql"
        error "Or if you're using Docker:"
        error "  docker start postgres"
        return 1
    fi
}

# Run the platform startup script
run_startup_script() {
    log "Starting the Forex Trading Platform..."
    
    # Parse command line arguments
    ENV="development"
    TIMEOUT=30
    SKIP_DEPS=false
    SKIP_HEALTH_CHECK=false
    SERVICES=""
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                ENV="$2"
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
            --services)
                SERVICES="$2"
                shift 2
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
    CMD="python3 $SCRIPT_DIR/start_platform.py --env $ENV --timeout $TIMEOUT"
    
    if [ "$SKIP_DEPS" = true ]; then
        CMD="$CMD --skip-deps"
    fi
    
    if [ "$SKIP_HEALTH_CHECK" = true ]; then
        CMD="$CMD --skip-health-check"
    fi
    
    if [ -n "$SERVICES" ]; then
        CMD="$CMD --services $SERVICES"
    fi
    
    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi
    
    # Run the script
    log "Executing: $CMD"
    eval "$CMD"
    
    # Check exit code
    if [ $? -eq 0 ]; then
        success "Forex Trading Platform started successfully"
    else
        error "Failed to start Forex Trading Platform"
        exit 1
    fi
}

# Main function
main() {
    log "Starting Forex Trading Platform..."
    
    # Check requirements
    check_python
    check_pip
    
    # Install required packages
    install_packages
    
    # Check if database is running
    check_database || exit 1
    
    # Run startup script
    run_startup_script "$@"
    
    success "Forex Trading Platform startup process completed"
}

# Run main function with all arguments
main "$@"
