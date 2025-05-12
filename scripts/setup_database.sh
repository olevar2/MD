#!/bin/bash
# Database Setup Script for Forex Trading Platform
# This script is a wrapper for setup_database.py that handles environment setup and error handling

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
    pip3 install psycopg2-binary
    success "Required Python packages installed successfully"
}

# Create infrastructure directories if they don't exist
create_directories() {
    log "Creating infrastructure directories..."
    mkdir -p "$PROJECT_ROOT/infrastructure/database/init_scripts"
    success "Infrastructure directories created successfully"
}

# Run the database setup script
run_setup_script() {
    log "Running database setup script..."
    
    # Parse command line arguments
    HOST="localhost"
    PORT="5432"
    ADMIN_USER="postgres"
    ADMIN_PASSWORD="postgres"
    SKIP_INSTALL=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --host)
                HOST="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --admin-user)
                ADMIN_USER="$2"
                shift 2
                ;;
            --admin-password)
                ADMIN_PASSWORD="$2"
                shift 2
                ;;
            --skip-install)
                SKIP_INSTALL=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Build command
    CMD="python3 $SCRIPT_DIR/setup_database.py --host $HOST --port $PORT --admin-user $ADMIN_USER --admin-password $ADMIN_PASSWORD"
    
    if [ "$SKIP_INSTALL" = true ]; then
        CMD="$CMD --skip-install"
    fi
    
    # Run the script
    log "Executing: $CMD"
    eval "$CMD"
    
    # Check exit code
    if [ $? -eq 0 ]; then
        success "Database setup completed successfully"
    else
        error "Database setup failed"
        exit 1
    fi
}

# Main function
main() {
    log "Starting database setup for Forex Trading Platform..."
    
    # Check requirements
    check_python
    check_pip
    
    # Install required packages
    install_packages
    
    # Create directories
    create_directories
    
    # Run setup script
    run_setup_script "$@"
    
    success "Database setup process completed"
}

# Run main function with all arguments
main "$@"
