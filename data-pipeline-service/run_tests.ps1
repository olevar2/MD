# Run tests for data-pipeline-service
$ErrorActionPreference = "Stop"

# Set the workspace root directory
$workspaceRoot = Split-Path -Parent $PSCommandPath
Write-Host "Workspace root: $workspaceRoot"

# Ensure we're in the right directory
Set-Location $workspaceRoot
Write-Host "Current directory: $(Get-Location)"

# Try to use Poetry if installed
try {
    Write-Host "Attempting to run tests with Poetry..."
    poetry run pytest -v tests/services/test_timeseries_aggregator.py tests/test_basic.py
} catch {
    Write-Host "Poetry execution failed: $($_.Exception.Message)"
    
    # Fallback to direct pytest invocation
    Write-Host "Falling back to direct pytest invocation..."
    
    # Make sure pytest is installed
    python -m pip install --upgrade pytest pytest-asyncio pandas

    # Run pytest directly
    python -m pytest -v tests/services/test_timeseries_aggregator.py tests/test_basic.py
}
