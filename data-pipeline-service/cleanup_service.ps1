# Cleanup and Validation Script for Data Pipeline Service
# This script performs cleanup operations and validates the code structure

# Variables
$serviceDir = "D:\MD\forex_trading_platform\data-pipeline-service"
$logFile = Join-Path $serviceDir "cleanup_log.txt"

function Write-Log {
    param (
        [string]$message,
        [string]$type = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$type] $message"
    
    Write-Host $logMessage
    Add-Content -Path $logFile -Value $logMessage
}

function Remove-EmptyFolders {
    param (
        [string]$basePath
    )
    
    Write-Log "Checking for empty folders in $basePath"
    
    # Find empty directories
    $emptyDirs = Get-ChildItem -Path $basePath -Directory -Recurse | 
                Where-Object { (Get-ChildItem -Path $_.FullName -Recurse -File).Count -eq 0 }
    
    foreach ($dir in $emptyDirs) {
        Write-Log "Empty folder found: $($dir.FullName)" "WARNING"
        
        # Ask user if they want to remove the folder or add a README
        Write-Host ""
        Write-Host "Options for empty folder $($dir.FullName):"
        Write-Host "1. Remove folder"
        Write-Host "2. Add README.md explaining purpose"
        Write-Host "3. Skip"
        $choice = Read-Host "Enter choice (default: 3)"
        
        if ($choice -eq "1") {
            try {
                Remove-Item -Path $dir.FullName -Force -Recurse
                Write-Log "Removed empty folder: $($dir.FullName)" "SUCCESS"
            } catch {
                Write-Log "Failed to remove folder: $($dir.FullName). Error: $($_.Exception.Message)" "ERROR"
            }
        } elseif ($choice -eq "2") {
            $readmePath = Join-Path $dir.FullName "README.md"
            $content = @"
# Purpose of this Directory

This directory is a placeholder for future development. 
It's part of the data-pipeline-service architecture but is not currently used.

## Intended Usage

[Add description of intended usage here]

## When to Use

[Add information about when this component will be implemented]
"@
            try {
                Set-Content -Path $readmePath -Value $content
                Write-Log "Added README.md to: $($dir.FullName)" "SUCCESS"
            } catch {
                Write-Log "Failed to add README.md. Error: $($_.Exception.Message)" "ERROR"
            }
        } else {
            Write-Log "Skipped folder: $($dir.FullName)" "INFO"
        }
    }
}

function Check-DuplicateLogic {
    param (
        [string]$basePath
    )
    
    Write-Log "Checking for potential duplicate logic in the codebase"
    
    # Check for validation engine duplication
    $basicEngine = Join-Path $basePath "data_pipeline_service\validation\validation_engine.py"
    $advancedEngine = Join-Path $basePath "data_pipeline_service\validation\advanced_validation_engine.py"
    
    if ((Test-Path $basicEngine) -and (Test-Path $advancedEngine)) {
        Write-Log "Potential duplicate validation logic found:" "WARNING"
        Write-Log "  - $basicEngine" "WARNING"
        Write-Log "  - $advancedEngine" "WARNING"
        Write-Log "Consider consolidating these components or documenting their differences" "SUGGESTION"
    }
    
    # Check for cleaning engine duplication
    $basicCleaning = Join-Path $basePath "data_pipeline_service\cleaning\cleaning_engine.py"
    $advancedCleaning = Join-Path $basePath "data_pipeline_service\cleaning\advanced_cleaning_engine.py"
    
    if ((Test-Path $basicCleaning) -and (Test-Path $advancedCleaning)) {
        Write-Log "Potential duplicate cleaning logic found:" "WARNING"
        Write-Log "  - $basicCleaning" "WARNING"
        Write-Log "  - $advancedCleaning" "WARNING"
        Write-Log "Consider consolidating these components or documenting their differences" "SUGGESTION"
    }
}

function Run-PythonTests {
    param (
        [string]$basePath
    )
    
    Write-Log "Running the TimeseriesAggregator tests"
    
    try {
        Set-Location $basePath
        $output = python -m pytest tests/services/test_timeseries_aggregator.py tests/test_basic.py -v
        Write-Log "Test execution completed" "INFO"
        Write-Log $output "TEST"
    } catch {
        Write-Log "Failed to run tests. Error: $($_.Exception.Message)" "ERROR"
    }
}

function Validate-PydanticUsage {
    param (
        [string]$basePath
    )
    
    Write-Log "Checking for deprecated Pydantic validators"
    
    $files = Get-ChildItem -Path $basePath -Recurse -Include "*.py" | 
             Select-Object -ExpandProperty FullName
             
    $pattern = '@validator\('
    
    foreach ($file in $files) {
        $content = Get-Content -Path $file -Raw
        if ($content -match $pattern) {
            Write-Log "Deprecated Pydantic validator found in: $file" "WARNING"
            Write-Log "Consider updating to @field_validator for Pydantic v2 compatibility" "SUGGESTION"
        }
    }
}

# Main execution
Write-Log "Starting cleanup and validation process" "INFO"

# Remove the log file if it exists
if (Test-Path $logFile) {
    Remove-Item -Path $logFile -Force
}

# Run the checks and cleanup
Remove-EmptyFolders -basePath "$serviceDir\data_pipeline_service"
Check-DuplicateLogic -basePath $serviceDir
Validate-PydanticUsage -basePath $serviceDir
Run-PythonTests -basePath $serviceDir

Write-Log "Cleanup and validation process completed" "INFO"

# Display summary
Write-Host ""
Write-Host "----- Summary -----"
Write-Host "Check the log file for detailed results: $logFile"
Write-Host "Review the SERVICE_ANALYSIS.md file for comprehensive recommendations"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Address any warnings or errors found"
Write-Host "2. Follow the recommendations in the SERVICE_ANALYSIS.md file"
Write-Host "3. Run the full test suite to verify all fixes"
