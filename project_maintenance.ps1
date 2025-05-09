# PowerShell script for project maintenance and cleanup
# This script combines functionality from the previous cleanup scripts
# and adds new features for ongoing project maintenance

param (
    [switch]$CheckOnly = $false,
    [switch]$RemoveEmptyDirs = $false,
    [switch]$RemoveDuplicateFiles = $false,
    [switch]$All = $false
)

# Show help if no parameters provided
if (-not ($CheckOnly -or $RemoveEmptyDirs -or $RemoveDuplicateFiles -or $All)) {
    Write-Host "Project Maintenance Script"
    Write-Host "Usage:"
    Write-Host "  .\project_maintenance.ps1 -CheckOnly              # Only check for issues without making changes"
    Write-Host "  .\project_maintenance.ps1 -RemoveEmptyDirs        # Remove empty directories"
    Write-Host "  .\project_maintenance.ps1 -RemoveDuplicateFiles   # Remove duplicate files"
    Write-Host "  .\project_maintenance.ps1 -All                    # Perform all maintenance tasks"
    exit
}

# If -All is specified, enable all operations
if ($All) {
    $RemoveEmptyDirs = $true
    $RemoveDuplicateFiles = $true
}

# Function to check if a file is empty or has only whitespace
function Test-FileIsEmpty {
    param (
        [string]$FilePath
    )
    
    if (-not (Test-Path $FilePath)) {
        return $true
    }
    
    $content = Get-Content $FilePath -Raw
    return [string]::IsNullOrWhiteSpace($content)
}

# Function to find empty directories
function Find-EmptyDirectories {
    param (
        [string]$RootPath = "."
    )
    
    $emptyDirs = @()
    
    # Get all directories
    $allDirs = Get-ChildItem -Path $RootPath -Directory -Recurse
    
    foreach ($dir in $allDirs) {
        # Check if directory is empty (no files and no subdirectories)
        $items = Get-ChildItem -Path $dir.FullName -Force
        if ($items.Count -eq 0) {
            $emptyDirs += $dir.FullName
        }
    }
    
    return $emptyDirs
}

# Function to find potential duplicate files
function Find-PotentialDuplicateFiles {
    param (
        [string]$RootPath = "."
    )
    
    $potentialDuplicates = @()
    
    # Find files with "copy" in the name
    $copyFiles = Get-ChildItem -Path $RootPath -File -Recurse | Where-Object { $_.Name -like "*copy*" }
    foreach ($file in $copyFiles) {
        $potentialDuplicates += @{
            Path = $file.FullName
            Reason = "Has 'copy' in the name"
        }
    }
    
    # Find empty files
    $emptyFiles = Get-ChildItem -Path $RootPath -File -Recurse | Where-Object { $_.Length -eq 0 }
    foreach ($file in $emptyFiles) {
        $potentialDuplicates += @{
            Path = $file.FullName
            Reason = "Empty file (0 bytes)"
        }
    }
    
    # Find duplicate test files (this is a simplified check, might need refinement)
    $testFiles = Get-ChildItem -Path $RootPath -File -Recurse -Filter "test_*.py"
    $testFileNames = $testFiles | Group-Object -Property Name
    
    foreach ($group in $testFileNames) {
        if ($group.Count -gt 1) {
            foreach ($file in $group.Group) {
                $potentialDuplicates += @{
                    Path = $file.FullName
                    Reason = "Potential duplicate test file (same name in different directories)"
                }
            }
        }
    }
    
    return $potentialDuplicates
}

# Main script execution

# Check for empty directories
$emptyDirs = Find-EmptyDirectories
if ($emptyDirs.Count -gt 0) {
    Write-Host "`nFound $($emptyDirs.Count) empty directories:"
    foreach ($dir in $emptyDirs) {
        Write-Host "  $dir"
    }
    
    if ($RemoveEmptyDirs -and -not $CheckOnly) {
        Write-Host "`nRemoving empty directories..."
        foreach ($dir in $emptyDirs) {
            try {
                Remove-Item -Path $dir -Force -Recurse -ErrorAction Stop
                Write-Host "  Removed: $dir"
            }
            catch {
                Write-Host "  Failed to remove: $dir - $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}
else {
    Write-Host "`nNo empty directories found."
}

# Check for potential duplicate files
$potentialDuplicates = Find-PotentialDuplicateFiles
if ($potentialDuplicates.Count -gt 0) {
    Write-Host "`nFound $($potentialDuplicates.Count) potential duplicate or problematic files:"
    foreach ($file in $potentialDuplicates) {
        Write-Host "  $($file.Path) - $($file.Reason)"
    }
    
    if ($RemoveDuplicateFiles -and -not $CheckOnly) {
        Write-Host "`nWARNING: Automatic removal of duplicate files is risky."
        $confirmation = Read-Host "Are you sure you want to remove these files? (y/n)"
        
        if ($confirmation -eq 'y') {
            Write-Host "`nRemoving files..."
            foreach ($file in $potentialDuplicates) {
                try {
                    Remove-Item -Path $file.Path -Force -ErrorAction Stop
                    Write-Host "  Removed: $($file.Path)"
                }
                catch {
                    Write-Host "  Failed to remove: $($file.Path) - $($_.Exception.Message)" -ForegroundColor Red
                }
            }
        }
        else {
            Write-Host "File removal cancelled."
        }
    }
}
else {
    Write-Host "`nNo potential duplicate files found."
}

# Check for .pytest_cache directories
$pytestCacheDirs = Get-ChildItem -Path "." -Directory -Recurse -Filter ".pytest_cache"
if ($pytestCacheDirs.Count -gt 0) {
    Write-Host "`nFound $($pytestCacheDirs.Count) .pytest_cache directories."
    Write-Host "Recommendation: Add .pytest_cache/ to your .gitignore file if not already present."
}

# Final summary
if ($CheckOnly) {
    Write-Host "`nCheck completed. Use -RemoveEmptyDirs or -RemoveDuplicateFiles to perform cleanup actions."
}
else {
    Write-Host "`nMaintenance tasks completed."
}
