# Try to find Poetry in various locations
$poetryPaths = @(
    "$env:USERPROFILE\.poetry\bin\poetry.exe",
    "$env:APPDATA\Python\Scripts\poetry.exe",
    "$env:LOCALAPPDATA\pypoetry\venv\Scripts\poetry.exe",
    "$env:USERPROFILE\AppData\Roaming\Python\Scripts\poetry.exe",
    "$env:USERPROFILE\AppData\Local\Programs\Python\Scripts\poetry.exe"
)

$poetryExe = $null
$useModule = $false # Initialize $useModule

foreach ($path in $poetryPaths) {
    Write-Verbose "Checking path: $path" # Add verbose output
    if (Test-Path $path) {
        $poetryExe = $path
        Write-Host "Found Poetry at: $poetryExe"
        break
    }
}

if (-not $poetryExe) {
    Write-Host "Poetry not found in common locations. Attempting to use module..."
    try {
        Write-Verbose "Attempting 'python -m poetry --version'"
        # Use Invoke-Expression to capture potential errors properly
        Invoke-Expression "python -m poetry --version" | Out-Null
        $useModule = $true
        Write-Host "Poetry module accessible."
    } catch {
        Write-Host "Poetry module not accessible: $($_.Exception.Message)"
        # Don't exit immediately, allow fallback if needed or manual intervention
        Write-Warning "Could not find or access Poetry automatically."
        # Optionally, provide manual instructions here
        # exit 1 # Removed exit to allow script completion
    }
}

# Run Poetry install only if found/accessible
if ($poetryExe -or $useModule) {
    Write-Host "Running Poetry install..."
    if ($useModule) {
        Write-Verbose "Using module: python -m poetry install"
        python -m poetry install
    } else {
        Write-Verbose "Using executable: & '$poetryExe' install"
        & $poetryExe install
    }
    Write-Host "Dependencies installation completed."
} else {
    Write-Error "Failed to find Poetry executable or module. Cannot install dependencies."
}
