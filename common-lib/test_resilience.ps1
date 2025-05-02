# PowerShell script to run the simple resilience test

Write-Host "Running simple resilience test..." -ForegroundColor Cyan

# Execute the test script
python usage_demos/simple_resilience_test.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "Test completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Test failed with exit code $LASTEXITCODE" -ForegroundColor Red
}

Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
