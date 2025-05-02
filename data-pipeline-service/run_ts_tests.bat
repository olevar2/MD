@echo off
echo Running Data Pipeline Service tests...
cd /d "%~dp0"

:: Set Python path to include parent directory for proper imports
set PYTHONPATH=%PYTHONPATH%;%~dp0..

echo Running timeseries aggregator tests with PYTHONPATH=%PYTHONPATH%
python -m pytest tests/services/test_timeseries_aggregator.py tests/test_basic.py -v
if %ERRORLEVEL% neq 0 (
    echo Tests failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo All tests passed successfully!
pause
