@echo off
REM Run the simple resilience test to validate the implementation

echo Running simple resilience test...
python usage_demos/simple_resilience_test.py

if %ERRORLEVEL% EQU 0 (
    echo Test completed successfully!
) else (
    echo Test failed with error level %ERRORLEVEL%
)

echo.
pause
