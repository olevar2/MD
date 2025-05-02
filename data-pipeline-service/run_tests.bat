@echo off
REM Run tests for data-pipeline-service
powershell -ExecutionPolicy Bypass -File "%~dp0run_tests.ps1"
pause
