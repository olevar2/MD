@echo off
REM Script to run individual services in Docker for isolated testing

setlocal enabledelayedexpansion

REM Root directory of the forex trading platform
set ROOT_DIR=D:\MD\forex_trading_platform

REM Parse command line arguments
set SERVICE=
set BUILD=0
set STOP=0
set LOGS=0
set DETACH=0

:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--service" (
    set SERVICE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--build" (
    set BUILD=1
    shift
    goto :parse_args
)
if "%~1"=="--stop" (
    set STOP=1
    shift
    goto :parse_args
)
if "%~1"=="--logs" (
    set LOGS=1
    shift
    goto :parse_args
)
if "%~1"=="--detach" (
    set DETACH=1
    shift
    goto :parse_args
)
echo Unknown option: %~1
goto :print_usage

:end_parse_args

REM Check if service is provided
if "%SERVICE%"=="" (
    echo Error: Service name is required
    goto :print_usage
)

REM Check if service is valid
set VALID_SERVICE=0
for %%s in (causal-analysis-service backtesting-service market-analysis-service analysis-coordinator-service data-management-service monitoring-alerting-service strategy-execution-engine analysis-engine-service api-gateway data-pipeline-service feature-store-service ml-integration-service ml-workbench-service model-registry-service portfolio-management-service risk-management-service trading-gateway-service ui-service) do (
    if "%%s"=="%SERVICE%" set VALID_SERVICE=1
)

if %VALID_SERVICE%==0 (
    echo Error: Invalid service name: %SERVICE%
    goto :print_usage
)

REM Get service directory
set SERVICE_DIR=%ROOT_DIR%\%SERVICE%
if not exist "%SERVICE_DIR%" (
    echo Error: Service directory not found: %SERVICE_DIR%
    exit /b 1
)

REM Change to service directory
cd /d "%SERVICE_DIR%"

REM Check if docker-compose.yml exists, create if not
if exist docker-compose.yml (
    echo Using existing docker-compose.yml in %SERVICE%
) else (
    if not exist Dockerfile (
        echo Error: Dockerfile not found in %SERVICE%
        exit /b 1
    )
    
    echo Creating docker-compose.yml for %SERVICE%
    
    REM Determine port based on service
    set PORT=8000
    if "%SERVICE%"=="causal-analysis-service" set PORT=8000
    if "%SERVICE%"=="backtesting-service" set PORT=8002
    if "%SERVICE%"=="market-analysis-service" set PORT=8001
    if "%SERVICE%"=="analysis-coordinator-service" set PORT=8003
    if "%SERVICE%"=="data-management-service" set PORT=8004
    if "%SERVICE%"=="monitoring-alerting-service" set PORT=8005
    if "%SERVICE%"=="strategy-execution-engine" set PORT=8006
    if "%SERVICE%"=="analysis-engine-service" set PORT=8002
    if "%SERVICE%"=="api-gateway" set PORT=8080
    if "%SERVICE%"=="data-pipeline-service" set PORT=8004
    if "%SERVICE%"=="feature-store-service" set PORT=8001
    if "%SERVICE%"=="ml-integration-service" set PORT=8005
    if "%SERVICE%"=="ml-workbench-service" set PORT=8006
    if "%SERVICE%"=="model-registry-service" set PORT=8007
    if "%SERVICE%"=="portfolio-management-service" set PORT=8008
    if "%SERVICE%"=="risk-management-service" set PORT=8009
    if "%SERVICE%"=="trading-gateway-service" set PORT=8010
    if "%SERVICE%"=="ui-service" set PORT=80
    
    (
        echo version: '3.8'
        echo.
        echo services:
        echo   %SERVICE%:
        echo     build:
        echo       context: .
        echo       dockerfile: Dockerfile
        echo     ports:
        echo       - "%PORT%:%PORT%"
        echo     environment:
        echo       - DEBUG_MODE=true
        echo       - LOG_LEVEL=DEBUG
        echo       - HOST=0.0.0.0
        echo       - PORT=%PORT%
        echo     volumes:
        echo       - ./:/app
        echo     networks:
        echo       - forex-platform-network
        echo.
        echo networks:
        echo   forex-platform-network:
        echo     driver: bridge
    ) > docker-compose.yml
    
    echo Created docker-compose.yml for %SERVICE%
)

REM Stop the service
if %STOP%==1 (
    echo Stopping %SERVICE%
    docker-compose down
    exit /b 0
)

REM Show logs
if %LOGS%==1 (
    echo Showing logs for %SERVICE%
    docker-compose logs -f
    exit /b 0
)

REM Build the Docker image
if %BUILD%==1 (
    echo Building Docker image for %SERVICE%
    docker-compose build
)

REM Run the service
echo Running %SERVICE%
if %DETACH%==1 (
    docker-compose up -d
) else (
    docker-compose up
)

exit /b 0

:print_usage
echo Usage: %0 --service SERVICE_NAME [--build] [--stop] [--logs] [--detach]
echo.
echo Options:
echo   --service SERVICE_NAME  Service to run in Docker
echo   --build                 Build the Docker image before running
echo   --stop                  Stop the running service
echo   --logs                  View logs for the service
echo   --detach                Run in detached mode
echo.
echo Available services:
echo   - causal-analysis-service
echo   - backtesting-service
echo   - market-analysis-service
echo   - analysis-coordinator-service
echo   - data-management-service
echo   - monitoring-alerting-service
echo   - strategy-execution-engine
echo   - analysis-engine-service
echo   - api-gateway
echo   - data-pipeline-service
echo   - feature-store-service
echo   - ml-integration-service
echo   - ml-workbench-service
echo   - model-registry-service
echo   - portfolio-management-service
echo   - risk-management-service
echo   - trading-gateway-service
echo   - ui-service
exit /b 1
