# Troubleshooting Guide

This guide provides solutions for common issues you might encounter when setting up, configuring, or running the Forex Trading Platform.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Database Issues](#database-issues)
3. [Service Startup Issues](#service-startup-issues)
4. [Configuration Issues](#configuration-issues)
5. [API Issues](#api-issues)
6. [Performance Issues](#performance-issues)
7. [Data Issues](#data-issues)
8. [Docker Issues](#docker-issues)
9. [Common Error Messages](#common-error-messages)
10. [Getting Help](#getting-help)

## Installation Issues

### Python Dependencies Installation Fails

**Problem**: When installing Python dependencies, you encounter errors.

**Solutions**:
- Ensure you're using Python 3.10 or later: `python --version`
- Update pip: `pip install --upgrade pip`
- Install build dependencies: `sudo apt-get install python3-dev build-essential` (Linux)
- Try installing dependencies one by one to identify the problematic package
- Check for conflicting dependencies in your environment

### TimescaleDB Installation Fails

**Problem**: TimescaleDB installation fails or doesn't work correctly.

**Solutions**:
- Ensure PostgreSQL is installed and running: `pg_isready -h localhost`
- Check PostgreSQL version (should be 14 or later): `psql --version`
- Verify TimescaleDB extension is available: `psql -c "SELECT * FROM pg_available_extensions WHERE name = 'timescaledb';"`
- Check PostgreSQL logs for errors: `sudo tail -f /var/log/postgresql/postgresql-14-main.log` (Linux)
- Restart PostgreSQL after installing TimescaleDB: `sudo systemctl restart postgresql` (Linux)

### Docker Installation Issues

**Problem**: Docker or Docker Compose installation fails.

**Solutions**:
- Ensure your system meets Docker requirements
- Check if Docker daemon is running: `docker info`
- Verify Docker Compose installation: `docker-compose --version`
- Add your user to the docker group: `sudo usermod -aG docker $USER` (Linux)
- Restart your system after installing Docker

## Database Issues

### Database Connection Fails

**Problem**: Services can't connect to the database.

**Solutions**:
- Verify PostgreSQL is running: `pg_isready -h localhost`
- Check database credentials in `.env` files
- Ensure the database exists: `psql -U postgres -c "SELECT datname FROM pg_database;"`
- Check PostgreSQL is accepting connections: `sudo netstat -tulpn | grep postgres`
- Verify firewall settings allow connections to PostgreSQL port (usually 5432)

### TimescaleDB Extension Not Available

**Problem**: TimescaleDB functions are not available in the database.

**Solutions**:
- Verify TimescaleDB is installed: `psql -U postgres -c "SELECT * FROM pg_available_extensions WHERE name = 'timescaledb';"`
- Create the extension in your database: `psql -U postgres -d your_database -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"`
- Check if shared_preload_libraries includes timescaledb in postgresql.conf
- Restart PostgreSQL after modifying postgresql.conf

### Database Schema Migration Fails

**Problem**: Database schema migration fails during setup.

**Solutions**:
- Check database logs for detailed error messages
- Verify database user has sufficient privileges
- Ensure no conflicting schema exists
- Try running migrations manually: `python scripts/setup_database.py`
- Check for database version compatibility issues

## Service Startup Issues

### Service Fails to Start

**Problem**: A service fails to start or crashes immediately.

**Solutions**:
- Check service logs for error messages: `tail -f logs/service-name.log`
- Verify all required environment variables are set
- Ensure dependencies (database, Redis, Kafka) are running
- Check if the port is already in use: `netstat -tulpn | grep <port>`
- Increase log level to DEBUG for more detailed logs

### Service Dependencies Not Available

**Problem**: Service can't connect to its dependencies.

**Solutions**:
- Use the health check script to verify dependencies: `./scripts/check_platform_health.sh`
- Ensure dependency services are running
- Check network connectivity between services
- Verify service URLs in configuration are correct
- Check for firewall or network issues

### Port Already in Use

**Problem**: Service can't bind to its port because it's already in use.

**Solutions**:
- Find the process using the port: `netstat -tulpn | grep <port>`
- Kill the process: `kill <pid>` or `sudo kill <pid>`
- Change the port in the service configuration
- Restart the service

## Configuration Issues

### Missing Environment Variables

**Problem**: Service complains about missing environment variables.

**Solutions**:
- Check if `.env` file exists for the service
- Verify all required variables are set in the `.env` file
- Run the environment validation script: `python scripts/validate_env_config.py --service <service-name>`
- Generate environment files: `python scripts/generate_env_files.py --env development`

### Configuration File Not Found

**Problem**: Service can't find its configuration file.

**Solutions**:
- Verify the configuration file exists in the expected location
- Check file permissions
- Ensure the service is running from the correct directory
- Specify the configuration file path explicitly when starting the service

### Invalid Configuration Values

**Problem**: Service rejects configuration values as invalid.

**Solutions**:
- Check the service logs for specific validation errors
- Verify the format of configuration values (e.g., URLs, ports, timeouts)
- Ensure numeric values are within acceptable ranges
- Check for typos in configuration keys or values

## API Issues

### API Returns 404 Not Found

**Problem**: API endpoint returns 404 Not Found.

**Solutions**:
- Verify the URL is correct
- Check if the service is running
- Ensure the API version is correct
- Verify the endpoint path is correct
- Check service logs for routing issues

### API Returns 500 Internal Server Error

**Problem**: API endpoint returns 500 Internal Server Error.

**Solutions**:
- Check service logs for detailed error messages
- Verify the request payload is valid
- Ensure the database is accessible
- Check if dependencies are available
- Increase log level to DEBUG for more detailed logs

### Authentication Fails

**Problem**: API authentication fails.

**Solutions**:
- Verify API key or token is correct
- Check if the token has expired
- Ensure the authentication header is formatted correctly
- Verify the user has sufficient permissions
- Check service logs for authentication issues

## Performance Issues

### Slow API Responses

**Problem**: API responses are slow.

**Solutions**:
- Check database performance
- Verify service has sufficient resources (CPU, memory)
- Look for bottlenecks in dependencies
- Enable caching where appropriate
- Optimize database queries
- Check for network latency issues

### High CPU or Memory Usage

**Problem**: Service uses excessive CPU or memory.

**Solutions**:
- Check for memory leaks
- Optimize resource-intensive operations
- Increase service resources if necessary
- Enable profiling to identify bottlenecks
- Implement pagination for large data sets
- Optimize database queries

### Database Performance Issues

**Problem**: Database operations are slow.

**Solutions**:
- Check database indexes
- Optimize queries
- Increase database resources
- Enable query caching
- Use connection pooling
- Implement database sharding or partitioning
- Tune PostgreSQL configuration

## Data Issues

### Missing or Incomplete Data

**Problem**: Data is missing or incomplete.

**Solutions**:
- Verify data sources are available
- Check data pipeline logs for errors
- Ensure data loaders are running correctly
- Verify data validation rules
- Check for data transformation issues
- Run data reconciliation to identify inconsistencies

### Data Inconsistency Across Services

**Problem**: Data is inconsistent across services.

**Solutions**:
- Run data reconciliation: `python scripts/validate_data_integrity.py`
- Check for failed data synchronization
- Verify event propagation between services
- Ensure data is properly validated
- Check for race conditions in data updates

### Historical Data Not Available

**Problem**: Historical data is not available.

**Solutions**:
- Verify data retention policies
- Check if data has been archived
- Ensure historical data loaders are running
- Verify data sources provide historical data
- Check for data purging or cleanup jobs

## Docker Issues

### Docker Containers Won't Start

**Problem**: Docker containers fail to start.

**Solutions**:
- Check Docker logs: `docker logs <container-id>`
- Verify Docker Compose configuration
- Ensure required environment variables are set
- Check for port conflicts
- Verify Docker has sufficient resources

### Docker Networking Issues

**Problem**: Docker containers can't communicate with each other.

**Solutions**:
- Check Docker network configuration
- Verify container names and hostnames
- Ensure containers are on the same network
- Check for firewall or security group issues
- Verify DNS resolution within Docker network

### Docker Volume Issues

**Problem**: Docker volumes are not working correctly.

**Solutions**:
- Check volume mount points
- Verify file permissions
- Ensure host directories exist
- Check for disk space issues
- Verify Docker volume configuration

## Common Error Messages

### "Connection refused"

**Problem**: Service can't connect to a dependency.

**Solutions**:
- Verify the dependency is running
- Check network connectivity
- Ensure firewall allows the connection
- Verify hostname and port are correct
- Check for network namespace issues in containerized environments

### "Authentication failed"

**Problem**: Service can't authenticate with a dependency.

**Solutions**:
- Verify credentials are correct
- Check if credentials have expired
- Ensure authentication method is supported
- Verify SSL/TLS configuration if applicable
- Check for permission issues

### "Resource not found"

**Problem**: Service can't find a required resource.

**Solutions**:
- Verify the resource exists
- Check file paths or URLs
- Ensure the service has permission to access the resource
- Verify resource naming is correct
- Check for case sensitivity issues

## Getting Help

If you're still experiencing issues after trying the solutions in this guide, you can:

1. Check the service logs for detailed error messages
2. Run the platform health check: `./scripts/check_platform_health.sh`
3. Increase log levels to DEBUG for more detailed logs
4. Check for known issues in the GitHub repository
5. Submit an issue on GitHub with detailed information about the problem

When reporting issues, please include:

- Detailed description of the problem
- Steps to reproduce the issue
- Service logs
- Platform health check output
- Environment information (OS, Python version, etc.)
- Configuration (with sensitive information redacted)
