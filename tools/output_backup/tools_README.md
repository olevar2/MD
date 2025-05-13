# Forex Trading Platform Analysis Tools

This directory contains a set of focused code analysis tools for the Forex Trading Platform. Each tool is designed to analyze a specific aspect of the codebase and generate reports in the `tools/reports` directory.

## Available Tools

### 1. Code Statistics Tool

Analyzes the codebase and generates statistics about file counts, lines of code, and identifies the largest files.

**Usage:**
```bash
python tools/code_statistics.py [--root DIR] [--format {json,markdown,both}]
```

**Features:**
- Counts files by type/language
- Counts lines of code by language
- Identifies largest files in the project
- Generates service-level statistics

### 2. Dependency Analyzer

Analyzes dependencies between services in the codebase and generates a dependency graph.

**Usage:**
```bash
python tools/dependency_analyzer.py [--root DIR] [--format {json,markdown,dot,all}]
```

**Features:**
- Identifies imports between services
- Generates a dependency graph (DOT format)
- Detects potential circular dependencies
- Provides detailed import information

### 3. API Endpoint Detector

Finds API endpoints in FastAPI, Flask, Express, and NestJS services and generates API documentation.

**Usage:**
```bash
python tools/api_endpoint_detector.py [--root DIR] [--format {json,markdown,both}]
```

**Features:**
- Detects API endpoints across multiple frameworks
- Identifies HTTP methods and routes
- Analyzes endpoint patterns and naming conventions
- Generates API documentation by service

### 4. Error Handler Analyzer

Checks for consistent error handling patterns and identifies services missing proper error handling.

**Usage:**
```bash
python tools/error_handler_analyzer.py [--root DIR] [--format {json,markdown,both}]
```

**Features:**
- Identifies error handling patterns
- Detects custom error classes
- Calculates error handling coverage
- Finds files missing error handling

## Common Parameters

All tools support the following parameters:

- `--root`: Root directory of the project (default: D:/MD/forex_trading_platform)
- `--format`: Output format for the report (varies by tool)

## Reports

All reports are saved to the `tools/reports` directory with the following naming convention:

- `{tool_name}_{timestamp}.{format}` - Timestamped reports
- `{tool_name}_latest.{format}` - Latest version of each report

## Example Workflow

1. Generate code statistics:
   ```bash
   python tools/code_statistics.py
   ```

2. Analyze dependencies between services:
   ```bash
   python tools/dependency_analyzer.py
   ```

3. Detect API endpoints:
   ```bash
   python tools/api_endpoint_detector.py
   ```

4. Analyze error handling:
   ```bash
   python tools/error_handler_analyzer.py
   ```

5. Review all reports in the `tools/reports` directory

## Integration with Improvement Plan

These tools can be used as part of the Forex Trading Platform Improvement Plan to:

1. Establish a baseline of the current architecture
2. Identify areas for improvement
3. Measure progress after implementing changes

After making improvements, run the tools again and compare the new reports with the initial ones to measure progress.