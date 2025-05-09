# Forex Trading Platform Visualization Tools

This directory contains tools for visualizing the architecture of the Forex Trading Platform.

## Available Tools

### 1. Current Architecture Analyzer

Analyzes the current architecture of the platform by scanning the codebase and generates visualizations and reports.

**Usage:**
```bash
python tools/visualizations/current_architecture_analyzer.py [--root DIR] [--output-dir DIR]
```

**Features:**
- Scans the directory structure to identify all services
- Analyzes dependencies between services
- Identifies API endpoints and their relationships
- Generates visual representations of the current architecture
- Outputs a comprehensive report with key metrics and insights

**Outputs:**
- `current_architecture.png` - Graph visualization of the current architecture
- `current_layered_architecture.png` - Layered visualization of the current architecture
- `current_architecture_report.md` - Detailed report of the current architecture

### 2. Improved Architecture Visualizer

Generates visualizations of an improved architecture for the Forex Trading Platform.

**Usage:**
```bash
python tools/visualizations/improved_architecture_visualizer.py [--output-dir DIR]
```

**Features:**
- Visualizes the recommended improved architecture
- Shows service organization and dependencies
- Generates both graph and layered visualizations

**Outputs:**
- `improved_architecture.png` - Graph visualization of the improved architecture
- `layered_architecture.png` - Layered visualization of the improved architecture

### 3. Structure Visualizer

Generates an interactive HTML visualization of the project structure.

**Usage:**
```bash
python tools/visualizations/structure_visualizer.py [--data FILE] [--output-dir DIR]
```

**Features:**
- Creates an interactive HTML visualization
- Shows hierarchy, components, and relationships
- Provides a user-friendly interface for exploring the structure

## Example Workflow

1. Analyze the current architecture:
   ```bash
   python tools/visualizations/current_architecture_analyzer.py
   ```

2. View the generated reports and visualizations in the `tools/reports` directory

3. Compare with the improved architecture:
   ```bash
   python tools/visualizations/improved_architecture_visualizer.py
   ```
