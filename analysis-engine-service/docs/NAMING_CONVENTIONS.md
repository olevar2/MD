# Naming Conventions for Analysis Engine Service

This document outlines the naming conventions to be followed within the Analysis Engine Service codebase to ensure consistency, readability, and maintainability.

## General Principles

*   **Clarity and Conciseness:** Names should clearly indicate the purpose of the component, variable, or function without being excessively long.
*   **Consistency:** Apply the same naming patterns across the entire service.
*   **PEP 8 Compliance:** Adhere to Python's PEP 8 style guide for variable, function, method, and module naming.

## Component Naming

*   **Modules/Files:** Use lowercase `snake_case` (e.g., `technical_indicators.py`, `market_regime_analyzer.py`).
*   **Classes:** Use `PascalCase` (also known as `CapWords`).
    *   **Analyzers/Detectors:** Suffix with `Analyzer` (e.g., `TechnicalIndicatorAnalyzer`, `PatternRecognitionAnalyzer`, `MarketRegimeAnalyzer`).
    *   **Services:** Suffix with `Service` (e.g., `MultiAssetService`, `AnalysisIntegrationService`).
    *   **Repositories:** Suffix with `Repository` (e.g., `FeedbackRepository`).
    *   **Models/Data Structures:** Use descriptive `PascalCase` names (e.g., `MarketData`, `AnalysisResult`).
    *   **Exceptions:** Suffix with `Error` or `Exception` (e.g., `DataNotFoundError`, `ConfigurationError`).
    *   **Factories:** Suffix with `Factory` (e.g., `SchedulerFactory`).
    *   **Adapters:** Suffix with `Adapter` (e.g., `FeatureStoreAdapter`).
    *   **Integrators:** Suffix with `Integrator` (e.g., `MLPredictionIntegrator`).

## Variable and Function Naming

*   **Variables:** Use lowercase `snake_case` (e.g., `market_data`, `confidence_threshold`).
*   **Functions/Methods:** Use lowercase `snake_case` (e.g., `analyze_asset`, `_run_technical_analysis`).
*   **Private Methods/Attributes:** Prefix with a single underscore `_` (e.g., `_integrate_results`).
*   **Constants:** Use uppercase `SNAKE_CASE` (e.g., `DEFAULT_TIMEFRAME = '1h'`).

## API Endpoint Naming (RESTful Conventions)

*   **Resource-Oriented:** Use nouns to represent resources (e.g., `/analysis`, `/feedback`, `/assets`).
*   **Plural Nouns:** Prefer plural nouns for collections (e.g., `/assets` instead of `/asset`).
*   **Specific Resources:** Use resource IDs in the path for specific items (e.g., `/assets/{symbol}`).
*   **Actions:** Use standard HTTP methods (GET, POST, PUT, DELETE) to represent actions on resources. Avoid verbs in the URL path where possible (e.g., use `POST /analysis` instead of `/perform_analysis`).
*   **Versioning:** Prefix API routes with a version identifier (e.g., `/v1/analysis`).
*   **Path Parameters:** Use `snake_case` for path parameters (e.g., `/assets/{symbol}/analysis`).
*   **Query Parameters:** Use `snake_case` for query parameters (e.g., `/analysis?timeframe=1h&include_components=technical`).
*   **Router Tags:** Use descriptive tags in `PascalCase` or `Title Case` for grouping endpoints in documentation (e.g., `Tags=["Analysis